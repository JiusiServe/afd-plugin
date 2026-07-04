# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU FFN-side model runner for AFD execution."""

from __future__ import annotations

from typing import Any

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner, graph_capture

from afd_plugin.compat.ascend import (
    ascend_forward_context,
    ensure_vllm_config_has_afd_proxy,
    fail_if_unsupported_npu_afd_features,
    mirror_afd_metadata_on_forward_context,
)
from afd_plugin.compat.ascend.profiler import (
    create_afd_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDConnectorMetadata,
    AFDDPMetadataPayload,
    AFDMetadata,
    AFDRecvOutput,
)
from afd_plugin.v1.worker.attention_model_runner import (
    _resolve_world_ranks,
    _with_dp_derived_afd_rank,
)
from afd_plugin.v1.worker.cuda_graph import (
    AFDGraphRunMode,
    graph_run_mode,
    make_ffn_graph_key,
)
from afd_plugin.v1.worker.ffn_model_runner import _set_moe_layer_index

logger = init_logger(__name__)


class AFDNPUFFNModelRunner(NPUModelRunner):
    """Connector-driven NPU FFN runner for AFD execution."""

    afd_expected_role = "ffn"

    def __init__(self, vllm_config: object, device: object) -> None:
        afd_config = self.parse_config(vllm_config)
        ensure_vllm_config_has_afd_proxy(vllm_config, afd_config)
        super().__init__(vllm_config, device)

        self.afd_config = afd_config
        if not self.afd_config.enabled:
            raise ValueError("AFD NPU FFN runtime requires enabled=true")
        fail_if_unsupported_npu_afd_features(vllm_config)
        self.afd_config = _with_dp_derived_afd_rank(vllm_config, self.afd_config)
        rank, local_rank = _resolve_world_ranks()
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        self.num_layers = int(self.model_config.hf_config.num_hidden_layers)
        self.use_aclgraph = _use_npu_aclgraph(vllm_config, self)
        self._acl_graphs: dict[tuple, dict[str, Any]] = {}
        self.graph_pool = (
            current_platform.get_global_graph_pool() if self.use_aclgraph else None
        )
        self.prof = create_afd_npu_profiler("ffn")

    @staticmethod
    def parse_config(vllm_config: object) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="ffn")

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def get_kv_cache_spec(self) -> dict[str, Any]:
        return {}

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        return None

    def profile_run(self) -> None:
        return None

    def execute_ffn_step(
        self,
        *,
        dp_metadata_list: dict[int, Any],
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        if dp_metadata_list is None:
            raise RuntimeError("AFD NPU FFN requires dp_metadata_list")
        if bool(self.use_aclgraph) and (is_graph_capturing or is_warmup):
            logger.debug(
                "AFD NPU FFN execute_ffn_step enters capture_model; "
                "key=%s is_graph_capturing=%s is_warmup=%s",
                self._make_graph_key(dp_metadata_list),
                is_graph_capturing,
                is_warmup,
            )
            self.capture_model(
                dp_metadata_list=dp_metadata_list,
                is_warmup=is_warmup,
                is_attn_graph_capturing=is_graph_capturing,
            )
            return None
        self.execute_model(dp_metadata_list=dp_metadata_list)
        return None

    def execute_model(
        self,
        scheduler_output: Any = None,
        intermediate_tensors: Any = None,
        *,
        dp_metadata_list: dict[int, Any] | None = None,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        step_afd_npu_profiler(self.prof)
        if dp_metadata_list is None:
            raise RuntimeError("AFD NPU FFN is connector-driven")
        graph_key = self._make_graph_key(dp_metadata_list)
        acl_graphs = self._acl_graphs
        graph_info = acl_graphs.get(graph_key)
        graph_enabled = bool(self.use_aclgraph)
        run_mode = graph_run_mode(
            is_warmup=is_warmup and graph_enabled,
            is_graph_capturing=is_graph_capturing and graph_enabled,
            graph_enabled=graph_enabled,
            graph_exists=graph_info is not None,
        )
        _log_graph_key_lookup(
            graph_key=graph_key,
            graph_enabled=graph_enabled,
            graph_exists=graph_info is not None,
            run_mode=run_mode,
            cached_graph_count=len(acl_graphs),
        )
        if run_mode is AFDGraphRunMode.REPLAY:
            logger.debug(
                "AFD NPU FFN replaying ACL graph; key=%s cached_graphs=%d",
                graph_key,
                len(acl_graphs),
            )
            graph_info["graph"].replay()
            return None
        if run_mode in (AFDGraphRunMode.WARMUP, AFDGraphRunMode.CAPTURE):
            return self.execute_ffn_step(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
            )

        self._ffn_forward(dp_metadata_list=dp_metadata_list)
        return None

    def _make_graph_key(self, dp_metadata_list: dict[int, Any]) -> tuple:
        return make_ffn_graph_key(
            dp_metadata_list,
            attention_size=int(self.connector.attn_size),
            ffn_size=int(self.connector.ffn_size),
            fallback=int(self.max_num_tokens),
        )

    def _ffn_forward(
        self,
        *,
        dp_metadata_list: dict[int, Any],
        aclgraph_runtime_mode: Any = None,
        is_graph_capturing: bool = False,
        update_connector_state: bool = True,
    ) -> Any:
        if update_connector_state:
            self.connector.update_state_from_dp_metadata(
                _make_dp_metadata_payload(
                    dp_metadata_list,
                    is_graph_capturing=is_graph_capturing,
                ),
            )
        num_stages = max(len(dp_metadata_list), 1)
        afd_metadata = AFDMetadata(
            afd_tokens_start_loc=[],
            afd_reqs_start_loc=[],
            afd_stage_idx=0,
            afd_connector=self.connector,
            afd_tokens_lens=[],
            num_of_stages=num_stages,
        )
        stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
        num_tokens_across_dp = _ffn_token_counts_across_ranks(
            self.connector,
            dp_metadata_list,
            stage_ids[0],
            fallback=self.max_num_tokens,
        )
        num_tokens = _ffn_token_count_for_rank(self.connector, num_tokens_across_dp)
        rank_ffn_output = None

        # Build DP-level token counts for vLLM's forward context.
        # num_tokens_across_dp has ffn_size entries (AFD-level, one per
        # role_rank = dp_rank * tp_size + tp_rank), but vLLM's DPMetadata
        # expects dp_size entries where [dp_rank] equals batchsize.
        dp_num_tokens_across_dp = _to_dp_level_token_counts(
            num_tokens_across_dp,
            dp_size=int(self.vllm_config.parallel_config.data_parallel_size),
        )

        with ascend_forward_context(
            vllm_config=self.vllm_config,
            afd_metadata=afd_metadata,
            model_instance=self.model,
            num_tokens=num_tokens,
            num_tokens_across_dp=dp_num_tokens_across_dp,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
        ) as forward_context:
            for layer_idx in range(max(int(self.num_layers or 0), 1)):
                for stage_idx in stage_ids:
                    recv_output = self._recv_attn_output(stage_idx, layer_idx)
                    hidden_states, metadata, payload = _normalize_recv_output(
                        recv_output,
                        stage_idx=stage_idx,
                        layer_idx=layer_idx,
                    )
                    self.connector.update_metadata(metadata, payload)
                    metadata.layer_idx = layer_idx
                    metadata.stage_idx = stage_idx
                    if forward_context is not None:
                        forward_context.dp_metadata = dp_metadata_list.get(stage_idx)
                        mirror_afd_metadata_on_forward_context(
                            forward_context,
                            metadata,
                        )
                        _set_moe_layer_index(forward_context, layer_idx)

                    rank_ffn_output = self._run_ffn_computation(
                        hidden_states=hidden_states,
                        layer_idx=layer_idx,
                        group_list=payload.group_list,
                        dynamic_scales=payload.dynamic_scales,
                        topk_weights=payload.topk_weights,
                        topk_ids=payload.topk_ids,
                        router_logits=payload.router_logits,
                        row_idx=payload.row_idx,
                        x_active_mask=payload.x_active_mask,
                        cam_p2p_ep_name=payload.cam_p2p_ep_name or "",
                    )
                    self.connector.send_ffn_output(
                        rank_ffn_output,
                        metadata,
                        ubatch_idx=stage_idx,
                    )
        return rank_ffn_output

    def capture_model(
        self,
        dp_metadata_list: dict[int, Any] | None = None,
        is_warmup: bool = False,
        is_attn_graph_capturing: bool = True,
    ) -> int:
        if not self.use_aclgraph:
            return 0
        if dp_metadata_list is None:
            raise RuntimeError("AFD NPU FFN capture requires dp_metadata_list")

        logger.debug(
            "AFD NPU FFN capture_model start; key=%s is_warmup=%s "
            "is_attn_graph_capturing=%s",
            self._make_graph_key(dp_metadata_list),
            is_warmup,
            is_attn_graph_capturing,
        )
        start_free_memory = int(torch.npu.mem_get_info()[0])
        set_cudagraph_capturing_enabled(True)
        try:
            if is_warmup:
                self._ffn_forward(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=False,
                )
            else:
                with graph_capture(device=self.device):
                    self._capture_graphs(
                        aclgraph_runtime_mode=CUDAGraphMode.FULL,
                        dp_metadata_list=dp_metadata_list,
                        is_attn_graph_capturing=is_attn_graph_capturing,
                    )
        finally:
            set_cudagraph_capturing_enabled(False)

        end_free_memory = int(torch.npu.mem_get_info()[0])
        graph_size = max(0, int(start_free_memory - end_free_memory))
        logger.debug(
            "AFD NPU FFN capture_model end; key=%s graph_size=%d",
            self._make_graph_key(dp_metadata_list),
            graph_size,
        )
        return graph_size

    def _capture_graphs(
        self,
        *,
        aclgraph_runtime_mode: Any,
        dp_metadata_list: dict[int, Any],
        is_attn_graph_capturing: bool = True,
    ) -> None:
        graph_key = self._make_graph_key(dp_metadata_list)
        if graph_key in self._acl_graphs:
            logger.debug(
                "AFD NPU FFN ACL graph capture skipped for existing key=%s",
                graph_key,
            )
            return

        logger.debug("AFD NPU FFN capturing ACL graph for key=%s", graph_key)
        graph = torch.npu.NPUGraph()
        logger.debug("AFD NPU FFN created NPUGraph for key=%s", graph_key)
        self.connector.update_state_from_dp_metadata(
            _make_dp_metadata_payload(
                dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
            ),
        )
        logger.debug("AFD NPU FFN updated connector state for key=%s", graph_key)
        with torch.npu.graph(graph, pool=self.graph_pool):
            logger.debug(
                "AFD NPU FFN entered NPU graph context for key=%s",
                graph_key,
            )
            output = self._ffn_forward(
                dp_metadata_list=dp_metadata_list,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                is_graph_capturing=is_attn_graph_capturing,
                update_connector_state=False,
            )
            logger.debug("AFD NPU FFN left _ffn_forward for key=%s", graph_key)
        self._acl_graphs[graph_key] = {
            "graph": graph,
            "output": output,
        }
        logger.debug("AFD NPU FFN captured ACL graph for key=%s", graph_key)

    def _recv_attn_output(self, stage_idx: int, layer_idx: int) -> Any:
        logger.debug(
            "AFD NPU FFN recv_attn_output start; stage_idx=%d layer_idx=%d",
            stage_idx,
            layer_idx,
        )
        metadata = self.connector.create_recv_metadata(
            dp_metadata_list=self.connector.dp_metadata_list,
            ubatch_idx=stage_idx,
            layer_idx=layer_idx,
            max_num_tokens=self.max_num_tokens,
        )
        output = self.connector.recv_attn_output(
            metadata=metadata,
            ubatch_idx=stage_idx,
        )
        logger.debug(
            "AFD NPU FFN recv_attn_output end; stage_idx=%d layer_idx=%d",
            stage_idx,
            layer_idx,
        )
        return output

    def _run_ffn_computation(
        self,
        *,
        hidden_states: Any,
        layer_idx: int,
        group_list: Any = None,
        dynamic_scales: Any = None,
        topk_weights: Any = None,
        topk_ids: Any = None,
        router_logits: Any = None,
        row_idx: Any = None,
        x_active_mask: Any = None,
        cam_p2p_ep_name: str = "",
    ) -> Any:
        compute = self.model.compute_ffn_output
        return compute(
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            group_list=group_list,
            dynamic_scales=dynamic_scales,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
            row_idx=row_idx,
            x_active_mask=x_active_mask,
            cam_p2p_ep_name=cam_p2p_ep_name,
        )

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        raise RuntimeError("AFD NPU FFN runners do not sample tokens")

    def shutdown(self) -> None:
        stop_afd_npu_profiler(self.prof)
        self.connector.close()
        try:
            super().shutdown()
        except AttributeError:
            logger.debug("AFD NPU FFN parent model runner has no shutdown()")


def _normalize_recv_output(
    recv_output: Any,
    *,
    stage_idx: int,
    layer_idx: int,
) -> tuple[Any, AFDConnectorMetadata, AFDRecvOutput]:
    if isinstance(recv_output, tuple):
        hidden_states, metadata = recv_output
        payload = AFDRecvOutput(hidden_states=hidden_states, metadata=metadata)
        return hidden_states, metadata, payload
    hidden_states = recv_output.hidden_states
    metadata = recv_output.metadata
    if metadata is None:
        metadata = AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[
                1
                if getattr(hidden_states, "shape", None) is None
                else max(1, int(hidden_states.shape[0])),
            ],
        )
        recv_output.metadata = metadata
    return hidden_states, metadata, recv_output


def _make_dp_metadata_payload(
    dp_metadata_list: dict[int, Any],
    *,
    is_graph_capturing: bool = False,
    is_warmup: bool = False,
) -> AFDDPMetadataPayload:
    return AFDDPMetadataPayload(
        dp_metadata_list=dp_metadata_list,
        is_graph_capturing=is_graph_capturing,
        is_warmup=is_warmup,
    )


def _ffn_token_counts_across_ranks(
    connector: Any,
    dp_metadata_list: dict[int, Any],
    stage_idx: int,
    *,
    fallback: int,
) -> Any:
    dp_metadata = dp_metadata_list.get(int(stage_idx))
    if dp_metadata is None:
        values = [max(1, int(fallback))] * int(connector.ffn_size)
    else:
        attention_counts = _to_int_list(dp_metadata.num_tokens_across_dp_cpu)
        # Expand DP-level counts to AFD-level counts when TP > 1.
        # With TP, attn_size = num_attention_ranks includes TP workers
        # but num_tokens_across_dp_cpu only has dp_size entries.
        # Each DP rank's token count is replicated tp_size times because
        # all TP workers within the same DP rank process the same tokens.
        if (
            len(attention_counts) < int(connector.attn_size)
            and int(connector.attn_size) % len(attention_counts) == 0
        ):
            tp_size = int(connector.attn_size) // len(attention_counts)
            attention_counts = [
                attention_counts[i // tp_size] for i in range(int(connector.attn_size))
            ]
        if (
            len(attention_counts) >= int(connector.attn_size)
            and int(connector.attn_size) >= int(connector.ffn_size)
            and int(connector.attn_size) % int(connector.ffn_size) == 0
        ):
            group_size = int(connector.attn_size) // int(connector.ffn_size)
            values = [
                max(1, sum(attention_counts[idx * group_size : (idx + 1) * group_size]))
                for idx in range(int(connector.ffn_size))
            ]
        else:
            values = [max(1, int(fallback))] * int(connector.ffn_size)
    return torch.tensor(values, dtype=torch.int32, device="cpu")


def _ffn_token_count_for_rank(connector: Any, num_tokens_across_dp: Any) -> int:
    values = _to_int_list(num_tokens_across_dp)
    role_rank = int(connector.topology.role_rank)
    if role_rank >= len(values):
        return max(1, values[0] if values else 1)
    return max(1, int(values[role_rank]))


def _to_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item) for item in value.tolist()]


def _to_dp_level_token_counts(
    num_tokens_across_dp: Any,
    *,
    dp_size: int,
) -> Any:
    """Project AFD-level token counts back to DP-level for vLLM's forward context.

    ``num_tokens_across_dp`` has ``ffn_size`` entries (one per AFD role_rank).
    With TP > 1 each DP rank's count is replicated ``tp_size`` times, so the
    layout is ``[dp0, dp0, dp1, dp1, ...]``.  vLLM's ``DPMetadata.make()``
    expects exactly ``dp_size`` entries where ``[dp_rank] == batchsize``.
    """
    numel = len(num_tokens_across_dp)
    if numel == dp_size:
        return num_tokens_across_dp
    if dp_size <= 0 or numel % dp_size != 0:
        return num_tokens_across_dp
    tp_size = numel // dp_size
    # Take the first TP slot of each DP group (all TP slots are identical).
    indices = [dp_idx * tp_size for dp_idx in range(dp_size)]
    return num_tokens_across_dp[indices].contiguous()


def _use_npu_aclgraph(vllm_config: object, runner: object) -> bool:
    inherited = bool(runner.use_aclgraph)
    if bool(vllm_config.model_config.enforce_eager):
        return False

    mode_name = vllm_config.compilation_config.cudagraph_mode.name
    return inherited or mode_name in {
        "FULL",
        "FULL_AND_PIECEWISE",
        "FULL_DECODE_ONLY",
        "PIECEWISE",
    }


def _log_graph_key_lookup(
    *,
    graph_key: tuple,
    graph_enabled: bool,
    graph_exists: bool,
    run_mode: AFDGraphRunMode,
    cached_graph_count: int,
) -> None:
    if not graph_enabled:
        logger.debug("AFD NPU FFN ACL graph disabled; key=%s", graph_key)
        return

    if run_mode is AFDGraphRunMode.REPLAY:
        logger.debug(
            "AFD NPU FFN ACL graph key hit; key=%s cached_graphs=%d",
            graph_key,
            cached_graph_count,
        )
        return

    if run_mode is AFDGraphRunMode.EAGER:
        logger.debug(
            "AFD NPU FFN ACL graph key miss; key=%s cached_graphs=%d, "
            "falling back to eager",
            graph_key,
            cached_graph_count,
        )
        return

    logger.debug(
        "AFD NPU FFN ACL graph key lookup during %s; key=%s hit=%s cached_graphs=%d",
        run_mode.value,
        graph_key,
        graph_exists,
        cached_graph_count,
    )


__all__ = ["AFDNPUFFNModelRunner"]
