# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-side model runner for AFD GPU execution."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode
from vllm.config import update_config as update_vllm_config
from vllm.distributed.parallel_state import get_world_group, graph_capture
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from afd_plugin.compat.profiler import (
    create_afd_gpu_profiler,
    step_afd_gpu_profiler,
    stop_afd_gpu_profiler,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDConnectorMetadata,
    AFDRecvOutput,
)
from afd_plugin.v1.worker.attention_model_runner import (
    _with_dp_derived_afd_rank,
    fail_if_unsupported_ubatching,
)
from afd_plugin.v1.worker.cuda_graph import (
    AFDGraphRunMode,
    graph_run_mode,
    make_ffn_graph_key,
    validate_cuda_graph_mode,
)


class GPUFFNModelRunner(LoRAModelRunnerMixin):
    """FFN model runner for connector-driven AFD GPU execution."""

    afd_expected_role = "ffn"

    def __init__(self, vllm_config: object, device: object) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.load_config = vllm_config.load_config
        self.device = device
        self.dtype = self.model_config.dtype
        self.afd_config = self.parse_config(vllm_config)
        if not self.afd_config.enabled:
            raise ValueError("AFD FFN runtime requires enabled=true")
        fail_if_unsupported_ubatching(vllm_config)
        self.afd_cudagraph_policy = validate_cuda_graph_mode(
            vllm_config,
            role="ffn",
        )
        self.afd_config = _with_dp_derived_afd_rank(vllm_config, self.afd_config)

        rank, local_rank = _resolve_world_ranks()
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        self.model: Any | None = None
        self.model_memory_usage = 0
        self.num_layers = int(self.model_config.hf_config.num_hidden_layers)
        self.use_cuda_graph = bool(
            self.afd_cudagraph_policy.enable_ffn_graph_cache,
        )
        self._cuda_graphs: dict[tuple, dict[str, Any]] = {}
        self._graph_memory_pool: Any | None = None
        self.prof = create_afd_gpu_profiler("ffn")

    @staticmethod
    def parse_config(vllm_config: object) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="ffn")

    def get_model(self) -> Any:
        return self.model

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def load_model(self, *, load_dummy_weights: bool = False, **kwargs: Any) -> None:
        """Load the vLLM model."""

        model_loader = get_model_loader(self.load_config)
        with DeviceMemoryProfiler() as profiler:
            if self.model is None:
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config,
                    model_config=self.model_config,
                )
            else:
                model_loader.load_weights(
                    self.model,
                    model_config=self.model_config,
                )
        self.model_memory_usage = profiler.consumed_memory

    def profile_run(self) -> None:
        return None

    def get_kv_cache_spec(self) -> dict[str, Any]:
        return {}

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
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
        step_afd_gpu_profiler(self.prof)
        if dp_metadata_list is None:
            raise RuntimeError("GPUFFNModelRunner requires dp_metadata_list")
        graph_key = make_ffn_graph_key(dp_metadata_list)
        cuda_graph_info = self._cuda_graphs.get(graph_key)
        run_mode = graph_run_mode(
            is_warmup=is_warmup,
            is_graph_capturing=is_graph_capturing,
            graph_enabled=bool(self.use_cuda_graph),
            graph_exists=cuda_graph_info is not None,
        )
        if run_mode is AFDGraphRunMode.REPLAY:
            cuda_graph_info["graph"].replay()
            return None

        self._ffn_forward(
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
        )
        return None

    def _ffn_forward(
        self,
        *,
        dp_metadata_list: dict[int, Any],
        is_graph_capturing: bool = False,
        update_connector_state: bool = True,
    ) -> Any:
        if update_connector_state:
            self.connector.update_state_from_dp_metadata(
                dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
            )

        rank_ffn_output = None
        num_layers = max(int(self.num_layers or 0), 1)
        stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
        with _ffn_forward_context(self.vllm_config) as forward_context:
            for layer_idx in range(num_layers):
                for stage_idx in stage_ids:
                    recv_output = self._recv_attn_output(stage_idx)
                    hidden_states, metadata, _payload = _normalize_recv_output(
                        recv_output,
                        stage_idx=stage_idx,
                        layer_idx=layer_idx,
                    )
                    metadata.layer_idx = layer_idx
                    metadata.stage_idx = stage_idx
                    if forward_context is not None:
                        forward_context.dp_metadata = dp_metadata_list.get(
                            metadata.stage_idx,
                        )
                        forward_context.additional_kwargs["afd_metadata"] = metadata
                        _set_moe_layer_index(forward_context, layer_idx)
                    recv_handle_list = metadata.recv_handle_list
                    if recv_handle_list is not None:
                        for work in recv_handle_list:
                            work.wait()
                        metadata.recv_handle_list = None
                    rank_ffn_output = self._execute_eager_mode(hidden_states, layer_idx)
                    self.connector.send_ffn_output(rank_ffn_output, metadata)
        return rank_ffn_output

    def _execute_eager_mode(self, hidden_states: Any, layer_idx: int) -> Any:
        model = self.model
        compute = getattr(model, "compute_ffn_output", None)
        if callable(compute):
            return compute(hidden_states, layer_idx)
        return hidden_states

    def _recv_attn_output(self, stage_idx: int) -> Any:
        try:
            return self.connector.recv_attn_output(ubatch_idx=stage_idx)
        except TypeError:
            return self.connector.recv_attn_output()

    def update_config(self, overrides: dict[str, Any]) -> None:
        for config_name, config_overrides in overrides.items():
            config = getattr(self, config_name)
            updated_config = update_vllm_config(config, config_overrides)
            setattr(self, config_name, updated_config)

    def reload_weights(self) -> None:
        if self.model is None:
            raise RuntimeError("Cannot reload weights before model is loaded")
        self.load_model()

    def _dummy_run(
        self,
        cudagraph_runtime_mode: Any,
        dp_metadata_list: dict[int, Any],
        is_attn_graph_capturing: bool,
    ) -> None:
        mode_name = getattr(cudagraph_runtime_mode, "name", str(cudagraph_runtime_mode))
        if mode_name.endswith(".FULL"):
            mode_name = "FULL"

        if mode_name == "FULL":
            if self._graph_memory_pool is None:
                self._graph_memory_pool = torch.cuda.graph_pool_handle()
            graph_key = make_ffn_graph_key(dp_metadata_list)
            cudagraph = torch.cuda.CUDAGraph()
            # DP metadata receive/update is a control-plane side effect and must
            # complete before CUDA graph capture starts.
            self.connector.update_state_from_dp_metadata(
                dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
            )
            with torch.cuda.graph(cudagraph, pool=self._graph_memory_pool):
                output = self._ffn_forward(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=is_attn_graph_capturing,
                    update_connector_state=False,
                )
            self._cuda_graphs[graph_key] = {
                "graph": cudagraph,
                "input_hidden_states": output,
                "output": output,
            }
        else:
            self._ffn_forward(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
            )

    def capture_model(
        self,
        dp_metadata_list: dict[int, Any] | None = None,
        is_warmup: bool = False,
        is_attn_graph_capturing: bool = True,
    ) -> int:
        if not self.use_cuda_graph:
            return 0
        if dp_metadata_list is None:
            raise RuntimeError("GPUFFNModelRunner.capture_model requires metadata")

        start_free_gpu_memory = torch.cuda.mem_get_info()[0]
        if self._graph_memory_pool is None:
            self._graph_memory_pool = torch.cuda.graph_pool_handle()

        set_cudagraph_capturing_enabled(True)
        try:
            with graph_capture(device=self.device):
                if is_warmup:
                    self.connector.update_state_from_dp_metadata(
                        dp_metadata_list,
                        is_graph_capturing=False,
                    )
                    self._ffn_forward(
                        dp_metadata_list=dp_metadata_list,
                        is_graph_capturing=False,
                        update_connector_state=False,
                    )
                else:
                    self._dummy_run(
                        cudagraph_runtime_mode=CUDAGraphMode.FULL,
                        dp_metadata_list=dp_metadata_list,
                        is_attn_graph_capturing=is_attn_graph_capturing,
                    )
        finally:
            set_cudagraph_capturing_enabled(False)

        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        return int(cuda_graph_size)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        raise RuntimeError("FFN runners do not sample tokens")

    def add_lora(self, lora_request: Any) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    @property
    def lora_config(self) -> None:
        return None

    @property
    def is_pooling_model(self) -> bool:
        return False

    def get_supported_tasks(self) -> tuple[Any, ...]:
        return ()

    def shutdown(self) -> None:
        stop_afd_gpu_profiler(self.prof)
        self.connector.close()


def _resolve_world_ranks() -> tuple[int, int]:
    group = get_world_group()
    return int(group.rank), int(group.local_rank)


@contextmanager
def _ffn_forward_context(vllm_config: object):
    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        yield get_forward_context()


def _set_moe_layer_index(forward_context: object, layer_idx: int) -> None:
    all_moe_layers = forward_context.all_moe_layers
    if not all_moe_layers:
        return

    target = f".layers.{int(layer_idx)}."
    for idx, layer_name in enumerate(all_moe_layers):
        if target in f".{layer_name}.":
            forward_context.moe_layer_index = idx
            return


def _normalize_recv_output(
    recv_output: Any,
    *,
    stage_idx: int,
    layer_idx: int,
) -> tuple[Any, AFDConnectorMetadata, Any]:
    if isinstance(recv_output, tuple):
        hidden_states, metadata = recv_output
        return hidden_states, metadata, recv_output

    if isinstance(recv_output, AFDRecvOutput):
        return recv_output.hidden_states, recv_output.metadata, recv_output

    hidden_states = recv_output.hidden_states
    metadata = getattr(recv_output, "metadata", None)
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
    return hidden_states, metadata, recv_output


__all__ = ["GPUFFNModelRunner"]
