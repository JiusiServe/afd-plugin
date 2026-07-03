# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side model runner for the Phase 2 MVP."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import torch
import vllm.v1.worker.gpu_model_runner as gpu_model_runner
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_world_group,
)
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds

from afd_plugin.compat.profiler import (
    create_afd_gpu_profiler,
    step_afd_gpu_profiler,
    stop_afd_gpu_profiler,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDDPMetadata,
    AFDMetadata,
)
from afd_plugin.model_executor.models.forward_context import use_afd_metadata_provider
from afd_plugin.v1.worker.cuda_graph import validate_cuda_graph_mode
from afd_plugin.v1.worker.ubatch_wrapper import (
    AFDUBatchWrapper,
    build_ubatch_dp_metadata_list,
)


class AFDAttentionModelRunner(GPUModelRunner):
    """Attention model runner that injects AFD metadata into forward context."""

    afd_expected_role = "attention"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.afd_config = self.parse_config(self.vllm_config)
        if not self.afd_config.enabled:
            raise ValueError("AFD Attention runtime requires enabled=true")
        fail_if_unsupported_ubatching(self.vllm_config)
        self.afd_cudagraph_policy = validate_cuda_graph_mode(
            self.vllm_config,
            role="attention",
        )
        self.afd_config = _with_dp_derived_afd_rank(
            self.vllm_config,
            self.afd_config,
        )
        rank, local_rank = _resolve_world_ranks()
        self.afd_connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            self.vllm_config,
            self.afd_config,
        )
        self.afd_connector.init_afd_connector()
        self._is_warmup = False
        self._afd_is_graph_capturing = False
        self._afd_pending_metadata: AFDMetadata | None = None
        self._afd_suppress_metadata_send = False
        self._afd_transaction_counter = 0
        self.prof = create_afd_gpu_profiler("attention")

    @staticmethod
    def parse_config(vllm_config: object) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    def _build_afd_metadata(
        self,
        ubatch_slices: Any,
        num_tokens_unpadded: int,
    ) -> AFDMetadata:
        if ubatch_slices and len(ubatch_slices) > 1:
            afd_tokens_start_loc = [ub.token_slice.start for ub in ubatch_slices]
            afd_reqs_start_loc = [ub.request_slice.start for ub in ubatch_slices]
            afd_tokens_lens = [ub.num_tokens for ub in ubatch_slices]
            afd_tokens_unpadded_lens = [int(ub.num_tokens) for ub in ubatch_slices]
            num_of_stages = len(ubatch_slices)
        else:
            afd_tokens_start_loc = [0]
            afd_reqs_start_loc = [0]
            afd_tokens_lens = [num_tokens_unpadded]
            afd_tokens_unpadded_lens = [num_tokens_unpadded]
            num_of_stages = 1

        return AFDMetadata(
            afd_tokens_start_loc=afd_tokens_start_loc,
            afd_reqs_start_loc=afd_reqs_start_loc,
            afd_stage_idx=0,
            afd_connector=self.afd_connector,
            afd_tokens_lens=afd_tokens_lens,
            num_of_stages=num_of_stages,
            transaction_id=self._next_afd_transaction_id(),
            afd_tokens_unpadded_lens=afd_tokens_unpadded_lens,
        )

    def _send_dp_metadata(self, dp_metadata: Any, ubatch_slices: Any) -> None:
        if ubatch_slices and len(ubatch_slices) > 1:
            dp_metadata_list = {
                idx: metadata
                for idx, metadata in enumerate(
                    build_ubatch_dp_metadata_list(self.vllm_config, ubatch_slices),
                )
            }
        else:
            dp_metadata = self._ensure_dp_metadata(dp_metadata)
            dp_metadata_list = {0: dp_metadata}
        is_warmup = bool(self._is_warmup)
        is_graph_capturing = bool(getattr(self, "_afd_is_graph_capturing", False))
        self.afd_connector.update_state_from_dp_metadata(
            dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
        )

        should_send = True
        rank = self.afd_connector.world_rank
        should_send = bool(self.afd_connector.is_attn_top_min_size_rank(rank))

        if should_send:
            self.afd_connector.send_dp_metadata_list(
                dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
            )

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        use_ubatching = _is_ubatching_enabled(self.vllm_config)
        with _use_afd_ubatch_wrapper_during_load(use_ubatching):
            result = super().load_model(*args, **kwargs)
        if use_ubatching:
            self._install_afd_ubatch_wrapper()
        return result

    def _install_afd_ubatch_wrapper(self) -> None:
        if isinstance(self.model, AFDUBatchWrapper):
            self.model.configure_afd_context_provider(self)
            return

        model = self.model
        if isinstance(model, UBatchWrapper):
            model = model.unwrap()
        self.model = AFDUBatchWrapper(
            model,
            self.vllm_config,
            CUDAGraphMode.NONE,
            self.device,
        )
        self.model.configure_afd_context_provider(self)

    def _ensure_dp_metadata(self, dp_metadata: Any) -> Any:
        if dp_metadata is not None:
            return dp_metadata

        dp_size = int(self.vllm_config.parallel_config.data_parallel_size)
        if dp_size != 1:
            raise RuntimeError("AFD expected vLLM DPMetadata for attention DP > 1")

        if self._afd_pending_metadata is None:
            raise RuntimeError("AFD metadata is not available for DP metadata fallback")
        if len(self._afd_pending_metadata.afd_tokens_lens) != 1:
            raise RuntimeError("AFD DP=1 fallback only supports one stage")

        num_tokens = int(self._afd_pending_metadata.afd_tokens_lens[0])
        num_tokens_across_dp_cpu = torch.tensor(
            [num_tokens],
            dtype=torch.int32,
            device="cpu",
        )
        return AFDDPMetadata(
            num_tokens_across_dp_cpu=num_tokens_across_dp_cpu,
            max_tokens_across_dp_cpu=torch.max(num_tokens_across_dp_cpu),
        )

    def _build_capture_dp_metadata(self, num_tokens: int) -> Any:
        dp_size = int(self.vllm_config.parallel_config.data_parallel_size)
        num_tokens_across_dp_cpu = torch.full(
            (dp_size,),
            int(num_tokens),
            dtype=torch.int32,
            device="cpu",
        )
        if dp_size > 1:
            return DPMetadata.make(
                self.vllm_config.parallel_config,
                int(num_tokens),
                num_tokens_across_dp_cpu,
            )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp_cpu)
        return AFDDPMetadata(
            num_tokens_across_dp_cpu=num_tokens_across_dp_cpu,
            max_tokens_across_dp_cpu=max_tokens_across_dp_cpu,
        )

    def _install_afd_metadata_on_forward_context(
        self,
        forward_context: object,
    ) -> None:
        if getattr(forward_context, "additional_kwargs", None) is None:
            forward_context.additional_kwargs = {}
        existing_metadata = (
            getattr(forward_context, "additional_kwargs", {}) or {}
        ).get("afd_metadata")
        if existing_metadata is not None and _is_ubatch_child_afd_context(
            forward_context,
            existing_metadata,
        ):
            return

        if self._afd_pending_metadata is None:
            self._afd_pending_metadata = self._build_afd_metadata(
                forward_context.ubatch_slices,
                _forward_context_num_tokens(forward_context, self.vllm_config),
            )
        if self._afd_pending_metadata is not None:
            forward_context.additional_kwargs["afd_metadata"] = (
                self._afd_pending_metadata
            )
        if bool(getattr(self, "_afd_suppress_metadata_send", False)):
            return
        dp_metadata = forward_context.dp_metadata
        ubatch_slices = forward_context.ubatch_slices
        padded_graph_tokens = _full_cudagraph_padded_tokens(forward_context)
        if padded_graph_tokens is not None and not ubatch_slices:
            dp_metadata = self._build_capture_dp_metadata(padded_graph_tokens)
        self._send_dp_metadata(dp_metadata, ubatch_slices)

    def _build_attention_metadata(self, *args: Any, **kwargs: Any) -> Any:
        num_tokens = kwargs.get("num_tokens", 0)
        ubatch_slices = kwargs.get("ubatch_slices")
        self._afd_pending_metadata = self._build_afd_metadata(
            ubatch_slices,
            int(num_tokens),
        )
        return super()._build_attention_metadata(*args, **kwargs)

    def _determine_batch_execution_and_padding(self, *args: Any, **kwargs: Any) -> Any:
        result = super()._determine_batch_execution_and_padding(*args, **kwargs)
        (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        ) = result
        values = _batch_execution_values(args, kwargs)
        if should_ubatch and not _has_enough_tokens_for_ubatches(
            self.vllm_config,
            int(values.get("num_tokens", 0)),
        ):
            should_ubatch = False
            return (
                cudagraph_mode,
                batch_descriptor,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            )
        if should_ubatch:
            return result
        should_ubatch = self._should_ubatch_without_vllm_dp(*args, **kwargs)
        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _should_ubatch_without_vllm_dp(self, *args: Any, **kwargs: Any) -> bool:
        parallel_config = self.vllm_config.parallel_config
        if int(parallel_config.data_parallel_size) > 1:
            return False
        if not bool(parallel_config.use_ubatching):
            return False
        if not bool(kwargs.get("allow_microbatching", True)):
            return False

        values = _batch_execution_values(args, kwargs)
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=values["max_num_scheduled_tokens"],
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=values["num_tokens"],
            num_reqs=values["num_reqs"],
            force_uniform_decode=values.get("force_uniform_decode"),
        )
        if not _has_enough_tokens_for_ubatches(
            self.vllm_config,
            int(values["num_tokens"]),
        ):
            return False
        return _check_ubatch_thresholds(
            parallel_config,
            int(values["num_tokens"]),
            bool(uniform_decode),
        )

    def _model_forward(self, *args: Any, **kwargs: Any) -> Any:
        forward_context = get_forward_context()
        self._install_afd_metadata_on_forward_context(forward_context)
        return super()._model_forward(*args, **kwargs)

    def execute_model(self, *args: Any, **kwargs: Any) -> Any:
        step_afd_gpu_profiler(self.prof)
        return super().execute_model(*args, **kwargs)

    def _dummy_run(self, *args: Any, **kwargs: Any) -> Any:
        """Run vLLM's DP dummy batch through the AFD model path.

        vLLM uses ``execute_dummy_batch`` on idle DP ranks while another DP rank
        is serving a request. The native dummy path calls the model directly,
        bypassing ``_model_forward()``, so we provide AFD metadata lazily when
        the plugin-owned model reads the current forward context. Do not force
        native attention metadata here: profiling dummy runs can happen before
        vLLM initializes ``kv_cache_config``.
        """

        previous_metadata = self._afd_pending_metadata
        previous_is_graph_capturing = getattr(
            self,
            "_afd_is_graph_capturing",
            False,
        )
        self._afd_is_graph_capturing = bool(
            kwargs.get("is_graph_capturing", False),
        )
        try:
            with use_afd_metadata_provider(self):
                return super()._dummy_run(*args, **kwargs)
        finally:
            self._afd_is_graph_capturing = previous_is_graph_capturing
            self._afd_pending_metadata = previous_metadata

    def _warmup_and_capture(self, *args: Any, **kwargs: Any) -> Any:
        """Mirror vLLM warmup/capture while marking AFD warmup metadata.

        The native implementation calls ``self._dummy_run`` for warmups and
        formal capture. We keep that flow intact and only set ``_is_warmup``
        around the warmup calls so FFN ranks can distinguish warmup metadata
        from graph-capture metadata.
        """

        names = [
            "desc",
            "cudagraph_runtime_mode",
            "profile_seq_lens",
            "allow_microbatching",
            "num_warmups",
        ]
        values = dict(zip(names, args, strict=False))
        values.update(kwargs)
        desc = values.get("desc")
        cudagraph_runtime_mode = values.get("cudagraph_runtime_mode")
        if desc is None or cudagraph_runtime_mode is None:
            return super()._warmup_and_capture(*args, **kwargs)

        num_warmups = values.get("num_warmups")
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups
        allow_microbatching = bool(values.get("allow_microbatching", False))
        profile_seq_lens = values.get("profile_seq_lens")
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL

        previous_is_warmup = bool(self._is_warmup)
        try:
            self._is_warmup = True
            for _ in range(int(num_warmups)):
                self._dummy_run(
                    desc.num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    force_attention=force_attention,
                    uniform_decode=desc.uniform,
                    allow_microbatching=allow_microbatching,
                    skip_eplb=True,
                    remove_lora=False,
                    num_active_loras=desc.num_active_loras,
                )
        finally:
            self._is_warmup = previous_is_warmup

        previous_metadata = self._afd_pending_metadata
        previous_suppress_send = self._afd_suppress_metadata_send
        previous_is_graph_capturing = self._afd_is_graph_capturing
        try:
            # DP metadata transfer is a control-plane side effect.  The original
            # AFD path sends it before formal CUDA graph capture so the capture
            # contains only replayable model/data-plane work.
            self._afd_is_graph_capturing = True
            if allow_microbatching:
                # The AFD-aware ubatch wrapper builds the exact padded ubatch
                # slices used by vLLM and sends per-stage DP metadata before it
                # enters torch.cuda.graph(...).  Avoid sending a single-stage
                # capture payload here.
                self._afd_pending_metadata = None
                self._afd_suppress_metadata_send = False
            else:
                self._afd_pending_metadata = self._build_afd_metadata(
                    None,
                    int(desc.num_tokens),
                )
                self._send_dp_metadata(
                    self._build_capture_dp_metadata(int(desc.num_tokens)),
                    None,
                )
                self._afd_suppress_metadata_send = True
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                uniform_decode=desc.uniform,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
                is_graph_capturing=True,
                profile_seq_lens=profile_seq_lens,
            )
        finally:
            self._afd_is_graph_capturing = previous_is_graph_capturing
            self._afd_suppress_metadata_send = previous_suppress_send
            self._afd_pending_metadata = previous_metadata

    def shutdown(self) -> None:
        stop_afd_gpu_profiler(self.prof)
        self.afd_connector.close()
        super().shutdown()

    def _next_afd_transaction_id(self) -> str:
        counter = self._afd_transaction_counter
        self._afd_transaction_counter = counter + 1
        return f"afd-{counter}"


def fail_if_unsupported_ubatching(vllm_config: object) -> None:
    parallel_config = vllm_config.parallel_config
    num_ubatches = int(parallel_config.num_ubatches)
    if _is_ubatching_enabled(vllm_config) and num_ubatches != 2:
        raise RuntimeError(
            "AFD Phase 5 currently supports exactly two ubatches; "
            f"got num_ubatches={num_ubatches}",
        )


fail_if_ubatching_enabled = fail_if_unsupported_ubatching


def fail_if_cuda_graph_enabled(vllm_config: object) -> None:
    validate_cuda_graph_mode(vllm_config)


def _resolve_world_ranks() -> tuple[int, int]:
    group = get_world_group()
    return int(group.rank), int(group.local_rank)


def _is_ubatch_child_afd_context(
    forward_context: object,
    afd_metadata: object,
) -> bool:
    if getattr(forward_context, "ubatch_slices", None) is not None:
        return False
    if int(getattr(afd_metadata, "num_of_stages", 1) or 1) <= 1:
        return False
    return len(getattr(afd_metadata, "afd_tokens_lens", []) or []) == 1


def _with_dp_derived_afd_rank(
    vllm_config: object,
    afd_config: AFDConfig,
) -> AFDConfig:
    parallel_config = vllm_config.parallel_config
    dp_size = int(parallel_config.data_parallel_size)
    tp_size = int(parallel_config.tensor_parallel_size)
    if dp_size <= 1 and tp_size <= 1:
        return afd_config
    dp_rank = int(parallel_config.data_parallel_rank) if dp_size > 1 else 0
    tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
    role_size = (
        afd_config.num_attention_servers
        if afd_config.role == "attention"
        else afd_config.num_ffn_servers
    )
    role_rank = afd_config.afd_server_rank + dp_rank * tp_size + tp_rank
    if role_rank >= role_size:
        raise ValueError(
            "AFD role rank derived from distributed ranks is out of range: "
            f"base={afd_config.afd_server_rank}, dp_rank={dp_rank}, "
            f"tp_rank={tp_rank}, role_size={role_size}",
        )
    return replace(afd_config, afd_server_rank=role_rank)


def _is_ubatching_enabled(vllm_config: object) -> bool:
    return bool(vllm_config.parallel_config.use_ubatching)


def _check_ubatch_thresholds(
    parallel_config: object,
    num_tokens: int,
    uniform_decode: bool,
) -> bool:
    return bool(check_ubatch_thresholds(parallel_config, num_tokens, uniform_decode))


def _batch_execution_values(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    names = [
        "num_tokens",
        "num_reqs",
        "num_scheduled_tokens_np",
        "max_num_scheduled_tokens",
        "use_cascade_attn",
        "allow_microbatching",
        "force_eager",
        "force_uniform_decode",
    ]
    values = dict(zip(names, args, strict=False))
    values.update(kwargs)
    return values


def _has_enough_tokens_for_ubatches(vllm_config: object, num_tokens: int) -> bool:
    num_ubatches = int(vllm_config.parallel_config.num_ubatches)
    return int(num_tokens) >= max(num_ubatches, 1)


def _forward_context_num_tokens(
    forward_context: object,
    vllm_config: object,
) -> int:
    dp_metadata = forward_context.dp_metadata
    dp_rank = int(vllm_config.parallel_config.data_parallel_rank)
    if dp_metadata is not None:
        return max(1, int(dp_metadata.num_tokens_across_dp_cpu[dp_rank]))

    return max(1, int(forward_context.batch_descriptor.num_tokens))


def _full_cudagraph_padded_tokens(forward_context: object) -> int | None:
    mode = getattr(forward_context, "cudagraph_runtime_mode", None)
    name = getattr(mode, "name", None)
    if isinstance(name, str):
        is_full = name == "FULL"
    else:
        is_full = str(mode).rsplit(".", 1)[-1] == "FULL"
    if not is_full:
        return None
    batch_descriptor = getattr(forward_context, "batch_descriptor", None)
    num_tokens = getattr(batch_descriptor, "num_tokens", None)
    return None if num_tokens is None else max(1, int(num_tokens))


@contextmanager
def _use_afd_ubatch_wrapper_during_load(enabled: bool):
    if not enabled:
        yield
        return

    original = gpu_model_runner.UBatchWrapper
    gpu_model_runner.UBatchWrapper = AFDUBatchWrapper
    try:
        yield
    finally:
        gpu_model_runner.UBatchWrapper = original


__all__ = [
    "AFDAttentionModelRunner",
    "fail_if_cuda_graph_enabled",
    "fail_if_ubatching_enabled",
    "fail_if_unsupported_ubatching",
]
