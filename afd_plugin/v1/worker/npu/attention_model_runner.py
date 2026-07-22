# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Thin NPU Attention-side ModelRunner for AFD execution."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    BatchDescriptor,
    DPMetadata,
    ForwardContext,
    get_forward_context,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.worker.model_runner_v1 import (
    NPUModelRunner,
    PerLayerAttnMetadata,
)

from afd_plugin.compat.npu import (
    assert_vllm_ascend_version_supported,
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.compat.npu.profiler import (
    create_afd_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
)
from afd_plugin.compat.npu.v0191rc1.attention_metadata_fanout import (
    AttentionMetadataFanoutV0191rc1,
)
from afd_plugin.compat.npu.v0191rc1.dp_coordination import (
    NPUDBOBatchDecisionV0191rc1,
)
from afd_plugin.compat.npu.v0191rc1.model_runner import (
    AscendDBOCompatV0191rc1,
)
from afd_plugin.config import AFD_ASYNC_CONNECTOR, AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDDPMetadata,
    AFDForwardContextMetadata,
)
from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo
from afd_plugin.v1.worker.attention_model_runner import (
    _resolve_world_ranks,
    _with_dp_derived_afd_rank,
)
from afd_plugin.v1.worker.npu.async_moe_ubatch import AsyncMoeUbatch
from afd_plugin.v1.worker.npu.attention_metadata_adapter import (
    AFDNPUAttentionMetadataAdapter,
)
from afd_plugin.v1.worker.npu.graph_capture import AFDNPUGraphCapture
from afd_plugin.v1.worker.npu.npu_ubatch_wrapper import AscendUBatchWrapper
from afd_plugin.v1.worker.npu.ubatch_plan import (
    UbatchMode,
    UbatchPlanScope,
    ensure_ubatch_plan_scope,
    get_ubatch_plan,
    install_ubatch_plan_on_forward_context,
)
from afd_plugin.v1.worker.npu.ubatch_utils import pad_out_ubatch_slices


class AFDNPUAttentionModelRunner(NPUModelRunner):
    """Lifecycle hooks joining AFD behavior to the pinned Ascend adapter."""

    afd_expected_role = "attention"

    def __init__(self, vllm_config: VllmConfig, device: object) -> None:
        assert_vllm_ascend_version_supported()
        afd_config = self.parse_config(vllm_config)
        super().__init__(vllm_config, device)

        self.afd_config = afd_config
        fail_if_unsupported_npu_afd_features(
            vllm_config,
            afd_config=afd_config,
        )
        self.afd_config = _with_dp_derived_afd_rank(vllm_config, self.afd_config)
        rank, local_rank = _resolve_world_ranks()
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        self.afd_async_extra_info = AFDAsyncExtraInfo()
        if afd_config.connector == AFD_ASYNC_CONNECTOR:
            connector_extra_info = self.connector.extra_info
            if not isinstance(connector_extra_info, AFDAsyncExtraInfo):
                raise TypeError(
                    "CAMAsyncAFDConnector requires AFDAsyncExtraInfo, got "
                    f"{type(connector_extra_info).__name__}",
                )
            self.afd_async_extra_info = connector_extra_info
        self.connector.init_afd_connector()
        self._is_warmup = False
        self._afd_is_graph_capturing = False
        self._afd_pending_metadata: AFDForwardContextMetadata | None = None
        self._afd_suppress_metadata_send = False
        self._afd_transaction_counter = 0
        self._afd_async_moe_ubatch_metadata = None
        self._afd_ubatch_plan_scope: UbatchPlanScope | None = None
        self.prof = create_afd_npu_profiler("attention")

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    # Patch reason: the upstream execute hook does not step the AFD profiler.
    # Patch functionality: step once, then preserve the upstream call unchanged.
    # Signature: matches vLLM-Ascend v0.19.1rc1.
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",  # noqa: UP037
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        # ### PATCH START: AFD profiler step
        step_afd_npu_profiler(self.prof)
        # ### PATCH END: AFD profiler step
        # ### PATCH START: one explicit uBatch plan per execution
        with UbatchPlanScope(self):
            return super().execute_model(scheduler_output, intermediate_tensors)
        # ### PATCH END: one explicit uBatch plan per execution

    # Patch reason: AFD metadata must be installed before the upstream model call.
    # Patch functionality: decorate the forward context, then delegate upstream.
    # Signature: matches vLLM-Ascend v0.19.1rc1.
    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ):
        # ### PATCH START: AFD forward context and Ascend DBO model call
        forward_context = get_forward_context()
        install_ubatch_plan_on_forward_context(self, forward_context)
        try:
            forward_context.dbo_enabled = bool(forward_context.dbo_enabled)
        except AttributeError:
            forward_context.dbo_enabled = False
        self._install_afd_metadata_on_forward_context(forward_context)
        self._install_async_moe_ubatch_metadata_on_forward_context(forward_context)

        return super()._model_forward(
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )
        # ### PATCH END: AFD forward context and Ascend DBO model call

    # Patch reason: the pinned upstream model-forward always performs the final
    # flash-comm all-gather, while the AFD uBatch wrapper merges stage outputs.
    # Patch functionality: suppress only that all-gather while DBO is active.
    # Signature: matches vLLM-Ascend v0.19.1rc1.
    @staticmethod
    def _all_gather_hidden_states_and_aux(hidden_states):
        # ### PATCH START: skip duplicate DBO flash-comm gather
        forward_context = get_forward_context()
        if bool(forward_context.dbo_enabled):
            return hidden_states
        # ### PATCH END: skip duplicate DBO flash-comm gather
        return NPUModelRunner._all_gather_hidden_states_and_aux(hidden_states)

    # Patch reason: the pinned Ascend runner does not build per-uBatch metadata.
    # Patch functionality: select native, async-MoE, or upstream metadata path.
    # Signature: matches vLLM-Ascend v0.19.1rc1.
    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        num_scheduled_tokens_np: np.ndarray | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        # ### PATCH START: AFD NPU uBatch metadata dispatch
        plan = get_ubatch_plan(self)
        if ubatch_slices is None and plan.mode is UbatchMode.NATIVE_DBO:
            ubatch_slices = (
                plan.padded_ubatch_slices
                if for_cudagraph_capture
                else plan.ubatch_slices
            )
        values = {
            "num_tokens": num_tokens,
            "num_reqs": num_reqs,
            "max_query_len": max_query_len,
            "num_tokens_padded": num_tokens_padded,
            "num_reqs_padded": num_reqs_padded,
            "ubatch_slices": ubatch_slices,
            "logits_indices": logits_indices,
            "use_spec_decode": use_spec_decode,
            "for_cudagraph_capture": for_cudagraph_capture,
            "num_scheduled_tokens": num_scheduled_tokens,
            "num_scheduled_tokens_np": num_scheduled_tokens_np,
            "cascade_attn_prefix_lens": cascade_attn_prefix_lens,
        }
        ubatch_slices = _normalize_metadata_ubatch_slices(ubatch_slices, values)
        values["ubatch_slices"] = ubatch_slices
        if self.afd_async_extra_info.async_moe_ubatching:
            return AsyncMoeUbatch.build_attention_metadata(
                self,
                values,
            )
        self._afd_pending_metadata = self._build_afd_metadata(
            ubatch_slices,
            num_tokens,
        )
        scope = self._afd_ubatch_plan_scope
        if scope is not None and scope.plan.mode is UbatchMode.NATIVE_DBO:
            if for_cudagraph_capture:
                scope.set_plan(
                    scope.plan.with_slices(
                        scope.plan.ubatch_slices,
                        ubatch_slices,
                    )
                )
            else:
                scope.set_plan(
                    scope.plan.with_slices(
                        ubatch_slices,
                        scope.plan.padded_ubatch_slices,
                    )
                )
        if ubatch_slices is not None:
            return self._build_attention_metadata_with_ubatches(**values)
        # ### PATCH END: AFD NPU uBatch metadata dispatch
        return super()._build_attention_metadata(**values)

    def _build_attention_metadata_with_ubatches(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        num_scheduled_tokens_np: np.ndarray | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
    ) -> tuple[list[dict[str, Any]], CommonAttentionMetadata | None]:
        return AttentionMetadataFanoutV0191rc1.build(
            self,
            num_tokens,
            num_reqs,
            max_query_len,
            num_tokens_padded,
            num_reqs_padded,
            ubatch_slices,
            logits_indices,
            use_spec_decode,
            for_cudagraph_capture,
            num_scheduled_tokens,
            num_scheduled_tokens_np,
            cascade_attn_prefix_lens,
        )

    # Patch reason: the pinned Ascend dummy run does not support AFD NPU DBO.
    # Patch functionality: scope the plan/capture flags around the upstream run.
    # Signature: matches vLLM-Ascend v0.19.1rc1.
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
        profile_cpp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ### PATCH START: AFD NPU uBatch dummy run
        previous_graph_capturing = self._afd_is_graph_capturing
        self._afd_is_graph_capturing = is_graph_capturing
        with ensure_ubatch_plan_scope(self), torch.inference_mode():
            try:
                return super()._dummy_run(
                    num_tokens,
                    with_prefill=with_prefill,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    force_attention=force_attention,
                    uniform_decode=uniform_decode,
                    is_profile=is_profile,
                    create_mixed_batch=create_mixed_batch,
                    allow_microbatching=allow_microbatching and not is_profile,
                    skip_eplb=skip_eplb,
                    remove_lora=remove_lora,
                    is_graph_capturing=is_graph_capturing,
                    num_active_loras=num_active_loras,
                    profile_seq_lens=profile_seq_lens,
                    profile_cpp=profile_cpp,
                )
            finally:
                self._afd_is_graph_capturing = previous_graph_capturing
                self._afd_pending_metadata = None
                self._afd_async_moe_ubatch_metadata = None
        # ### PATCH END: AFD NPU uBatch dummy run

    # Patch reason: AFD captures separate single-stage and uBatch graph keys.
    # Patch functionality: delegate capture ordering and control-plane flags.
    # Signature: matches vLLM v0.19.1.
    def _warmup_and_capture(
        self,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None = None,
        allow_microbatching: bool = False,
        num_warmups: int | None = None,
    ) -> None:
        AFDNPUGraphCapture.warmup_and_capture(
            self,
            desc,
            cudagraph_runtime_mode,
            profile_seq_lens,
            allow_microbatching,
            num_warmups,
        )

    def _build_afd_metadata(
        self,
        ubatch_slices: UBatchSlices | None,
        num_tokens_unpadded: int,
    ) -> AFDForwardContextMetadata:
        return AFDNPUAttentionMetadataAdapter._build_afd_metadata(
            self,
            ubatch_slices,
            num_tokens_unpadded,
        )

    def _install_afd_metadata_on_forward_context(
        self,
        forward_context: ForwardContext,
    ) -> None:
        AFDNPUAttentionMetadataAdapter._install_afd_metadata_on_forward_context(
            self,
            forward_context,
        )

    def _install_async_moe_ubatch_metadata_on_forward_context(
        self,
        forward_context: ForwardContext,
    ) -> None:
        AsyncMoeUbatch._install_async_moe_ubatch_metadata_on_forward_context(
            self,
            forward_context,
        )

    def _send_dp_metadata(
        self,
        dp_metadata: DPMetadata | AFDDPMetadata | None,
        ubatch_slices: UBatchSlices | None,
    ) -> None:
        AFDNPUAttentionMetadataAdapter._send_dp_metadata(
            self, dp_metadata, ubatch_slices
        )

    def _ensure_dp_metadata(
        self,
        dp_metadata: DPMetadata | AFDDPMetadata | None,
    ) -> DPMetadata | AFDDPMetadata:
        return AFDNPUAttentionMetadataAdapter._ensure_dp_metadata(self, dp_metadata)

    def _build_capture_dp_metadata(
        self,
        num_tokens: int,
    ) -> DPMetadata | AFDDPMetadata:
        return AFDNPUAttentionMetadataAdapter._build_capture_dp_metadata(
            self, num_tokens
        )

    # Patch reason: the AFD NPU model must be decorated with its uBatch wrapper.
    # Patch functionality: preserve upstream loading, then install the wrapper.
    # Signature: matches vLLM-Ascend v0.19.1rc1.
    def load_model(self) -> None:
        super().load_model()
        # ### PATCH START: AFD NPU uBatch model wrapper
        if bool(self.vllm_config.parallel_config.use_ubatching):
            self._install_ascend_ubatch_wrapper()
        # ### PATCH END: AFD NPU uBatch model wrapper

    def _install_ascend_ubatch_wrapper(self) -> None:
        if isinstance(self.model, AscendUBatchWrapper):
            return
        model = self.model
        runtime_mode = CUDAGraphMode.NONE
        if (
            isinstance(
                model,
                ACLGraphWrapper,
            )
            or self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            runtime_mode = CUDAGraphMode.FULL
        self.model = AscendUBatchWrapper(
            model,
            self.vllm_config,
            runtime_mode,
            self.device,
        )

    def get_model(self) -> nn.Module:
        if isinstance(self.model, AscendUBatchWrapper):
            return self.model.unwrap()
        return super().get_model()

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_attn_backend(kv_cache_config)
        if (
            bool(self.vllm_config.parallel_config.use_ubatching)
            or self.afd_async_extra_info.async_moe_ubatching
        ):
            self._ensure_two_metadata_builders()

    def _ensure_two_metadata_builders(self) -> None:
        for attn_groups in self.attn_groups:
            for attn_group in attn_groups:
                if len(attn_group.metadata_builders) >= 2:
                    continue
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    num_metadata_builders=2,
                )

    # Patch reason: AFD's DBO coordinator needs unpadded-token information.
    # Patch functionality: adapt the scoped decision to one packed CPU collective.
    # Signature: matches vLLM-Ascend v0.19.1rc1.
    def _sync_metadata_across_dp(
        self,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        allow_dp_padding: bool = False,
    ) -> tuple[int, torch.Tensor | None, CUDAGraphMode]:
        # ### PATCH START: AFD NPU DP coordination adapter
        return NPUDBOBatchDecisionV0191rc1.sync_metadata_across_dp(
            self,
            num_tokens,
            is_draft_model,
            cudagraph_mode,
            allow_dp_padding,
        )
        # ### PATCH END: AFD NPU DP coordination adapter

    # Patch reason: pinned Ascend always disables uBatch in its batch decision.
    # Patch functionality: preserve upstream graph/padding and publish an AFD plan.
    # Signature: matches vLLM-Ascend v0.19.1rc1 except the documented True default.
    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        # ### PATCH START: explicit native DBO plan
        return NPUDBOBatchDecisionV0191rc1.determine_batch_execution_and_padding(
            self,
            num_tokens,
            num_reqs,
            num_scheduled_tokens_np,
            max_num_scheduled_tokens,
            use_cascade_attn,
            allow_microbatching,
            force_eager,
            force_uniform_decode,
            force_has_lora,
            force_num_active_loras,
            num_encoder_reqs,
        )
        # ### PATCH END: explicit native DBO plan

    # Patch reason: pinned PP slicing does not read the active AFD uBatch plan.
    # Patch functionality: delegate only this missing seam to the pinned adapter.
    # Signature: matches vLLM-Ascend v0.19.1rc1.
    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        # ### PATCH START: uBatch-aware PP intermediate slicing
        result = AscendDBOCompatV0191rc1.sync_and_slice_intermediate_tensors(
            self,
            num_tokens,
            intermediate_tensors,
            sync_self,
        )
        # ### PATCH END: uBatch-aware PP intermediate slicing
        return result

    def shutdown(self) -> None:
        stop_afd_npu_profiler(self.prof)
        self.connector.close()
        super().shutdown()

    def _next_afd_transaction_id(self) -> str:
        return AFDNPUAttentionMetadataAdapter._next_afd_transaction_id(self)


def _normalize_metadata_ubatch_slices(
    ubatch_slices: UBatchSlices | None,
    values: dict[str, Any],
) -> UBatchSlices | None:
    if not ubatch_slices:
        return ubatch_slices
    num_tokens_padded = values.get("num_tokens_padded")
    num_reqs_padded = values.get("num_reqs_padded")
    if num_tokens_padded is None or num_reqs_padded is None:
        return ubatch_slices

    last_slice = ubatch_slices[-1]
    if int(last_slice.token_slice.stop) != int(num_tokens_padded) or int(
        last_slice.request_slice.stop
    ) == int(num_reqs_padded):
        return ubatch_slices

    return pad_out_ubatch_slices(
        ubatch_slices,
        int(num_tokens_padded),
        int(num_reqs_padded),
    )


__all__ = ["AFDNPUAttentionModelRunner"]
