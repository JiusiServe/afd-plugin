# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD-owned vLLM ubatch wrapper.

This runtime module depends on vLLM's native ubatching stack.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import DPMetadata, create_forward_context, get_forward_context
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.worker.gpu_ubatch_wrapper import UbatchMetadata, UBatchWrapper
from vllm.v1.worker.ubatching import make_ubatch_contexts

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors import AFDDPMetadata, AFDForwardContextMetadata


class AFDUBatchWrapper(UBatchWrapper):
    """Thin AFD-aware subclass of vLLM's native ``UBatchWrapper``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._afd_context_provider: Any | None = None

    def configure_afd_context_provider(self, provider: Any) -> None:
        self._afd_context_provider = provider

    @staticmethod
    def _create_sm_control_context(
        vllm_config: VllmConfig,
    ) -> AbstractContextManager[None]:
        if parse_afd_config(vllm_config).enabled:
            return nullcontext()
        return UBatchWrapper._create_sm_control_context(vllm_config)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        forward_context = get_forward_context()
        ubatch_slices = forward_context.ubatch_slices
        if ubatch_slices is None:
            return super().__call__(*args, **kwargs)

        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        parent_additional_kwargs = dict(forward_context.additional_kwargs)
        if "afd_metadata" not in parent_additional_kwargs:
            self._install_missing_afd_metadata(forward_context, ubatch_slices)
            parent_additional_kwargs = dict(forward_context.additional_kwargs)

        num_tokens = sum(int(ubatch_slice.num_tokens) for ubatch_slice in ubatch_slices)
        dp_metadata = build_ubatch_dp_metadata_list(
            self.vllm_config,
            ubatch_slices,
        )

        if (
            num_tokens not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=forward_context.attn_metadata,
                slot_mapping=forward_context.slot_mapping,
                input_ids=kwargs["input_ids"],
                positions=kwargs["positions"],
                inputs_embeds=kwargs["inputs_embeds"],
                intermediate_tensors=kwargs["intermediate_tensors"],
                compute_stream=torch.cuda.current_stream(),
                dp_metadata=dp_metadata,
                batch_descriptor=forward_context.batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._capture_ubatches(ubatch_metadata, self.runnable)

        if (
            num_tokens in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            get_offloader().sync_prev_onload()
            cudagraph_metadata = self.cudagraphs[num_tokens]
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs

        ubatch_metadata = self._make_ubatch_metadata(
            ubatch_slices=ubatch_slices,
            attn_metadata=forward_context.attn_metadata,
            slot_mapping=forward_context.slot_mapping,
            input_ids=kwargs["input_ids"],
            positions=kwargs["positions"],
            inputs_embeds=kwargs["inputs_embeds"],
            intermediate_tensors=kwargs["intermediate_tensors"],
            compute_stream=torch.cuda.current_stream(),
            dp_metadata=dp_metadata,
            batch_descriptor=forward_context.batch_descriptor,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        with self.sm_control:
            return self._run_ubatches(ubatch_metadata, self.runnable)

    def _install_missing_afd_metadata(
        self,
        forward_context: Any,
        ubatch_slices: Any,
    ) -> None:
        provider = self._afd_context_provider
        if provider is None:
            self._afd_use_native_ubatch_metadata = True
            return

        num_tokens_unpadded = sum(int(ub.num_tokens) for ub in ubatch_slices)
        afd_metadata = provider._build_afd_metadata(
            ubatch_slices,
            num_tokens_unpadded,
        )
        forward_context.additional_kwargs["afd_metadata"] = afd_metadata
        provider._afd_pending_metadata = afd_metadata
        if not bool(getattr(provider, "_afd_suppress_metadata_send", False)):
            provider._send_dp_metadata(
                forward_context.dp_metadata,
                ubatch_slices,
            )

    def _make_ubatch_metadata(
        self,
        ubatch_slices: Any,
        attn_metadata: Any,
        slot_mapping: Any,
        input_ids: Any,
        positions: Any,
        inputs_embeds: Any,
        intermediate_tensors: Any,
        compute_stream: Any,
        dp_metadata: list[DPMetadata | AFDDPMetadata],
        batch_descriptor: Any,
        cudagraph_runtime_mode: Any,
    ) -> list[UbatchMetadata]:
        parent_forward_context = get_forward_context()
        parent_additional_kwargs = dict(parent_forward_context.additional_kwargs)
        afd_metadata = parent_additional_kwargs.get("afd_metadata")
        if afd_metadata is None:
            if getattr(self, "_afd_use_native_ubatch_metadata", False):
                try:
                    return UBatchWrapper._make_ubatch_metadata(
                        self,
                        ubatch_slices,
                        attn_metadata,
                        slot_mapping,
                        input_ids,
                        positions,
                        inputs_embeds,
                        intermediate_tensors,
                        compute_stream,
                        dp_metadata,
                        batch_descriptor,
                        cudagraph_runtime_mode,
                    )
                finally:
                    self._afd_use_native_ubatch_metadata = False
            raise RuntimeError(
                "AFDUBatchWrapper requires "
                "ForwardContext.additional_kwargs['afd_metadata']",
            )

        forward_contexts = []
        has_slot_mapping = slot_mapping and isinstance(slot_mapping, list)
        for idx, _ubatch_slice in enumerate(ubatch_slices):
            ubatch_afd_metadata = build_ubatch_afd_metadata(
                afd_metadata,
                ubatch_slices,
                idx,
            )
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[idx] if attn_metadata is not None else None,
                    self.vllm_config,
                    dp_metadata=dp_metadata[idx],
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=slot_mapping[idx] if has_slot_mapping else None,
                    additional_kwargs=build_ubatch_additional_kwargs(
                        parent_additional_kwargs,
                        ubatch_afd_metadata,
                    ),
                ),
            )

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[UbatchMetadata] = []
        for idx, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[idx],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.num_tokens,
                ),
            )

        return ubatch_metadata


def build_ubatch_afd_metadata(
    afd_metadata: AFDForwardContextMetadata,
    ubatch_slices: Any,
    ubatch_idx: int,
) -> AFDForwardContextMetadata:
    """Clone parent AFD metadata for one vLLM ubatch."""

    if ubatch_idx < 0 or ubatch_idx >= len(ubatch_slices):
        raise IndexError(f"ubatch_idx {ubatch_idx} out of range")

    ubatch_slice = ubatch_slices[ubatch_idx]
    clone = afd_metadata.clone()
    clone.stage_idx = ubatch_idx
    clone.num_stages = len(ubatch_slices)
    clone.tokens_start_loc = [int(ubatch_slice.token_slice.start)]
    clone.requests_start_loc = [int(ubatch_slice.request_slice.start)]
    clone.tokens_lens = [int(ubatch_slice.num_tokens)]
    clone.tokens_unpadded_lens = [
        _resolve_ubatch_unpadded_tokens(afd_metadata, ubatch_slice, ubatch_idx),
    ]
    return clone


def build_ubatch_additional_kwargs(
    parent_additional_kwargs: dict[str, Any],
    afd_metadata: AFDForwardContextMetadata,
) -> dict[str, Any]:
    child_kwargs = dict(parent_additional_kwargs)
    child_kwargs["afd_metadata"] = afd_metadata
    return child_kwargs


def build_ubatch_dp_metadata_list(
    vllm_config: VllmConfig,
    ubatch_slices: Any,
) -> list[DPMetadata | AFDDPMetadata]:
    """Create DP metadata for each ubatch.

    For DP=1 we use the plugin-owned metadata object to stay independent of
    vLLM internals. For DP>1 we delegate to vLLM's native ``DPMetadata.make``.
    """

    parallel_config = vllm_config.parallel_config
    dp_size = int(parallel_config.data_parallel_size)
    if dp_size <= 1:
        return [
            AFDDPMetadata(
                num_tokens_across_dp_cpu=torch.tensor(
                    [ubatch_slice.num_tokens],
                    dtype=torch.int32,
                    device="cpu",
                ),
                max_tokens_across_dp_cpu=torch.tensor(
                    [ubatch_slice.num_tokens],
                    dtype=torch.int32,
                    device="cpu",
                ),
            )
            for ubatch_slice in ubatch_slices
        ]

    ubatch_dp_metadata = []
    for ubatch_slice in ubatch_slices:
        num_tokens_across_dp_cpu = torch.tensor(
            [ubatch_slice.num_tokens] * dp_size,
            device="cpu",
            dtype=torch.int32,
        )
        ubatch_dp_metadata.append(
            DPMetadata.make(
                parallel_config,
                ubatch_slice.num_tokens,
                num_tokens_across_dp_cpu,
            ),
        )
    return ubatch_dp_metadata


def _resolve_ubatch_unpadded_tokens(
    afd_metadata: AFDForwardContextMetadata,
    ubatch_slice: Any,
    ubatch_idx: int,
) -> int:
    unpadded_lens = afd_metadata.tokens_unpadded_lens
    if ubatch_idx < len(unpadded_lens):
        return int(unpadded_lens[ubatch_idx])
    return int(ubatch_slice.num_tokens)


__all__ = [
    "AFDUBatchWrapper",
    "build_ubatch_additional_kwargs",
    "build_ubatch_afd_metadata",
    "build_ubatch_dp_metadata_list",
]
