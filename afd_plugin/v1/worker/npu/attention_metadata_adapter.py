# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bridge NPU attention metadata to the connector-owned control plane."""

from __future__ import annotations

from typing import Any

import torch
from vllm.forward_context import DPMetadata, ForwardContext
from vllm.logger import init_logger

from afd_plugin.connectors import (
    AFDControlPayload,
    AFDDPMetadata,
    AFDForwardContextMetadata,
)
from afd_plugin.v1.worker.attention_model_runner import (
    _forward_context_num_tokens,
    _full_cudagraph_padded_tokens,
)
from afd_plugin.v1.worker.ubatch_wrapper import build_ubatch_dp_metadata_list

logger = init_logger("afd_plugin.v1.worker.npu.attention_model_runner")


class AFDNPUAttentionMetadataAdapter:
    """Unbound helpers adapting runner metadata to the connector control plane."""

    # AFD source: pre-refactor NPU attention ModelRunner control plane.
    # Move reason: isolate AFD metadata transport from Ascend DBO backports.
    # Functionality: moved without changing payload construction or ordering.
    # Signature: retained from the pre-refactor AFD implementation.
    def _build_afd_metadata(
        self,
        ubatch_slices: Any,
        num_tokens_unpadded: int,
    ) -> AFDForwardContextMetadata:
        # ### PATCH START: AFD NPU control plane
        if ubatch_slices and len(ubatch_slices) > 1:
            tokens_start_loc = [ub.token_slice.start for ub in ubatch_slices]
            requests_start_loc = [ub.request_slice.start for ub in ubatch_slices]
            tokens_lens = [ub.num_tokens for ub in ubatch_slices]
            tokens_unpadded_lens = [int(ub.num_tokens) for ub in ubatch_slices]
            num_stages = len(ubatch_slices)
        else:
            tokens_start_loc = [0]
            requests_start_loc = [0]
            tokens_lens = [num_tokens_unpadded]
            tokens_unpadded_lens = [num_tokens_unpadded]
            num_stages = 1

        return AFDForwardContextMetadata(
            tokens_start_loc=tokens_start_loc,
            requests_start_loc=requests_start_loc,
            stage_idx=0,
            connector=self.connector,
            tokens_lens=tokens_lens,
            num_stages=num_stages,
            transaction_id=self._next_afd_transaction_id(),
            tokens_unpadded_lens=tokens_unpadded_lens,
        )
        # ### PATCH END: AFD NPU control plane

    # AFD source: pre-refactor NPU attention ModelRunner control plane.
    # Move reason: isolate AFD metadata transport from Ascend DBO backports.
    # Functionality: moved without changing payload construction or ordering.
    # Signature: retained from the pre-refactor AFD implementation.
    def _install_afd_metadata_on_forward_context(
        self,
        forward_context: ForwardContext,
    ) -> None:
        # ### PATCH START: AFD NPU control plane
        if self._afd_pending_metadata is None:
            self._afd_pending_metadata = self._build_afd_metadata(
                forward_context.ubatch_slices,
                _forward_context_num_tokens(forward_context, self.vllm_config),
            )

        if forward_context.additional_kwargs is None:
            forward_context.additional_kwargs = {}
        forward_context.additional_kwargs["afd_metadata"] = self._afd_pending_metadata
        if self.connector.control_plane is None:
            return
        if self._afd_suppress_metadata_send:
            return
        dp_metadata = forward_context.dp_metadata
        ubatch_slices = forward_context.ubatch_slices
        padded_graph_tokens = _full_cudagraph_padded_tokens(forward_context)
        if padded_graph_tokens is not None and not ubatch_slices:
            dp_metadata = self._build_capture_dp_metadata(padded_graph_tokens)
        self._send_dp_metadata(dp_metadata, ubatch_slices)
        # ### PATCH END: AFD NPU control plane

    # AFD source: pre-refactor NPU attention ModelRunner control plane.
    # Move reason: isolate AFD metadata transport from Ascend DBO backports.
    # Functionality: moved without changing payload construction or ordering.
    # Signature: retained from the pre-refactor AFD implementation.
    def _send_dp_metadata(
        self,
        dp_metadata: DPMetadata | AFDDPMetadata | None,
        ubatch_slices: Any,
    ) -> None:
        # ### PATCH START: AFD NPU control plane
        assert self.connector.control_plane is not None, (
            "_send_dp_metadata needs a control-plane connector"
        )
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
        is_graph_capturing = bool(self._afd_is_graph_capturing)
        payload = AFDControlPayload(
            dp_metadata_list=dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
        )
        self.connector.control_plane.update_state_from_dp_metadata(payload)
        logger.warning(
            "AFD NPU Attention send_dp_metadata decision; world_rank=%d "
            "key=%s is_graph_capturing=%s is_warmup=%s",
            self.connector.world_rank,
            _dp_metadata_debug_key(dp_metadata_list),
            is_graph_capturing,
            is_warmup,
        )
        self.connector.control_plane.send_dp_metadata_list(payload)
        # ### PATCH END: AFD NPU control plane

    # AFD source: pre-refactor NPU attention ModelRunner control plane.
    # Move reason: isolate AFD metadata transport from Ascend DBO backports.
    # Functionality: moved without changing payload construction or ordering.
    # Signature: retained from the pre-refactor AFD implementation.
    def _ensure_dp_metadata(
        self,
        dp_metadata: DPMetadata | AFDDPMetadata | None,
    ) -> DPMetadata | AFDDPMetadata:
        # ### PATCH START: AFD NPU control plane
        if dp_metadata is not None:
            return dp_metadata

        dp_size = int(self.vllm_config.parallel_config.data_parallel_size)
        if dp_size != 1:
            raise RuntimeError("AFD NPU Attention expected DPMetadata for DP > 1")
        if self._afd_pending_metadata is None:
            raise RuntimeError("AFD metadata is not available for DP fallback")

        num_tokens = int(self._afd_pending_metadata.tokens_lens[0])
        return _make_uniform_dp_metadata(dp_size, num_tokens)
        # ### PATCH END: AFD NPU control plane

    # AFD source: pre-refactor NPU attention ModelRunner control plane.
    # Move reason: isolate AFD metadata transport from Ascend DBO backports.
    # Functionality: moved without changing payload construction or ordering.
    # Signature: retained from the pre-refactor AFD implementation.
    def _build_capture_dp_metadata(self, num_tokens: int) -> DPMetadata | AFDDPMetadata:
        # ### PATCH START: AFD NPU control plane
        dp_size = int(self.vllm_config.parallel_config.data_parallel_size)
        return _make_uniform_dp_metadata(dp_size, int(num_tokens))
        # ### PATCH END: AFD NPU control plane

    # AFD source: pre-refactor NPU attention ModelRunner control plane.
    # Move reason: isolate AFD metadata transport from Ascend DBO backports.
    # Functionality: moved without changing payload construction or ordering.
    # Signature: retained from the pre-refactor AFD implementation.
    def _next_afd_transaction_id(self) -> str:
        # ### PATCH START: AFD NPU control plane
        counter = self._afd_transaction_counter
        self._afd_transaction_counter = counter + 1
        return f"afd-npu-{counter}"
        # ### PATCH END: AFD NPU control plane


# AFD source: pre-refactor NPU attention ModelRunner control-plane helper.
# Move reason: keep the payload key and fallback metadata with their owner.
# Functionality: moved without behavior changes.
# Signature: retained from the pre-refactor helper.
def _make_uniform_dp_metadata(dp_size: int, num_tokens: int) -> AFDDPMetadata:
    # ### PATCH START: AFD NPU control-plane helper
    num_tokens_across_dp_cpu = torch.full(
        (int(dp_size),),
        int(num_tokens),
        dtype=torch.int32,
        device="cpu",
    )
    return AFDDPMetadata(num_tokens_across_dp_cpu=num_tokens_across_dp_cpu)
    # ### PATCH END: AFD NPU control-plane helper


# AFD source: pre-refactor NPU attention ModelRunner control-plane helper.
# Move reason: keep the payload key and fallback metadata with their owner.
# Functionality: moved without behavior changes.
# Signature: retained from the pre-refactor helper.
def _dp_metadata_debug_key(
    dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
) -> tuple[tuple[int, tuple]]:
    # ### PATCH START: AFD NPU control-plane helper
    key_parts: list[tuple[int, tuple]] = []
    for stage_idx, metadata in sorted(dp_metadata_list.items()):
        values = metadata.num_tokens_across_dp_cpu
        tolist = getattr(values, "tolist", None)
        if callable(tolist):
            values = tolist()
        elif hasattr(values, "item"):
            values = [values.item()]
        try:
            values_tuple = tuple(int(value) for value in values)
        except TypeError:
            values_tuple = (int(values),)
        key_parts.append((int(stage_idx), values_tuple))
    return tuple(key_parts)
    # ### PATCH END: AFD NPU control-plane helper


__all__ = ["AFDNPUAttentionMetadataAdapter"]
