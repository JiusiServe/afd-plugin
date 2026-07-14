# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 async CAM forward orchestration helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from itertools import islice
from typing import TYPE_CHECKING

import torch
from vllm.forward_context import get_forward_context
from vllm.v1.worker.ubatch_utils import UBatchSlices

from afd_plugin.connectors import AFDForwardContextMetadata, AFDTransferMetadata
from afd_plugin.model_executor.models import AsyncMoeUbatchMetadata
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.deepseek_v2 import (
        AFDDeepseekV2DecoderLayer,
        AFDDeepseekV2Model,
    )


def run_attention_gate_afd_forward(
    model: AFDDeepseekV2Model,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    afd_metadata: AFDForwardContextMetadata,
    llama_4_scaling: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the Attention-side gate AFD path used by async CAM."""

    afd_connector = afd_metadata.afd_connector
    forward_context = get_forward_context()
    stage_idx = int(
        getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx),
    )
    pending_ffn_recv = False

    for layer_offset, layer in enumerate(
        islice(model.layers, model.start_layer, model.end_layer),
    ):
        stage_idx = int(
            getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx),
        )
        afd_metadata.stage_idx = stage_idx
        if layer_offset > 0 and pending_ffn_recv:
            hidden_states = afd_connector.recv_ffn_output(
                ref_tensor=hidden_states,
                ubatch_idx=stage_idx,
            )
            pending_ffn_recv = False

        if not layer.is_moe_layer:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
            )
            continue

        (
            hidden_states,
            residual,
            topk_weights,
            topk_ids,
            router_logits,
        ) = layer.compute_attn_output(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        )

        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer.layer_idx,
            stage_idx=stage_idx,
            seq_len=int(hidden_states.shape[0]),
        )
        afd_connector.send_attn_output(
            hidden_states,
            metadata,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )
        pending_ffn_recv = True
        hidden_states = maybe_apply_dbo_yield(
            hidden_states,
            role="attention",
        )

    if pending_ffn_recv:
        hidden_states = afd_connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=stage_idx,
        )
    return hidden_states, residual


def run_async_moe_ubatch_afd_forward(
    model: AFDDeepseekV2Model,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    afd_metadata: AFDForwardContextMetadata,
    async_moe_ubatch_metadata: AsyncMoeUbatchMetadata,
    llama_4_scaling: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the two-stage async MoE ubatch pipeline used by async CAM."""

    forward_context = get_forward_context()
    ubatch_slices = async_moe_ubatch_metadata["ubatch_slices"]
    afd_connector = afd_metadata.afd_connector
    first_moe_layer = int(model.config.first_k_dense_replace)
    dense_end_layer = min(model.end_layer, first_moe_layer)
    for layer in islice(model.layers, model.start_layer, dense_end_layer):
        hidden_states, residual = layer(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        )
    if dense_end_layer == model.end_layer:
        return hidden_states, residual

    stage_hidden_states = [
        hidden_states[ubatch_slice.token_slice] for ubatch_slice in ubatch_slices
    ]
    stage_residual = [
        _slice_optional_first_dim(residual, ubatch_slice.token_slice)
        for ubatch_slice in ubatch_slices
    ]
    stage_positions = [
        _slice_positions(positions, ubatch_slice.token_slice)
        for ubatch_slice in ubatch_slices
    ]
    stage_llama_4_scaling = [
        _slice_llama_4_scaling(
            llama_4_scaling,
            ubatch_slice.token_slice,
            num_tokens=int(hidden_states.shape[0]),
        )
        for ubatch_slice in ubatch_slices
    ]

    moe_start_layer = max(model.start_layer, first_moe_layer)
    moe_layers = list(islice(model.layers, moe_start_layer, model.end_layer))

    def compute_stage_attention(
        layer: AFDDeepseekV2DecoderLayer,
        stage_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ubatch_slice = ubatch_slices[stage_idx]
        with _use_async_moe_ubatch_forward_context(
            forward_context=forward_context,
            parent_afd_metadata=afd_metadata,
            async_moe_ubatch_metadata=async_moe_ubatch_metadata,
            stage_idx=stage_idx,
        ):
            (
                stage_hidden_states[stage_idx],
                stage_residual[stage_idx],
                topk_weights,
                topk_ids,
                router_logits,
            ) = layer.compute_attn_output(
                stage_positions[stage_idx],
                stage_hidden_states[stage_idx],
                stage_residual[stage_idx],
                stage_llama_4_scaling[stage_idx],
            )
        if topk_weights is None or topk_ids is None:
            raise RuntimeError(
                "async_moe_ubatching requires Attention-side topk payloads",
            )
        expected_tokens = int(ubatch_slice.num_tokens)
        if int(stage_hidden_states[stage_idx].shape[0]) != expected_tokens:
            raise RuntimeError(
                "async_moe_ubatching stage output token count mismatch: "
                f"expected {expected_tokens}, got "
                f"{int(stage_hidden_states[stage_idx].shape[0])}",
            )
        return topk_weights, topk_ids, router_logits

    def send_stage_attention(
        layer: AFDDeepseekV2DecoderLayer,
        stage_idx: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor | None,
    ) -> None:
        expected_tokens = int(ubatch_slices[stage_idx].num_tokens)
        stage_metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer.layer_idx,
            stage_idx=stage_idx,
            seq_len=expected_tokens,
        )
        afd_connector.send_attn_output(
            stage_hidden_states[stage_idx],
            stage_metadata,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )

    def recv_stage_ffn(stage_idx: int) -> None:
        stage_hidden_states[stage_idx] = afd_connector.recv_ffn_output(
            ref_tensor=stage_hidden_states[stage_idx],
            ubatch_idx=stage_idx,
        )

    last_moe_layer_offset = len(moe_layers) - 1
    first_layer = moe_layers[0]
    topk_weights, topk_ids, router_logits = compute_stage_attention(
        first_layer,
        0,
    )
    send_stage_attention(
        first_layer,
        0,
        topk_weights,
        topk_ids,
        router_logits,
    )

    for moe_layer_offset in range(last_moe_layer_offset):
        current_layer = moe_layers[moe_layer_offset]
        next_layer = moe_layers[moe_layer_offset + 1]

        topk_weights, topk_ids, router_logits = compute_stage_attention(
            current_layer,
            1,
        )
        recv_stage_ffn(0)
        send_stage_attention(
            current_layer,
            1,
            topk_weights,
            topk_ids,
            router_logits,
        )

        topk_weights, topk_ids, router_logits = compute_stage_attention(
            next_layer,
            0,
        )
        recv_stage_ffn(1)
        send_stage_attention(
            next_layer,
            0,
            topk_weights,
            topk_ids,
            router_logits,
        )

    last_layer = moe_layers[last_moe_layer_offset]
    topk_weights, topk_ids, router_logits = compute_stage_attention(
        last_layer,
        1,
    )
    recv_stage_ffn(0)
    send_stage_attention(
        last_layer,
        1,
        topk_weights,
        topk_ids,
        router_logits,
    )
    recv_stage_ffn(1)
    output_hidden_states = torch.cat(stage_hidden_states, dim=0)
    return (
        output_hidden_states,
        _cat_optional_async_moe_stage_outputs(
            stage_residual,
            stage_hidden_states,
        ),
    )


_MISSING_FORWARD_CONTEXT_ATTR = object()


@contextmanager
def _use_async_moe_ubatch_forward_context(
    *,
    forward_context: object,
    parent_afd_metadata: AFDForwardContextMetadata,
    async_moe_ubatch_metadata: AsyncMoeUbatchMetadata,
    stage_idx: int,
) -> Iterator[None]:
    ubatch_slices = async_moe_ubatch_metadata["ubatch_slices"]
    attn_metadata = async_moe_ubatch_metadata["attn_metadata"]
    stage_afd_metadata = _build_async_moe_stage_afd_metadata(
        parent_afd_metadata,
        ubatch_slices,
        stage_idx,
    )

    saved_attrs = {
        "attn_metadata": _read_forward_context_attr(
            forward_context,
            "attn_metadata",
        ),
        "additional_kwargs": _read_forward_context_attr(
            forward_context,
            "additional_kwargs",
        ),
        "ubatch_idx": _read_forward_context_attr(forward_context, "ubatch_idx"),
        "num_ubatches": _read_forward_context_attr(
            forward_context,
            "num_ubatches",
        ),
        "num_tokens": _read_forward_context_attr(forward_context, "num_tokens"),
    }

    original_kwargs = (
        forward_context.additional_kwargs
        if saved_attrs["additional_kwargs"] is not _MISSING_FORWARD_CONTEXT_ATTR
        else None
    )
    stage_kwargs = dict(original_kwargs or {})
    stage_kwargs["afd_metadata"] = stage_afd_metadata

    try:
        forward_context.attn_metadata = attn_metadata[stage_idx]
        forward_context.additional_kwargs = stage_kwargs
        forward_context.ubatch_idx = stage_idx
        forward_context.num_ubatches = len(ubatch_slices)
        forward_context.num_tokens = int(ubatch_slices[stage_idx].num_tokens)
        yield
    finally:
        for name, value in saved_attrs.items():
            _restore_forward_context_attr(forward_context, name, value)


def _build_async_moe_stage_afd_metadata(
    parent_afd_metadata: AFDForwardContextMetadata,
    ubatch_slices: UBatchSlices,
    stage_idx: int,
) -> AFDForwardContextMetadata:
    ubatch_slice = ubatch_slices[stage_idx]
    stage_metadata = parent_afd_metadata.clone()
    stage_metadata.stage_idx = stage_idx
    stage_metadata.num_stages = len(ubatch_slices)
    stage_metadata.tokens_start_loc = [ubatch_slice.token_slice.start]
    stage_metadata.requests_start_loc = [ubatch_slice.request_slice.start]
    stage_metadata.tokens_lens = [ubatch_slice.num_tokens]
    if len(parent_afd_metadata.tokens_unpadded_lens) > stage_idx:
        unpadded_len = parent_afd_metadata.tokens_unpadded_lens[stage_idx]
    else:
        unpadded_len = ubatch_slice.num_tokens
    stage_metadata.tokens_unpadded_lens = [int(unpadded_len)]
    return stage_metadata


def _cat_optional_async_moe_stage_outputs(
    stage_outputs: list[torch.Tensor | None],
    fallback_outputs: list[torch.Tensor],
) -> torch.Tensor | None:
    if all(stage_output is None for stage_output in stage_outputs):
        return None
    return torch.cat(
        [
            stage_output if stage_output is not None else fallback_output
            for stage_output, fallback_output in zip(
                stage_outputs,
                fallback_outputs,
                strict=True,
            )
        ],
        dim=0,
    )


def _slice_optional_first_dim(
    tensor: torch.Tensor | None,
    token_slice: slice,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[token_slice]


def _slice_positions(positions: torch.Tensor, token_slice: slice) -> torch.Tensor:
    if positions.dim() <= 1:
        return positions[token_slice]
    return positions[..., token_slice]


def _slice_llama_4_scaling(
    llama_4_scaling: torch.Tensor | None,
    token_slice: slice,
    *,
    num_tokens: int,
) -> torch.Tensor | None:
    if llama_4_scaling is None:
        return None
    if llama_4_scaling.shape[0] == num_tokens:
        return llama_4_scaling[token_slice]
    if llama_4_scaling.dim() > 1 and llama_4_scaling.shape[1] == num_tokens:
        return llama_4_scaling[:, token_slice]
    return llama_4_scaling


def _read_forward_context_attr(forward_context: object, name: str) -> object:
    try:
        return getattr(forward_context, name)
    except AttributeError:
        return _MISSING_FORWARD_CONTEXT_ATTR


def _restore_forward_context_attr(
    forward_context: object,
    name: str,
    value: object,
) -> None:
    if value is _MISSING_FORWARD_CONTEXT_ATTR:
        with suppress(AttributeError):
            delattr(forward_context, name)
        return
    setattr(forward_context, name, value)


__all__ = [
    "run_async_moe_ubatch_afd_forward",
    "run_attention_gate_afd_forward",
]
