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

from afd_plugin.connectors import (
    AFDForwardContextMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.model_executor.models import AsyncMoeUbatchMetadata
from afd_plugin.model_executor.models.npu.ubatch_sp import (
    build_async_moe_stage_inputs,
    restore_async_moe_stage_outputs,
    sp_local_actual_token_count,
)
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

    afd_connector = afd_metadata.connector
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
        context = AFDTransferContext(metadata=metadata)
        afd_connector.send_attn_output(
            hidden_states,
            context,
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
    afd_connector = afd_metadata.connector
    first_moe_layer = int(model.config.first_k_dense_replace)
    dense_end_layer = min(model.end_layer, first_moe_layer)

    (
        stage_hidden_states,
        stage_residual,
        stage_positions,
        stage_llama_4_scaling,
        sp_local_stage_slices,
    ) = build_async_moe_stage_inputs(
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        llama_4_scaling=llama_4_scaling,
        ubatch_slices=ubatch_slices,
        use_sp_stage_resharding=bool(
            async_moe_ubatch_metadata.get("use_sp_stage_resharding", False),
        ),
    )
    async_moe_ubatch_metadata["sp_local_stage_slices"] = sp_local_stage_slices
    stage_actual_token_counts = async_moe_ubatch_metadata.get(
        "stage_actual_token_counts",
        [int(ubatch_slice.num_tokens) for ubatch_slice in ubatch_slices],
    )
    use_sp_stage_resharding = (
        bool(
            async_moe_ubatch_metadata.get("use_sp_stage_resharding", False),
        )
        and sp_local_stage_slices is not ubatch_slices
    )
    if use_sp_stage_resharding:
        sp_local_stage_actual_token_counts = [
            sp_local_actual_token_count(
                stage_actual_tokens=int(stage_actual_tokens),
                stage_input_tokens=int(ubatch_slice.num_tokens),
            )
            for stage_actual_tokens, ubatch_slice in zip(
                stage_actual_token_counts,
                ubatch_slices,
                strict=True,
            )
        ]
    else:
        sp_local_stage_actual_token_counts = [
            int(stage_actual_tokens)
            for stage_actual_tokens in stage_actual_token_counts
        ]
    async_moe_ubatch_metadata["sp_local_stage_actual_token_counts"] = (
        sp_local_stage_actual_token_counts
    )

    # Per-stage Ascend metadata is built before model execution. Run the dense
    # prefix under the matching stage context too, so its RoPE/KV inputs never
    # mix the full-batch context with buffers prepared by a stage builder.
    for stage_idx in range(len(ubatch_slices)):
        with _use_async_moe_ubatch_forward_context(
            forward_context=forward_context,
            parent_afd_metadata=afd_metadata,
            async_moe_ubatch_metadata=async_moe_ubatch_metadata,
            stage_idx=stage_idx,
        ):
            for layer in islice(
                model.layers,
                model.start_layer,
                dense_end_layer,
            ):
                (
                    stage_hidden_states[stage_idx],
                    stage_residual[stage_idx],
                ) = layer(
                    stage_positions[stage_idx],
                    stage_hidden_states[stage_idx],
                    stage_residual[stage_idx],
                    stage_llama_4_scaling[stage_idx],
                )

    if dense_end_layer == model.end_layer:
        return _restore_async_moe_stage_state(
            stage_hidden_states,
            stage_residual,
            ubatch_slices,
            use_sp_stage_resharding=use_sp_stage_resharding,
        )

    moe_start_layer = max(model.start_layer, first_moe_layer)
    moe_layers = list(islice(model.layers, moe_start_layer, model.end_layer))

    def compute_stage_attention(
        layer: AFDDeepseekV2DecoderLayer,
        stage_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        sp_local_stage_slice = sp_local_stage_slices[stage_idx]
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
        expected_tokens = int(sp_local_stage_slice.num_tokens)
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
        expected_tokens = int(sp_local_stage_slices[stage_idx].num_tokens)
        stage_metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer.layer_idx,
            stage_idx=stage_idx,
            seq_len=expected_tokens,
        )
        stage_context = AFDTransferContext(metadata=stage_metadata)
        afd_connector.send_attn_output(
            stage_hidden_states[stage_idx],
            stage_context,
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
    return _restore_async_moe_stage_state(
        stage_hidden_states,
        stage_residual,
        ubatch_slices,
        use_sp_stage_resharding=use_sp_stage_resharding,
    )


def _restore_async_moe_stage_state(
    stage_hidden_states: list[torch.Tensor],
    stage_residual: list[torch.Tensor | None],
    ubatch_slices: UBatchSlices,
    *,
    use_sp_stage_resharding: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    output_hidden_states = restore_async_moe_stage_outputs(
        stage_hidden_states,
        ubatch_slices,
        use_sp_stage_resharding=use_sp_stage_resharding,
    )
    resolved_stage_residual = [
        stage_output if stage_output is not None else fallback_output
        for stage_output, fallback_output in zip(
            stage_residual,
            stage_hidden_states,
            strict=True,
        )
    ]
    return (
        output_hidden_states,
        None
        if all(stage_output is None for stage_output in stage_residual)
        else restore_async_moe_stage_outputs(
            resolved_stage_residual,
            ubatch_slices,
            use_sp_stage_resharding=use_sp_stage_resharding,
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
    sp_local_stage_slices = async_moe_ubatch_metadata.get(
        "sp_local_stage_slices",
        ubatch_slices,
    )
    attn_metadata = async_moe_ubatch_metadata["attn_metadata"]
    stage_afd_metadata = _build_async_moe_stage_afd_metadata(
        parent_afd_metadata,
        sp_local_stage_slices,
        async_moe_ubatch_metadata.get(
            "sp_local_stage_actual_token_counts",
            [int(stage_slice.num_tokens) for stage_slice in sp_local_stage_slices],
        ),
        stage_idx,
    )

    stage_context_attr_names = (
        "attn_metadata",
        "additional_kwargs",
        "ubatch_idx",
        "num_ubatches",
        "num_tokens",
        "pad_size",
        "padded_length",
        "max_tokens_across_dp",
        "padded_num_tokens",
        "mc2_mask",
        "dbo_enabled",
    )
    saved_attrs = {
        name: _read_forward_context_attr(forward_context, name)
        for name in stage_context_attr_names
    }

    original_kwargs = (
        forward_context.additional_kwargs
        if saved_attrs["additional_kwargs"] is not _MISSING_FORWARD_CONTEXT_ATTR
        else None
    )
    stage_kwargs = dict(original_kwargs or {})
    stage_kwargs["afd_metadata"] = stage_afd_metadata
    stage_input_tokens = int(ubatch_slices[stage_idx].num_tokens)
    stage_actual_tokens = int(
        async_moe_ubatch_metadata.get(
            "stage_actual_token_counts",
            [int(ubatch_slice.num_tokens) for ubatch_slice in ubatch_slices],
        )[stage_idx],
    )

    try:
        forward_context.attn_metadata = attn_metadata[stage_idx]
        forward_context.additional_kwargs = stage_kwargs
        forward_context.ubatch_idx = stage_idx
        forward_context.num_ubatches = len(ubatch_slices)
        # Ascend MLA all-gathers the local stage shards before RoPE and uses
        # this value to allocate its global o-projection input.
        forward_context.num_tokens = stage_input_tokens
        forward_context.pad_size = 0
        forward_context.padded_length = stage_input_tokens
        forward_context.max_tokens_across_dp = stage_input_tokens
        forward_context.padded_num_tokens = stage_input_tokens
        forward_context.dbo_enabled = True
        saved_mc2_mask = saved_attrs["mc2_mask"]
        if isinstance(saved_mc2_mask, torch.Tensor):
            stage_mc2_mask = torch.zeros(
                (stage_input_tokens,),
                dtype=saved_mc2_mask.dtype,
                device=saved_mc2_mask.device,
            )
            stage_mc2_mask[:stage_actual_tokens] = True
            forward_context.mc2_mask = stage_mc2_mask
        yield
    finally:
        for name, value in saved_attrs.items():
            _restore_forward_context_attr(forward_context, name, value)


def _build_async_moe_stage_afd_metadata(
    parent_afd_metadata: AFDForwardContextMetadata,
    ubatch_slices: UBatchSlices,
    stage_actual_token_counts: list[int],
    stage_idx: int,
) -> AFDForwardContextMetadata:
    ubatch_slice = ubatch_slices[stage_idx]
    stage_metadata = parent_afd_metadata.clone()
    stage_metadata.stage_idx = stage_idx
    stage_metadata.num_stages = len(ubatch_slices)
    stage_metadata.tokens_start_loc = [ubatch_slice.token_slice.start]
    stage_metadata.requests_start_loc = [ubatch_slice.request_slice.start]
    stage_metadata.tokens_lens = [ubatch_slice.num_tokens]
    stage_metadata.tokens_unpadded_lens = [
        int(stage_actual_token_counts[stage_idx]),
    ]
    return stage_metadata


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
