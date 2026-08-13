# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side gate helpers for DeepSeek-V2 on CUDA.

The async GPU connector dispatches tokens that are already routed: one row per
``(token, topk_slot)`` partial, grouped by local expert, with a ``group_list``
of per-expert counts. The FFN side therefore must not run routing again -- it
only needs the grouped GEMM over its local experts.

That shape is a ``topk == 1`` problem: give every arriving row its own expert id
and a unit weight, and vLLM's ``fused_experts`` computes exactly the local
expert output. Topk weighting stays on the Attention side, applied during
combine, matching the NPU path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from afd_plugin.connectors.metadata import AFDF2ATransferPayload

if TYPE_CHECKING:
    from torch import nn


def compute_attention_gate_moe_ffn(
    layer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    expand_x_shared: torch.Tensor | None = None,
) -> AFDF2ATransferPayload:
    """Run this rank's local experts over pre-routed tokens.

    Args:
        layer: The AFD DeepSeek decoder layer owning the MoE module.
        hidden_states: ``[num_partials, hidden]`` rows sorted by local expert.
        group_list: ``[expert_per_rank]`` per-expert row counts. Must sum to
            ``hidden_states.shape[0]``.
        expand_x_shared: Optional ``[num_shared, hidden]`` shared-expert rows.

    Returns:
        Routed output in the same row order as ``hidden_states``, plus the
        shared-expert output when the model has shared experts.
    """
    # ``mlp.experts`` is a MoERunner; the weight parameters live on its
    # RoutedExperts, and the shared experts hang off the runner rather than
    # off the MoE module.
    runner = layer.mlp.experts
    routed_experts = runner.routed_experts
    counts = group_list.to(torch.int64)
    num_rows = int(hidden_states.shape[0])
    if int(counts.sum()) != num_rows:
        raise ValueError(
            f"group_list sums to {int(counts.sum())} but hidden_states has "
            f"{num_rows} rows",
        )

    num_local_experts = counts.numel()
    if num_rows == 0:
        # Routing can leave a peer with nothing -- common in decode, where a
        # single token's topk may land entirely on the other FFN rank.
        routed_output = hidden_states.new_empty((0, hidden_states.shape[-1]))
        shared = runner._shared_experts
        return AFDF2ATransferPayload(
            routed_output=routed_output,
            shared_output=(
                shared._layer(expand_x_shared)
                if shared is not None
                and expand_x_shared is not None
                and expand_x_shared.shape[0] > 0
                else None
            ),
        )
    expert_ids = torch.repeat_interleave(
        torch.arange(
            num_local_experts,
            device=hidden_states.device,
            dtype=torch.int32,
        ),
        counts,
    ).unsqueeze(1)
    # Unit weights: the real topk weighting happens in the connector's combine.
    unit_weights = torch.ones(
        (num_rows, 1),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    routed_output = fused_experts(
        hidden_states,
        routed_experts.w13_weight,
        routed_experts.w2_weight,
        unit_weights,
        expert_ids,
        global_num_experts=num_local_experts,
        expert_map=None,
    )

    shared_output = None
    shared_experts = runner._shared_experts
    if shared_experts is not None and expand_x_shared is not None:
        # Call the wrapped MLP rather than SharedExperts.forward: the wrapper is
        # a stateful scheduler for the runner's own multi-stream pipeline and
        # returns None unless its expected ordering matches. AFD feeds shared
        # tokens as their own batch, so that machinery does not apply.
        shared_output = shared_experts._layer(expand_x_shared)

    # Mirrors the NPU gate path: scale the routed branch unless fp16, where the
    # native model instead scales the shared branch down.
    routed_scaling_factor = runner.routed_scaling_factor
    if hidden_states.dtype != torch.float16:
        routed_output = routed_output * routed_scaling_factor
    elif shared_output is not None:
        shared_output = shared_output * (1.0 / routed_scaling_factor)

    return AFDF2ATransferPayload(
        routed_output=routed_output,
        shared_output=shared_output,
    )


__all__ = ["compute_attention_gate_moe_ffn"]
