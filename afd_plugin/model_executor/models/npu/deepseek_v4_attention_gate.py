# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4 Attention-side routing helpers for Async CAM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.deepseek_v4 import (
        AFDDeepseekV4AttentionGateRemoteMoE,
    )


def compute_attention_gate_topk(
    moe: AFDDeepseekV4AttentionGateRemoteMoE,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DSV4 routing without entering vLLM's native MoE communicator.

    AFD Async CAM owns the cross-role dispatch.  The vLLM-Ascend fused
    selector's hash path instead assumes native EP/SP communication and calls
    ``forward_context.moe_comm_method.pad_and_split_input_ids``.  That object
    is intentionally absent on Attention ranks, including the KV-cache profile
    forward.  Use the same CANN routing operators directly on local Attention
    tokens, then hand their IDs and weights to CAM dispatch.
    """

    router_logits, _ = moe.gate(hidden_states)
    if moe.scoring_func == "sqrtsoftplus":
        topk_weights, topk_ids = _compute_sqrtsoftplus_topk(moe, router_logits)
    else:
        topk_weights, topk_ids = _compute_standard_topk(moe, router_logits)
    return topk_weights.to(torch.float32), topk_ids


def _compute_sqrtsoftplus_topk(
    moe: AFDDeepseekV4AttentionGateRemoteMoE,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DSV4's sqrtsoftplus CANN router without native MoE communication."""

    if moe.scoring_func != "sqrtsoftplus":
        raise RuntimeError(
            "DSV4 Hash routing requires scoring_func='sqrtsoftplus', got "
            f"{moe.scoring_func!r}",
        )

    tid2eid = moe.gate.tid2eid
    input_ids = None
    if tid2eid is not None:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        input_ids = getattr(forward_context, "input_ids", None)
        if input_ids is None:
            raise RuntimeError(
                "DSV4 Hash routing requires local input_ids in the forward "
                "context",
            )
        input_ids = input_ids.reshape(-1).to(torch.int64)
        if input_ids.numel() != router_logits.shape[0]:
            raise RuntimeError(
                "DSV4 Hash routing input_ids/token count mismatch on Attention: "
                f"input_ids={input_ids.numel()} router_tokens={router_logits.shape[0]}",
            )
        input_ids = torch.where(input_ids == -1, 0, input_ids)
        tid2eid = tid2eid.to(torch.int32)
    correction_bias = moe.gate.e_score_correction_bias
    if correction_bias is not None and correction_bias.dtype != router_logits.dtype:
        correction_bias = correction_bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
        x=router_logits,
        k=moe.top_k,
        bias=correction_bias,
        input_ids=input_ids,
        tid2eid=tid2eid,
        k_group=moe.topk_group,
        group_count=moe.num_expert_group,
        routed_scaling_factor=moe.routed_scaling_factor,
        eps=1e-20,
        group_select_mode=1,
        renorm=0,
        norm_type=2,
        out_flag=False,
    )
    return topk_weights, topk_ids


def _compute_standard_topk(
    moe: AFDDeepseekV4AttentionGateRemoteMoE,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the non-Hash CANN selector without native MoE communication."""

    from vllm_ascend.device.device_op import DeviceOperator

    norm_type_by_scoring_func = {"softmax": 0, "sigmoid": 1}
    try:
        norm_type = norm_type_by_scoring_func[moe.scoring_func]
    except KeyError as exc:
        raise RuntimeError(
            "Unsupported non-Hash DSV4 routing scoring function: "
            f"{moe.scoring_func!r}",
        ) from exc
    correction_bias = moe.gate.e_score_correction_bias
    if correction_bias is not None and correction_bias.dtype != router_logits.dtype:
        correction_bias = correction_bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
        router_logits,
        k=moe.top_k,
        k_group=moe.topk_group,
        group_count=moe.num_expert_group,
        group_select_mode=1,
        renorm=int(moe.renormalize),
        norm_type=norm_type,
        out_flag=False,
        routed_scaling_factor=moe.routed_scaling_factor,
        eps=1e-20,
        bias_opt=correction_bias,
    )
    return topk_weights, topk_ids
