# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patch vllm-ascend W8A8 MoE to support force load balance.

This module patches only the Ascend W8A8 FusedMoE path. When
``enable_force_load_balance`` is set in ``additional_config``, routed
``topk_ids`` are replaced with deterministic fake expert ids before
``build_fused_experts_input``. This keeps routed-token volume evenly balanced
across EP ranks for communication profiling.

Force load balance changes model outputs. It is a benchmark/profiling switch,
not a production correctness feature.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.quantization.methods import w8a8_dynamic
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    build_fused_experts_input,
)
from vllm_ascend.quantization.quant_type import QuantType

if TYPE_CHECKING:
    from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEFusedExpertsInput

logger = logging.getLogger(__name__)

_ORIGINAL_FUSED_MOE_INIT_ATTR = "_afd_plugin_original_fused_moe_init"
_ORIGINAL_W8A8_APPLY_ATTR = "_afd_plugin_original_w8a8_apply"
_ORIGINAL_BUILD_FUSED_EXPERTS_INPUT_ATTR = (
    "_afd_plugin_original_build_fused_experts_input"
)
_FORCE_LB_DETERMINISTIC_SEED = 1024

_original_fused_moe_init: Callable[..., object] | None = None
_original_w8a8_apply: Callable[..., object] | None = None
_original_build_fused_experts_input: Callable[..., object] | None = None


@dataclass(frozen=True)
class ForceLoadBalanceConfig:
    """Force-load-balance parameters for one AscendFusedMoE layer.

    Args:
        n_routed_experts: Number of routed experts in the MoE layer.
        ep_size: Number of expert-parallel ranks.
        top_k: Number of routed experts selected for each token.
        topn_per_rank: Number of local experts per EP rank used by the fake
            routing cycle. A value of 0 means all routed experts participate.
    """

    n_routed_experts: int
    ep_size: int
    top_k: int
    topn_per_rank: int


def _get_force_lb_max_tokens(vllm_config: VllmConfig) -> int:
    max_tokens = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", None)
    if not isinstance(max_tokens, int):
        max_tokens = 128
    return max(max_tokens, 1)


def _get_force_lb_config(layer: object) -> ForceLoadBalanceConfig:
    return ForceLoadBalanceConfig(
        n_routed_experts=int(layer.n_routed_experts),
        ep_size=int(layer.ep_size),
        top_k=int(layer.top_k),
        topn_per_rank=int(layer.force_load_balance_topn_per_rank),
    )


def _validate_force_lb_config(config: ForceLoadBalanceConfig) -> None:
    if config.topn_per_rank == 0:
        return

    assert config.topn_per_rank > 0, "force_load_balance_topn_per_rank must be >= 0"
    assert config.ep_size > 0, "ep_size must be positive"
    assert config.n_routed_experts % config.ep_size == 0, (
        "force_load_balance_topn_per_rank requires n_routed_experts to be"
        " divisible by ep_size"
    )

    local_routed_experts = config.n_routed_experts // config.ep_size
    assert config.topn_per_rank <= local_routed_experts, (
        "force_load_balance_topn_per_rank exceeds routed experts on each FFN rank"
    )
    assert config.top_k <= config.topn_per_rank * config.ep_size, (
        "top_k must be <= force_load_balance_topn_per_rank * ep_size"
    )


def _build_expert_cycle(
    config: ForceLoadBalanceConfig,
    device: torch.device,
) -> torch.Tensor:
    if config.topn_per_rank > 0:
        local_routed_experts = config.n_routed_experts // config.ep_size
        per_rank_cycles = [
            torch.arange(
                rank * local_routed_experts,
                rank * local_routed_experts + config.topn_per_rank,
                device=device,
                dtype=torch.int32,
            )
            for rank in range(config.ep_size)
        ]
        return torch.cat(per_rank_cycles, dim=0)

    generator = torch.Generator()
    generator.manual_seed(_FORCE_LB_DETERMINISTIC_SEED)
    return torch.randperm(
        config.n_routed_experts,
        generator=generator,
        dtype=torch.int32,
    ).to(device=device, non_blocking=True)


def _build_topk_buffer(
    config: ForceLoadBalanceConfig,
    max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    expert_cycle = _build_expert_cycle(config, device)
    total_needed = max_tokens * config.top_k
    repeat_times = (total_needed + expert_cycle.numel() - 1) // expert_cycle.numel()
    expanded = expert_cycle.repeat(repeat_times)[:total_needed]
    return expanded.reshape(max_tokens, config.top_k)


def _init_force_lb_buffer(
    layer: object,
    max_tokens: int,
    device: torch.device,
) -> None:
    config = _get_force_lb_config(layer)
    _validate_force_lb_config(config)
    buffer = _build_topk_buffer(config, max_tokens, device)

    layer.force_lb_fake_topk_buffer = buffer
    layer.max_force_lb_tokens = max_tokens

    logger.info(
        "AFD force load balance buffer initialized: ep_size=%s top_k=%s"
        " topn_per_rank=%s shape=%s preview=%s",
        config.ep_size,
        config.top_k,
        config.topn_per_rank,
        tuple(buffer.shape),
        buffer[: min(8, max_tokens)].cpu().tolist(),
    )


def _get_force_lb_topk_ids(
    layer: object,
    batch_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    buffer: torch.Tensor | None = getattr(layer, "force_lb_fake_topk_buffer", None)
    if buffer is None:
        raise RuntimeError("force_lb_fake_topk_buffer is not initialized")

    if batch_tokens > buffer.size(0):
        new_max_tokens = max(batch_tokens, buffer.size(0) * 2)
        logger.warning(
            "Growing AFD force load balance buffer: old_tokens=%s new_tokens=%s",
            buffer.size(0),
            new_max_tokens,
        )
        _init_force_lb_buffer(layer, new_max_tokens, device)
        buffer = layer.force_lb_fake_topk_buffer

    if buffer.device != device:
        buffer = buffer.to(device, non_blocking=True)
        layer.force_lb_fake_topk_buffer = buffer

    top_k = int(layer.top_k)
    return buffer[:batch_tokens, :top_k]


# Patch reason: vllm-ascend's AscendFusedMoE does not initialize AFD profiling
# knobs for deterministic force-load-balance routing.
# Patch functionality: delegates upstream initialization, then builds the AFD
# fake top-k buffer for W8A8 layers when force load balance is enabled.
# Signature: matches upstream; no added parameters.
def __init__(self, *args, **kwargs):
    assert _original_fused_moe_init is not None
    _original_fused_moe_init(self, *args, **kwargs)

    # ### PATCH START: AFD force-load-balance layer initialization
    # Read plugin-owned profiling knobs and prebuild deterministic fake routed
    # expert ids for Ascend W8A8 MoE layers.
    vllm_config = self.vllm_config
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        additional_config = {}

    self.n_routed_experts = int(kwargs["num_experts"])
    self.enable_force_load_balance = bool(
        additional_config.get("enable_force_load_balance", False)
    )
    self.force_load_balance_topn_per_rank = int(
        additional_config.get("force_load_balance_topn_per_rank", 0)
    )
    self.max_force_lb_tokens = _get_force_lb_max_tokens(vllm_config)
    self.force_lb_fake_topk_buffer = None

    if self.enable_force_load_balance and self.quant_type == QuantType.W8A8:
        _init_force_lb_buffer(
            self,
            int(self.max_force_lb_tokens),
            self.w13_weight.device,
        )
    # ### PATCH END: AFD force-load-balance layer initialization


def _replace_topk_ids(
    layer: object,
    topk_ids: torch.Tensor,
    routed_top_k: int | None,
) -> torch.Tensor:
    fake_topk_ids = _get_force_lb_topk_ids(
        layer,
        batch_tokens=topk_ids.shape[0],
        device=topk_ids.device,
    ).to(topk_ids.dtype)

    if getattr(layer, "mix_placement", False) and routed_top_k is not None:
        shared_topk_ids = topk_ids[:, routed_top_k:]
        return torch.cat([fake_topk_ids, shared_topk_ids], dim=1)
    return fake_topk_ids


# Patch reason: vllm-ascend W8A8 MoE routes tokens with model-selected expert
# ids, but AFD profiling needs deterministic balanced expert ids.
# Patch functionality: temporarily replaces build_fused_experts_input so only
# routed top-k ids are swapped; all other W8A8 apply behavior stays upstream.
# Signature: matches upstream; no added parameters.
def apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    use_grouped_topk: bool = False,
    num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    is_prefill: bool = True,
    enable_force_load_balance: bool = False,
    log2phy: torch.Tensor | None = None,
    global_redundant_expert_num: int = 0,
    pertoken_scale: Any | None = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    mc2_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not (
        getattr(layer, "enable_force_load_balance", False)
        and getattr(layer, "force_lb_fake_topk_buffer", None) is not None
    ):
        assert _original_w8a8_apply is not None
        return _original_w8a8_apply(
            self,
            layer,
            x,
            router_logits,
            top_k,
            renormalize,
            use_grouped_topk=use_grouped_topk,
            num_experts=num_experts,
            expert_map=expert_map,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            pertoken_scale=pertoken_scale,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            mc2_mask=mc2_mask,
        )

    # ### PATCH START: AFD force-load-balance W8A8 routing override
    # Swap routed topk ids only around build_fused_experts_input so the rest of
    # vllm-ascend's W8A8 apply path stays upstream-compatible.
    # Patch reason: this wrapper is installed only during W8A8 apply to intercept
    # the exact point where routed expert ids become fused expert input.
    # Patch functionality: replaces topk_ids with deterministic AFD ids and
    # delegates all other fused expert input construction to upstream.
    # Signature: matches upstream; no added parameters.
    def _build_fused_experts_input(
        *,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w1: torch.Tensor | list[torch.Tensor],
        w2: torch.Tensor | list[torch.Tensor],
        quant_type: QuantType,
        dynamic_eplb: bool,
        expert_map: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        mc2_mask: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        log2phy: torch.Tensor | None = None,
        pertoken_scale: torch.Tensor | None = None,
        activation: str = "silu",
        need_trans: bool = False,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        comm_quant_mode: int | None = None,
        mxfp_act_quant_type: torch.dtype | None = None,
        mxfp_weight_quant_type: torch.dtype | None = None,
        mxfp_scale_dtype: torch.dtype | None = None,
        mxfp_per_token_scale_dtype: torch.dtype | None = None,
        mxfp_use_bf16: bool | None = None,
        w1_scale: list[torch.Tensor] | torch.Tensor | None = None,
        w2_scale: list[torch.Tensor] | torch.Tensor | None = None,
        w1_scale_bias: torch.Tensor | None = None,
        w2_scale_bias: torch.Tensor | None = None,
        w1_offset: torch.Tensor | None = None,
        w2_offset: torch.Tensor | None = None,
    ) -> MoEFusedExpertsInput:
        assert _original_build_fused_experts_input is not None
        return _original_build_fused_experts_input(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=_replace_topk_ids(layer, topk_ids, top_k),
            w1=w1,
            w2=w2,
            quant_type=quant_type,
            dynamic_eplb=dynamic_eplb,
            expert_map=expert_map,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            log2phy=log2phy,
            pertoken_scale=pertoken_scale,
            activation=activation,
            need_trans=need_trans,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            comm_quant_mode=comm_quant_mode,
            mxfp_act_quant_type=mxfp_act_quant_type,
            mxfp_weight_quant_type=mxfp_weight_quant_type,
            mxfp_scale_dtype=mxfp_scale_dtype,
            mxfp_per_token_scale_dtype=mxfp_per_token_scale_dtype,
            mxfp_use_bf16=mxfp_use_bf16,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
        )

    old_build_fused_experts_input = w8a8_dynamic.build_fused_experts_input
    w8a8_dynamic.build_fused_experts_input = _build_fused_experts_input
    try:
        assert _original_w8a8_apply is not None
        result = _original_w8a8_apply(
            self,
            layer,
            x,
            router_logits,
            top_k,
            renormalize,
            use_grouped_topk=use_grouped_topk,
            num_experts=num_experts,
            expert_map=expert_map,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            pertoken_scale=pertoken_scale,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            mc2_mask=mc2_mask,
        )
    finally:
        w8a8_dynamic.build_fused_experts_input = old_build_fused_experts_input
    # ### PATCH END: AFD force-load-balance W8A8 routing override
    return result


def _patch_force_load_balance() -> None:
    """Patch Ascend W8A8 force-load-balance behavior when imported."""

    global _original_build_fused_experts_input
    global _original_fused_moe_init
    global _original_w8a8_apply

    if not hasattr(AscendFusedMoE, _ORIGINAL_FUSED_MOE_INIT_ATTR):
        setattr(
            AscendFusedMoE,
            _ORIGINAL_FUSED_MOE_INIT_ATTR,
            AscendFusedMoE.__init__,
        )
        setattr(
            AscendFusedMoE,
            _ORIGINAL_W8A8_APPLY_ATTR,
            AscendW8A8DynamicFusedMoEMethod.apply,
        )
        setattr(
            AscendFusedMoE,
            _ORIGINAL_BUILD_FUSED_EXPERTS_INPUT_ATTR,
            build_fused_experts_input,
        )

    _original_fused_moe_init = getattr(
        AscendFusedMoE,
        _ORIGINAL_FUSED_MOE_INIT_ATTR,
    )
    _original_w8a8_apply = getattr(AscendFusedMoE, _ORIGINAL_W8A8_APPLY_ATTR)
    _original_build_fused_experts_input = getattr(
        AscendFusedMoE,
        _ORIGINAL_BUILD_FUSED_EXPERTS_INPUT_ATTR,
    )

    AscendFusedMoE.__init__ = __init__
    AscendW8A8DynamicFusedMoEMethod.apply = apply


_patch_force_load_balance()


__all__: list[str] = []
