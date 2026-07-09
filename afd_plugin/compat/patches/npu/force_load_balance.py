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

import torch
from vllm.config import VllmConfig
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.quantization.methods import w8a8_dynamic
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    build_fused_experts_input,
)
from vllm_ascend.quantization.quant_type import QuantType

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


def __init__(self: object, *args: object, **kwargs: object) -> None:
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


def apply(
    self: object,
    layer: object,
    *args: object,
    **kwargs: object,
) -> object:
    if not (
        getattr(layer, "enable_force_load_balance", False)
        and getattr(layer, "force_lb_fake_topk_buffer", None) is not None
    ):
        assert _original_w8a8_apply is not None
        return _original_w8a8_apply(self, layer, *args, **kwargs)

    routed_top_k = kwargs.get("top_k")
    routed_top_k = routed_top_k if isinstance(routed_top_k, int) else None

    # ### PATCH START: AFD force-load-balance W8A8 routing override
    # Swap routed topk ids only around build_fused_experts_input so the rest of
    # vllm-ascend's W8A8 apply path stays upstream-compatible.
    def _build_fused_experts_input(
        *inner_args: object,
        **inner_kwargs: object,
    ) -> object:
        topk_ids = inner_kwargs.get("topk_ids")
        if topk_ids is not None:
            inner_kwargs["topk_ids"] = _replace_topk_ids(
                layer,
                topk_ids,
                routed_top_k,
            )
        assert _original_build_fused_experts_input is not None
        return _original_build_fused_experts_input(*inner_args, **inner_kwargs)

    old_build_fused_experts_input = w8a8_dynamic.build_fused_experts_input
    w8a8_dynamic.build_fused_experts_input = _build_fused_experts_input
    try:
        assert _original_w8a8_apply is not None
        result = _original_w8a8_apply(self, layer, *args, **kwargs)
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
