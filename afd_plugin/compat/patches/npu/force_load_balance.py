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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import vllm_ascend.ops.fused_moe.fused_moe as fused_moe_module
from vllm.config import CompilationMode, VllmConfig, get_current_vllm_config
from vllm.logger import logger
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import (
    _EXTRA_CTX,
    _MEGA_MOE_SUPPORTED,
    MoECommType,
)
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.experts_selector import (
    select_experts,
    zero_experts_compute,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.methods.base import get_moe_num_logical_experts
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
)

_FORCE_LB_DETERMINISTIC_SEED = 1024


@dataclass(frozen=True)
class ForceLoadBalanceConfig:
    """Force-load-balance parameters for one AscendFusedMoE layer.

    Args:
        n_routed_experts: Number of routed experts in the MoE layer.
        ep_size: Number of expert-parallel ranks.
        ep_rank: Source expert-parallel rank that builds the fake routing cycle.
        top_k: Number of routed experts selected for each token.
        topn_per_rank: Number of local experts per EP rank used by the fake
            routing cycle. A value of 0 means all routed experts participate.
    """

    n_routed_experts: int
    ep_size: int
    ep_rank: int
    top_k: int
    topn_per_rank: int


def _get_force_lb_max_tokens(vllm_config: VllmConfig) -> int:
    max_tokens = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", None)
    if not isinstance(max_tokens, int):
        max_tokens = 128
    return max(max_tokens, 1)


def _validate_force_lb_config(config: ForceLoadBalanceConfig) -> None:
    assert config.ep_size > 0, "ep_size must be positive"
    assert 0 <= config.ep_rank < config.ep_size, (
        "ep_rank must be within the expert-parallel group"
    )
    assert config.n_routed_experts % config.ep_size == 0, (
        "force load balance requires n_routed_experts to be divisible by ep_size"
    )

    if config.topn_per_rank == 0:
        return

    assert config.topn_per_rank > 0, "force_load_balance_topn_per_rank must be >= 0"
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
    local_routed_experts = config.n_routed_experts // config.ep_size
    if config.topn_per_rank > 0:
        per_rank_cycles = [
            torch.arange(
                rank * local_routed_experts,
                rank * local_routed_experts + config.topn_per_rank,
                device=device,
                dtype=torch.int32,
            )
            for rank in range(config.ep_size)
        ]
        expert_cycle = torch.cat(per_rank_cycles, dim=0)
    else:
        generator_device = torch.device("cpu")
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(_FORCE_LB_DETERMINISTIC_SEED)
        expert_cycle = torch.randperm(
            config.n_routed_experts,
            generator=generator,
            device=generator_device,
            dtype=torch.int32,
        ).to(device=device, non_blocking=True)

    # Shift every expert by whole target-rank blocks so source EP ranks use
    # different phases without changing the deterministic cycle order.
    source_rank_expert_offset = config.ep_rank * local_routed_experts
    return (expert_cycle + source_rank_expert_offset) % config.n_routed_experts


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
    method: AscendW8A8DynamicFusedMoEMethod,
    config: ForceLoadBalanceConfig,
    max_tokens: int,
    device: torch.device,
) -> None:
    _validate_force_lb_config(config)
    buffer = _build_topk_buffer(config, max_tokens, device)

    method.force_lb_fake_topk_buffer = buffer
    method.max_force_lb_tokens = max_tokens

    fused_moe_module.logger.info(
        "AFD force load balance buffer initialized: ep_size=%s top_k=%s"
        " topn_per_rank=%s shape=%s preview=%s",
        config.ep_size,
        config.top_k,
        config.topn_per_rank,
        tuple(buffer.shape),
        buffer[: min(8, max_tokens)].cpu().tolist(),
    )


def _get_force_lb_topk_ids(
    method: AscendW8A8DynamicFusedMoEMethod,
    config: ForceLoadBalanceConfig,
    batch_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    buffer = method.force_lb_fake_topk_buffer
    if buffer is None:
        raise RuntimeError("force_lb_fake_topk_buffer is not initialized")

    if batch_tokens > buffer.size(0):
        new_max_tokens = max(batch_tokens, buffer.size(0) * 2)
        fused_moe_module.logger.warning(
            "Growing AFD force load balance buffer: old_tokens=%s new_tokens=%s",
            buffer.size(0),
            new_max_tokens,
        )
        _init_force_lb_buffer(method, config, new_max_tokens, device)
        buffer = method.force_lb_fake_topk_buffer
        assert buffer is not None

    if buffer.device != device:
        buffer = buffer.to(device, non_blocking=True)
        method.force_lb_fake_topk_buffer = buffer

    return buffer[:batch_tokens, : config.top_k]


# Patch reason: vllm-ascend's W8A8 method does not retain AFD's deterministic
# force-load-balance settings after leaving the model-construction config context.
# Patch functionality: preserves upstream initialization and captures the AFD
# profiling switch, local-expert limit, and initial buffer capacity for apply.
# Signature: matches upstream; no added parameters.
def __init__(self):
    vllm_config = get_current_vllm_config()
    ascend_config = get_ascend_config()
    self.use_aclgraph = (
        vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
        and not vllm_config.model_config.enforce_eager
    )
    self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb
    self.in_dtype = vllm_config.model_config.dtype
    self.supports_eplb = True

    try:
        device_group = get_mc2_group().device_group
        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
    except AttributeError:
        logger.warning_once(
            "[vllm-ascend/W8A8_DYNAMIC] MC2 group metadata unavailable, "
            "falling back to empty moe_all_to_all_group_name."
        )
        self.moe_all_to_all_group_name = ""

    # ### PATCH START: capture AFD force-load-balance configuration
    additional_config = vllm_config.additional_config or {}
    self.enable_force_load_balance = bool(
        additional_config.get("enable_force_load_balance", False)
    )
    self.force_load_balance_topn_per_rank = int(
        additional_config.get("force_load_balance_topn_per_rank", 0)
    )
    self.max_force_lb_tokens = _get_force_lb_max_tokens(vllm_config)
    self.force_lb_fake_topk_buffer: torch.Tensor | None = None
    # ### PATCH END: capture AFD force-load-balance configuration


# Patch reason: vllm-ascend W8A8 MoE routes tokens with model-selected expert
# ids, but AFD profiling needs deterministic balanced expert ids.
# Patch functionality: preserves the target upstream tag's W8A8 apply path and
# replaces model-selected top-k ids with the method-owned AFD deterministic
# ids.
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
    tid2eid: torch.Tensor | None = None,
) -> torch.Tensor:
    zero_expert_num = getattr(layer, "zero_expert_num", 0)
    zero_expert_type = getattr(layer, "zero_expert_type", None)
    n_shared_experts = getattr(layer, "n_shared_experts", 0)
    mix_placement = getattr(layer, "mix_placement", False)
    if n_shared_experts is None:
        n_shared_experts = 0
    num_logical_experts = get_moe_num_logical_experts(
        layer,
        num_experts,
        global_redundant_expert_num=global_redundant_expert_num,
        num_shared_experts=n_shared_experts,
    )
    if zero_expert_num == 0 or zero_expert_type is None:
        assert router_logits.shape[1] == num_logical_experts, (
            "[vllm-ascend/W8A8_DYNAMIC] Number of global experts mismatch "
            "(excluding redundancy). "
            f"router_experts={router_logits.shape[1]}, "
            f"expected_experts={num_logical_experts}, "
            f"zero_expert_num={zero_expert_num}, "
            f"zero_expert_type={zero_expert_type}"
        )

    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        mix_placement=mix_placement,
        num_logical_experts=router_logits.shape[1],
        num_shared_experts=n_shared_experts,
        num_experts=num_logical_experts,
        tid2eid=tid2eid,
    )
    assert topk_ids is not None
    assert topk_weights is not None
    if zero_expert_num > 0 and zero_expert_type is not None:
        topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
            expert_indices=topk_ids,
            expert_scales=topk_weights,
            num_experts=num_logical_experts,
            zero_expert_type=zero_expert_type,
            hidden_states=x,
        )

    # this is a naive implementation for experts load balance so as
    # to avoid accumulating too much tokens on a single rank.
    # currently it is only activated when doing profile runs.
    if enable_force_load_balance:
        random_matrix = torch.rand(
            topk_ids.size(0), num_logical_experts, device=topk_ids.device
        )
        topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(
            topk_ids.dtype
        )

    # ### PATCH START: AFD force-load-balance W8A8 routing override
    # Replace routed ids with a deterministic balanced cycle when explicitly
    # requested by the plugin configuration captured during construction.
    if not enable_force_load_balance and self.enable_force_load_balance:
        force_lb_config = ForceLoadBalanceConfig(
            n_routed_experts=num_logical_experts,
            ep_size=int(layer.moe_config.ep_size),
            ep_rank=int(layer.moe_config.ep_rank),
            top_k=top_k,
            topn_per_rank=self.force_load_balance_topn_per_rank,
        )
        if self.force_lb_fake_topk_buffer is None:
            _init_force_lb_buffer(
                self,
                force_lb_config,
                self.max_force_lb_tokens,
                topk_ids.device,
            )
        fake_routed_topk_ids = _get_force_lb_topk_ids(
            self,
            force_lb_config,
            topk_ids.shape[0],
            topk_ids.device,
        )
        fake_routed_topk_ids = fake_routed_topk_ids.to(topk_ids.dtype)
        if mix_placement:
            shared_topk_ids = topk_ids[:, top_k:]
            topk_ids = torch.cat([fake_routed_topk_ids, shared_topk_ids], dim=1)
        else:
            topk_ids = fake_routed_topk_ids
    # ### PATCH END: AFD force-load-balance W8A8 routing override

    assert topk_weights is not None
    topk_weights = topk_weights.to(self.in_dtype)

    act_name = getattr(activation, "value", activation)
    moe_comm_method = _EXTRA_CTX.moe_comm_method
    fused_scale_flag = (
        _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2
        and get_ascend_config().enable_fused_mc2 == 1
        and act_name != "swigluoai_uninterleave"
    )
    if self.dynamic_eplb:
        w1 = layer.w13_weight_list
        w1_scale = (
            layer.fused_w1_scale_list
            if fused_scale_flag
            else layer.w13_weight_scale_fp32_list
        )
        w2 = layer.w2_weight_list
        w2_scale = (
            layer.fused_w2_scale_list
            if fused_scale_flag
            else layer.w2_weight_scale_list
        )
        w1_scale_bias = (
            [torch.tensor([], dtype=torch.float32)] if fused_scale_flag else None
        )
        w2_scale_bias = (
            [torch.tensor([], dtype=torch.float32)] if fused_scale_flag else None
        )
    elif fused_scale_flag and _MEGA_MOE_SUPPORTED:
        w1 = layer.cann_mega_moe_w13_weight_list
        w1_scale = layer.cann_mega_moe_fused_w1_scale_list
        w2 = layer.cann_mega_moe_w2_weight_list
        w2_scale = layer.cann_mega_moe_fused_w2_scale_list
        w1_scale_bias = None
        w2_scale_bias = None
    else:
        w1 = [layer.w13_weight]
        w1_scale = (
            [layer.fused_w1_scale]
            if fused_scale_flag
            else [layer.w13_weight_scale_fp32]
        )
        w2 = [layer.w2_weight]
        w2_scale = (
            [layer.fused_w2_scale] if fused_scale_flag else [layer.w2_weight_scale]
        )
        w1_scale_bias = (
            [torch.tensor([], dtype=torch.float32)] if fused_scale_flag else None
        )
        w2_scale_bias = (
            [torch.tensor([], dtype=torch.float32)] if fused_scale_flag else None
        )

    final_hidden_states = moe_comm_method.fused_experts(
        fused_experts_input=build_fused_experts_input(
            hidden_states=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1=w1,
            w2=w2,
            quant_type=self.quant_type,
            dynamic_eplb=self.dynamic_eplb,
            expert_map=expert_map,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            log2phy=log2phy,
            pertoken_scale=pertoken_scale,
            activation=activation,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            swiglu_limit=layer.swiglu_limit,
            swiglu_alpha=getattr(layer, "swiglu_alpha", 1.0),
            swiglu_beta=getattr(layer, "swiglu_beta", 0.0),
        )
    )
    if zero_expert_num > 0 and zero_expert_type is not None:
        final_hidden_states += zero_expert_result
    return final_hidden_states


AscendW8A8DynamicFusedMoEMethod.__init__ = __init__
AscendW8A8DynamicFusedMoEMethod.apply = apply


__all__: list[str] = []
