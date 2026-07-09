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
from typing import Any

import torch
from vllm.config import VllmConfig
import vllm_ascend.envs as envs_ascend
import vllm_ascend.ops.fused_moe.fused_moe as fused_moe_module
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.flash_common3_context import get_flash_common3_context
from vllm_ascend.ops.fused_moe.experts_selector import (
    select_experts,
    zero_experts_compute,
)
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    build_fused_experts_input,
)
from vllm_ascend.quantization.quant_type import QuantType

logger = logging.getLogger(__name__)

_FORCE_LB_DETERMINISTIC_SEED = 1024


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
# Patch functionality: preserves the target upstream tag's AscendFusedMoE
# initialization and replaces the force-load-balance buffer setup with AFD's
# deterministic fake top-k buffer.
# Signature: matches upstream; no added parameters.
def __init__(self, *args, **kwargs):
    super(AscendFusedMoE, self).__init__(*args, **kwargs)

    num_experts = kwargs["num_experts"]
    intermediate_size = kwargs["intermediate_size"]
    num_shared_experts = kwargs.get("n_shared_experts", 0)
    self.n_routed_experts = num_experts

    AscendFusedMoE.moe_counter += 1
    self.moe_instance_id = AscendFusedMoE.moe_counter

    self._expert_map = None
    self.log2phy = None

    if self.quant_config is None:
        self.quant_method = fused_moe_module.AscendUnquantizedFusedMoEMethod(
            self.moe_config
        )
    else:
        self.quant_method = self.quant_config.get_quant_method(self, self.layer_name)

    assert self.quant_method is not None

    self.moe_config.tp_group = fused_moe_module.get_tp_group()
    self.moe_config.dp_group = fused_moe_module.get_dp_group()
    self.moe_config.ep_group = fused_moe_module.get_ep_group()
    self.moe_config.mc2_group = fused_moe_module.get_mc2_group()
    self.moe_config.supports_eplb = self.quant_method.supports_eplb
    ascend_config = fused_moe_module.get_ascend_config()
    vllm_config = fused_moe_module.get_current_vllm_config()
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        additional_config = {}
    # flashcommon3 gate stream
    self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
    if self.multistream_overlap_gate and AscendFusedMoE.gate_stream is None:
        AscendFusedMoE.gate_stream = torch.npu.Stream()
    if (
        self.custom_routing_function is None
        and self.e_score_correction_bias is not None
    ):
        self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(
            dtype=vllm_config.model_config.dtype
        )

    # ### PATCH START: AFD force-load-balance layer initialization
    # Read plugin-owned profiling knobs and prebuild deterministic fake routed
    # expert ids for Ascend W8A8 MoE layers.
    self.enable_force_load_balance = bool(
        additional_config.get("enable_force_load_balance", False)
    )
    self.force_load_balance_topn_per_rank = int(
        additional_config.get("force_load_balance_topn_per_rank", 0)
    )
    self.max_force_lb_tokens = _get_force_lb_max_tokens(vllm_config)
    self.force_lb_fake_topk_buffer = None
    # ### PATCH END: AFD force-load-balance layer initialization

    # init moe
    eplb_config = ascend_config.eplb_config
    self.mix_placement = getattr(ascend_config, "mix_placement", False)
    self.n_shared_experts = num_shared_experts
    num_experts += num_shared_experts if self.mix_placement else 0
    self.moe_config.num_experts = num_experts
    (
        self.global_expert_map,
        self._expert_map,
        self.log2phy,
        self.global_redundant_expert_num,
    ) = fused_moe_module.init_eplb_config(
        eplb_config,
        self.moe_instance_id,
        self.moe_config,
        self.mix_placement,
        num_shared_experts,
    )
    self.global_num_experts = num_experts + self.global_redundant_expert_num
    self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy is not None)
    self.local_num_experts = self.global_num_experts // self.ep_size
    if self._expert_map is not None:
        fused_moe_module.logger.info_once(
            "[EP Rank %s/%s] Expert parallelism is enabled. Local/global"
            " number of experts: %s/%s. Experts local to global index map:"
            " %s.",
            self.ep_rank,
            self.ep_size,
            self.local_num_experts,
            self.global_num_experts,
            fused_moe_module.get_compressed_expert_map(self._expert_map),
        )
    if self.dynamic_eplb:
        self.multi_stage = False
        self.moe_load = torch.zeros(self.local_num_experts, dtype=torch.int64).npu()
        if eplb_config.eplb_policy_type == 3:
            self.multi_stage = True
            self.load_counter = torch.tensor(0, dtype=torch.int32, device="npu")
            self.num_iter = eplb_config.expert_heat_collection_interval
            self.moe_load = torch.zeros(
                (self.num_iter, self.local_num_experts),
                dtype=torch.int32,
                device="npu",
            )

    self.moe_config.num_experts = self.global_num_experts
    self.moe_config.num_local_experts = self.local_num_experts
    self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num

    moe_quant_params = {
        "num_experts": self.local_num_experts,
        "hidden_size": self.hidden_size,
        "intermediate_size_per_partition": self.intermediate_size_per_partition,
        "params_dtype": self.params_dtype,
        "weight_loader": self.weight_loader,
    }
    # need full intermediate size pre-sharding for WNA16 act order
    if self.quant_method.__class__.__name__ in (
        "GPTQMarlinMoEMethod",
        "CompressedTensorsWNA16MoEMethod",
    ):
        moe_quant_params["intermediate_size_full"] = intermediate_size
    self.quant_method.create_weights(layer=self, **moe_quant_params)

    self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
    self.enable_npugraph_ex_static_kernel = (
        ascend_config.ascend_compilation_config.enable_static_kernel
    )

    fused_moe_module.setup_moe_comm_method(self.moe_config)
    self.quant_type = self._get_quant_type()

    # ### PATCH START: AFD force-load-balance layer initialization
    # Initialize the deterministic fake top-k buffer after W8A8 weights exist.
    if self.enable_force_load_balance and self.quant_type == QuantType.W8A8:
        _init_force_lb_buffer(
            self,
            int(self.max_force_lb_tokens),
            self.w13_weight.device,
        )
    # ### PATCH END: AFD force-load-balance layer initialization

    is_legacy = fused_moe_module.vllm_version_is("0.19.1")
    self.runner = fused_moe_module.AscendMoERunner(
        self if is_legacy else self.layer_name,
        self.moe_config,
        self.router,
        self._routed_input_transform,
        self.gate if is_legacy else kwargs.pop("gate", None),
        self.shared_experts if is_legacy else kwargs.pop("shared_experts", None),
        self.quant_method,
        self.reduce_results,
        self.vllm_config.parallel_config.enable_dbo,
    )


# Patch reason: vllm-ascend W8A8 MoE routes tokens with model-selected expert
# ids, but AFD profiling needs deterministic balanced expert ids.
# Patch functionality: preserves the target upstream tag's W8A8 apply path and
# replaces only layer-owned force-load-balance top-k ids with AFD deterministic
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
) -> torch.Tensor:
    zero_expert_num = getattr(layer, "zero_expert_num", 0)
    zero_expert_type = getattr(layer, "zero_expert_type", None)
    n_shared_experts = getattr(layer, "n_shared_experts", 0)
    mix_placement = getattr(layer, "mix_placement", False)
    if n_shared_experts is None:
        n_shared_experts = 0
    valid_global_expert_num = num_experts - n_shared_experts
    if zero_expert_num == 0 or zero_expert_type is None:
        assert router_logits.shape[1] == valid_global_expert_num, (
            "Number of global experts mismatch (excluding redundancy)"
        )

    if self.multistream_overlap_gate:
        fc3_context = get_flash_common3_context()
        assert fc3_context is not None
        topk_weights = fc3_context.topk_weights
        topk_ids = fc3_context.topk_ids
    else:
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
            num_experts=num_experts,
        )
    assert topk_ids is not None
    assert topk_weights is not None
    if zero_expert_num > 0 and zero_expert_type is not None:
        topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
            expert_indices=topk_ids,
            expert_scales=topk_weights,
            num_experts=num_experts,
            zero_expert_type=zero_expert_type,
            hidden_states=x,
        )

    # this is a naive implementation for experts load balance so as
    # to avoid accumulating too much tokens on a single rank.
    # currently it is only activated when doing profile runs.
    if enable_force_load_balance:
        random_matrix = torch.rand(topk_ids.size(0), num_experts, device=topk_ids.device)
        topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(
            topk_ids.dtype
        )

    # ### PATCH START: AFD force-load-balance W8A8 routing override
    # Replace layer-owned profiling topk ids with deterministic balanced ids.
    elif getattr(layer, "enable_force_load_balance", False):
        fake_routed_topk_ids = _get_force_lb_topk_ids(
            layer,
            batch_tokens=topk_ids.shape[0],
            device=topk_ids.device,
        )
        fake_routed_topk_ids = fake_routed_topk_ids.to(topk_ids.dtype)
        if getattr(layer, "mix_placement", False):
            shared_topk_ids = topk_ids[:, top_k:]
            topk_ids = torch.cat([fake_routed_topk_ids, shared_topk_ids], dim=1)
        else:
            topk_ids = fake_routed_topk_ids
    # ### PATCH END: AFD force-load-balance W8A8 routing override

    assert topk_weights is not None
    topk_weights = topk_weights.to(self.in_dtype)

    moe_comm_method = _EXTRA_CTX.moe_comm_method
    fused_scale_flag = (
        _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2
        and envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1
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
            layer.fused_w2_scale_list if fused_scale_flag else layer.w2_weight_scale_list
        )
    else:
        w1 = [layer.w13_weight]
        w1_scale = (
            [layer.fused_w1_scale] if fused_scale_flag else [layer.w13_weight_scale_fp32]
        )
        w2 = [layer.w2_weight]
        w2_scale = [layer.fused_w2_scale] if fused_scale_flag else [layer.w2_weight_scale]

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
        )
    )
    if zero_expert_num > 0 and zero_expert_type is not None:
        final_hidden_states += zero_expert_result
    return final_hidden_states


AscendFusedMoE.__init__ = __init__
AscendW8A8DynamicFusedMoEMethod.apply = apply


__all__: list[str] = []
