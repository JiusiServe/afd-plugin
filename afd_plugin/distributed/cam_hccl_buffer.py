# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CAM HCCL buffer sizing and memory-headroom warnings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.config import AFDConfig

CAM_BFLOAT16_ELEMENT_SIZE_BYTES = 2
CAM_INT8_ELEMENT_SIZE_BYTES = 1
CAM_DYNAMIC_SCALE_SIZE_BYTES = 4
CAM_BUFFER_SAFETY_FACTOR_NUMERATOR = 11
CAM_BUFFER_SAFETY_FACTOR_DENOMINATOR = 10
CAM_MEMORY_RESERVE_FACTOR_NUMERATOR = 5
CAM_MEMORY_RESERVE_FACTOR_DENOMINATOR = 2
MEBIBYTE = 1024**2

logger = logging.getLogger(__name__)


def _ceil_div(dividend: int, divisor: int) -> int:
    return (dividend + divisor - 1) // divisor


@dataclass(frozen=True, slots=True)
class CAMHCCLBufferPlan:
    """Derived role-local HCCL buffer requirements."""

    attention_required_bytes: int
    ffn_required_bytes: int
    attention_buffer_size_mb: int
    ffn_buffer_size_mb: int

    def buffer_size_mb_for_role(self, role: str) -> int:
        """Return the independently derived HCCL setting for one AFD role."""
        if role == "attention":
            return self.attention_buffer_size_mb
        if role == "ffn":
            return self.ffn_buffer_size_mb
        raise ValueError(f"unsupported AFD role for CAM buffer sizing: {role!r}")

    def required_bytes_for_role(self, role: str) -> int:
        """Return the pre-headroom byte requirement for one AFD role."""
        if role == "attention":
            return self.attention_required_bytes
        if role == "ffn":
            return self.ffn_required_bytes
        raise ValueError(f"unsupported AFD role for CAM buffer sizing: {role!r}")


def _cam_slot_row_size_bytes(
    hidden_size: int,
    *,
    compute_gate_on_attention: bool,
    dynamic_quant: int,
) -> int:
    """Return dispatch plus combine storage for one hidden-state row."""
    dispatch_element_size_bytes = CAM_BFLOAT16_ELEMENT_SIZE_BYTES
    dispatch_scale_size_bytes = 0
    if compute_gate_on_attention and dynamic_quant:
        dispatch_element_size_bytes = CAM_INT8_ELEMENT_SIZE_BYTES
        dispatch_scale_size_bytes = CAM_DYNAMIC_SCALE_SIZE_BYTES

    dispatch_row_size_bytes = (
        hidden_size * dispatch_element_size_bytes + dispatch_scale_size_bytes
    )
    combine_row_size_bytes = hidden_size * CAM_BFLOAT16_ELEMENT_SIZE_BYTES
    return dispatch_row_size_bytes + combine_row_size_bytes


def derive_cam_hccl_buffer_plan(
    *,
    hidden_size: int,
    max_batch_tokens: int,
    num_npus_per_dp_group: int,
    topk: int,
    num_routed_experts: int,
    attention_rank_size: int,
    ffn_rank_size: int,
    compute_gate_on_attention: bool,
    dynamic_quant: int,
) -> CAMHCCLBufferPlan:
    """Derive role-local one-slot CAM buffers with 10% headroom.

    A slot belongs to one Attention source rank. Its token capacity is the
    scheduler capacity divided across the TP/SP ranks in that Attention DP
    group. When the gate runs on Attention, the source slot reserves ``topk``
    routed rows and one fused shared-expert row, while an FFN slot reserves one
    fixed-length queue for every local routed expert. When the gate runs on
    FFN, its role-local size covers both the unexpanded cross-role transfer and
    the local-expert communication group.

    Dynamic quantization applies only to an Attention-gated dispatch row: its
    payload is INT8 hidden data plus one FP32 per-token scale. Combine output
    remains BF16.
    """
    if dynamic_quant not in (0, 1):
        raise ValueError(f"dynamic_quant must be 0 or 1, got {dynamic_quant}")
    positive_dimensions = {
        "hidden_size": hidden_size,
        "max_batch_tokens": max_batch_tokens,
        "num_npus_per_dp_group": num_npus_per_dp_group,
        "topk": topk,
        "num_routed_experts": num_routed_experts,
        "attention_rank_size": attention_rank_size,
        "ffn_rank_size": ffn_rank_size,
    }
    for dimension_name, dimension_value in positive_dimensions.items():
        if dimension_value <= 0:
            raise ValueError(
                f"{dimension_name} must be positive, got {dimension_value}",
            )

    slot_row_size_bytes = _cam_slot_row_size_bytes(
        hidden_size,
        compute_gate_on_attention=compute_gate_on_attention,
        dynamic_quant=dynamic_quant,
    )
    tokens_per_attention_rank = _ceil_div(
        max_batch_tokens,
        num_npus_per_dp_group,
    )
    local_routed_experts = _ceil_div(num_routed_experts, ffn_rank_size)
    if compute_gate_on_attention:
        attention_rows = tokens_per_attention_rank * (topk + 1)
        ffn_rows = tokens_per_attention_rank * local_routed_experts
    else:
        attention_sources_per_ffn_rank = _ceil_div(
            attention_rank_size,
            ffn_rank_size,
        )
        attention_rows = tokens_per_attention_rank
        ffn_rows = tokens_per_attention_rank * max(
            attention_sources_per_ffn_rank,
            local_routed_experts,
        )

    attention_required_bytes = attention_rows * slot_row_size_bytes
    ffn_required_bytes = ffn_rows * slot_row_size_bytes
    attention_buffered_bytes = _ceil_div(
        attention_required_bytes * CAM_BUFFER_SAFETY_FACTOR_NUMERATOR,
        CAM_BUFFER_SAFETY_FACTOR_DENOMINATOR,
    )
    ffn_buffered_bytes = _ceil_div(
        ffn_required_bytes * CAM_BUFFER_SAFETY_FACTOR_NUMERATOR,
        CAM_BUFFER_SAFETY_FACTOR_DENOMINATOR,
    )
    return CAMHCCLBufferPlan(
        attention_required_bytes=attention_required_bytes,
        ffn_required_bytes=ffn_required_bytes,
        attention_buffer_size_mb=_ceil_div(
            attention_buffered_bytes,
            MEBIBYTE,
        ),
        ffn_buffer_size_mb=_ceil_div(ffn_buffered_bytes, MEBIBYTE),
    )


def derive_cam_hccl_buffer_plan_from_config(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> CAMHCCLBufferPlan:
    """Derive CAM buffer sizes for either Ascend CAM connector."""
    from afd_plugin.config import (
        AFD_ASYNC_CONNECTOR,
        connector_extra_config_from_source,
    )
    from afd_plugin.config_utils import (
        coerce_extra_int,
        coerce_extra_positive_int,
    )

    extra_config = connector_extra_config_from_source(vllm_config)
    if afd_config.connector == AFD_ASYNC_CONNECTOR:
        num_npus_per_dp_group = coerce_extra_positive_int(
            extra_config.get("attn_ranks_per_dp", 1),
            field_name="attn_ranks_per_dp",
        )
        dynamic_quant = coerce_extra_int(
            extra_config.get("dynamicQuant", 0),
            field_name="dynamicQuant",
        )
        compute_gate_on_attention = True
    elif afd_config.connector == "CAMP2pAFDConnector":
        # CAMP2P currently supports TP as the only intra-DP NPU dimension.
        num_npus_per_dp_group = int(
            vllm_config.parallel_config.tensor_parallel_size,
        )
        dynamic_quant = 0
        compute_gate_on_attention = False
    else:
        raise ValueError(
            "CAM HCCL buffer sizing requires CAMAsyncAFDConnector or "
            f"CAMP2pAFDConnector, got {afd_config.connector!r}",
        )

    hf_config = vllm_config.model_config.hf_config
    return derive_cam_hccl_buffer_plan(
        hidden_size=hf_config.hidden_size,
        max_batch_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        num_npus_per_dp_group=num_npus_per_dp_group,
        topk=hf_config.num_experts_per_tok,
        num_routed_experts=hf_config.n_routed_experts,
        attention_rank_size=afd_config.num_attention_ranks,
        ffn_rank_size=afd_config.num_ffn_ranks,
        compute_gate_on_attention=compute_gate_on_attention,
        dynamic_quant=dynamic_quant,
    )


def warn_if_cam_memory_headroom_is_low(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    total_device_memory_bytes: int,
) -> None:
    """Warn when configured utilization leaves less than 2.5 CAM buffers."""
    from afd_plugin.config import AFD_ASYNC_CONNECTOR

    if afd_config.connector not in (AFD_ASYNC_CONNECTOR, "CAMP2pAFDConnector"):
        return

    buffer_plan = derive_cam_hccl_buffer_plan_from_config(vllm_config, afd_config)
    buffer_size_mb = buffer_plan.buffer_size_mb_for_role(afd_config.role)
    buffer_size_bytes = buffer_size_mb * MEBIBYTE
    required_reserve_bytes = _ceil_div(
        buffer_size_bytes * CAM_MEMORY_RESERVE_FACTOR_NUMERATOR,
        CAM_MEMORY_RESERVE_FACTOR_DENOMINATOR,
    )
    gpu_memory_utilization = vllm_config.cache_config.gpu_memory_utilization
    configured_memory_bytes = int(total_device_memory_bytes * gpu_memory_utilization)
    available_reserve_bytes = max(
        0,
        total_device_memory_bytes - configured_memory_bytes,
    )
    if available_reserve_bytes >= required_reserve_bytes:
        return

    recommended_maximum_utilization = max(
        0.0,
        (total_device_memory_bytes - required_reserve_bytes)
        / total_device_memory_bytes,
    )
    logger.warning(
        "CAM %s %s rank has %d bytes outside gpu_memory_utilization, below "
        "the recommended %d bytes (2.5x its %d MB HCCL buffer); consider "
        "setting gpu_memory_utilization to %.6f or lower. The configured "
        "value %.6f is unchanged.",
        afd_config.connector,
        afd_config.role,
        available_reserve_bytes,
        required_reserve_bytes,
        buffer_size_mb,
        recommended_maximum_utilization,
        gpu_memory_utilization,
    )


__all__ = [
    "CAMHCCLBufferPlan",
    "derive_cam_hccl_buffer_plan",
    "derive_cam_hccl_buffer_plan_from_config",
    "warn_if_cam_memory_headroom_is_low",
]
