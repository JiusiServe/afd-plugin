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

CAM_ATTENTION_ELEMENT_SIZE_BYTES = 2
CAM_DYNAMIC_QUANT_MOE_TOKEN_SIZE_BYTES = 6176
CAM_NON_QUANT_MOE_TOKEN_SIZE_BYTES = 12288
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
    moe_required_bytes: int
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
            return self.moe_required_bytes
        raise ValueError(f"unsupported AFD role for CAM buffer sizing: {role!r}")


def derive_cam_hccl_buffer_plan(
    *,
    hidden_size: int,
    max_batch_tokens: int,
    num_npus_per_dp_group: int,
    topk: int,
    attention_rank_size: int,
    dynamic_quant: int,
) -> CAMHCCLBufferPlan:
    """Derive independent Attention and FFN HCCL buffers with 10% headroom.

    The Attention side includes routed and shared-expert payloads, represented
    by ``topk + 1``. The MoE side uses the CAM-provided per-token byte width and
    intentionally has no ``topk + 1`` multiplier.
    """
    if dynamic_quant not in (0, 1):
        raise ValueError(f"dynamic_quant must be 0 or 1, got {dynamic_quant}")
    if num_npus_per_dp_group <= 0:
        raise ValueError(
            f"num_npus_per_dp_group must be positive, got {num_npus_per_dp_group}",
        )

    tokens_per_npu = _ceil_div(max_batch_tokens, num_npus_per_dp_group)
    attention_required_bytes = (
        CAM_ATTENTION_ELEMENT_SIZE_BYTES * hidden_size * tokens_per_npu * (topk + 1)
    )
    moe_token_size_bytes = (
        CAM_DYNAMIC_QUANT_MOE_TOKEN_SIZE_BYTES
        if dynamic_quant
        else CAM_NON_QUANT_MOE_TOKEN_SIZE_BYTES
    )
    moe_required_bytes = attention_rank_size * moe_token_size_bytes * tokens_per_npu
    attention_buffered_bytes = _ceil_div(
        attention_required_bytes * CAM_BUFFER_SAFETY_FACTOR_NUMERATOR,
        CAM_BUFFER_SAFETY_FACTOR_DENOMINATOR,
    )
    moe_buffered_bytes = _ceil_div(
        moe_required_bytes * CAM_BUFFER_SAFETY_FACTOR_NUMERATOR,
        CAM_BUFFER_SAFETY_FACTOR_DENOMINATOR,
    )
    return CAMHCCLBufferPlan(
        attention_required_bytes=attention_required_bytes,
        moe_required_bytes=moe_required_bytes,
        attention_buffer_size_mb=_ceil_div(
            attention_buffered_bytes,
            MEBIBYTE,
        ),
        ffn_buffer_size_mb=_ceil_div(moe_buffered_bytes, MEBIBYTE),
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
    elif afd_config.connector == "CAMP2pAFDConnector":
        # CAMP2P currently supports TP as the only intra-DP NPU dimension.
        num_npus_per_dp_group = int(
            vllm_config.parallel_config.tensor_parallel_size,
        )
        dynamic_quant = 0
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
        attention_rank_size=afd_config.num_attention_ranks,
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
