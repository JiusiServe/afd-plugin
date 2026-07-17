# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Validation for AFD features supported by the Ascend runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from afd_plugin.config import (
    ASYNC_MOE_REQUEST_SPLIT,
    AFDConfig,
    async_moe_num_ubatches,
    async_moe_split,
    async_moe_ubatching_enabled,
    is_afd_async_dp,
    parse_afd_config,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def fail_if_unsupported_npu_afd_features(vllm_config: VllmConfig) -> None:
    """Fail fast for NPU AFD settings that are not currently supported."""

    afd_config = parse_afd_config(vllm_config)
    extra = afd_config.extra_config or {}
    if afd_config.connector == "CAMAsyncAFDConnector":
        _fail_if_unsupported_npu_afd_async_features(vllm_config, afd_config)
        return

    if afd_config.compute_gate_on_attention or _truthy(
        extra.get("compute_gate_on_attention"),
    ):
        raise RuntimeError(
            "AFD NPU runtime does not support compute_gate_on_attention=true yet",
        )

    quant_mode = extra.get("quant_mode", 0)
    if quant_mode not in (None, "", 0, "0"):
        raise RuntimeError("AFD NPU runtime currently supports only quant_mode=0")

    if bool(vllm_config.parallel_config.use_ubatching) and (
        int(vllm_config.parallel_config.num_ubatches) != 2
    ):
        raise RuntimeError(
            "AFD NPU runtime supports exactly two ubatches when DBO is enabled",
        )


def _fail_if_unsupported_npu_afd_async_features(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    extra = afd_config.extra_config or {}
    parallel_config = vllm_config.parallel_config
    if not is_afd_async_dp(vllm_config):
        raise RuntimeError(
            "CAMAsyncAFDConnector requires additional_config['afd'] "
            "with async=true and connector='CAMAsyncAFDConnector'",
        )
    if not bool(vllm_config.model_config.enforce_eager):
        raise RuntimeError(
            "CAMAsyncAFDConnector supports only eager Attention/FFN execution",
        )
    if bool(parallel_config.use_ubatching):
        raise RuntimeError(
            "CAMAsyncAFDConnector does not support vLLM native ubatching/DBO",
        )
    if async_moe_ubatching_enabled(afd_config):
        _fail_if_unsupported_npu_async_moe_ubatching_features(
            vllm_config,
            afd_config,
        )
    quant_mode = extra.get("dynamicQuant", 0)
    if quant_mode not in (None, "", 0, "0", 1, "1"):
        raise RuntimeError(
            "CAMAsyncAFDConnector currently supports only dynamicQuant 0 or 1",
        )


def _fail_if_unsupported_npu_async_moe_ubatching_features(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    parallel_config = vllm_config.parallel_config
    if not bool(afd_config.compute_gate_on_attention):
        raise RuntimeError(
            "async_moe_ubatching requires compute_gate_on_attention=true",
        )
    num_ubatches = async_moe_num_ubatches(afd_config)
    if num_ubatches != 2:
        raise RuntimeError(
            "async_moe_ubatching currently supports exactly two stages; "
            f"got async_moe_num_ubatches={num_ubatches}",
        )
    split = async_moe_split(afd_config)
    if split != ASYNC_MOE_REQUEST_SPLIT:
        raise RuntimeError(
            "async_moe_ubatching currently supports only request-boundary split; "
            f"got async_moe_split={split!r}",
        )
    if int(parallel_config.decode_context_parallel_size) > 1:
        raise RuntimeError(
            "async_moe_ubatching does not support decode context parallel metadata yet",
        )


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = ["fail_if_unsupported_npu_afd_features"]
