# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD async-DP compatibility helpers."""

from __future__ import annotations

from afd_plugin.config import AFDConfig, parse_afd_config

AFD_ASYNC_CONNECTOR: str = "afdasyncconnector"


def is_afd_async_dp(vllm_config: object) -> bool:
    """Return whether ``vllm_config`` selects AFD's async connector mode."""

    config = parse_afd_config(vllm_config, validate=False)
    return _is_async_connector_config(config)


def is_afd_async_attention_dp(vllm_config: object) -> bool:
    """Return whether this config is an async-DP Attention-side engine."""

    config = parse_afd_config(vllm_config, validate=False)
    return _is_async_connector_config(config) and config.role == "attention"


def ensure_async_dp_compat_attr(vllm_config: object) -> bool:
    """Mirror plugin async-DP mode onto ``parallel_config.async_dp``.

    The public configuration remains ``additional_config["afd"]``.  This
    instance-local attribute is only a bridge for vLLM/vLLM-Ascend code paths
    that were originally written against an in-tree ``ParallelConfig.async_dp``
    field.
    """

    enabled = is_afd_async_dp(vllm_config)
    vllm_config.parallel_config.async_dp = enabled
    return enabled


def parallel_config_async_dp(parallel_config: object) -> bool:
    """Return the mirrored async-DP flag from a vLLM ``ParallelConfig``."""

    try:
        return bool(parallel_config.async_dp)
    except AttributeError:
        return False


def _is_async_connector_config(config: AFDConfig) -> bool:
    return config.enabled and config.connector == AFD_ASYNC_CONNECTOR


__all__ = [
    "AFD_ASYNC_CONNECTOR",
    "ensure_async_dp_compat_attr",
    "is_afd_async_attention_dp",
    "is_afd_async_dp",
    "parallel_config_async_dp",
]
