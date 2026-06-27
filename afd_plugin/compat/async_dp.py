# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD async-DP compatibility helpers."""

from __future__ import annotations

from afd_plugin.config import AFD_ASYNC_CONNECTOR, AFDConfig, parse_afd_config


def is_afd_async_dp(vllm_config: object) -> bool:
    """Return whether ``vllm_config`` selects AFD's async connector mode."""

    config = parse_afd_config(vllm_config, validate=False)
    return _is_async_dp_config(config)


def is_afd_async_attention_dp(vllm_config: object) -> bool:
    """Return whether this config is an async-DP Attention-side engine."""

    config = parse_afd_config(vllm_config, validate=False)
    return _is_async_dp_config(config) and config.role == "attention"


def _is_async_dp_config(config: AFDConfig) -> bool:
    return (
        config.enabled
        and config.async_dp
        and config.connector == AFD_ASYNC_CONNECTOR
    )


__all__ = [
    "AFD_ASYNC_CONNECTOR",
    "is_afd_async_attention_dp",
    "is_afd_async_dp",
]
