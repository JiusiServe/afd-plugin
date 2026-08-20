# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit coverage for the narrow DSV4 Async CAM feature contract."""

from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.feature_validation import (
    _fail_if_unsupported_dsv4_async_features,
)
from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo


def _afd_config(*, compute_gate_on_attention: bool) -> SimpleNamespace:
    return SimpleNamespace(compute_gate_on_attention=compute_gate_on_attention)


def test_dsv4_async_requires_attention_side_gate() -> None:
    with pytest.raises(RuntimeError, match="compute_gate_on_attention"):
        _fail_if_unsupported_dsv4_async_features(
            _afd_config(compute_gate_on_attention=False),
            AFDAsyncExtraInfo(dynamic_quant=1),
        )


@pytest.mark.parametrize(
    ("extra_info", "message"),
    [
        (
            AFDAsyncExtraInfo(dynamic_quant=0),
            "dynamicQuant=1",
        ),
        (
            AFDAsyncExtraInfo(dynamic_quant=1, async_moe_ubatching=True),
            "async_moe_ubatching",
        ),
    ],
)
def test_dsv4_async_rejects_deferred_features(
    extra_info: AFDAsyncExtraInfo,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _fail_if_unsupported_dsv4_async_features(
            _afd_config(compute_gate_on_attention=True),
            extra_info,
        )


def test_dsv4_async_accepts_target_baseline() -> None:
    _fail_if_unsupported_dsv4_async_features(
        _afd_config(compute_gate_on_attention=True),
        AFDAsyncExtraInfo(dynamic_quant=1, attn_ranks_per_dp=2),
    )
