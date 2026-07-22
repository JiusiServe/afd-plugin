# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Frozen vLLM-Ascend version gate tests."""

from __future__ import annotations

import pytest

from afd_plugin.compat.npu import version as ascend_version


@pytest.mark.parametrize(
    ("installed_version", "supported"),
    [
        ("0.19.1rc1", True),
        ("0.19.1rc1+vendor.1", True),
        ("0.19.1", False),
        ("0.19.1rc2", False),
        ("0.20.0", False),
        (None, False),
    ],
)
def test_vllm_ascend_version_support_is_exact(installed_version, supported) -> None:
    assert (
        ascend_version.is_vllm_ascend_version_supported(installed_version) is supported
    )


def test_vllm_ascend_version_gate_reports_installed_version(monkeypatch) -> None:
    monkeypatch.setattr(
        ascend_version,
        "get_installed_vllm_ascend_version",
        lambda: "0.20.0",
    )

    with pytest.raises(RuntimeError, match="0.19.1rc1.*0.20.0"):
        ascend_version.assert_vllm_ascend_version_supported()
