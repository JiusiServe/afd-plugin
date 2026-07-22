# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exact vLLM-Ascend version gate for the pinned NPU adapters."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from typing import Final

TARGET_VLLM_ASCEND_VERSION: Final[str] = "0.19.1rc1"
_SUPPORTED_VERSION = re.compile(r"^0\.19\.1rc1(?:\+[a-z0-9.-]+)?$", re.IGNORECASE)


def get_installed_vllm_ascend_version() -> str | None:
    try:
        return version("vllm-ascend")
    except PackageNotFoundError:
        return None


def is_vllm_ascend_version_supported(
    installed_version: str | None = None,
) -> bool:
    if installed_version is None:
        installed_version = get_installed_vllm_ascend_version()
    return bool(installed_version and _SUPPORTED_VERSION.fullmatch(installed_version))


def assert_vllm_ascend_version_supported() -> None:
    installed_version = get_installed_vllm_ascend_version()
    if is_vllm_ascend_version_supported(installed_version):
        return
    raise RuntimeError(
        "AFD NPU adapters currently support exactly vLLM-Ascend "
        f"{TARGET_VLLM_ASCEND_VERSION}; installed vLLM-Ascend version is "
        f"{installed_version or 'not installed'}"
    )


__all__ = [
    "assert_vllm_ascend_version_supported",
    "get_installed_vllm_ascend_version",
    "is_vllm_ascend_version_supported",
    "TARGET_VLLM_ASCEND_VERSION",
]
