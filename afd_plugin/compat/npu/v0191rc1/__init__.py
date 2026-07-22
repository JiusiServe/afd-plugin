# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compatibility boundary for vLLM-Ascend v0.19.1rc1."""

from afd_plugin.compat.npu.v0191rc1.model_runner import (
    AscendDBOCompatV0191rc1,
)

__all__ = ["AscendDBOCompatV0191rc1"]
