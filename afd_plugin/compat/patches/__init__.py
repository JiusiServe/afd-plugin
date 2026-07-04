# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Isolated monkey-patch namespace.

Patches in this package must remain idempotent, version-aware, documented, and
covered by CPU-safe tests whenever possible.
"""

from afd_plugin.compat.patches.config_validation import apply_config_validation_patch
from afd_plugin.compat.patches.engine_core import apply_engine_core_patch

__all__ = [
    "apply_async_dp_engine_patch",
    "apply_async_dp_forward_context_patch",
    "apply_config_validation_patch",
    "apply_engine_core_patch",
]


def __getattr__(name: str):
    if name == "apply_async_dp_engine_patch":
        from afd_plugin.compat.patches.async_dp_engine import (
            apply_async_dp_engine_patch,
        )

        return apply_async_dp_engine_patch
    if name == "apply_async_dp_forward_context_patch":
        from afd_plugin.compat.patches.async_dp_forward_context import (
            apply_async_dp_forward_context_patch,
        )

        return apply_async_dp_forward_context_patch
    if name == "apply_engine_core_patch":
        from afd_plugin.compat.patches.engine_core import apply_engine_core_patch

        return apply_engine_core_patch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
