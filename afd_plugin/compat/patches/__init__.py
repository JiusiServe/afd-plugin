# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Isolated monkey-patch namespace.

Patches in this package must remain idempotent, version-aware, documented, and
covered by CPU-safe tests whenever possible.
"""

__all__: list[str] = []


def __getattr__(name: str):
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
