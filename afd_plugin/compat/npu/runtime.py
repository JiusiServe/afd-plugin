# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Public Ascend runtime compatibility facade."""

from __future__ import annotations

from afd_plugin.compat.npu.feature_validation import (
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.compat.npu.forward_context import (
    ascend_forward_context,
)
from afd_plugin.compat.npu.runtime_config import (
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)

_PATCHES_APPLIED = False


def apply_afd_ascend_patches_if_needed() -> None:
    """Apply plugin-owned, AFD-scoped Ascend patches."""

    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    from afd_plugin.compat.patches.npu.ascend_platform import (
        apply_afd_ascend_dbo_config_patch,
    )

    if apply_afd_ascend_dbo_config_patch():
        _PATCHES_APPLIED = True


__all__ = [
    "apply_afd_ascend_patches_if_needed",
    "ascend_forward_context",
    "fail_if_unsupported_npu_afd_features",
    "fix_all2all_backend_for_afd",
    "npu_afd_num_ubatches",
]
