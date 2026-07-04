# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD model wrapper namespace."""

from afd_plugin.model_executor.models.forward_context import (
    ASYNC_MOE_UBATCH_METADATA_KEY,
    get_afd_metadata_from_forward_context,
    get_async_moe_ubatch_metadata_from_forward_context,
)

__all__ = [
    "ASYNC_MOE_UBATCH_METADATA_KEY",
    "get_afd_metadata_from_forward_context",
    "get_async_moe_ubatch_metadata_from_forward_context",
]
