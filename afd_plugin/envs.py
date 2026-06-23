# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Environment-variable helpers for AFD plugin runtime diagnostics."""

from __future__ import annotations

import os

AFD_CAMP2P_STUB_IO = "AFD_CAMP2P_STUB_IO"
AFD_FORCE_BALANCED_TOPK_IDS = "AFD_FORCE_BALANCED_TOPK_IDS"
AFD_OFFLINE_SCHEDULER_CSV = "AFD_OFFLINE_SCHEDULER_CSV"
AFD_OFFLINE_SCHEDULER_DP_RANK = "AFD_OFFLINE_SCHEDULER_DP_RANK"
AFD_OFFLINE_SCHEDULER_DP_SIZE = "AFD_OFFLINE_SCHEDULER_DP_SIZE"
AFD_OFFLINE_SCHEDULER_REQUEST_INDICES = "AFD_OFFLINE_SCHEDULER_REQUEST_INDICES"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def camp2p_stub_io_enabled() -> bool:
    return os.environ.get(AFD_CAMP2P_STUB_IO, "").lower() in ENV_TRUE_VALUES


def force_balanced_topk_ids_enabled() -> bool:
    return os.environ.get(AFD_FORCE_BALANCED_TOPK_IDS, "").lower() in ENV_TRUE_VALUES


__all__ = [
    "AFD_CAMP2P_STUB_IO",
    "AFD_FORCE_BALANCED_TOPK_IDS",
    "AFD_OFFLINE_SCHEDULER_CSV",
    "AFD_OFFLINE_SCHEDULER_DP_RANK",
    "AFD_OFFLINE_SCHEDULER_DP_SIZE",
    "AFD_OFFLINE_SCHEDULER_REQUEST_INDICES",
    "camp2p_stub_io_enabled",
    "force_balanced_topk_ids_enabled",
]
