# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU stream helpers used by CAMP2P communication."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def npu_stream_switch_within_graph(
    current_stream: torch.npu.Stream | None,
    target_stream: torch.npu.Stream | None,
    enabled: bool,
) -> AbstractContextManager[None]:
    """Switch to ``target_stream`` after ``current_stream`` when enabled.

    Return a no-op context when multi-stream execution is disabled. Both
    streams are required when it is enabled.
    """
    if not enabled:
        return nullcontext()
    if current_stream is None or target_stream is None:
        raise RuntimeError(
            "CAMP2P multistream requires compute and communication streams",
        )
    target_stream.wait_stream(current_stream)
    return torch.npu.stream(target_stream)


__all__ = ["npu_stream_switch_within_graph"]
