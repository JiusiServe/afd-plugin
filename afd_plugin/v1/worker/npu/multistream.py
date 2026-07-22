# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU stream helpers used by CAMP2P communication."""

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch


def npu_stream_switch_within_graph(
    current_stream: Any,
    target_stream: Any,
    enabled: bool,
) -> AbstractContextManager:
    """Make ``target_stream`` depend on compute and execute on that stream."""
    if not enabled:
        return nullcontext()
    if current_stream is None or target_stream is None:
        raise RuntimeError(
            "CAMP2P multistream requires compute and communication streams",
        )
    target_stream.wait_stream(current_stream)
    return torch.npu.stream(target_stream)


__all__ = ["npu_stream_switch_within_graph"]
