# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend microbatch context switching for plugin-owned DBO."""

import threading

import torch
import torch_npu  # noqa: F401
from vllm import forward_context
from vllm.forward_context import ForwardContext

_THREAD_ID_TO_CONTEXT: dict[int, int] = {}
_CURRENT_CONTEXTS: list["AscendUBatchContext | None"] = []
_DBO_CURRENT_STREAM = threading.local()


class AscendUBatchContext:
    """Minimal context for sequential two-ubatch execution on Ascend."""

    def __init__(
        self,
        id: int,
        compute_stream: torch.npu.Stream,
        forward_context: ForwardContext,
        ready_barrier: threading.Barrier,
        cpu_wait_event: threading.Event,
        cpu_signal_event: threading.Event,
    ):
        self.id = id
        self.compute_stream = compute_stream
        self.forward_context = forward_context
        self.ready_barrier = ready_barrier
        self.cpu_wait_event = cpu_wait_event
        self.cpu_signal_event = cpu_signal_event
        self.current_stream = compute_stream

    def __enter__(self):
        _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
        _CURRENT_CONTEXTS[self.id] = self
        self.ready_barrier.wait()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()
        self.update_stream(self.compute_stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _CURRENT_CONTEXTS[self.id] = None
        del _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()
        return False

    def _restore_context(self):
        forward_context._forward_context = self.forward_context

    def update_stream(self, stream: torch.npu.Stream):
        self.current_stream = stream
        if dbo_current_stream() != stream:
            dbo_set_stream(stream)

    def _cpu_yield(self):
        assert forward_context._forward_context == self.forward_context
        assert dbo_current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()

        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()
        self.update_stream(self.current_stream)

    def yield_(self):
        self.current_stream = dbo_current_stream()
        self._cpu_yield()


def dbo_current_stream() -> torch.npu.Stream:
    if not hasattr(_DBO_CURRENT_STREAM, "value") or _DBO_CURRENT_STREAM.value is None:
        _DBO_CURRENT_STREAM.value = torch.npu.current_stream()
    return _DBO_CURRENT_STREAM.value


def dbo_set_stream(stream: torch.npu.Stream) -> None:
    _DBO_CURRENT_STREAM.value = stream
    torch.npu.set_stream(stream)


def dbo_enabled() -> bool:
    return len(_THREAD_ID_TO_CONTEXT) > 0


def dbo_current_ubatch_id() -> int:
    if not _THREAD_ID_TO_CONTEXT:
        return 0
    return _THREAD_ID_TO_CONTEXT[threading.get_ident()]


def _register_ubatch_function(func):
    def wrapper(*args, **kwargs):
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            assert ctx is not None
            return func(ctx, *args, **kwargs)
        return None

    return wrapper


dbo_yield = _register_ubatch_function(AscendUBatchContext.yield_)


def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.npu.Stream,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
) -> list[AscendUBatchContext]:
    assert num_micro_batches == 2, (
        "Ascend ubatching currently supports exactly 2 ubatches."
    )
    if len(_CURRENT_CONTEXTS) < num_micro_batches:
        _CURRENT_CONTEXTS.extend([None] * (num_micro_batches - len(_CURRENT_CONTEXTS)))

    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    return [
        AscendUBatchContext(
            id=i,
            compute_stream=compute_stream,
            forward_context=forward_contexts[i],
            ready_barrier=ready_barrier,
            cpu_wait_event=cpu_events[i],
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
        )
        for i in range(num_micro_batches)
    ]


__all__ = [
    "AscendUBatchContext",
    "dbo_current_ubatch_id",
    "dbo_current_stream",
    "dbo_enabled",
    "dbo_set_stream",
    "dbo_yield",
    "make_ubatch_contexts",
]
