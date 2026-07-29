from contextlib import nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from afd_plugin.v1.worker.npu import multistream


class _FakeStream:
    def __init__(self):
        self.waited_for = None

    def wait_stream(self, stream):
        self.waited_for = stream


def test_stream_switch_waits_for_compute_stream(monkeypatch):
    comm_stream = _FakeStream()
    monkeypatch.setattr(
        multistream.torch,
        "npu",
        SimpleNamespace(stream=nullcontext),
        raising=False,
    )

    with multistream.npu_stream_switch_within_graph(
        "compute",
        comm_stream,
        enabled=True,
    ):
        pass

    assert comm_stream.waited_for == "compute"


def test_stream_switch_disabled_accepts_missing_streams():
    with multistream.npu_stream_switch_within_graph(None, None, enabled=False):
        pass


@pytest.mark.parametrize(
    ("compute_stream", "comm_stream"),
    [(None, object()), (object(), None)],
)
def test_stream_switch_requires_both_streams(compute_stream, comm_stream):
    with pytest.raises(RuntimeError, match="requires compute and communication"):
        multistream.npu_stream_switch_within_graph(
            compute_stream,
            comm_stream,
            enabled=True,
        )
