# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The Attention-side transfer traced once must be right for every token count.

vLLM compiles the model forward once with a symbolic token dimension and
drops all guards (``TorchCompileWithNoGuardsWrapper``), then reuses that
trace for every later batch. ``send_attn_output`` and ``recv_ffn_output``
run inside that trace, so their share arithmetic must not depend on
anything frozen at trace time: no Python-int sizes from the metadata and no
branch on the remainder (a ``divmod``-based split traced at 64 tokens would
keep sending (n//2, n//2) — two rows for five tokens).

These tests compile the connector methods with the same options vLLM uses,
trace at 64 tokens, and then call the compiled code with other token counts
through custom ops that record what actually went on the wire.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.distributed import split_send_sizes  # noqa: E402

_HIDDEN = 8
_SENT: list[tuple[int, int]] = []
_RECEIVED: list[tuple[int, int]] = []


# Custom ops with fake implementations stand in for the NCCL ops so the
# transfer survives tracing and the recorded sizes are runtime values. Like
# the real slice ops they take whole tensors plus (start, size).
@torch.library.custom_op("afd_test::record_send_slice", mutates_args=("base",))
def _record_send_slice(base: torch.Tensor, start: int, size: int, dst: int) -> None:
    _SENT.append((dst, int(size)))


@_record_send_slice.register_fake
def _(base, start, size, dst):
    return None


@torch.library.custom_op("afd_test::record_recv_slice", mutates_args=("out",))
def _record_recv_slice(
    base: torch.Tensor, out: torch.Tensor, start: int, size: int, src: int
) -> None:
    _RECEIVED.append((src, int(size)))
    out.narrow(0, start, size).fill_(float(src))


@_record_recv_slice.register_fake
def _(base, out, start, size, src):
    return None


@torch.library.custom_op("afd_test::record_send", mutates_args=("tensor",))
def _record_send(tensor: torch.Tensor, dst: int) -> None:
    _SENT.append((dst, int(tensor.shape[0])))


@_record_send.register_fake
def _(tensor, dst):
    return None


def _attention_connector(attention, ffn):
    text_config = SimpleNamespace(hidden_size=_HIDDEN, num_hidden_layers=2)
    vllm_config = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            dtype=torch.float32,
            enforce_eager=False,
            hf_config=text_config,
            hf_text_config=text_config,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=attention,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
    )
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        vllm_config,
        AFDConfig(
            role="attention",
            connector="P2pNcclAFDConnector",
            num_attention_ranks=attention,
            num_ffn_ranks=ffn,
        ),
    )
    connector.control_plane.update_state_from_dp_metadata(
        AFDControlPayload(
            dp_metadata_list={0: AFDDPMetadata([64] * attention)},
            is_graph_capturing=False,
            is_warmup=False,
        ),
    )
    connector._send_hidden_states = lambda hidden, dst, group, comm_id: (
        torch.ops.afd_test.record_send(hidden, dst)
    )
    connector._send_slice = lambda base, start, size, dst: (
        torch.ops.afd_test.record_send_slice(base, start, size, dst)
    )
    connector._recv_slice = lambda base, out, start, size, src: (
        torch.ops.afd_test.record_recv_slice(base, out, start, size, src)
    )
    return connector


def _compile_like_vllm(fn):
    return torch.compile(
        fn,
        fullgraph=True,
        dynamic=False,
        backend="aot_eager",
        options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    )


def _context(seq_len):
    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=seq_len,
        ),
    )


@pytest.mark.parametrize("ffn", [2, 4])
def test_compiled_send_splits_every_token_count_like_the_receivers_expect(ffn):
    torch._dynamo.reset()
    connector = _attention_connector(1, ffn)
    context = _context(64)
    compiled = _compile_like_vllm(
        lambda hidden: connector.send_attn_output(hidden, context),
    )
    traced = torch.zeros((64, _HIDDEN))
    torch._dynamo.mark_dynamic(traced, 0)
    compiled(traced)
    for tokens in (64, 5, 7, 1, 128, 0):
        _SENT.clear()
        compiled(torch.zeros((tokens, _HIDDEN)))
        assert list(enumerate(split_send_sizes(tokens, ffn))) == _SENT, tokens


def test_compiled_receive_fills_each_share_in_member_order():
    torch._dynamo.reset()
    connector = _attention_connector(1, 2)
    compiled = _compile_like_vllm(
        lambda buffer: connector.recv_ffn_output(ref_tensor=buffer, ubatch_idx=0),
    )
    traced = torch.zeros((64, _HIDDEN))
    torch._dynamo.mark_dynamic(traced, 0)
    compiled(traced)
    for tokens in (5, 7, 1):
        _RECEIVED.clear()
        output = compiled(torch.full((tokens, _HIDDEN), -1.0))
        first, second = split_send_sizes(tokens, 2)
        assert [(0, first), (1, second)] == _RECEIVED, tokens
        assert output.shape == (tokens, _HIDDEN)
        assert torch.equal(output[:first], torch.zeros((first, _HIDDEN)))
        assert torch.equal(output[first:], torch.ones((second, _HIDDEN)))


def test_single_member_compiled_send_is_one_whole_send():
    torch._dynamo.reset()
    connector = _attention_connector(2, 1)
    context = _context(64)
    compiled = _compile_like_vllm(
        lambda hidden: connector.send_attn_output(hidden, context),
    )
    traced = torch.zeros((64, _HIDDEN))
    torch._dynamo.mark_dynamic(traced, 0)
    compiled(traced)
    for tokens in (5, 1, 128):
        _SENT.clear()
        compiled(torch.zeros((tokens, _HIDDEN)))
        assert [(0, tokens)] == _SENT
