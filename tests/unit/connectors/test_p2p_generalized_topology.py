# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Transport behavior of ``P2pNcclAFDConnector`` on generalized topologies.

The subgroup partition is now defined for any positive (A, F). The
existing connector tests cover the historical A >= F layouts; this file
covers the new shapes — A < F (one attention splitting across several FFN
members) and non-divisible A > F (uneven subgroups) — plus the historical
4A2F layout run through the same checks as a regression anchor.

All communication is mocked at the ``_send_hidden_states`` /
``_recv_hidden_states`` seam, the same one the existing tests use, so the
connector's own logic (who sends how many rows to whom, in which order)
runs unchanged on the CPU. The round-trip tests connect all ranks of a
topology through a fake wire (a dict of queues keyed by subgroup and
member ranks) so that sizes, addressing, and ordering are checked end to
end exactly as the real NCCL exchange would pair them.
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
from afd_plugin.connectors.gpu.p2p import _TensorMetadata  # noqa: E402

_HIDDEN = 16


def _fake_vllm_config(*, dp_size=1, dp_rank=0, enforce_eager=True):
    text_config = SimpleNamespace(hidden_size=_HIDDEN, num_hidden_layers=2)
    return SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            dtype=torch.float32,
            enforce_eager=enforce_eager,
            hf_config=text_config,
            hf_text_config=text_config,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
    )


def _connector(role, role_rank, attention, ffn, **config_kwargs):
    # The factory derives the role rank from the vLLM DP coordinates, so the
    # fake config carries this connector's own DP size/rank.
    role_size = attention if role == "attention" else ffn
    return AFDConnectorFactory.create_connector(
        role_rank,
        0,
        _fake_vllm_config(dp_size=role_size, dp_rank=role_rank, **config_kwargs),
        AFDConfig(
            role=role,
            connector="P2pNcclAFDConnector",
            num_attention_ranks=attention,
            num_ffn_ranks=ffn,
        ),
    )


def _all_connectors(attention, ffn):
    attentions = [_connector("attention", a, attention, ffn) for a in range(attention)]
    ffns = [_connector("ffn", f, attention, ffn) for f in range(ffn)]
    return attentions, ffns


def _payload(token_counts):
    return AFDControlPayload(
        dp_metadata_list={0: AFDDPMetadata(token_counts)},
        is_graph_capturing=False,
        is_warmup=False,
    )


def _apply(connectors, token_counts):
    for connector in connectors:
        connector.control_plane.update_state_from_dp_metadata(_payload(token_counts))
        # The dummy batch allocates on the stage metadata's device; point
        # it at the CPU so the tests can run without CUDA.
        old = connector.tensor_metadata_list[0]
        connector.tensor_metadata_list[0] = _TensorMetadata(
            torch.device("cpu"),
            old.dtype,
            old.size,
        )


def _attention_context(seq_len):
    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=seq_len,
        ),
    )


def _ffn_context(seq_lens):
    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_ffn_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_lens=seq_lens,
        ),
    )


def _record_sends(monkeypatch, connector):
    """Replace both send seams with a recorder of (rows, dst, tensor).

    Single-member subgroups send whole tensors through ``_send_hidden_states``;
    multi-member subgroups send (base, start, size) slices through
    ``_send_slice`` — recorded as the sliced rows.
    """
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_hidden_states",
        lambda hidden, dst, group, comm_id: sent.append((hidden.shape[0], dst, hidden)),
    )
    monkeypatch.setattr(
        connector,
        "_send_slice",
        lambda base, start, size, dst: sent.append(
            (size, dst, base.narrow(0, start, size))
        ),
    )
    return sent


def _fake_recvs(monkeypatch, connector, fill=None):
    """Replace the receive seam; returns the (src, rows) call log.

    Received tensors are filled with ``fill(src)`` (default: the source
    rank) so callers can tell which source each row came from.
    """
    received = []

    def fake_recv(src, group, comm_id, tensor_metadata, *, ref_tensor=None):
        received.append((src, tensor_metadata.size[0]))
        value = float(src) if fill is None else fill(src)
        tensor = torch.full(
            tuple(tensor_metadata.size),
            value,
            dtype=tensor_metadata.dtype,
        )
        if ref_tensor is not None:
            # The real op receives in place into the caller's buffer.
            ref_tensor.copy_(tensor)
            return ref_tensor
        return tensor

    def fake_recv_slice(base, out, start, size, src):
        received.append((src, size))
        value = float(src) if fill is None else fill(src)
        out.narrow(0, start, size).fill_(value)

    monkeypatch.setattr(connector, "_recv_hidden_states", fake_recv)
    monkeypatch.setattr(connector, "_recv_slice", fake_recv_slice)
    return received


# --------------------------------------------------------------------------
# 1. Construction: every topology is accepted in eager and graph mode alike
# --------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_multi_ffn_subgroup_is_accepted_in_both_modes(role, enforce_eager):
    connector = _connector(role, 0, 1, 2, enforce_eager=enforce_eager)
    assert connector.subgroup_ffn_count == 2


def test_registration_defines_the_four_transfer_ops():
    """Every topology registers the same ops, so the schemas are pinned here.

    ``base`` is declared mutated on the send and only read by the receive,
    which is what keeps receives ordered after sends in a compiled graph.
    """
    from afd_plugin.connectors.gpu import p2p

    p2p._register_p2p_custom_ops()
    schemas = {
        name: str(getattr(torch.ops.vllm, name).default._schema)
        for name in (
            "afd_p2p_send",
            "afd_p2p_recv",
            "afd_p2p_send_slice",
            "afd_p2p_recv_slice",
        )
    }
    assert schemas["afd_p2p_send"] == (
        "vllm::afd_p2p_send(Tensor(a0!) tensor, SymInt dst, SymInt comm_id) -> ()"
    )
    assert schemas["afd_p2p_recv"] == (
        "vllm::afd_p2p_recv(Tensor(a0!) out, SymInt src, SymInt comm_id) -> ()"
    )
    assert schemas["afd_p2p_send_slice"] == (
        "vllm::afd_p2p_send_slice(Tensor(a0!) base, SymInt start, SymInt size, "
        "SymInt dst, SymInt comm_id) -> ()"
    )
    assert schemas["afd_p2p_recv_slice"] == (
        "vllm::afd_p2p_recv_slice(Tensor base, Tensor(a1!) out, SymInt start, "
        "SymInt size, SymInt src, SymInt comm_id) -> ()"
    )


def test_historical_layout_still_allows_graph_mode():
    connector = _connector("attention", 1, 4, 2, enforce_eager=False)
    assert connector.subgroup_ffn_count == 1


# --------------------------------------------------------------------------
# 2. Size pre-computation (update_state_from_dp_metadata)
# --------------------------------------------------------------------------


def test_sizes_1a2f_split_across_two_ffn_members():
    # Group (F0, F1, A0): A0 sits at group rank 2 and splits 5 tokens 3/2.
    attentions, ffns = _all_connectors(1, 2)
    _apply(attentions + ffns, [5])
    assert attentions[0].tensor_metadata_list[0].size == torch.Size([5, _HIDDEN])
    expected = ffns[0]._recv_attn_tensor_metadata_list[(0, 2)].size
    assert expected == torch.Size([3, _HIDDEN])
    expected = ffns[1]._recv_attn_tensor_metadata_list[(0, 2)].size
    assert expected == torch.Size([2, _HIDDEN])
    assert ffns[0].tensor_metadata_list[0].size == torch.Size([3, _HIDDEN])
    assert ffns[1].tensor_metadata_list[0].size == torch.Size([2, _HIDDEN])
    assert all(not ffn._dummy_stages for ffn in ffns)


def test_sizes_1a4f_with_fewer_tokens_than_members_marks_dummy_stages():
    # 2 tokens over 4 FFN members: shares (1, 1, 0, 0). F2 and F3 receive
    # nothing this step and are marked for the dummy batch.
    attentions, ffns = _all_connectors(1, 4)
    _apply(attentions + ffns, [2])
    shares = [ffn._recv_attn_tensor_metadata_list[(0, 4)].size[0] for ffn in ffns]
    assert shares == [1, 1, 0, 0]
    assert [sorted(ffn._dummy_stages) for ffn in ffns] == [[], [], [0], [0]]


def test_sizes_6a4f_uneven_subgroups():
    # Subgroups: {F0,A0,A1} {F1,A2} {F2,A3,A4} {F3,A5}; token count = rank+1.
    attentions, ffns = _all_connectors(6, 4)
    _apply(attentions + ffns, [1, 2, 3, 4, 5, 6])

    def peers(ffn):
        return sorted(
            (src, md.size[0])
            for (_, src), md in ffn._recv_attn_tensor_metadata_list.items()
        )

    assert peers(ffns[0]) == [(1, 1), (2, 2)]
    assert peers(ffns[1]) == [(1, 3)]
    assert peers(ffns[2]) == [(1, 4), (2, 5)]
    assert peers(ffns[3]) == [(1, 6)]
    assert [ffn.tensor_metadata_list[0].size[0] for ffn in ffns] == [3, 3, 9, 6]


def test_sizes_4a2f_match_historical_layout():
    attentions, ffns = _all_connectors(4, 2)
    _apply(attentions + ffns, [3, 5, 7, 11])
    assert ffns[0]._recv_attn_tensor_metadata_list[(0, 1)].size[0] == 3
    assert ffns[0]._recv_attn_tensor_metadata_list[(0, 2)].size[0] == 5
    assert ffns[1]._recv_attn_tensor_metadata_list[(0, 1)].size[0] == 7
    assert ffns[1]._recv_attn_tensor_metadata_list[(0, 2)].size[0] == 11


# --------------------------------------------------------------------------
# 3. Attention send
# --------------------------------------------------------------------------


def test_attention_sends_slices_to_ffn_members_in_order(monkeypatch):
    attention = _connector("attention", 0, 1, 2)
    _apply([attention], [5])
    sent = _record_sends(monkeypatch, attention)
    hidden = torch.arange(5 * _HIDDEN, dtype=torch.float32).reshape(5, _HIDDEN)
    attention.send_attn_output(hidden, _attention_context(5))
    assert [(rows, dst) for rows, dst, _ in sent] == [(3, 0), (2, 1)]
    assert torch.equal(sent[0][2], hidden[:3])
    assert torch.equal(sent[1][2], hidden[3:])


def test_attention_sends_every_share_including_zero_size_ones(monkeypatch):
    # 2 tokens over 4 members: shares (1, 1, 0, 0). The zero-size shares are
    # sent too — the send path has no branch on the token count so that it
    # traces correctly under torch.compile (see test_p2p_compiled_send).
    attention = _connector("attention", 0, 1, 4)
    _apply([attention], [2])
    sent = _record_sends(monkeypatch, attention)
    attention.send_attn_output(torch.zeros((2, _HIDDEN)), _attention_context(2))
    assert [(rows, dst) for rows, dst, _ in sent] == [(1, 0), (1, 1), (0, 2), (0, 3)]


def test_attention_with_single_ffn_member_sends_the_whole_tensor(monkeypatch):
    # One FFN member: the upstream path — a single send of the tensor
    # object itself (no slicing, no view).
    attention = _connector("attention", 1, 4, 2)
    _apply([attention], [3, 5, 7, 11])
    sent = _record_sends(monkeypatch, attention)
    hidden = torch.zeros((5, _HIDDEN))
    attention.send_attn_output(hidden, _attention_context(5))
    assert len(sent) == 1
    assert sent[0][1] == 0
    assert sent[0][2] is hidden


# --------------------------------------------------------------------------
# 4. FFN receive (incl. the dummy batch)
# --------------------------------------------------------------------------


def test_ffn_receives_its_share_from_the_attention_member(monkeypatch):
    ffn1 = _connector("ffn", 1, 1, 2)
    _apply([ffn1], [5])
    received = _fake_recvs(monkeypatch, ffn1)
    payload = ffn1.recv_attn_output(ubatch_idx=0)
    assert received == [(2, 2)]  # attention member at group rank 2, 2 rows
    assert payload.hidden_states.shape == (2, _HIDDEN)
    assert payload.context.metadata.seq_lens == [2]


def test_ffn_with_nothing_to_receive_runs_on_a_dummy_batch(monkeypatch):
    ffn2 = _connector("ffn", 2, 1, 4)
    _apply([ffn2], [2])
    received = _fake_recvs(monkeypatch, ffn2)
    payload = ffn2.recv_attn_output(ubatch_idx=0)
    # The zero-size share is still received (the attention member sent it).
    assert received == [(4, 0)]
    assert payload.hidden_states.shape == (1, _HIDDEN)
    assert torch.equal(payload.hidden_states, torch.zeros((1, _HIDDEN)))
    assert payload.context.metadata.seq_lens == [1]
    assert 0 in ffn2._dummy_stages


# --------------------------------------------------------------------------
# 5. FFN return send
# --------------------------------------------------------------------------


def test_ffn_returns_whole_output_to_its_attention_member(monkeypatch):
    ffn1 = _connector("ffn", 1, 1, 2)
    _apply([ffn1], [5])
    sent = _record_sends(monkeypatch, ffn1)
    ffn1.send_ffn_output(torch.zeros((2, _HIDDEN)), _ffn_context([2]))
    assert [(rows, dst) for rows, dst, _ in sent] == [(2, 2)]


def test_ffn_on_a_dummy_stage_returns_a_zero_size_tensor(monkeypatch):
    # The dummy result is discarded; a zero-size tensor goes back to the
    # attention member (group rank 4) to match its posted receive.
    ffn2 = _connector("ffn", 2, 1, 4)
    _apply([ffn2], [2])
    _fake_recvs(monkeypatch, ffn2)
    payload = ffn2.recv_attn_output(ubatch_idx=0)
    sent = _record_sends(monkeypatch, ffn2)
    ffn2.send_ffn_output(payload.hidden_states, payload.context)
    assert [(rows, dst) for rows, dst, _ in sent] == [(0, 4)]


def test_ffn_with_several_attention_members_splits_by_seq_lens(monkeypatch):
    ffn0 = _connector("ffn", 0, 6, 4)  # subgroup {F0, A0, A1}
    _apply([ffn0], [1, 2, 3, 4, 5, 6])
    sent = _record_sends(monkeypatch, ffn0)
    ffn0.send_ffn_output(torch.zeros((3, _HIDDEN)), _ffn_context([1, 2]))
    assert [(rows, dst) for rows, dst, _ in sent] == [(1, 1), (2, 2)]


# --------------------------------------------------------------------------
# 6. Attention return receive
# --------------------------------------------------------------------------


def test_attention_reassembles_returns_in_member_order(monkeypatch):
    attention = _connector("attention", 0, 1, 2)
    _apply([attention], [5])
    received = _fake_recvs(monkeypatch, attention)
    buffer = torch.full((5, _HIDDEN), -1.0)
    output = attention.recv_ffn_output(ref_tensor=buffer, ubatch_idx=0)
    assert received == [(0, 3), (1, 2)]
    assert output.shape == (5, _HIDDEN)
    assert torch.equal(output[:3], torch.zeros((3, _HIDDEN)))
    assert torch.equal(output[3:], torch.ones((2, _HIDDEN)))


# --------------------------------------------------------------------------
# 7 + 8. Round trip over a fake wire
# --------------------------------------------------------------------------


def _wire(monkeypatch, connectors):
    """Connect every connector through queues keyed by subgroup and ranks.

    A send from member ``s`` to member ``d`` of subgroup ``g`` lands on
    ``wire[(g, s, d)]``; a receive on member ``d`` from ``s`` pops the same
    queue and checks the tensor has the size the receiver expected.
    """
    wire: dict[tuple[int, int, int], list[torch.Tensor]] = {}
    for connector in connectors:
        group = connector.mapping.subgroup_index
        me = connector.mapping.rank_in_subgroup

        def fake_send(hidden, dst, process_group, comm_id, *, group=group, me=me):
            wire.setdefault((group, me, dst), []).append(hidden.clone())

        def fake_recv(
            src,
            process_group,
            comm_id,
            tensor_metadata,
            *,
            ref_tensor=None,
            group=group,
            me=me,
        ):
            queue = wire.get((group, src, me))
            assert queue, f"nothing on the wire from {src} to {me} in subgroup {group}"
            tensor = queue.pop(0)
            assert tuple(tensor.shape) == tuple(tensor_metadata.size)
            if ref_tensor is not None:
                ref_tensor.copy_(tensor)
                return ref_tensor
            return tensor

        def fake_send_slice(base, start, size, dst, *, group=group, me=me):
            wire.setdefault((group, me, dst), []).append(
                base.narrow(0, start, size).clone()
            )

        def fake_recv_slice(base, out, start, size, src, *, group=group, me=me):
            queue = wire.get((group, src, me))
            assert queue, f"nothing on the wire from {src} to {me} in subgroup {group}"
            tensor = queue.pop(0)
            assert tuple(tensor.shape) == (size, out.shape[1])
            out.narrow(0, start, size).copy_(tensor)

        monkeypatch.setattr(connector, "_send_hidden_states", fake_send)
        monkeypatch.setattr(connector, "_recv_hidden_states", fake_recv)
        monkeypatch.setattr(connector, "_send_slice", fake_send_slice)
        monkeypatch.setattr(connector, "_recv_slice", fake_recv_slice)
    return wire


@pytest.mark.parametrize(
    ("attention", "ffn", "tokens"),
    [
        (1, 2, [5]),  # A < F: one attention splits 3/2
        (2, 3, [5, 3]),  # A < F: one subgroup splits, one does not
        (6, 4, [1, 2, 3, 4, 5, 6]),  # non-divisible A > F: uneven subgroups
        (4, 2, [3, 5, 7, 11]),  # historical divisible layout (regression)
        (1, 4, [2]),  # fewer tokens than members: dummy stages
    ],
)
def test_round_trip_preserves_every_token(monkeypatch, attention, ffn, tokens):
    attentions, ffns = _all_connectors(attention, ffn)
    _apply(attentions + ffns, tokens)
    wire = _wire(monkeypatch, attentions + ffns)

    # Distinct values per attention rank and per token so any misrouting or
    # reordering shows up in the final comparison.
    inputs = [
        torch.arange(t * _HIDDEN, dtype=torch.float32).reshape(t, _HIDDEN) + 1000 * a
        for a, t in enumerate(tokens)
    ]
    for connector, hidden in zip(attentions, inputs, strict=True):
        connector.send_attn_output(hidden, _attention_context(hidden.shape[0]))

    # Every FFN rank receives what arrived (or a dummy batch), "computes"
    # by adding one, and returns the result.
    for connector in ffns:
        payload = connector.recv_attn_output(ubatch_idx=0)
        connector.send_ffn_output(payload.hidden_states + 1, payload.context)

    for connector, hidden in zip(attentions, inputs, strict=True):
        output = connector.recv_ffn_output(
            ref_tensor=torch.empty_like(hidden),
            ubatch_idx=0,
        )
        assert torch.equal(output, hidden + 1)

    # Nothing sent but unreceived, nothing expected but unsent.
    assert all(not queue for queue in wire.values())


def test_round_trip_zero_shares_travel_as_zero_size_tensors(monkeypatch):
    # 1A4F with 2 tokens: F2 and F3 get zero-size shares. Both directions
    # still post the transfer (as zero-row tensors), so the pairs appear on
    # the wire, the dummy members return zero rows, and nothing is left over.
    attentions, ffns = _all_connectors(1, 4)
    _apply(attentions + ffns, [2])
    wire = _wire(monkeypatch, attentions + ffns)
    hidden = torch.arange(2 * _HIDDEN, dtype=torch.float32).reshape(2, _HIDDEN)
    attentions[0].send_attn_output(hidden, _attention_context(2))
    assert [t.shape[0] for t in wire[(0, 4, 2)]] == [0]
    assert [t.shape[0] for t in wire[(0, 4, 3)]] == [0]
    for connector in ffns:
        payload = connector.recv_attn_output(ubatch_idx=0)
        connector.send_ffn_output(payload.hidden_states + 1, payload.context)
    assert [t.shape[0] for t in wire[(0, 2, 4)]] == [0]
    assert [t.shape[0] for t in wire[(0, 3, 4)]] == [0]
    output = attentions[0].recv_ffn_output(
        ref_tensor=torch.empty_like(hidden),
        ubatch_idx=0,
    )
    assert torch.equal(output, hidden + 1)
    assert all(not queue for queue in wire.values())
