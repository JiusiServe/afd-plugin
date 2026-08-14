# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit tests for the async GPU connector's wire format and routing math."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from afd_plugin.connectors.factory import AFDConnectorFactory  # noqa: E402
from afd_plugin.connectors.gpu.async_gpu import (  # noqa: E402
    GpuAsyncAFDConnector,
    GpuAsyncExtraInfo,
    plan_dispatch,
)
from afd_plugin.connectors.gpu.symm_window import (  # noqa: E402
    FLAG_SHUTDOWN_BIT,
    HEADER_FIXED_WORDS,
    SlotLayout,
    decode_header,
    encode_header,
)


@pytest.fixture
def layout() -> SlotLayout:
    return SlotLayout.build(
        expert_per_rank=4,
        partial_cap=100,
        token_cap=32,
        hidden_size=8,
        payload_itemsize=2,
    )


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


def test_connector_is_registered_and_has_no_control_plane():
    connector_cls = AFDConnectorFactory.get_connector_class("GpuAsyncAFDConnector")
    assert connector_cls is GpuAsyncAFDConnector
    assert connector_cls.control_plane is None


def test_ring_depth_defaults_to_the_number_of_live_stages():
    assert GpuAsyncExtraInfo.from_mapping(None).ring_depth == 1
    assert GpuAsyncExtraInfo.from_mapping({"async_moe_ubatching": True}).ring_depth == 2
    assert GpuAsyncExtraInfo.from_mapping({"ring_depth": 4}).ring_depth == 4


def test_unknown_extra_config_field_is_rejected():
    with pytest.raises(ValueError, match="unknown AFD async GPU"):
        GpuAsyncExtraInfo.from_mapping({"nope": 1})


# ----------------------------------------------------------------------
# Slot layout
# ----------------------------------------------------------------------


def test_slot_fields_are_disjoint_and_fit_inside_the_slot(layout: SlotLayout):
    assert layout.header_words == HEADER_FIXED_WORDS + 4
    assert layout.expand_idx_off >= layout.header_off + layout.header_words * 4
    assert layout.weights_off >= layout.expand_idx_off + 100 * 4
    assert layout.routed_x_off >= layout.weights_off + 100 * 4
    # The payload is sized by distinct tokens, not by partials, and the shared
    # rows are a contiguous range so no index rides along with them.
    assert layout.shared_x_off >= layout.routed_x_off + 32 * 8 * 2
    assert layout.slot_bytes >= layout.shared_x_off + 32 * 8 * 2


def test_every_field_offset_is_viewable_as_int32_and_payload(layout: SlotLayout):
    # get_buffer takes an element offset, so a byte offset that is not a
    # multiple of the element size would silently land on the wrong address.
    for offset in (
        layout.header_off,
        layout.expand_idx_off,
        layout.weights_off,
        layout.routed_x_off,
        layout.shared_x_off,
    ):
        assert offset % 4 == 0
        assert offset % layout.payload_itemsize == 0


# ----------------------------------------------------------------------
# Header codec
# ----------------------------------------------------------------------


def test_header_round_trip(layout: SlotLayout):
    header = encode_header(
        layout,
        seq=7,
        src_role_rank=3,
        layer_idx=11,
        stage_idx=1,
        num_tokens=32,
        routed_tokens=90,
        shared_tokens=16,
        topk=6,
        flags=0,
        expert_counts=[10, 20, 30, 30],
        uniq_tokens=25,
    )
    decoded = decode_header(header)
    assert decoded.seq == 7
    assert decoded.src_role_rank == 3
    assert decoded.layer_idx == 11
    assert decoded.stage_idx == 1
    assert decoded.num_tokens == 32
    assert decoded.routed_tokens == 90
    assert decoded.shared_tokens == 16
    assert decoded.topk == 6
    assert decoded.expert_counts == [10, 20, 30, 30]
    assert decoded.uniq_tokens == 25
    assert sum(decoded.expert_counts) == decoded.routed_tokens
    assert not decoded.is_shutdown


def test_shutdown_flag_survives_the_round_trip(layout: SlotLayout):
    header = encode_header(
        layout,
        seq=8,
        src_role_rank=0,
        layer_idx=0,
        stage_idx=0,
        num_tokens=0,
        routed_tokens=0,
        shared_tokens=0,
        topk=6,
        flags=FLAG_SHUTDOWN_BIT,
        expert_counts=[0, 0, 0, 0],
    )
    assert decode_header(header).is_shutdown


def test_corrupt_magic_is_rejected(layout: SlotLayout):
    header = encode_header(
        layout,
        seq=1,
        src_role_rank=0,
        layer_idx=0,
        stage_idx=0,
        num_tokens=1,
        routed_tokens=0,
        shared_tokens=0,
        topk=6,
        flags=0,
        expert_counts=[0, 0, 0, 0],
    )
    header[0] = 0
    with pytest.raises(RuntimeError, match="magic mismatch"):
        decode_header(header)


def test_expert_counts_length_must_match_the_layout(layout: SlotLayout):
    with pytest.raises(ValueError, match="expert_counts"):
        encode_header(
            layout,
            seq=1,
            src_role_rank=0,
            layer_idx=0,
            stage_idx=0,
            num_tokens=1,
            routed_tokens=0,
            shared_tokens=0,
            topk=6,
            flags=0,
            expert_counts=[1, 2],
        )


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------

_NUM_TOKENS = 7
_TOPK = 3
_FFN_SIZE = 2
_EXPERT_PER_RANK = 4
_HIDDEN = 5


@pytest.fixture
def routing_inputs():
    generator = torch.Generator().manual_seed(0)
    num_experts = _FFN_SIZE * _EXPERT_PER_RANK
    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, generator=generator)[:_TOPK]
            for _ in range(_NUM_TOKENS)
        ],
    ).to(torch.int32)
    hidden_states = torch.randn(_NUM_TOKENS, _HIDDEN, generator=generator)
    topk_weights = torch.rand(_NUM_TOKENS, _TOPK, generator=generator)
    return topk_ids, hidden_states, topk_weights


def _destination_slices(plan, ffn_size, expert_per_rank):
    """Walk the plan the way send_attn_output does: one slice per destination."""
    counts, offsets = plan.counts.tolist(), plan.offsets.tolist()
    uniq = plan.uniq_per_rank.tolist()
    uniq_start = 0
    for ffn_rank in range(ffn_size):
        base = ffn_rank * expert_per_rank
        total = sum(counts[base : base + expert_per_rank])
        partials = slice(offsets[base], offsets[base] + total)
        rows = plan.uniq_token_ids[uniq_start : uniq_start + uniq[ffn_rank]]
        uniq_start += uniq[ffn_rank]
        yield ffn_rank, partials, rows


def test_every_partial_is_routed_exactly_once(routing_inputs):
    topk_ids, _, topk_weights = routing_inputs
    plan = plan_dispatch(
        topk_ids,
        topk_weights,
        ffn_size=_FFN_SIZE,
        expert_per_rank=_EXPERT_PER_RANK,
    )
    assert plan.expand_idx.shape == (_NUM_TOKENS * _TOPK,)
    assert plan.weights.shape == (_NUM_TOKENS * _TOPK,)
    assert int(plan.counts.sum()) == _NUM_TOKENS * _TOPK
    assert int(plan.uniq_per_rank.sum()) <= _NUM_TOKENS * _TOPK


def test_each_destination_segment_is_grouped_by_local_expert(routing_inputs):
    topk_ids, _, topk_weights = routing_inputs
    plan = plan_dispatch(
        topk_ids,
        topk_weights,
        ffn_size=_FFN_SIZE,
        expert_per_rank=_EXPERT_PER_RANK,
    )
    counts, offsets = plan.counts.tolist(), plan.offsets.tolist()
    for ffn_rank, partials, rows in _destination_slices(
        plan,
        _FFN_SIZE,
        _EXPERT_PER_RANK,
    ):
        base = ffn_rank * _EXPERT_PER_RANK
        # The token behind each partial, in the order the receiver sees them.
        tokens = rows.index_select(0, plan.expand_idx[partials].to(torch.int64))
        cursor = 0
        for local_expert in range(_EXPERT_PER_RANK):
            for _ in range(counts[base + local_expert]):
                token_idx = int(tokens[cursor])
                assert base + local_expert in topk_ids[token_idx].tolist()
                cursor += 1
        assert cursor == partials.stop - partials.start
        assert offsets[base] == partials.start


def test_each_token_is_shipped_once_per_destination(routing_inputs):
    topk_ids, _, topk_weights = routing_inputs
    plan = plan_dispatch(
        topk_ids,
        topk_weights,
        ffn_size=_FFN_SIZE,
        expert_per_rank=_EXPERT_PER_RANK,
    )
    for ffn_rank, partials, rows in _destination_slices(
        plan,
        _FFN_SIZE,
        _EXPERT_PER_RANK,
    ):
        shipped = rows.tolist()
        assert len(set(shipped)) == len(shipped), "a token was shipped twice"
        base = ffn_rank * _EXPERT_PER_RANK
        expected = {
            token
            for token in range(_NUM_TOKENS)
            for expert in topk_ids[token].tolist()
            if base <= expert < base + _EXPERT_PER_RANK
        }
        assert set(shipped) == expected
        # Every partial must land inside its destination's own payload.
        expand = plan.expand_idx[partials]
        assert int(expand.min()) >= 0
        assert int(expand.max()) < len(shipped)


def test_identity_experts_recombine_to_the_weighted_sum(routing_inputs):
    """The full chain: ship distinct rows, expand, weight, reduce, scatter."""
    topk_ids, hidden_states, topk_weights = routing_inputs
    plan = plan_dispatch(
        topk_ids,
        topk_weights,
        ffn_size=_FFN_SIZE,
        expert_per_rank=_EXPERT_PER_RANK,
    )
    accumulator = torch.zeros(_NUM_TOKENS, _HIDDEN, dtype=torch.float32)
    for _, partials, rows in _destination_slices(plan, _FFN_SIZE, _EXPERT_PER_RANK):
        expand = plan.expand_idx[partials].to(torch.int64)
        # What crosses the wire, and what the FFN side does with it.
        expanded = hidden_states.index_select(0, rows).index_select(0, expand)
        reduced = torch.zeros(rows.numel(), _HIDDEN, dtype=torch.float32)
        reduced.index_add_(0, expand, expanded * plan.weights[partials].unsqueeze(1))
        accumulator.index_add_(0, rows, reduced)
    expected = hidden_states * topk_weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(accumulator, expected.to(torch.float32))


def test_routing_handles_experts_not_divisible_by_ffn_size():
    # expert_per_rank is a ceiling division, so the padded tail must stay empty
    # instead of silently absorbing real partials.
    ffn_size, expert_per_rank, num_experts = 3, 2, 5
    topk_ids = torch.tensor([[0, 4], [1, 3], [2, 4]], dtype=torch.int32)
    plan = plan_dispatch(
        topk_ids,
        torch.ones(3, 2),
        ffn_size=ffn_size,
        expert_per_rank=expert_per_rank,
    )
    assert plan.counts.numel() == ffn_size * expert_per_rank
    assert int(plan.counts.sum()) == topk_ids.numel()
    assert int(plan.counts[num_experts:].sum()) == 0


def test_routing_can_leave_one_destination_empty():
    """A single decode token's topk can land entirely on one FFN rank.

    The peer that gets nothing must still see a well-formed, empty segment --
    zero-length windows are what crashed a 2A2F decode step.
    """
    ffn_size, expert_per_rank = 2, 4
    # Every partial targets experts owned by FFN rank 0.
    topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    plan = plan_dispatch(
        topk_ids,
        torch.ones(1, 3),
        ffn_size=ffn_size,
        expert_per_rank=expert_per_rank,
    )
    slices = list(_destination_slices(plan, ffn_size, expert_per_rank))

    _, busy_partials, busy_rows = slices[0]
    assert busy_partials.stop - busy_partials.start == 3
    # One token, three of its partials: it is shipped once, read three times.
    assert busy_rows.tolist() == [0]

    _, empty_partials, empty_rows = slices[1]
    assert empty_partials.stop - empty_partials.start == 0
    assert empty_rows.numel() == 0

    # Reducing an empty destination must be a no-op, not an error.
    accumulator = torch.zeros(1, 4, dtype=torch.float32)
    accumulator.index_add_(0, empty_rows, torch.zeros(0, 4))
    assert torch.count_nonzero(accumulator) == 0
