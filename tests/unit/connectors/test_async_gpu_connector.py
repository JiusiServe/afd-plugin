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
    combine_scatter,
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
        routed_cap=100,
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


def test_routed_cap_multiplier_must_be_positive():
    with pytest.raises(ValueError, match="routed_cap_multiplier"):
        GpuAsyncExtraInfo.from_mapping({"routed_cap_multiplier": 0})


# ----------------------------------------------------------------------
# Slot layout
# ----------------------------------------------------------------------


def test_slot_fields_are_disjoint_and_fit_inside_the_slot(layout: SlotLayout):
    assert layout.header_words == HEADER_FIXED_WORDS + 4
    assert layout.route_table_off >= layout.header_off + layout.header_words * 4
    assert layout.routed_x_off >= layout.route_table_off + 2 * 100 * 4
    assert layout.shared_idx_off >= layout.routed_x_off + 100 * 8 * 2
    assert layout.shared_x_off >= layout.shared_idx_off + 32 * 4
    assert layout.slot_bytes >= layout.shared_x_off + 32 * 8 * 2


def test_every_field_offset_is_viewable_as_int32_and_payload(layout: SlotLayout):
    # get_buffer takes an element offset, so a byte offset that is not a
    # multiple of the element size would silently land on the wrong address.
    for offset in (
        layout.header_off,
        layout.route_table_off,
        layout.routed_x_off,
        layout.shared_idx_off,
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


def test_every_partial_is_routed_exactly_once(routing_inputs):
    topk_ids, _, _ = routing_inputs
    route_table, counts, _ = plan_dispatch(
        topk_ids,
        ffn_size=_FFN_SIZE,
        expert_per_rank=_EXPERT_PER_RANK,
    )
    assert route_table.shape == (_NUM_TOKENS * _TOPK, 2)
    assert int(counts.sum()) == _NUM_TOKENS * _TOPK
    seen = {(token_idx, slot) for token_idx, slot in route_table.tolist()}
    assert len(seen) == _NUM_TOKENS * _TOPK


def test_each_destination_segment_is_grouped_by_local_expert(routing_inputs):
    topk_ids, _, _ = routing_inputs
    route_table, counts, offsets = plan_dispatch(
        topk_ids,
        ffn_size=_FFN_SIZE,
        expert_per_rank=_EXPERT_PER_RANK,
    )
    counts_host, offsets_host = counts.tolist(), offsets.tolist()
    for ffn_rank in range(_FFN_SIZE):
        base = ffn_rank * _EXPERT_PER_RANK
        cursor = offsets_host[base]
        for local_expert in range(_EXPERT_PER_RANK):
            for _ in range(counts_host[base + local_expert]):
                token_idx, slot = route_table[cursor].tolist()
                assert int(topk_ids[token_idx, slot]) == base + local_expert
                cursor += 1
        assert cursor == offsets_host[base] + sum(
            counts_host[base : base + _EXPERT_PER_RANK],
        )


def test_identity_experts_recombine_to_the_weighted_sum(routing_inputs):
    topk_ids, hidden_states, topk_weights = routing_inputs
    route_table, counts, offsets = plan_dispatch(
        topk_ids,
        ffn_size=_FFN_SIZE,
        expert_per_rank=_EXPERT_PER_RANK,
    )
    counts_host, offsets_host = counts.tolist(), offsets.tolist()
    accumulator = torch.zeros(_NUM_TOKENS, _HIDDEN, dtype=torch.float32)
    for ffn_rank in range(_FFN_SIZE):
        base = ffn_rank * _EXPERT_PER_RANK
        start = offsets_host[base]
        total = sum(counts_host[base : base + _EXPERT_PER_RANK])
        segment = route_table[start : start + total]
        combine_scatter(
            accumulator,
            routed_out=hidden_states.index_select(0, segment[:, 0].to(torch.int64)),
            route_table=segment,
            topk_weights=topk_weights,
        )
    expected = hidden_states * topk_weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(accumulator, expected.to(torch.float32))


def test_routing_handles_experts_not_divisible_by_ffn_size():
    # expert_per_rank is a ceiling division, so the padded tail must stay empty
    # instead of silently absorbing real partials.
    ffn_size, expert_per_rank, num_experts = 3, 2, 5
    topk_ids = torch.tensor([[0, 4], [1, 3], [2, 4]], dtype=torch.int32)
    _, counts, _ = plan_dispatch(
        topk_ids,
        ffn_size=ffn_size,
        expert_per_rank=expert_per_rank,
    )
    assert counts.numel() == ffn_size * expert_per_rank
    assert int(counts.sum()) == topk_ids.numel()
    assert int(counts[num_experts:].sum()) == 0


def test_routing_can_leave_one_destination_empty():
    """A single decode token's topk can land entirely on one FFN rank.

    The peer that gets nothing must still see a well-formed, empty segment --
    zero-length windows are what crashed a 2A2F decode step.
    """
    ffn_size, expert_per_rank = 2, 4
    # Every partial targets experts owned by FFN rank 0.
    topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    route_table, counts, offsets = plan_dispatch(
        topk_ids,
        ffn_size=ffn_size,
        expert_per_rank=expert_per_rank,
    )
    counts_host, offsets_host = counts.tolist(), offsets.tolist()

    base_zero = 0
    assert sum(counts_host[base_zero : base_zero + expert_per_rank]) == 3
    base_one = expert_per_rank
    empty_total = sum(counts_host[base_one : base_one + expert_per_rank])
    assert empty_total == 0
    empty_segment = route_table[
        offsets_host[base_one] : offsets_host[base_one] + empty_total
    ]
    assert empty_segment.shape == (0, 2)

    # Combining an empty segment must be a no-op, not an error.
    accumulator = torch.zeros(1, 4, dtype=torch.float32)
    combine_scatter(
        accumulator,
        routed_out=torch.zeros(0, 4),
        route_table=empty_segment,
        topk_weights=torch.ones(1, 3),
    )
    assert torch.count_nonzero(accumulator) == 0
