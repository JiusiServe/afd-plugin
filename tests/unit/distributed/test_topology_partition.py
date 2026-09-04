# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unified subgroup partition tests for arbitrary positive ``(A, F)``.

``build_rank_mapping`` partitions the world into ``min(A, F)`` subgroups
(attention ``a`` -> subgroup ``a*G//A``, FFN ``f`` -> subgroup ``f*G//F``).
The historical ``A >= F`` divisible layout is the ``G == F`` special case,
pinned byte-for-byte by ``test_topology_snapshot.py``; this file covers what
the snapshots cannot: the structural properties that must hold on every
positive pair.

Also guards the DP metadata channel numbering: attention ``p2p_rank`` uses
the ``ffn_size`` offset that the receiver side assumes (``p2p.py``:
``src = p2p_rank % min_size + ffn_size``); under ``A >= F`` it equals the
historical ``min_size`` offset, and under ``A < F`` it is what keeps the
numbering collision-free.
"""

from __future__ import annotations

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.distributed.topology import (
    build_rank_mapping,
    split_send_sizes,
    validate_p2p_topology,
)

# Every positive pair with A, F in 1..6 — divisible, non-divisible, and
# A < F shapes (36 combinations).
_FULL_GRID = [(a, f) for a in range(1, 7) for f in range(1, 7)]


def _config(role: str, attention: int, ffn: int) -> AFDConfig:
    return AFDConfig(
        role=role,
        num_attention_ranks=attention,
        num_ffn_ranks=ffn,
    )


def _mapping(role, attention, ffn, role_rank):
    return build_rank_mapping(_config(role, attention, ffn), role_rank)


def _all_mappings(attention, ffn):
    for role, size in (("ffn", ffn), ("attention", attention)):
        for role_rank in range(size):
            yield _mapping(role, attention, ffn, role_rank)


def _receiver_expected_source_rank(ffn_rank, attention, ffn):
    """The p2p_rank the FFN receiver polls for its metadata sender.

    Mirrors the receiver convention in ``connectors/gpu/p2p.py``
    (``recv_dp_metadata_list``): ``src = p2p_rank % min_size + ffn_size``
    with ``p2p_rank == ffn_rank`` on FFN ranks. Kept in sync by review; the
    connector cannot be imported here without torch/vLLM.
    """
    return ffn_rank % min(attention, ffn) + ffn


# --------------------------------------------------------------------------
# Validation — positive rank counts only
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("attention", "ffn"), [(1, 2), (3, 2), (2, 4), (6, 4)])
def test_validation_accepts_any_positive_pair(attention, ffn):
    # The historical A >= F and divisibility constraints are removed; the
    # previously rejected shapes validate cleanly.
    validate_p2p_topology(_config("attention", attention, ffn))


@pytest.mark.parametrize(("attention", "ffn"), [(0, 2), (2, 0)])
def test_validation_rejects_non_positive_pair(attention, ffn):
    with pytest.raises(ValueError, match="positive rank counts"):
        validate_p2p_topology(_config("attention", attention, ffn))


# --------------------------------------------------------------------------
# Partition structure on the full grid
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("attention", "ffn"), _FULL_GRID)
def test_partition_properties(attention, ffn):
    group_count = min(attention, ffn)
    groups: dict[int, tuple[int, ...]] = {}

    for mapping in _all_mappings(attention, ffn):
        # Index formulas and membership consistency: every member computes
        # the identical member tuple, and sits at its claimed position.
        if mapping.role == "ffn":
            assert mapping.subgroup_index == mapping.role_rank * group_count // ffn
        else:
            assert (
                mapping.subgroup_index == mapping.role_rank * group_count // attention
            )
        prior = groups.setdefault(mapping.subgroup_index, mapping.subgroup_ranks)
        assert prior == mapping.subgroup_ranks
        assert mapping.subgroup_ranks[mapping.rank_in_subgroup] == mapping.world_rank
        # ratio is this subgroup's attention member count.
        attention_members = [rank for rank in mapping.subgroup_ranks if rank >= ffn]
        assert mapping.ratio == len(attention_members)

    # Exactly min(A, F) subgroups, each holding at least one rank of both
    # roles, members FFN-first in ascending world order, and together they
    # partition the world with no overlap.
    assert sorted(groups) == list(range(group_count))
    seen: list[int] = []
    for members in groups.values():
        ffn_members = [rank for rank in members if rank < ffn]
        attention_members = [rank for rank in members if rank >= ffn]
        assert ffn_members and attention_members
        assert list(members) == sorted(ffn_members) + sorted(attention_members)
        seen.extend(members)
    assert sorted(seen) == list(range(ffn + attention))


def test_example_2a3f_partition():
    # The shape that motivated the unified partition: 2 attention, 3 FFN
    # (world order [F0, F1, F2, A0, A1]) splits into {F0, F1, A0} and
    # {F2, A1}.
    members = {
        mapping.subgroup_index: mapping.subgroup_ranks
        for mapping in _all_mappings(2, 3)
    }
    assert members == {0: (0, 1, 3), 1: (2, 4)}


@pytest.mark.parametrize(
    ("attention", "ffn", "expected"),
    [
        (4, 2, {0: (0, 2, 3), 1: (1, 4, 5)}),  # historical divisible layout
        (3, 2, {0: (0, 2, 3), 1: (1, 4)}),  # non-divisible A > F
        (1, 2, {0: (0, 1, 2)}),  # capacity-bound target shape
    ],
)
def test_partition_literal_examples(attention, ffn, expected):
    members = {
        mapping.subgroup_index: mapping.subgroup_ranks
        for mapping in _all_mappings(attention, ffn)
    }
    assert members == expected


# --------------------------------------------------------------------------
# DP metadata channel numbering
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("attention", "ffn"), _FULL_GRID)
def test_metadata_p2p_ranks_are_unique_and_contiguous(attention, ffn):
    ranks = sorted(
        mapping.p2p_rank
        for mapping in _all_mappings(attention, ffn)
        if mapping.participates_in_dp_metadata_group
    )
    assert ranks == list(range(ffn)) + [
        ffn + attention_rank for attention_rank in range(min(attention, ffn))
    ]


@pytest.mark.parametrize(("attention", "ffn"), _FULL_GRID)
def test_sender_p2p_rank_matches_receiver_formula(attention, ffn):
    min_size = min(attention, ffn)
    for ffn_rank in range(ffn):
        representative = ffn_rank % min_size
        sender = _mapping("attention", attention, ffn, representative)
        assert sender.p2p_rank == _receiver_expected_source_rank(
            ffn_rank,
            attention,
            ffn,
        )


@pytest.mark.parametrize(("attention", "ffn"), _FULL_GRID)
def test_dp_metadata_destinations_cover_every_ffn_once(attention, ffn):
    destinations: list[int] = []
    for role_rank in range(attention):
        mapping = _mapping("attention", attention, ffn, role_rank)
        destinations.extend(mapping.dp_metadata_destinations)
    assert sorted(destinations) == list(range(ffn))


# --------------------------------------------------------------------------
# split_send_sizes — the per-peer size arithmetic the transport shares
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token_count", "parts", "expected"),
    [
        (5, 2, (3, 2)),  # remainder rides on the leading shares
        (6, 3, (2, 2, 2)),  # even split
        (7, 1, (7,)),  # parts=1 = whole count (every A >= F subgroup)
        (2, 4, (1, 1, 0, 0)),  # fewer tokens than peers -> trailing zeros
        (0, 3, (0, 0, 0)),  # nothing to send
    ],
)
def test_split_send_sizes_literal_cases(token_count, parts, expected):
    assert split_send_sizes(token_count, parts) == expected


@pytest.mark.parametrize("token_count", range(9))
@pytest.mark.parametrize("parts", range(1, 5))
def test_split_send_sizes_properties(token_count, parts):
    sizes = split_send_sizes(token_count, parts)
    # Conservation, near-evenness, and deterministic front-loading — the
    # contract that lets sender and receivers agree without communication.
    assert sum(sizes) == token_count
    assert len(sizes) == parts
    assert max(sizes) - min(sizes) <= 1
    assert list(sizes) == sorted(sizes, reverse=True)


@pytest.mark.parametrize(("token_count", "parts"), [(1, 0), (1, -1), (-1, 2)])
def test_split_send_sizes_rejects_invalid_inputs(token_count, parts):
    with pytest.raises(ValueError):
        split_send_sizes(token_count, parts)
