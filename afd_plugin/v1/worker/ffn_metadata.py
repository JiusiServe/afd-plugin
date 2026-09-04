# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe token metadata helpers for AFD FFN runtimes."""

from __future__ import annotations

from afd_plugin.distributed import split_send_sizes


def aggregate_ffn_token_counts(
    attention_counts: tuple[int, ...],
    *,
    attention_size: int,
    ffn_size: int,
    fallback: int = 1,
) -> tuple[int, ...]:
    """Token count each FFN rank receives for a stage, from Attention counts.

    An FFN rank sums the shares its subgroup's Attention members send it,
    splitting with ``split_send_sizes`` as the connector does. For example,
    ``4A2F`` counts ``(0, 4, 5, 6)`` become ``(5, 11)`` because every
    zero-token Attention peer contributes one placeholder row, and ``1A2F``
    count ``(5,)`` becomes ``(3, 2)``. No count is below one, since an FFN
    rank that receives nothing runs a one-token dummy batch. Missing peers
    use the same per-peer fallback.
    """

    fallback_count = max(1, int(fallback))
    if ffn_size <= 0 or attention_size <= 0:
        return tuple(fallback_count for _ in range(max(0, ffn_size)))

    expanded_counts = attention_counts
    if (
        attention_counts
        and len(attention_counts) < attention_size
        and attention_size % len(attention_counts) == 0
    ):
        attention_ranks_per_count = attention_size // len(attention_counts)
        expanded_counts = tuple(
            attention_counts[rank // attention_ranks_per_count]
            for rank in range(attention_size)
        )

    def peer_count(attention_rank: int) -> int:
        if attention_rank < len(expanded_counts):
            return max(1, int(expanded_counts[attention_rank]))
        return fallback_count

    group_count = min(attention_size, ffn_size)
    ffn_counts = []
    for ffn_rank in range(ffn_size):
        group = ffn_rank * group_count // ffn_size
        ffn_members = [
            member
            for member in range(ffn_size)
            if member * group_count // ffn_size == group
        ]
        position = ffn_members.index(ffn_rank)
        total = sum(
            split_send_sizes(peer_count(attention_rank), len(ffn_members))[position]
            for attention_rank in range(attention_size)
            if attention_rank * group_count // attention_size == group
        )
        ffn_counts.append(max(1, total))
    return tuple(ffn_counts)


def project_ffn_token_counts_to_dp(
    ffn_counts: tuple[int, ...],
    *,
    dp_size: int,
) -> tuple[int, ...]:
    """Project FFN role-rank counts to vLLM data-parallel rank counts.

    For example, two FFN TP ranks per DP rank project
    ``(8, 8, 13, 13)`` to ``(8, 13)``.
    """

    if len(ffn_counts) == dp_size or dp_size <= 0 or len(ffn_counts) % dp_size != 0:
        return ffn_counts

    ffn_ranks_per_dp = len(ffn_counts) // dp_size
    return tuple(ffn_counts[rank * ffn_ranks_per_dp] for rank in range(dp_size))


__all__ = ["aggregate_ffn_token_counts", "project_ffn_token_counts_to_dp"]
