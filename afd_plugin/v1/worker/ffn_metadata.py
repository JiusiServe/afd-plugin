# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe token metadata helpers for AFD FFN runtimes."""

from __future__ import annotations


def aggregate_ffn_token_counts(
    attention_counts: tuple[int, ...],
    *,
    attention_size: int,
    ffn_size: int,
    fallback: int = 1,
) -> tuple[int, ...]:
    """Aggregate consecutive Attention-rank counts for each FFN rank."""

    fallback_counts = tuple(max(1, int(fallback)) for _ in range(ffn_size))
    if not attention_counts:
        return fallback_counts

    expanded_counts = attention_counts
    if (
        len(attention_counts) < attention_size
        and attention_size % len(attention_counts) == 0
    ):
        parallel_size = attention_size // len(attention_counts)
        expanded_counts = tuple(
            attention_counts[rank // parallel_size] for rank in range(attention_size)
        )

    if (
        len(expanded_counts) < attention_size
        or attention_size < ffn_size
        or attention_size % ffn_size != 0
    ):
        return fallback_counts

    group_size = attention_size // ffn_size
    return tuple(
        max(
            1,
            sum(expanded_counts[rank * group_size : (rank + 1) * group_size]),
        )
        for rank in range(ffn_size)
    )


def project_ffn_token_counts_to_dp(
    ffn_counts: tuple[int, ...],
    *,
    dp_size: int,
) -> tuple[int, ...]:
    """Project FFN role-rank counts to vLLM data-parallel rank counts."""

    if len(ffn_counts) == dp_size or dp_size <= 0 or len(ffn_counts) % dp_size != 0:
        return ffn_counts

    parallel_size = len(ffn_counts) // dp_size
    return tuple(ffn_counts[rank * parallel_size] for rank in range(dp_size))


__all__ = ["aggregate_ffn_token_counts", "project_ffn_token_counts_to_dp"]
