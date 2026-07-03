# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe AFD rank topology helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from afd_plugin.config import AFDConfig


@dataclass(frozen=True, slots=True)
class AFDRankMapping:
    """Rank mapping for the Phase 4 P2P connector.

    The P2P world always places FFN ranks first, followed by Attention ranks:
    ``[F0, F1, ..., A0, A1, ...]``. Each FFN rank owns one subgroup containing
    itself at subgroup rank 0 and one or more consecutive Attention ranks.
    """

    role: str
    role_rank: int
    world_rank: int
    p2p_rank: int
    attention_size: int
    ffn_size: int
    min_size: int
    ratio: int
    subgroup_index: int
    rank_in_subgroup: int
    subgroup_ranks: tuple[int, ...]
    dp_metadata_destinations: tuple[int, ...] = field(default_factory=tuple)

    @property
    def is_attention_top_min_size_rank(self) -> bool:
        return self.ffn_size <= self.world_rank < self.ffn_size + self.min_size

    @property
    def participates_in_dp_metadata_group(self) -> bool:
        return self.world_rank < self.ffn_size or self.is_attention_top_min_size_rank


def topology_from_config(config: AFDConfig) -> tuple[int, int]:
    """Return ``(attention_size, ffn_size)`` for an AFD config."""

    return config.num_attention_servers, config.num_ffn_servers


def validate_p2p_topology(config: AFDConfig) -> None:
    attention_size, ffn_size = topology_from_config(config)
    if attention_size < ffn_size:
        raise ValueError(
            "p2pconnector currently requires num_attention_servers >= "
            f"num_ffn_servers, got {attention_size} < {ffn_size}",
        )
    if attention_size % ffn_size != 0:
        raise ValueError(
            "p2pconnector currently requires num_attention_servers to be a "
            "multiple of num_ffn_servers, got "
            f"{attention_size} and {ffn_size}",
        )


def build_rank_mapping(
    config: AFDConfig,
    role_rank: int | None = None,
) -> AFDRankMapping:
    """Build the P2P rank mapping for one Attention or FFN process."""

    validate_p2p_topology(config)
    attention_size, ffn_size = topology_from_config(config)
    role_rank = config.afd_server_rank if role_rank is None else role_rank
    if role_rank < 0:
        raise ValueError(f"AFD role rank must be non-negative, got {role_rank}")

    if config.role == "attention":
        if role_rank >= attention_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attention_size})",
            )
        world_rank = ffn_size + role_rank
        subgroup_index = role_rank // (attention_size // ffn_size)
    elif config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={ffn_size})",
            )
        world_rank = role_rank
        subgroup_index = role_rank
    else:
        raise ValueError(f"unknown AFD role {config.role!r}")

    ratio = attention_size // ffn_size
    min_size = min(ffn_size, attention_size)
    ffn_ranks = list(range(ffn_size))
    attention_ranks = list(range(ffn_size, ffn_size + attention_size))
    subgroup_ranks = tuple(
        [ffn_ranks[subgroup_index]]
        + [attention_ranks[subgroup_index * ratio + offset] for offset in range(ratio)],
    )
    rank_in_subgroup = subgroup_ranks.index(world_rank)
    p2p_rank = role_rank + min_size if config.role == "attention" else role_rank

    destinations: list[int] = []
    if ffn_size <= world_rank < ffn_size + min_size:
        local_attention_rank = world_rank - ffn_size
        destination = local_attention_rank
        while destination < ffn_size:
            destinations.append(destination)
            destination += min_size

    return AFDRankMapping(
        role=config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        p2p_rank=p2p_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
        min_size=min_size,
        ratio=ratio,
        subgroup_index=subgroup_index,
        rank_in_subgroup=rank_in_subgroup,
        subgroup_ranks=subgroup_ranks,
        dp_metadata_destinations=tuple(destinations),
    )


__all__ = [
    "AFDRankMapping",
    "build_rank_mapping",
    "topology_from_config",
    "validate_p2p_topology",
]
