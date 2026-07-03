# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD metadata objects shared by runtime classes and model wrappers."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class AFDDPMetadata:
    """Serializable DPMetadata-compatible payload for AFD control traffic."""

    num_tokens_across_dp_cpu: Any
    max_tokens_across_dp_cpu: Any | None = None
    local_sizes: list[int] | None = None

    def __post_init__(self) -> None:
        self.num_tokens_across_dp_cpu = _cpu_int_tensor_or_list(
            self.num_tokens_across_dp_cpu,
        )
        if self.max_tokens_across_dp_cpu is None:
            self.max_tokens_across_dp_cpu = _max_token_count(
                self.num_tokens_across_dp_cpu,
            )
        else:
            self.max_tokens_across_dp_cpu = _cpu_scalar_tensor_or_int(
                self.max_tokens_across_dp_cpu,
            )

    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int) -> Iterator[list[int]]:
        self.local_sizes = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu,
            sequence_parallel_size,
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        return self.local_sizes

    def cu_tokens_across_sp(self, sp_size: int) -> Any:
        num_tokens = _cpu_int_tensor_or_list(self.num_tokens_across_dp_cpu)
        if isinstance(num_tokens, list):
            num_tokens = torch.tensor(num_tokens, dtype=torch.int32, device="cpu")
        num_tokens_across_sp_cpu = (num_tokens - 1 + sp_size) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(
            sp_size,
        )
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)

    @contextmanager
    def chunked_sizes(
        self,
        sequence_parallel_size: int,
        max_chunk_size_per_rank: int,
        chunk_idx: int,
    ) -> Iterator[list[int]]:
        sp_tokens = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu,
            sequence_parallel_size,
        )
        self.local_sizes = [
            max(
                1,
                min(
                    max_chunk_size_per_rank,
                    size - max_chunk_size_per_rank * chunk_idx,
                ),
            )
            for size in sp_tokens
        ]
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None


AFDSingleDPMetadata = AFDDPMetadata


def _cpu_int_tensor_or_list(value: Any) -> Any:
    values = _to_int_list(value)
    return torch.tensor(values, dtype=torch.int32, device="cpu")


def _cpu_scalar_tensor_or_int(value: Any) -> Any:
    if isinstance(value, (int, float)):
        value = int(value)
    elif isinstance(value, (list, tuple)):
        value = max(int(item) for item in value)
    else:
        value = int(value.item())
    return torch.tensor(value, dtype=torch.int32, device="cpu")


def _max_token_count(value: Any) -> Any:
    if isinstance(value, list):
        return max(_to_int_list(value))
    return value.max()


def _to_int_list(value: Any) -> list[int]:
    if isinstance(value, (int, float)):
        value = [value]
    elif isinstance(value, (list, tuple)):
        pass
    else:
        value = value.tolist()
    return [int(item) for item in value]


def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: Any,
    sequence_parallel_size: int,
) -> list[int]:
    if not isinstance(num_tokens_across_dp_cpu, (int, float, list, tuple)):
        sp_tokens = (
            num_tokens_across_dp_cpu + sequence_parallel_size - 1
        ) // sequence_parallel_size
        return sp_tokens.repeat_interleave(sequence_parallel_size).tolist()

    if isinstance(num_tokens_across_dp_cpu, (int, float)):
        values = [int(num_tokens_across_dp_cpu)]
    else:
        values = [int(value) for value in num_tokens_across_dp_cpu]
    local_sizes: list[int] = []
    for value in values:
        local_sizes.extend(
            [max(1, (value + sequence_parallel_size - 1) // sequence_parallel_size)]
            * sequence_parallel_size,
        )
    return local_sizes


@dataclass(slots=True)
class AFDConnectorMetadata:
    """Communication metadata for one AFD Attention/FFN exchange."""

    layer_idx: int
    stage_idx: int
    seq_lens: list[int]
    recv_handle_list: list[Any] | None = None
    connector_data: Any = None

    def __post_init__(self) -> None:
        if not self.seq_lens:
            raise ValueError("seq_lens cannot be empty")
        if any(length <= 0 for length in self.seq_lens):
            raise ValueError("all sequence lengths must be positive")

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @classmethod
    def create_attention_metadata(
        cls,
        *,
        layer_idx: int,
        stage_idx: int,
        seq_len: int,
    ) -> AFDConnectorMetadata:
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[seq_len],
        )

    @classmethod
    def create_ffn_metadata(
        cls,
        *,
        layer_idx: int,
        stage_idx: int,
        seq_lens: list[int],
    ) -> AFDConnectorMetadata:
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=list(seq_lens),
        )

    def validate_tensor_shape(self, tensor_shape: tuple[int, ...]) -> bool:
        return len(tensor_shape) > 0 and tensor_shape[0] == self.total_tokens


@dataclass(slots=True)
class AFDRecvOutput:
    """Unified Attention -> FFN payload returned by connector recv paths."""

    hidden_states: Any
    metadata: AFDConnectorMetadata
    group_list: Any = None
    topk_weights: Any = None
    topk_ids: Any = None
    router_logits: Any = None
    row_idx: Any = None
    x_active_mask: Any = None
    dynamic_scales: Any = None
    cam_p2p_ep_name: str | None = None
    atten_batch_size: Any = None
    expand_idx: Any = None
    ep_recv_counts: Any = None


@dataclass(slots=True)
class AFDMetadata:
    """Forward-context metadata visible to plugin-owned model wrappers."""

    afd_tokens_start_loc: list[int]
    afd_reqs_start_loc: list[int]
    afd_stage_idx: int
    afd_connector: Any
    afd_tokens_lens: list[int]
    num_of_stages: int
    ubatch_idx: int = 0
    transaction_id: str | None = None
    afd_tokens_unpadded_lens: list[int] = field(default_factory=list)

    def clone(self) -> AFDMetadata:
        cloned = copy.copy(self)
        cloned.afd_tokens_start_loc = list(self.afd_tokens_start_loc)
        cloned.afd_reqs_start_loc = list(self.afd_reqs_start_loc)
        cloned.afd_tokens_lens = list(self.afd_tokens_lens)
        cloned.afd_tokens_unpadded_lens = list(self.afd_tokens_unpadded_lens)
        return cloned


__all__ = [
    "AFDConnectorMetadata",
    "AFDDPMetadata",
    "AFDMetadata",
    "AFDRecvOutput",
    "AFDSingleDPMetadata",
]
