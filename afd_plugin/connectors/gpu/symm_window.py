# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Symmetric-memory window used by the async GPU AFD connector.

The window mirrors the CAM shared-window substrate: every rank allocates an
identically sized symmetric buffer, senders write one-sided into the receiver's
buffer, and arrival is announced by a magic-stamped flag word that the receiver
polls. Only the allocation is collective; once it is done a sender never needs
the receiver to participate.

Layout of one window::

    [flag[num_regions * ring_depth] | slot(0, 0) | slot(0, 1) | ...]

``slot(region, ring)`` holds a fixed header followed by the routed/shared
payloads. Dispatch (A -> F) and combine (F -> A) use the same slot layout, so a
single spec sizes both directions.

Flag words are written *after* the payload on the same stream. Same-stream
device-to-device copies complete in issue order, so a visible flag implies a
complete payload. That holds for NVLink-mapped peer memory; a cross-node
transport would need an explicit fence here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from afd_plugin.connectors.gpu import nvshmem_rt

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

# Header words shared by dispatch and combine, followed by expert_counts.
HEADER_MAGIC = 0x41464447  # "AFDG"
HEADER_VERSION = 1
_H_MAGIC = 0
_H_VERSION = 1
_H_SEQ = 2
_H_SRC_ROLE_RANK = 3
_H_LAYER_IDX = 4
_H_STAGE_IDX = 5
_H_NUM_TOKENS = 6
_H_ROUTED_TOKENS = 7
_H_SHARED_TOKENS = 8
_H_TOPK = 9
_H_FLAGS = 10
_H_ECHO_SEQ = 11  # combine only: the dispatch seq being answered
HEADER_FIXED_WORDS = 12

FLAG_EMPTY = 0
FLAG_SHUTDOWN_BIT = 1 << 1

# Every field starts on this boundary so a byte offset stays divisible by the
# element size of whichever dtype views it.
_FIELD_ALIGN = 256


def _align(offset: int) -> int:
    return (offset + _FIELD_ALIGN - 1) // _FIELD_ALIGN * _FIELD_ALIGN


@dataclass(frozen=True, slots=True)
class SlotLayout:
    """Byte offsets and element counts for the fields inside one slot."""

    header_words: int
    routed_cap: int
    token_cap: int
    hidden_size: int
    payload_itemsize: int

    header_off: int
    route_table_off: int
    routed_x_off: int
    shared_idx_off: int
    shared_x_off: int
    slot_bytes: int

    @classmethod
    def build(
        cls,
        *,
        expert_per_rank: int,
        routed_cap: int,
        token_cap: int,
        hidden_size: int,
        payload_itemsize: int,
    ) -> SlotLayout:
        header_words = HEADER_FIXED_WORDS + expert_per_rank
        header_off = 0
        route_table_off = _align(header_off + header_words * 4)
        routed_x_off = _align(route_table_off + 2 * routed_cap * 4)
        shared_idx_off = _align(
            routed_x_off + routed_cap * hidden_size * payload_itemsize
        )
        shared_x_off = _align(shared_idx_off + token_cap * 4)
        slot_bytes = _align(shared_x_off + token_cap * hidden_size * payload_itemsize)
        return cls(
            header_words=header_words,
            routed_cap=routed_cap,
            token_cap=token_cap,
            hidden_size=hidden_size,
            payload_itemsize=payload_itemsize,
            header_off=header_off,
            route_table_off=route_table_off,
            routed_x_off=routed_x_off,
            shared_idx_off=shared_idx_off,
            shared_x_off=shared_x_off,
            slot_bytes=slot_bytes,
        )


def encode_header(
    layout: SlotLayout,
    *,
    seq: int,
    src_role_rank: int,
    layer_idx: int,
    stage_idx: int,
    num_tokens: int,
    routed_tokens: int,
    shared_tokens: int,
    topk: int,
    flags: int,
    expert_counts: list[int],
    echo_seq: int = 0,
) -> torch.Tensor:
    """Build the fixed header for one slot as a CPU int32 tensor."""
    expert_per_rank = layout.header_words - HEADER_FIXED_WORDS
    if len(expert_counts) != expert_per_rank:
        raise ValueError(
            f"expert_counts must have {expert_per_rank} entries, "
            f"got {len(expert_counts)}",
        )
    header = torch.zeros(layout.header_words, dtype=torch.int32)
    header[_H_MAGIC] = HEADER_MAGIC
    header[_H_VERSION] = HEADER_VERSION
    header[_H_SEQ] = seq
    header[_H_SRC_ROLE_RANK] = src_role_rank
    header[_H_LAYER_IDX] = layer_idx
    header[_H_STAGE_IDX] = stage_idx
    header[_H_NUM_TOKENS] = num_tokens
    header[_H_ROUTED_TOKENS] = routed_tokens
    header[_H_SHARED_TOKENS] = shared_tokens
    header[_H_TOPK] = topk
    header[_H_FLAGS] = flags
    header[_H_ECHO_SEQ] = echo_seq
    if expert_per_rank:
        header[HEADER_FIXED_WORDS:] = torch.tensor(expert_counts, dtype=torch.int32)
    return header


@dataclass(frozen=True, slots=True)
class SlotHeader:
    """Decoded slot header."""

    seq: int
    src_role_rank: int
    layer_idx: int
    stage_idx: int
    num_tokens: int
    routed_tokens: int
    shared_tokens: int
    topk: int
    flags: int
    echo_seq: int
    expert_counts: list[int]

    @property
    def is_shutdown(self) -> bool:
        return bool(self.flags & FLAG_SHUTDOWN_BIT)


def decode_header(header: torch.Tensor) -> SlotHeader:
    """Decode a CPU int32 header tensor, validating magic and version."""
    values = header.tolist()
    if values[_H_MAGIC] != HEADER_MAGIC:
        raise RuntimeError(
            f"AFD async GPU header magic mismatch: got {values[_H_MAGIC]:#x}, "
            f"expected {HEADER_MAGIC:#x}",
        )
    if values[_H_VERSION] != HEADER_VERSION:
        raise RuntimeError(
            f"AFD async GPU header version {values[_H_VERSION]} is not supported "
            f"(expected {HEADER_VERSION})",
        )
    return SlotHeader(
        seq=values[_H_SEQ],
        src_role_rank=values[_H_SRC_ROLE_RANK],
        layer_idx=values[_H_LAYER_IDX],
        stage_idx=values[_H_STAGE_IDX],
        num_tokens=values[_H_NUM_TOKENS],
        routed_tokens=values[_H_ROUTED_TOKENS],
        shared_tokens=values[_H_SHARED_TOKENS],
        topk=values[_H_TOPK],
        flags=values[_H_FLAGS],
        echo_seq=values[_H_ECHO_SEQ],
        expert_counts=values[HEADER_FIXED_WORDS:],
    )


@dataclass(slots=True)
class ArrivedSlot:
    """One arrival found by ``SymmWindow.poll``."""

    region: int
    ring: int
    header: SlotHeader


class SymmWindow:
    """Symmetric receive window plus the one-sided writes that fill peers'."""

    def __init__(
        self,
        *,
        num_regions: int,
        ring_depth: int,
        layout: SlotLayout,
        payload_dtype: torch.dtype,
        device: torch.device,
        group: ProcessGroup,
        rank: int,
        world_size: int,
    ) -> None:
        if num_regions <= 0 or ring_depth <= 0:
            raise ValueError("num_regions and ring_depth must be positive")
        self.num_regions = num_regions
        self.ring_depth = ring_depth
        self.layout = layout
        self.payload_dtype = payload_dtype
        self.device = device
        self.rank = rank

        self.num_flags = num_regions * ring_depth
        self._flag_bytes = _align(self.num_flags * 4)
        self.total_bytes = self._flag_bytes + self.num_flags * layout.slot_bytes

        nvshmem_rt.init(group, rank, world_size)
        self._base = nvshmem_rt.malloc(self.total_bytes)
        # Peer mappings are stable for the life of the allocation, so resolve
        # them once instead of per transfer.
        self._peer_base = {
            pe: (self._base if pe == rank else nvshmem_rt.peer_ptr(self._base, pe))
            for pe in range(world_size)
        }
        self.local_bytes_view().zero_()
        torch.cuda.synchronize()

        # The layout is static, so every window view is built once and then
        # sliced. Rebuilding them per transfer meant a __cuda_array_interface__
        # import on every field of every message, which dominated the data path.
        self._view_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}
        self._flag_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._flags_local = nvshmem_rt.tensor_from_ptr(
            self._base,
            byte_offset=0,
            sizes=(self.num_flags,),
            dtype=torch.int32,
            device=device,
        )

        # Pinned staging keeps header transfers asynchronous; a pageable source
        # forces a blocking copy, and there is one header per peer per layer.
        # One row per (peer, ring): a single shared row would be overwritten on
        # the host by the next peer in the send loop while its own asynchronous
        # copy was still in flight, delivering another peer's token counts.
        self._header_send = torch.zeros(
            (world_size, ring_depth, layout.header_words),
            dtype=torch.int32,
        ).pin_memory()
        self._header_recv = torch.zeros(
            layout.header_words,
            dtype=torch.int32,
        ).pin_memory()
        # Host mirror of the local flag array; one D2H per poll refreshes it.
        self._flag_host = torch.zeros(self.num_flags, dtype=torch.int32).pin_memory()
        self._seen = [FLAG_EMPTY] * self.num_flags

    def local_bytes_view(self) -> torch.Tensor:
        return nvshmem_rt.tensor_from_ptr(
            self._base,
            byte_offset=0,
            sizes=(self.total_bytes,),
            dtype=torch.uint8,
            device=self.device,
        )

    def _slot_byte_off(self, region: int, ring: int) -> int:
        return (
            self._flag_bytes
            + (region * self.ring_depth + ring) * self.layout.slot_bytes
        )

    def _capacity_view(
        self,
        peer: int,
        region: int,
        ring: int,
        field_off: int,
    ) -> torch.Tensor:
        """Return the cached full-capacity view of one slot field."""
        key = (peer, region, ring, field_off)
        view = self._view_cache.get(key)
        if view is not None:
            return view

        layout = self.layout
        hidden = layout.hidden_size
        if field_off == layout.header_off:
            sizes, dtype = (layout.header_words,), torch.int32
        elif field_off == layout.route_table_off:
            sizes, dtype = (layout.routed_cap, 2), torch.int32
        elif field_off == layout.routed_x_off:
            sizes, dtype = (layout.routed_cap, hidden), self.payload_dtype
        elif field_off == layout.shared_idx_off:
            sizes, dtype = (layout.token_cap,), torch.int32
        elif field_off == layout.shared_x_off:
            sizes, dtype = (layout.token_cap, hidden), self.payload_dtype
        else:
            raise ValueError(f"unknown slot field offset {field_off}")

        view = nvshmem_rt.tensor_from_ptr(
            self._peer_base[peer],
            byte_offset=self._slot_byte_off(region, ring) + field_off,
            sizes=sizes,
            dtype=dtype,
            device=self.device,
        )
        self._view_cache[key] = view
        return view

    def _view(
        self,
        peer: int,
        region: int,
        ring: int,
        field_off: int,
        sizes: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Slicing a cached capacity view costs no CUDA calls, unlike importing
        # a fresh pointer for every field of every message.
        return self._capacity_view(peer, region, ring, field_off)[: sizes[0]]

    def _flag_view(self, peer: int, flag_idx: int) -> torch.Tensor:
        key = (peer, flag_idx)
        view = self._flag_cache.get(key)
        if view is None:
            view = nvshmem_rt.tensor_from_ptr(
                self._peer_base[peer],
                byte_offset=flag_idx * 4,
                sizes=(1,),
                dtype=torch.int32,
                device=self.device,
            )
            self._flag_cache[key] = view
        return view

    # ------------------------------------------------------------------
    # Send side: every write targets ``peer``'s window, one-sided.
    # ------------------------------------------------------------------

    def write_slot(
        self,
        *,
        peer: int,
        region: int,
        ring: int,
        header: torch.Tensor,
        route_table: torch.Tensor | None,
        routed_x: torch.Tensor | None,
        shared_idx: torch.Tensor | None,
        shared_x: torch.Tensor | None,
    ) -> None:
        """Write one slot into ``peer``'s window, then stamp its flag.

        The flag copy is issued last on the same stream, so a peer that observes
        the flag also observes the payload.
        """
        layout = self.layout
        if routed_x is not None and routed_x.shape[0] > layout.routed_cap:
            raise RuntimeError(
                f"routed tokens {routed_x.shape[0]} exceed routed_cap "
                f"{layout.routed_cap}; raise routed_cap_multiplier",
            )
        if shared_x is not None and shared_x.shape[0] > layout.token_cap:
            raise RuntimeError(
                f"shared tokens {shared_x.shape[0]} exceed token_cap "
                f"{layout.token_cap}",
            )

        # Stage through pinned memory so the copy is asynchronous: a pageable
        # source would force a blocking transfer, and there is one header per
        # peer per layer.
        staging = self._header_send[peer][ring]
        staging.copy_(header)
        self._capacity_view(peer, region, ring, layout.header_off).copy_(
            staging,
            non_blocking=True,
        )

        if route_table is not None and route_table.numel():
            n = route_table.shape[0]
            self._view(
                peer,
                region,
                ring,
                layout.route_table_off,
                (n, 2),
                torch.int32,
            ).copy_(route_table, non_blocking=True)
        if routed_x is not None and routed_x.numel():
            n = routed_x.shape[0]
            self._view(
                peer,
                region,
                ring,
                layout.routed_x_off,
                (n, layout.hidden_size),
                self.payload_dtype,
            ).copy_(routed_x, non_blocking=True)
        if shared_idx is not None and shared_idx.numel():
            n = shared_idx.shape[0]
            self._view(
                peer,
                region,
                ring,
                layout.shared_idx_off,
                (n,),
                torch.int32,
            ).copy_(shared_idx, non_blocking=True)
        if shared_x is not None and shared_x.numel():
            n = shared_x.shape[0]
            self._view(
                peer,
                region,
                ring,
                layout.shared_x_off,
                (n, layout.hidden_size),
                self.payload_dtype,
            ).copy_(shared_x, non_blocking=True)

        seq = int(header[_H_SEQ].item())
        flag_idx = region * self.ring_depth + ring
        self._flag_view(peer, flag_idx).fill_(seq)

    # ------------------------------------------------------------------
    # Receive side.
    # ------------------------------------------------------------------

    def poll(self) -> ArrivedSlot | None:
        """Return the first slot whose flag advanced past what we consumed.

        ponytail: host-side poll, one D2H per call. Correct but it burns a
        synchronize per attempt; replace with a device-side ``wait_any`` kernel
        spinning on the flag array when the poll shows up in a profile.
        """
        self._flag_host.copy_(self._flags_local, non_blocking=False)
        host = self._flag_host.tolist()
        for idx in range(self.num_flags):
            if host[idx] != self._seen[idx]:
                self._seen[idx] = host[idx]
                region, ring = divmod(idx, self.ring_depth)
                return ArrivedSlot(
                    region=region,
                    ring=ring,
                    header=self.read_header(region, ring),
                )
        return None

    def read_header(self, region: int, ring: int) -> SlotHeader:
        # Reuse the pinned mirror instead of allocating a fresh host tensor on
        # every arrival.
        self._header_recv.copy_(
            self._capacity_view(self.rank, region, ring, self.layout.header_off),
        )
        return decode_header(self._header_recv)

    def local_route_table(self, region: int, ring: int, count: int) -> torch.Tensor:
        return self._view(
            self.rank,
            region,
            ring,
            self.layout.route_table_off,
            (count, 2),
            torch.int32,
        )

    def local_routed(self, region: int, ring: int, count: int) -> torch.Tensor:
        return self._view(
            self.rank,
            region,
            ring,
            self.layout.routed_x_off,
            (count, self.layout.hidden_size),
            self.payload_dtype,
        )

    def local_shared_idx(self, region: int, ring: int, count: int) -> torch.Tensor:
        return self._view(
            self.rank,
            region,
            ring,
            self.layout.shared_idx_off,
            (count,),
            torch.int32,
        )

    def local_shared(self, region: int, ring: int, count: int) -> torch.Tensor:
        return self._view(
            self.rank,
            region,
            ring,
            self.layout.shared_x_off,
            (count, self.layout.hidden_size),
            self.payload_dtype,
        )

    def close(self) -> None:
        # ponytail: the symmetric allocation is left to process teardown.
        # nvshmem_free is collective, so freeing here would need both roles to
        # shut down in lockstep; add it if windows are ever recreated in-process.
        self._peer_base = {}
