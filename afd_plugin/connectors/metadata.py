# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD metadata objects shared by runtime classes and model wrappers."""

from __future__ import annotations

import copy
import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

    from afd_plugin.connectors.base import AFDConnectorBase


@dataclass(slots=True)
class AFDDPMetadata:
    """Serializable DPMetadata-compatible payload for AFD control traffic."""

    num_tokens_across_dp_cpu: torch.Tensor
    max_tokens_across_dp_cpu: torch.Tensor | None = None
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
    def sp_local_sizes(self, sequence_parallel_size: int) -> Generator[list[int]]:
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

    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        num_tokens = _cpu_int_tensor_or_list(self.num_tokens_across_dp_cpu)
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
    ) -> Generator[list[int]]:
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


@dataclass(slots=True)
class AFDControlPayload:
    """Structured DP metadata control-plane payload.

    This object is the envelope used by connector control-plane methods:
    ``update_state_from_dp_metadata()``, ``send_dp_metadata_list()``, and
    ``recv_dp_metadata_list()``.

    Attributes:
        dp_metadata_list: Mapping from stage or ubatch index to DP token
            metadata. Values are normalized to ``AFDDPMetadata`` in
            ``__post_init__`` so connector implementations can rely on a stable
            plugin-owned representation instead of vLLM-internal DP metadata
            objects.
        is_graph_capturing: Whether the Attention side is currently executing
            a graph-capture path. Connectors use this to decide how to prepare
            local buffers or backend state before data-path operations.
        is_warmup: Whether the payload belongs to a warmup step. This is
            separate from graph capture because warmup may prepare state without
            representing a real serving step.
    """

    dp_metadata_list: dict[int, AFDDPMetadata]
    is_graph_capturing: bool
    is_warmup: bool

    def __post_init__(self) -> None:
        self.dp_metadata_list = {
            int(stage_idx): _ensure_afd_dp_metadata(dp_metadata)
            for stage_idx, dp_metadata in self.dp_metadata_list.items()
        }
        self.is_graph_capturing = bool(self.is_graph_capturing)
        self.is_warmup = bool(self.is_warmup)


def _ensure_afd_dp_metadata(value: object) -> AFDDPMetadata:
    if isinstance(value, AFDDPMetadata):
        return value
    token_counts = getattr(value, "num_tokens_across_dp_cpu", None)
    if token_counts is None:
        raise TypeError(
            "AFD DP metadata must expose num_tokens_across_dp_cpu",
        )
    max_token_count = getattr(value, "max_tokens_across_dp_cpu", None)
    return AFDDPMetadata(
        num_tokens_across_dp_cpu=_cpu_int_tensor_or_list(token_counts),
        max_tokens_across_dp_cpu=(
            None
            if max_token_count is None
            else _cpu_scalar_tensor_or_int(max_token_count)
        ),
    )


def _cpu_int_tensor_or_list(value: object) -> torch.Tensor:
    values = _to_int_list(value)
    return torch.tensor(values, dtype=torch.int32, device="cpu")


def _cpu_scalar_tensor_or_int(value: object) -> torch.Tensor:
    if isinstance(value, (int, float)):
        value = int(value)
    elif isinstance(value, (list, tuple)):
        value = max(int(item) for item in value)
    else:
        value = int(value.item())
    return torch.tensor(value, dtype=torch.int32, device="cpu")


def _max_token_count(value: object) -> torch.Tensor:
    if isinstance(value, list):
        return torch.tensor(max(_to_int_list(value)), dtype=torch.int32, device="cpu")
    if not isinstance(value, torch.Tensor):
        value = _cpu_int_tensor_or_list(value)
    return value.max()


def _to_int_list(value: object) -> list[int]:
    if isinstance(value, (int, float)):
        value = [value]
    elif isinstance(value, (list, tuple)):
        pass
    else:
        value = value.tolist()
    return [int(item) for item in value]


def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: object,
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
class AFDTransferState:
    """Base class for backend-specific connector metadata payloads."""


@dataclass(slots=True)
class AFDTransferMetadata:
    """Communication metadata for one AFD Attention/FFN exchange.

    ``AFDTransferMetadata`` describes the logical tensor transfer for a single
    layer/stage pair. It is intentionally backend-neutral, with
    ``transfer_state`` reserved for backend-specific state.

    Attributes:
        layer_idx: Model layer index associated with this transfer.
        stage_idx: Stage or ubatch index associated with this transfer.
        seq_lens: Per-peer or per-split token lengths. The sum of this list is
            the expected leading dimension for tensors validated against this
            metadata. For one-to-one transfers this is usually a single-item
            list; for fan-in/fan-out paths it can describe split sizes.
        transfer_state: Optional backend-specific payload. For example,
            CAMP2P stores ``CAMP2PTransferState`` here so receive-time
            results can be reused by the matching send path. P2P does not
            currently require connector-specific data.
    """

    layer_idx: int
    stage_idx: int
    seq_lens: list[int]
    transfer_state: AFDTransferState | None = None

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
    ) -> AFDTransferMetadata:
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
    ) -> AFDTransferMetadata:
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=list(seq_lens),
        )

    def validate_tensor_shape(self, tensor_shape: tuple[int, ...]) -> bool:
        return len(tensor_shape) > 0 and tensor_shape[0] == self.total_tokens


@dataclass(slots=True)
class AFDA2FTransferPayload:
    """Unified Attention-to-FFN receive payload.

    ``recv_attn_output()`` returns this object on the FFN side. The first two
    fields are the common contract; the remaining fields carry backend-specific
    outputs produced by NPU/CAM-style custom ops.

    Attributes:
        hidden_states: Hidden-state tensor received from the Attention side.
        metadata: Metadata describing the received transfer. FFN runners pass
            this metadata through the FFN compute and back into
            ``send_ffn_output()``.
        group_list: Optional expert/group token-count payload. CAMP2P and
            async CAM style connectors use this for MoE routing metadata.
        topk_weights: Optional top-k routing weights produced or forwarded by
            the backend receive path.
        topk_ids: Optional top-k expert ids produced or forwarded by the
            backend receive path.
        router_logits: Optional router logits for backends that forward router
            outputs through the connector payload.
        row_idx: Optional row-index payload for backend-specific token routing.
        x_active_mask: Optional active-token mask returned by CAMP2P/CAM ops.
        dynamic_scales: Optional dynamic quantization scales for routed expert
            tokens.
        cam_p2p_ep_name: Optional CAM/HCCL endpoint name associated with the
            receive path.
        atten_batch_size: Optional backend-reported Attention batch-size or
            token-count tensor. CAMP2P stores it in connector data for the
            matching FFN-to-Attention send.
        expand_idx: Optional expanded-token index payload for MoE routing.
        ep_recv_counts: Optional expert-parallel receive counts.
    """

    hidden_states: torch.Tensor
    metadata: AFDTransferMetadata
    group_list: object = None
    topk_weights: torch.Tensor | None = None
    topk_ids: torch.Tensor | None = None
    router_logits: torch.Tensor | None = None
    row_idx: torch.Tensor | None = None
    x_active_mask: torch.Tensor | None = None
    dynamic_scales: torch.Tensor | None = None
    expand_x_shared: torch.Tensor | None = None
    dynamic_scales_shared: torch.Tensor | None = None
    cam_p2p_ep_name: str | None = None
    atten_batch_size: torch.Tensor | None = None
    expand_idx: torch.Tensor | None = None
    ep_recv_counts: torch.Tensor | None = None
    ep_recv_counts_shared: torch.Tensor | None = None


@dataclass(slots=True)
class AFDF2ATransferPayload:
    """Unified FFN -> Attention payload for separated routed/shared outputs."""

    routed_output: torch.Tensor
    shared_output: torch.Tensor | None = None


@dataclass(slots=True)
class AFDForwardContextMetadata:
    """Forward-context metadata visible to plugin-owned model wrappers."""

    tokens_start_loc: list[int]
    requests_start_loc: list[int]
    stage_idx: int
    afd_connector: AFDConnectorBase
    tokens_lens: list[int]
    num_stages: int
    transaction_id: str | None = None
    tokens_unpadded_lens: list[int] = field(default_factory=list)

    def clone(self) -> AFDForwardContextMetadata:
        cloned = copy.copy(self)
        cloned.tokens_start_loc = list(self.tokens_start_loc)
        cloned.requests_start_loc = list(self.requests_start_loc)
        cloned.tokens_lens = list(self.tokens_lens)
        cloned.tokens_unpadded_lens = list(self.tokens_unpadded_lens)
        return cloned


def _to_int(value: object) -> int:
    item = getattr(value, "item", None)
    return int(item() if callable(item) else value)


def encode_control_payload(payload: AFDControlPayload) -> bytes:
    """Serialize an ``AFDControlPayload`` to a compact JSON byte string.

    Only ``num_tokens_across_dp_cpu`` / ``max_tokens_across_dp_cpu`` per stage
    and the graph-capturing / warmup flags are carried; these are the fields the
    FFN-side connectors read back after decode. Serializing to a plugin-owned
    minimal schema keeps the wire format decoupled from vLLM-internal DP
    metadata objects.
    """
    metadata_payload: dict[str, dict[str, int | list[int]]] = {}
    for stage_idx, dp_metadata in payload.dp_metadata_list.items():
        token_counts = getattr(dp_metadata, "num_tokens_across_dp_cpu", None)
        if token_counts is None:
            raise TypeError(
                "AFD DP metadata must expose num_tokens_across_dp_cpu "
                "for JSON serialization",
            )
        token_counts_list = _to_int_list(token_counts)
        max_token_count = getattr(dp_metadata, "max_tokens_across_dp_cpu", None)
        if max_token_count is None:
            max_token_count = max(token_counts_list)
        metadata_payload[str(int(stage_idx))] = {
            "num_tokens_across_dp_cpu": token_counts_list,
            "max_tokens_across_dp_cpu": _to_int(max_token_count),
        }

    wire_payload = {
        "dp_metadata_list": metadata_payload,
        "is_graph_capturing": bool(payload.is_graph_capturing),
        "is_warmup": bool(payload.is_warmup),
    }
    return json.dumps(wire_payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8",
    )


def decode_control_payload(payload_bytes: bytes) -> AFDControlPayload:
    """Rebuild an ``AFDControlPayload`` from ``encode_control_payload``."""
    payload = json.loads(payload_bytes.decode("utf-8"))
    dp_metadata_list = {
        int(stage_idx): AFDDPMetadata(
            num_tokens_across_dp_cpu=torch.tensor(
                [int(value) for value in metadata["num_tokens_across_dp_cpu"]],
                dtype=torch.int32,
                device="cpu",
            ),
            max_tokens_across_dp_cpu=torch.tensor(
                int(metadata["max_tokens_across_dp_cpu"]),
                dtype=torch.int32,
                device="cpu",
            ),
        )
        for stage_idx, metadata in payload["dp_metadata_list"].items()
    }
    return AFDControlPayload(
        dp_metadata_list=dp_metadata_list,
        is_graph_capturing=bool(payload.get("is_graph_capturing", False)),
        is_warmup=bool(payload.get("is_warmup", False)),
    )


def send_control_payload(
    payload: AFDControlPayload,
    *,
    dst: int | list[int] | tuple[int, ...],
    group: ProcessGroup,
    device: torch.device,
) -> None:
    """Encode and send a DP-metadata payload to ``dst`` over ``group``.

    The payload is sent as two messages: a ``long`` size tensor followed by the
    ``uint8`` object tensor, both staged on ``device`` so the caller controls
    whether the backend transport sees CUDA/NPU or CPU tensors.
    """
    object_bytes = encode_control_payload(payload)
    object_tensor_cpu = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)
    object_tensor = object_tensor_cpu.to(device)
    size_tensor = torch.tensor(
        [object_tensor.numel()],
        dtype=torch.long,
        device=device,
    )
    dsts = [dst] if isinstance(dst, int) else dst
    for d in dsts:
        torch.distributed.send(size_tensor, dst=d, group=group)
        torch.distributed.send(object_tensor, dst=d, group=group)


def recv_control_payload(
    *,
    src: int,
    group: ProcessGroup,
    device: torch.device,
) -> AFDControlPayload:
    """Receive and decode a DP-metadata payload from ``src`` over ``group``."""
    size_tensor = torch.empty(1, dtype=torch.long, device=device)
    rank_size = torch.distributed.recv(size_tensor, src=src, group=group)
    object_tensor = torch.empty(
        int(size_tensor.item()),
        dtype=torch.uint8,
        device=device,
    )
    rank_object = torch.distributed.recv(object_tensor, src=src, group=group)
    if rank_object != rank_size:
        raise RuntimeError("received AFD DP metadata fragments from different ranks")
    return decode_control_payload(object_tensor.cpu().numpy().tobytes())


__all__ = [
    "AFDTransferState",
    "AFDTransferMetadata",
    "AFDDPMetadata",
    "AFDControlPayload",
    "AFDF2ATransferPayload",
    "AFDForwardContextMetadata",
    "AFDA2FTransferPayload",
    "AFDSingleDPMetadata",
    "decode_control_payload",
    "encode_control_payload",
    "recv_control_payload",
    "send_control_payload",
]
