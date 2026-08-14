# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NVSHMEM-backed asynchronous connector for CUDA AFD.

``GpuAsyncAFDConnector`` is the CUDA counterpart of ``CAMAsyncAFDConnector``:
Attention ranks run MoE routing, write routed tokens one-sided into the FFN
ranks' symmetric windows, and later reduce the weighted expert output; FFN ranks
poll their window, run their local experts, and write the result back. There is
no DP metadata control plane (``control_plane`` stays ``None``) and FFN work is
driven directly by the connector receive loop, so Attention DP replicas never
wait for each other.

The world is Attention-first, ``[A0, A1, ..., F0, F1, ...]``, matching
``CAMAsyncAFDConnector``. Every Attention rank routes to every FFN rank, so an
FFN window holds one region per Attention rank and vice versa.

See ``docs/design/rfc_async_gpu_connector.md``. Supported deployment requires
``async=true``, ``compute_gate_on_attention=true``, eager execution, prefill
only, and a single node.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

import torch
from torch import Tensor
from vllm.logger import init_logger

from afd_plugin.config import AFDConfig
from afd_plugin.config_utils import (
    coerce_extra_bool,
    coerce_extra_positive_int,
    coerce_extra_str,
)
from afd_plugin.connectors.async_topology import (
    ASYNC_MOE_REQUEST_SPLIT,
    build_async_topology,
)
from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.connectors.gpu.symm_window import (
    FLAG_SHUTDOWN_BIT,
    SlotLayout,
    SymmWindow,
    encode_header,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDF2ATransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.distributed import init_afd_process_group

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup
    from vllm.config import VllmConfig

AFD_ASYNC_GPU_GROUP_NAME = "afd_async_gpu"

_GPU_ASYNC_EXTRA_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "attn_ranks_per_dp",
        "ring_depth",
        "routed_cap_multiplier",
        "recv_poll_timeout_ms",
        "async_moe_ubatching",
        "async_moe_num_ubatches",
        "async_moe_split",
    },
)

# Name the logger inside vLLM's tree: vLLM installs its handler on the "vllm"
# logger only, so a bare ``afd_plugin.*`` logger propagates to a handler-less
# root and every line is dropped -- which is how the window summary, the only
# report of a multi-GiB allocation, stayed invisible.
logger = init_logger(f"vllm.{__name__}")


def _coerce_extra_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be positive, got {result}")
    return result


@dataclass(frozen=True)
class GpuAsyncExtraInfo(ConnectorExtraInfo):
    """Typed async GPU connector configuration.

    Attributes:
        attn_ranks_per_dp: Number of Attention ranks in each data-parallel group.
        ring_depth: Slots per peer region. Derived from the send-then-recv
            invariant, not a performance knob: an Attention rank has at most one
            in-flight request per ``(peer, stage)``, so ``num_stages`` suffices.
        routed_cap_multiplier: Headroom over the balanced routed-token estimate.
            Real gates are not balanced -- DeepSeek-V2-Lite at 2 FFN ranks was
            observed 1.33x over the even split -- so the default leaves room.
            The true worst case is ``ffn_size`` (every partial to one rank).
        recv_poll_timeout_ms: Idle poll timeout on the FFN loop; bounds shutdown
            response time.
        async_moe_ubatching: Whether request-boundary async MoE ubatching is used.
        async_moe_num_ubatches: Number of stages used by async MoE ubatching.
        async_moe_split: Boundary at which async MoE work is split.
    """

    attn_ranks_per_dp: int = 1
    ring_depth: int = 0
    routed_cap_multiplier: float = 2.0
    recv_poll_timeout_ms: int = 50
    async_moe_ubatching: bool = False
    async_moe_num_ubatches: int = 2
    async_moe_split: str = ASYNC_MOE_REQUEST_SPLIT

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> GpuAsyncExtraInfo:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__} connector_extra_config must be a mapping, "
                f"got {type(raw).__name__}",
            )
        unknown = sorted(
            str(key) for key in raw if key not in _GPU_ASYNC_EXTRA_CONFIG_FIELDS
        )
        if unknown:
            raise ValueError(
                "unknown AFD async GPU connector_extra_config field(s): "
                + ", ".join(unknown),
            )

        ubatching = coerce_extra_bool(
            raw.get("async_moe_ubatching", False),
            field_name="async_moe_ubatching",
        )
        num_ubatches = coerce_extra_positive_int(
            raw.get("async_moe_num_ubatches", 2),
            field_name="async_moe_num_ubatches",
        )
        # Ring depth follows the number of live stages unless pinned explicitly.
        ring_depth = coerce_extra_positive_int(
            raw.get("ring_depth", num_ubatches if ubatching else 1),
            field_name="ring_depth",
        )
        return cls(
            attn_ranks_per_dp=coerce_extra_positive_int(
                raw.get("attn_ranks_per_dp", 1),
                field_name="attn_ranks_per_dp",
            ),
            ring_depth=ring_depth,
            routed_cap_multiplier=_coerce_extra_float(
                raw.get("routed_cap_multiplier", 2.0),
                field_name="routed_cap_multiplier",
            ),
            recv_poll_timeout_ms=coerce_extra_positive_int(
                raw.get("recv_poll_timeout_ms", 50),
                field_name="recv_poll_timeout_ms",
            ),
            async_moe_ubatching=ubatching,
            async_moe_num_ubatches=num_ubatches,
            async_moe_split=coerce_extra_str(
                raw.get("async_moe_split", ASYNC_MOE_REQUEST_SPLIT),
                field_name="async_moe_split",
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "attn_ranks_per_dp": self.attn_ranks_per_dp,
            "ring_depth": self.ring_depth,
            "routed_cap_multiplier": self.routed_cap_multiplier,
            "recv_poll_timeout_ms": self.recv_poll_timeout_ms,
            "async_moe_ubatching": self.async_moe_ubatching,
            "async_moe_num_ubatches": self.async_moe_num_ubatches,
            "async_moe_split": self.async_moe_split,
        }


class ConnectorShutdown(RuntimeError):  # noqa: N818
    """Raised on the FFN loop when a peer announced shutdown."""


@dataclass(slots=True)
class GpuAsyncTransferState(AFDTransferState):
    """FFN-side state carried from dispatch recv through combine send.

    ``region``/``ring`` locate the window slot so ``send_ffn_work_item_output``
    can write back to the originating Attention rank and release the slot;
    ``route_table`` is echoed so the Attention side can scatter the result.
    """

    region: int = 0
    ring: int = 0
    src_role_rank: int = 0
    layer_idx: int = 0
    stage_idx: int = 0
    seq: int = 0
    num_tokens: int = 0
    routed_tokens: int = 0
    shared_tokens: int = 0
    group_list: Tensor | None = None
    route_table: Tensor | None = None
    shared_idx: Tensor | None = None
    expand_x_shared: Tensor | None = None


@dataclass(slots=True)
class GpuAsyncFFNWorkItem:
    """Normalized FFN-side work item produced by a window arrival."""

    hidden_states: Tensor
    context: AFDTransferContext
    recv_output: AFDA2FTransferPayload
    layer_idx: int
    stage_idx: int
    num_tokens: int
    total_num_tokens: int
    shared_num_tokens: int


@dataclass(slots=True)
class _PendingDispatch:
    """Attention-side record of one in-flight layer, popped by combine recv."""

    context: AFDTransferContext
    topk_weights: Tensor
    num_tokens: int
    ring: int
    seq: int
    expected_ffn: list[int]


def plan_dispatch(
    topk_ids: Tensor,
    *,
    ffn_size: int,
    expert_per_rank: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Cluster ``(token, topk_slot)`` partials by destination FFN rank.

    Sorting by the global expert id groups partials by destination rank and, in
    the same pass, by local expert inside each destination -- the two orderings
    the receiver needs. Returns ``(route_table, counts, offsets)`` where
    ``route_table[i] = (token_idx, topk_slot)`` in that order, ``counts`` holds
    per-global-expert partial counts padded to ``ffn_size * expert_per_rank``,
    and ``offsets`` is the exclusive prefix sum of ``counts``.
    """
    num_slots = topk_ids.shape[1]
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=ffn_size * expert_per_rank)
    offsets = torch.cumsum(counts, dim=0) - counts
    route_table = torch.stack(
        (order // num_slots, order % num_slots),
        dim=1,
    ).to(torch.int32)
    return route_table, counts, offsets


def combine_scatter(
    accumulator: Tensor,
    *,
    routed_out: Tensor,
    route_table: Tensor,
    topk_weights: Tensor,
) -> None:
    """Weighted-scatter one FFN rank's routed output into ``accumulator``."""
    token_idx = route_table[:, 0].to(torch.int64)
    slot_idx = route_table[:, 1].to(torch.int64)
    weights = topk_weights[token_idx, slot_idx].to(accumulator.dtype)
    accumulator.index_add_(
        0,
        token_idx,
        routed_out.to(accumulator.dtype) * weights.unsqueeze(1),
    )


class GpuAsyncAFDConnector(AFDConnectorBase):
    """NVSHMEM symmetric-window asynchronous connector for CUDA AFD."""

    control_plane = None

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> GpuAsyncExtraInfo:
        return GpuAsyncExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self._initialized = False
        hf_config = vllm_config.model_config.hf_config
        self.hidden_size = hf_config.hidden_size
        self.topk = hf_config.num_experts_per_tok
        self.num_routed_experts = hf_config.n_routed_experts
        self.payload_dtype = vllm_config.model_config.dtype
        self.max_seq_len = vllm_config.scheduler_config.max_num_batched_tokens
        self.tp_size = self.extra_info.attn_ranks_per_dp

        self.topology = build_async_topology(
            afd_config,
            role_rank,
            num_routed_experts=self.num_routed_experts,
        )
        self.world_rank = self.topology.world_rank
        self.attn_size = self.topology.attn_size
        self.ffn_size = self.topology.ffn_size
        self.expert_per_rank = self.topology.expert_per_rank
        self.is_attention = afd_config.role == "attention"

        self.ring_depth = self.extra_info.ring_depth
        # Every Attention rank routes to every FFN rank, so a window carries one
        # region per opposite-role peer. Both roles allocate the larger of the
        # two so the symmetric allocation matches.
        self.num_regions = max(self.attn_size, self.ffn_size)
        routed_cap = int(
            -(-self.max_seq_len * self.topk // self.ffn_size)
            * self.extra_info.routed_cap_multiplier,
        )
        self.routed_cap = max(1, routed_cap)
        self.token_cap = max(1, self.max_seq_len)
        self.layout = SlotLayout.build(
            expert_per_rank=self.expert_per_rank,
            routed_cap=self.routed_cap,
            token_cap=self.token_cap,
            hidden_size=self.hidden_size,
            payload_itemsize=torch.empty(0, dtype=self.payload_dtype).element_size(),
        )

        self.pg: ProcessGroup | None = None
        self.window: SymmWindow | None = None
        self._seq = 0
        self._pending: dict[int, list[_PendingDispatch]] = {}
        self._free_rings: dict[int, list[int]] = {}

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def init_afd_connector(self) -> None:
        """Collectively create the AFD world group and the symmetric window.

        All Attention and FFN ranks must call this with identical rendezvous and
        topology settings; the window allocation is symmetric, so a mismatched
        size fails here rather than corrupting a later transfer.
        """
        if self._initialized:
            return

        self.pg = init_afd_process_group(
            backend="nccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.topology.world_size,
            rank=self.world_rank,
            group_name=AFD_ASYNC_GPU_GROUP_NAME,
            timeout=timedelta(minutes=30),
        )
        device = torch.device("cuda", self.local_rank)
        self.window = SymmWindow(
            num_regions=self.num_regions,
            ring_depth=self.ring_depth,
            layout=self.layout,
            payload_dtype=self.payload_dtype,
            device=device,
            group=self.pg,
            rank=self.world_rank,
            world_size=self.topology.world_size,
        )
        for stage in range(max(1, self.extra_info.async_moe_num_ubatches)):
            self._free_rings[stage] = list(range(self.ring_depth))
        logger.info(
            "AFD async GPU window ready: role=%s role_rank=%d world_rank=%d/%d "
            "regions=%d rings=%d routed_cap=%d slot=%.1fMiB total=%.1fMiB",
            self.afd_config.role,
            self.role_rank,
            self.world_rank,
            self.topology.world_size,
            self.num_regions,
            self.ring_depth,
            self.routed_cap,
            self.layout.slot_bytes / 2**20,
            self.window.total_bytes / 2**20,
        )
        self._initialized = True

    def close(self) -> None:
        if self.window is not None:
            self.window.close()
        self.window = None
        if self.pg is not None:
            import torch.distributed as dist

            dist.destroy_process_group(self.pg)
        self.pg = None
        self._pending.clear()
        self._free_rings.clear()
        self._initialized = False

    def select_experts(self, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Run vLLM's grouped top-k on the Attention side.

        ``compute_gate_topk`` delegates expert selection to the connector so the
        CAM and CUDA paths can share one gate; this is the CUDA half.
        """
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
            grouped_topk,
        )

        if kwargs.get("mix_placement"):
            raise RuntimeError(
                "AFD async GPU connector does not support mix_placement",
            )
        return grouped_topk(
            hidden_states=kwargs["hidden_states"],
            gating_output=kwargs["router_logits"],
            topk=kwargs["top_k"],
            renormalize=kwargs["renormalize"],
            num_expert_group=kwargs.get("num_expert_group", 0),
            topk_group=kwargs.get("topk_group", 0),
            scoring_func=kwargs.get("scoring_func", "softmax"),
            routed_scaling_factor=kwargs.get("routed_scaling_factor", 1.0),
            e_score_correction_bias=kwargs.get("e_score_correction_bias"),
        )

    def _require_initialized(self) -> SymmWindow:
        if not self._initialized or self.window is None:
            raise RuntimeError("AFD async GPU connector is not initialized")
        return self.window

    # ==================================================================
    # Attention-side data path
    # ==================================================================

    def send_attn_output(
        self,
        hidden_states: Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Route this layer's tokens and write them into every FFN window.

        ``topk_ids``/``topk_weights`` come from the Attention-side gate. Weights
        stay local -- only the routed activations and their route table go on the
        wire, and the weighting happens in ``recv_ffn_output``.
        """
        window = self._require_initialized()
        topk_ids: Tensor | None = kwargs.get("topk_ids")
        topk_weights: Tensor | None = kwargs.get("topk_weights")
        if topk_ids is None or topk_weights is None:
            raise RuntimeError(
                "AFD async GPU send_attn_output requires topk_ids and "
                "topk_weights from the Attention-side gate",
            )
        metadata = context.metadata
        num_tokens = metadata.total_tokens
        if hidden_states.shape[0] != num_tokens:
            raise ValueError(
                f"hidden_states has {hidden_states.shape[0]} rows but metadata "
                f"expects {num_tokens}",
            )
        if tuple(topk_ids.shape) != (num_tokens, self.topk):
            raise ValueError(
                f"topk_ids shape must be ({num_tokens}, {self.topk}), "
                f"got {tuple(topk_ids.shape)}",
            )

        stage_idx = metadata.stage_idx
        rings = self._free_rings.setdefault(stage_idx, list(range(self.ring_depth)))
        if not rings:
            raise RuntimeError(
                f"AFD async GPU ring exhausted on stage {stage_idx}; the "
                "send-then-recv invariant was violated or the topology config "
                "does not match the actual peer count",
            )
        ring = rings.pop(0)
        self._seq += 1

        route_table, counts, _ = plan_dispatch(
            topk_ids,
            ffn_size=self.ffn_size,
            expert_per_rank=self.expert_per_rank,
        )
        # One D2H per send: offsets are a prefix sum, cheaper to redo on host
        # than to fetch a second tensor.
        counts_host = counts.cpu().tolist()
        offsets_host = [0] * len(counts_host)
        for i in range(1, len(counts_host)):
            offsets_host[i] = offsets_host[i - 1] + counts_host[i - 1]
        token_ids = route_table[:, 0].to(torch.int64)

        # Every FFN rank gets a slot even when routing sends it nothing, and it
        # replies to every slot, so a reply is expected from all of them.
        # Expecting only the ranks that received data leaves the empty rank's
        # reply unmatched and its ring slot never released -- which is what a
        # single-token decode hits, since both the routed segment and the
        # round-robin shared slice can come out empty for one rank.
        expected_ffn = list(range(self.ffn_size))
        for ffn_rank in range(self.ffn_size):
            base = ffn_rank * self.expert_per_rank
            expert_counts = counts_host[base : base + self.expert_per_rank]
            start = offsets_host[base]
            routed_tokens = sum(expert_counts)
            segment = slice(start, start + routed_tokens)

            # Shared-expert tokens are split round-robin across FFN ranks.
            shared_idx = torch.arange(
                ffn_rank,
                num_tokens,
                self.ffn_size,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            header = encode_header(
                self.layout,
                seq=self._seq,
                src_role_rank=self.role_rank,
                layer_idx=metadata.layer_idx,
                stage_idx=stage_idx,
                num_tokens=num_tokens,
                routed_tokens=routed_tokens,
                shared_tokens=int(shared_idx.numel()),
                topk=self.topk,
                flags=0,
                expert_counts=expert_counts,
            )
            # Rows are gathered straight into the FFN rank's window; handing
            # ``write_slot`` the index instead of a gathered tensor saves a
            # local write and re-read of every payload byte.
            window.write_slot(
                peer=self.attn_size + ffn_rank,
                region=self.role_rank,
                ring=ring,
                header=header,
                route_table=route_table[segment],
                routed_x=hidden_states,
                routed_rows=token_ids[segment],
                shared_idx=shared_idx,
                shared_x=hidden_states,
                shared_rows=shared_idx.to(torch.int64),
            )

        logger.debug(
            "AFD dispatch sent: A%d layer=%d stage=%d tokens=%d ring=%d "
            "awaiting_ffn=%s",
            self.role_rank,
            metadata.layer_idx,
            stage_idx,
            num_tokens,
            ring,
            expected_ffn,
        )
        self._pending.setdefault(stage_idx, []).append(
            _PendingDispatch(
                context=context,
                topk_weights=topk_weights,
                num_tokens=num_tokens,
                ring=ring,
                seq=self._seq,
                expected_ffn=expected_ffn,
            ),
        )

    def recv_ffn_output(
        self,
        ref_tensor: Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> Tensor:
        """Wait for this layer's expert output and reduce it back to ``[B, H]``."""

        window = self._require_initialized()
        queue = self._pending.get(ubatch_idx)
        if not queue:
            raise RuntimeError(
                f"AFD async GPU recv_ffn_output has no pending dispatch on "
                f"stage {ubatch_idx}",
            )
        pending = queue.pop(0)

        accumulator = torch.zeros(
            (pending.num_tokens, self.hidden_size),
            dtype=torch.float32,
            device=ref_tensor.device,
        )
        outstanding = set(pending.expected_ffn)
        while outstanding:
            arrived = window.poll()
            if arrived is None:
                continue
            header = arrived.header
            if header.is_shutdown:
                raise ConnectorShutdown(
                    f"FFN rank {header.src_role_rank} announced shutdown",
                )
            if header.src_role_rank not in outstanding:
                raise RuntimeError(
                    "AFD async GPU combine received an unexpected FFN rank "
                    f"{header.src_role_rank}; expected one of {sorted(outstanding)}",
                )
            if header.echo_seq != pending.seq:
                raise RuntimeError(
                    "AFD async GPU combine answered dispatch seq "
                    f"{header.echo_seq} while waiting on {pending.seq} "
                    f"(F{header.src_role_rank}, layer {header.layer_idx}); the "
                    "pending FIFO and the wire have diverged",
                )
            outstanding.discard(header.src_role_rank)
            logger.debug(
                "AFD combine recv: A%d <- F%d layer=%d routed=%d still_waiting=%s",
                self.role_rank,
                header.src_role_rank,
                header.layer_idx,
                header.routed_tokens,
                sorted(outstanding),
            )

            if header.routed_tokens:
                combine_scatter(
                    accumulator,
                    routed_out=window.local_routed(
                        arrived.region,
                        arrived.ring,
                        header.routed_tokens,
                    ),
                    route_table=window.local_route_table(
                        arrived.region,
                        arrived.ring,
                        header.routed_tokens,
                    ),
                    topk_weights=pending.topk_weights,
                )
            if header.shared_tokens:
                accumulator.index_add_(
                    0,
                    window.local_shared_idx(
                        arrived.region,
                        arrived.ring,
                        header.shared_tokens,
                    ).to(torch.int64),
                    window.local_shared(
                        arrived.region,
                        arrived.ring,
                        header.shared_tokens,
                    ).to(accumulator.dtype),
                )

        self._free_rings.setdefault(ubatch_idx, []).append(pending.ring)
        return accumulator.to(ref_tensor.dtype)

    # ==================================================================
    # FFN-side data path
    # ==================================================================

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Block until one Attention rank's routed tokens arrive.

        The layer index, token counts, and per-expert group list all come from
        the arrived slot header; the FFN side knows none of them beforehand.
        """
        window = self._require_initialized()
        timeout_ms = int(kwargs.get("timeout_ms", 0))
        deadline = None
        if timeout_ms:
            import time

            deadline = time.monotonic() + timeout_ms / 1000.0

        while True:
            arrived = window.poll()
            if arrived is not None:
                break
            if deadline is not None:
                import time

                if time.monotonic() >= deadline:
                    raise TimeoutError("AFD async GPU dispatch recv timed out")

        header = arrived.header
        if header.is_shutdown:
            raise ConnectorShutdown(
                f"Attention rank {header.src_role_rank} announced shutdown",
            )

        states = GpuAsyncTransferState(
            region=arrived.region,
            ring=arrived.ring,
            seq=header.seq,
            src_role_rank=header.src_role_rank,
            layer_idx=header.layer_idx,
            stage_idx=header.stage_idx,
            num_tokens=header.num_tokens,
            routed_tokens=header.routed_tokens,
            shared_tokens=header.shared_tokens,
            group_list=torch.tensor(
                header.expert_counts,
                dtype=torch.int64,
                device=torch.device("cuda", self.local_rank),
            ),
            route_table=window.local_route_table(
                arrived.region,
                arrived.ring,
                header.routed_tokens,
            ),
            shared_idx=window.local_shared_idx(
                arrived.region,
                arrived.ring,
                header.shared_tokens,
            ),
            expand_x_shared=window.local_shared(
                arrived.region,
                arrived.ring,
                header.shared_tokens,
            ),
        )
        logger.debug(
            "AFD dispatch recv: F%d <- A%d layer=%d stage=%d routed=%d shared=%d "
            "region=%d ring=%d",
            self.role_rank,
            header.src_role_rank,
            header.layer_idx,
            header.stage_idx,
            header.routed_tokens,
            header.shared_tokens,
            arrived.region,
            arrived.ring,
        )
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=header.layer_idx,
            stage_idx=header.stage_idx,
            seq_lens=[max(1, header.routed_tokens)],
        )
        return AFDA2FTransferPayload(
            hidden_states=window.local_routed(
                arrived.region,
                arrived.ring,
                header.routed_tokens,
            ),
            context=AFDTransferContext(metadata=metadata, states=states),
        )

    def send_ffn_output(
        self,
        ffn_output: Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Write expert output back to the originating Attention rank."""
        window = self._require_initialized()
        states = context.states
        if not isinstance(states, GpuAsyncTransferState):
            raise RuntimeError(
                "AFD async GPU send_ffn_output requires GpuAsyncTransferState",
            )
        shared_output: Tensor | None = kwargs.get("shared_output")
        self._seq += 1
        header = encode_header(
            self.layout,
            seq=self._seq,
            src_role_rank=self.role_rank,
            layer_idx=states.layer_idx,
            stage_idx=states.stage_idx,
            num_tokens=states.num_tokens,
            routed_tokens=states.routed_tokens,
            shared_tokens=states.shared_tokens if shared_output is not None else 0,
            topk=self.topk,
            flags=0,
            expert_counts=list(header_counts(states)),
            echo_seq=states.seq,
        )
        window.write_slot(
            peer=states.src_role_rank,
            region=self.role_rank,
            ring=states.ring,
            header=header,
            route_table=states.route_table,
            routed_x=ffn_output,
            shared_idx=states.shared_idx if shared_output is not None else None,
            shared_x=shared_output,
        )

    # ==================================================================
    # Connector-driven FFN loop
    # ==================================================================

    def recv_ffn_work_item(
        self,
        *,
        stage_idx: int,
        max_num_tokens: int,
    ) -> GpuAsyncFFNWorkItem:
        """Receive and normalize one connector-driven FFN dispatch item."""
        recv_output = self.recv_attn_output(
            ubatch_idx=stage_idx,
            timeout_ms=self.extra_info.recv_poll_timeout_ms,
        )
        states = recv_output.context.states
        assert isinstance(states, GpuAsyncTransferState)
        return GpuAsyncFFNWorkItem(
            hidden_states=recv_output.hidden_states,
            context=recv_output.context,
            recv_output=recv_output,
            layer_idx=states.layer_idx,
            stage_idx=states.stage_idx,
            num_tokens=states.routed_tokens,
            total_num_tokens=states.num_tokens,
            shared_num_tokens=states.shared_tokens,
        )

    def send_ffn_work_item_output(
        self,
        work_item: GpuAsyncFFNWorkItem,
        ffn_output: Tensor | AFDF2ATransferPayload,
    ) -> Tensor:
        """Return one work item's expert output to its Attention rank."""
        if isinstance(ffn_output, AFDF2ATransferPayload):
            routed = ffn_output.routed_output
            shared = ffn_output.shared_output
        else:
            routed = ffn_output
            shared = None
        self.send_ffn_output(routed, work_item.context, shared_output=shared)
        return routed

    def announce_shutdown(self) -> None:
        """Tell every opposite-role peer to leave its receive loop."""
        window = self._require_initialized()
        peers = (
            range(self.attn_size, self.attn_size + self.ffn_size)
            if self.is_attention
            else range(self.attn_size)
        )
        self._seq += 1
        header = encode_header(
            self.layout,
            seq=self._seq,
            src_role_rank=self.role_rank,
            layer_idx=0,
            stage_idx=0,
            num_tokens=0,
            routed_tokens=0,
            shared_tokens=0,
            topk=self.topk,
            flags=FLAG_SHUTDOWN_BIT,
            expert_counts=[0] * self.expert_per_rank,
        )
        for peer in peers:
            window.write_slot(
                peer=peer,
                region=self.role_rank,
                ring=0,
                header=header,
                route_table=None,
                routed_x=None,
                shared_idx=None,
                shared_x=None,
            )


def header_counts(states: GpuAsyncTransferState) -> list[int]:
    """Echo the per-expert group list back on the combine header."""
    if states.group_list is None:
        return []
    return states.group_list.cpu().tolist()


__all__ = [
    "AFD_ASYNC_GPU_GROUP_NAME",
    "ConnectorShutdown",
    "GpuAsyncAFDConnector",
    "GpuAsyncExtraInfo",
    "GpuAsyncFFNWorkItem",
    "GpuAsyncTransferState",
    "combine_scatter",
    "plan_dispatch",
]
