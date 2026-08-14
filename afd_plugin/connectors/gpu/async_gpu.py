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
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class GpuAsyncExtraInfo(ConnectorExtraInfo):
    """Typed async GPU connector configuration.

    Attributes:
        attn_ranks_per_dp: Number of Attention ranks in each data-parallel group.
        ring_depth: Slots per peer region. Derived from the send-then-recv
            invariant, not a performance knob: an Attention rank has at most one
            in-flight request per ``(peer, stage)``, so ``num_stages`` suffices.
        recv_poll_timeout_ms: Idle poll timeout on the FFN loop; bounds shutdown
            response time.
        async_moe_ubatching: Whether request-boundary async MoE ubatching is used.
        async_moe_num_ubatches: Number of stages used by async MoE ubatching.
        async_moe_split: Boundary at which async MoE work is split.
    """

    attn_ranks_per_dp: int = 1
    ring_depth: int = 0
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
    can write back to the originating Attention rank and release the slot.

    ``group_list`` is a device view of the arrived header's trailing words;
    ``expert_counts_host`` is the same numbers as they were decoded on the host,
    kept so the combine header can be built without reading the device back.

    ``expand_idx`` and ``weights`` describe the partials: which shipped row each
    one reads on the way in, and what to weight it by when the expert output is
    reduced back to one row per shipped token on the way out.
    """

    region: int = 0
    ring: int = 0
    src_role_rank: int = 0
    layer_idx: int = 0
    stage_idx: int = 0
    seq: int = 0
    num_tokens: int = 0
    routed_tokens: int = 0
    uniq_tokens: int = 0
    shared_tokens: int = 0
    group_list: Tensor | None = None
    expert_counts_host: list[int] = field(default_factory=list)
    expand_idx: Tensor | None = None
    weights: Tensor | None = None
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
    """Attention-side record of one in-flight layer, popped by combine recv.

    ``uniq_ids[r]`` holds the token ids behind the rows shipped to FFN rank
    ``r``, in the order they were shipped. The reply comes back one row per
    shipped token, already weighted and summed, so combine is a scatter-add at
    exactly those ids.
    """

    context: AFDTransferContext
    uniq_ids: list[Tensor]
    num_tokens: int
    ring: int
    seq: int
    expected_ffn: list[int]


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    """One layer's routing plan, already in the order every consumer wants.

    Partial-indexed fields (``expand_idx``, ``weights``, length
    ``num_tokens * topk``) are sorted by global expert, so a destination's slice
    arrives grouped by local expert and feeds the grouped GEMM directly.
    Row-indexed fields (``uniq_token_ids``) are grouped by destination rank, so
    a destination's payload is also one contiguous slice.

    Attributes:
        counts: Partials per global expert, padded to
            ``ffn_size * expert_per_rank``.
        offsets: Exclusive prefix sum of ``counts``.
        uniq_per_rank: Distinct tokens each FFN rank receives.
        expand_idx: Per partial, which of its destination's shipped rows it
            reads. Already rank-local, so a slice needs no rebasing.
        uniq_token_ids: Per shipped row, the token it carries.
        weights: Per partial, its topk weight.
    """

    counts: Tensor
    offsets: Tensor
    uniq_per_rank: Tensor
    expand_idx: Tensor
    uniq_token_ids: Tensor
    weights: Tensor


def plan_dispatch(
    topk_ids: Tensor,
    topk_weights: Tensor,
    *,
    ffn_size: int,
    expert_per_rank: int,
) -> DispatchPlan:
    """Cluster ``(token, topk_slot)`` partials by destination and deduplicate.

    Sorting by the global expert id groups partials by destination rank and, in
    the same pass, by local expert inside each destination. A second sort by
    ``(destination, token)`` makes the partials that share a shipped row
    adjacent, which is what turns ``topk`` copies of a token into one.

    Every step is a whole-tensor op: the host learns nothing here, so the caller
    can fetch ``counts`` and ``uniq_per_rank`` in a single readback.
    """
    num_tokens, num_slots = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=ffn_size * expert_per_rank)
    offsets = torch.cumsum(counts, dim=0) - counts

    token_of_partial = order // num_slots
    weights = topk_weights.reshape(-1)[order].to(torch.float32)
    dest_rank = flat[order] // expert_per_rank

    # Partials sharing a (destination, token) share a shipped row. Sorting by
    # that key makes them adjacent, so "is this row new" is one neighbour
    # comparison and the row numbering is its running sum.
    key = dest_rank * num_tokens + token_of_partial
    key_order = torch.argsort(key, stable=True)
    sorted_key = key[key_order]
    is_new = torch.ones_like(sorted_key, dtype=torch.int32)
    is_new[1:] = (sorted_key[1:] != sorted_key[:-1]).to(torch.int32)
    row_of_partial = is_new.cumsum(0) - 1

    uniq_per_rank = torch.zeros(
        ffn_size,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    uniq_per_rank.index_add_(0, dest_rank[key_order], is_new)

    # Duplicates write the same token id to the same row, so the scatter is
    # well defined despite the repeated indices.
    uniq_token_ids = torch.zeros_like(token_of_partial)
    uniq_token_ids.scatter_(0, row_of_partial, token_of_partial[key_order])

    # Back to expert-sorted order, rebased so each destination's slice indexes
    # its own payload from zero.
    row_base = (torch.cumsum(uniq_per_rank, 0) - uniq_per_rank).to(torch.int32)
    expand_idx = torch.empty_like(row_of_partial, dtype=torch.int32)
    expand_idx.scatter_(0, key_order, row_of_partial.to(torch.int32))
    expand_idx -= row_base[dest_rank]

    return DispatchPlan(
        counts=counts,
        offsets=offsets,
        uniq_per_rank=uniq_per_rank,
        expand_idx=expand_idx,
        uniq_token_ids=uniq_token_ids,
        weights=weights,
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
        self.num_stages = (
            max(1, self.extra_info.async_moe_num_ubatches)
            if self.extra_info.async_moe_ubatching
            else 1
        )
        if self.ring_depth < self.num_stages:
            raise ValueError(
                f"ring_depth {self.ring_depth} cannot serve "
                f"{self.num_stages} async MoE stages; each stage needs a slot "
                "of its own",
            )
        # Every Attention rank routes to every FFN rank, so a window carries one
        # region per opposite-role peer. Both roles allocate the larger of the
        # two so the symmetric allocation matches.
        self.num_regions = max(self.attn_size, self.ffn_size)
        # Payload rows are distinct tokens, so a batch is their bound no matter
        # how skewed the gate is. Only the 4-byte-per-partial index arrays need
        # the every-partial-to-one-rank worst case.
        self.token_cap = max(1, self.max_seq_len)
        self.partial_cap = max(1, self.max_seq_len * self.topk)
        self.layout = SlotLayout.build(
            expert_per_rank=self.expert_per_rank,
            partial_cap=self.partial_cap,
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

    def _rings_for_stage(self, stage_idx: int) -> list[int]:
        """Ring slots this stage owns.

        A ring names a window slot, so stages must not share one: two stages of
        the same layer are in flight at the same time, and handing both the same
        slot means the second dispatch overwrites the first's payload and flag,
        after which the reply the first is waiting for never arrives.
        """
        first = stage_idx % self.num_stages
        return list(range(first, self.ring_depth, self.num_stages))

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
        for stage in range(self.num_stages):
            self._free_rings[stage] = self._rings_for_stage(stage)
        logger.info(
            "AFD async GPU window ready: role=%s role_rank=%d world_rank=%d/%d "
            "regions=%d rings=%d partial_cap=%d slot=%.1fMiB total=%.1fMiB",
            self.afd_config.role,
            self.role_rank,
            self.world_rank,
            self.topology.world_size,
            self.num_regions,
            self.ring_depth,
            self.partial_cap,
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
        expected_shape = (num_tokens, self.topk)
        if tuple(topk_ids.shape) != expected_shape:
            raise ValueError(
                f"topk_ids shape must be {expected_shape}, got {tuple(topk_ids.shape)}",
            )
        # The weights are flattened alongside the ids to give each partial its
        # own weight, so a mismatched shape would silently misalign them.
        if tuple(topk_weights.shape) != expected_shape:
            raise ValueError(
                f"topk_weights shape must be {expected_shape}, "
                f"got {tuple(topk_weights.shape)}",
            )

        stage_idx = metadata.stage_idx
        rings = self._free_rings.setdefault(
            stage_idx,
            self._rings_for_stage(stage_idx),
        )
        if not rings:
            raise RuntimeError(
                f"AFD async GPU ring exhausted on stage {stage_idx}; the "
                "send-then-recv invariant was violated or the topology config "
                "does not match the actual peer count",
            )
        ring = rings.pop(0)
        self._seq += 1

        plan = plan_dispatch(
            topk_ids,
            topk_weights,
            ffn_size=self.ffn_size,
            expert_per_rank=self.expert_per_rank,
        )
        # One D2H per send, carrying everything the host needs to slice the
        # plan: offsets are a prefix sum, cheaper to redo here than to fetch.
        # This is the last synchronize left on the layer path, and it is
        # intrinsic to slicing the segments on the host -- removing it means
        # computing the destination offsets on the device and writing
        # full-capacity segments instead.
        num_experts = self.ffn_size * self.expert_per_rank
        plan_host = (
            torch.cat((plan.counts.to(torch.int32), plan.uniq_per_rank)).cpu().tolist()
        )
        counts_host = plan_host[:num_experts]
        uniq_host = plan_host[num_experts:]
        offsets_host = [0] * num_experts
        for i in range(1, num_experts):
            offsets_host[i] = offsets_host[i - 1] + counts_host[i - 1]

        # Every FFN rank gets a slot even when routing sends it nothing, and it
        # replies to every slot, so a reply is expected from all of them.
        # Expecting only the ranks that received data leaves the empty rank's
        # reply unmatched and its ring slot never released -- which is what a
        # single-token decode hits, since both the routed segment and the
        # round-robin shared slice can come out empty for one rank.
        expected_ffn = list(range(self.ffn_size))
        uniq_ids: list[Tensor] = []
        uniq_start = 0
        for ffn_rank in range(self.ffn_size):
            base = ffn_rank * self.expert_per_rank
            expert_counts = counts_host[base : base + self.expert_per_rank]
            start = offsets_host[base]
            routed_tokens = sum(expert_counts)
            segment = slice(start, start + routed_tokens)
            # The shipped rows this rank owns, and the tokens they carry. The
            # ids stay here rather than going on the wire: combine scatters the
            # reply back to exactly these rows.
            uniq_tokens = uniq_host[ffn_rank]
            rows = plan.uniq_token_ids[uniq_start : uniq_start + uniq_tokens]
            uniq_start += uniq_tokens
            uniq_ids.append(rows)

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
                uniq_tokens=uniq_tokens,
            )
            # Rows are gathered straight into the FFN rank's window; handing
            # ``write_slot`` the index instead of a gathered tensor saves a
            # local write and re-read of every payload byte.
            window.write_slot(
                peer=self.attn_size + ffn_rank,
                region=self.role_rank,
                ring=ring,
                header=header,
                expand_idx=plan.expand_idx[segment],
                weights=plan.weights[segment],
                routed_x=hidden_states,
                routed_rows=rows,
                shared_idx=shared_idx,
                shared_x=hidden_states,
                shared_rows=shared_idx.to(torch.int64),
            )

        logger.debug(
            "AFD dispatch sent: A%d layer=%d stage=%d tokens=%d ring=%d "
            "rows_per_ffn=%s partials=%d awaiting_ffn=%s",
            self.role_rank,
            metadata.layer_idx,
            stage_idx,
            num_tokens,
            ring,
            uniq_host,
            num_tokens * self.topk,
            expected_ffn,
        )
        self._pending.setdefault(stage_idx, []).append(
            _PendingDispatch(
                context=context,
                uniq_ids=uniq_ids,
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

        # Accumulate in the payload dtype. Each row takes at most one
        # contribution per FFN rank plus its shared one -- the topk sum already
        # happened on the FFN side -- so there is little left for a wider
        # accumulator to protect, and float32 cost a widening pass over every
        # arriving block plus a narrowing one on the way out.
        accumulator = torch.zeros(
            (pending.num_tokens, self.hidden_size),
            dtype=self.payload_dtype,
            device=ref_tensor.device,
        )
        outstanding = set(pending.expected_ffn)
        while outstanding:
            arrived = window.wait()
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

            if header.uniq_tokens:
                # One row per token this rank was sent, already weighted and
                # summed over that token's partials on the FFN side.
                accumulator.index_add_(
                    0,
                    pending.uniq_ids[header.src_role_rank],
                    window.local_routed(
                        arrived.region,
                        arrived.ring,
                        header.uniq_tokens,
                    ),
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
                    ),
                )

        self._free_rings.setdefault(ubatch_idx, []).append(pending.ring)
        return accumulator

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

        The payload carries each token once, so the rows are expanded back to
        one per partial here -- a local gather that replaces the duplicate rows
        the sender used to put on the wire.
        """
        window = self._require_initialized()
        timeout_ms = int(kwargs.get("timeout_ms", 0))
        arrived = window.wait(timeout_s=timeout_ms / 1000.0 if timeout_ms else None)
        if arrived is None:
            raise TimeoutError("AFD async GPU dispatch recv timed out")

        header = arrived.header
        if header.is_shutdown:
            raise ConnectorShutdown(
                f"Attention rank {header.src_role_rank} announced shutdown",
            )
        # The group list is echoed back on the combine header, so the sender's
        # own accounting has to agree with it. Checking the decoded host values
        # here is free; the equivalent check on the device tensor cost a
        # synchronize per work item.
        if sum(header.expert_counts) != header.routed_tokens:
            raise RuntimeError(
                f"AFD async GPU dispatch header from A{header.src_role_rank} "
                f"has expert counts summing to {sum(header.expert_counts)} but "
                f"declares {header.routed_tokens} routed tokens",
            )

        expand_idx = window.local_expand_idx(
            arrived.region,
            arrived.ring,
            header.routed_tokens,
        ).to(torch.int64)
        states = GpuAsyncTransferState(
            region=arrived.region,
            ring=arrived.ring,
            seq=header.seq,
            src_role_rank=header.src_role_rank,
            layer_idx=header.layer_idx,
            stage_idx=header.stage_idx,
            num_tokens=header.num_tokens,
            routed_tokens=header.routed_tokens,
            uniq_tokens=header.uniq_tokens,
            shared_tokens=header.shared_tokens,
            group_list=window.local_expert_counts(arrived.region, arrived.ring),
            expert_counts_host=header.expert_counts,
            expand_idx=expand_idx,
            weights=window.local_weights(
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
                header.uniq_tokens,
            ).index_select(0, expand_idx),
            context=AFDTransferContext(metadata=metadata, states=states),
        )

    def send_ffn_output(
        self,
        ffn_output: Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Reduce expert output to one row per shipped token and write it back.

        Every partial of a token that landed on this rank is weighted and summed
        here, so the reply carries the same rows the dispatch did. Doing it on
        this side keeps the duplicates off the wire and leaves the Attention
        side a plain scatter-add.

        Both the weighting and the sum stay in the payload dtype. Widening to
        float32 first cost two extra passes over ``[partials, hidden]`` and made
        the scatter move twice the bytes, which a profile showed costing almost
        as much GPU time as the expert GEMM itself -- to protect a sum of at
        most ``topk`` terms whose result goes on the wire narrowed anyway.
        """
        window = self._require_initialized()
        states = context.states
        if not isinstance(states, GpuAsyncTransferState):
            raise RuntimeError(
                "AFD async GPU send_ffn_output requires GpuAsyncTransferState",
            )
        if states.expand_idx is None or states.weights is None:
            raise RuntimeError(
                "AFD async GPU send_ffn_output requires the dispatch expansion",
            )
        reduced = torch.zeros(
            (states.uniq_tokens, self.hidden_size),
            dtype=self.payload_dtype,
            device=ffn_output.device,
        )
        if states.routed_tokens:
            weighted = ffn_output * states.weights.unsqueeze(1).to(ffn_output.dtype)
            reduced.index_add_(0, states.expand_idx, weighted)

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
            expert_counts=states.expert_counts_host,
            uniq_tokens=states.uniq_tokens,
            echo_seq=states.seq,
        )
        # ``reduced`` is float32 and the slot is the payload dtype; the copy
        # inside write_slot casts on its way into the peer window, so the
        # narrowing costs no extra pass over the rows.
        window.write_slot(
            peer=states.src_role_rank,
            region=self.role_rank,
            ring=states.ring,
            header=header,
            expand_idx=None,
            weights=None,
            routed_x=reduced,
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
                routed_x=None,
                shared_idx=None,
                shared_x=None,
            )


__all__ = [
    "AFD_ASYNC_GPU_GROUP_NAME",
    "ConnectorShutdown",
    "DispatchPlan",
    "GpuAsyncAFDConnector",
    "GpuAsyncExtraInfo",
    "GpuAsyncFFNWorkItem",
    "GpuAsyncTransferState",
    "plan_dispatch",
]
