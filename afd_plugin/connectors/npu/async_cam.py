# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend CAM asynchronous connector for Attention/FFN disaggregation.

``CAMAsyncAFDConnector`` is the eager-only Ascend prefill data path. Attention
ranks run MoE routing, submit activations with CAM async dispatch-send, and
receive combined expert output with combine-recv. FFN ranks receive routed and
shared-expert activations with dispatch-recv, execute their local experts, and
return the results with combine-send.

The connector creates one HCCL world ordered as
``[A0, A1, ..., F0, F1, ...]``. Attention world ranks therefore equal their
role ranks, while FFN world ranks start at ``num_attention_ranks``. Unlike the
synchronous connectors, CAM async carries routing and token-count metadata in
the CAM operator payload and does not create a separate Gloo DP-metadata
control plane.

The supported deployment requires ``async=true``, eager execution, Ascend CAM
operator packages, and matching topology/configuration on every rank. vLLM
native DBO, ACL graph execution, and decode are not supported.
Optional AFD-managed MoE ubatching is a separate two-stage request-boundary
pipeline. See ``docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md`` for configuration,
rank derivation, launch guidance, and the full limitations.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, cast

import torch
from torch import Tensor
from vllm.logger import init_logger

from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available
from afd_plugin.config import AFDConfig
from afd_plugin.config_utils import (
    coerce_extra_bool,
    coerce_extra_int,
    coerce_extra_positive_int,
    coerce_extra_str,
)
from afd_plugin.connectors.base import (
    AFDConnectorBase,
    ConnectorExtraInfo,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDF2ATransferPayload,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.distributed import init_afd_process_group

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup
    from vllm.config import VllmConfig

AFD_ASYNC_CAM_GROUP_NAME = "afd_async_cam"
CAM_COMM_ID = 0
ATTN_RANKS_PER_DP_CONFIG_KEY = "attn_ranks_per_dp"
ASYNC_MOE_REQUEST_SPLIT = "request"

_AFD_ASYNC_EXTRA_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "dynamicQuant",
        "attn_ranks_per_dp",
        "async_moe_ubatching",
        "async_moe_num_ubatches",
        "async_moe_split",
    },
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class AFDAsyncExtraInfo(ConnectorExtraInfo):
    """Typed async CAM connector configuration.

    Attributes:
        dynamic_quant: Dynamic quantization mode accepted by CAM operators.
        attn_ranks_per_dp: Number of Attention ranks in each data-parallel group.
        async_moe_ubatching: Whether request-boundary async MoE ubatching is used.
        async_moe_num_ubatches: Number of stages used by async MoE ubatching.
        async_moe_split: Boundary at which async MoE work is split.
    """

    dynamic_quant: int = 0
    attn_ranks_per_dp: int = 1
    async_moe_ubatching: bool = False
    async_moe_num_ubatches: int = 2
    async_moe_split: str = ASYNC_MOE_REQUEST_SPLIT

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> AFDAsyncExtraInfo:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__} connector_extra_config must be a mapping, "
                f"got {type(raw).__name__}",
            )
        unknown = sorted(
            str(key) for key in raw if key not in _AFD_ASYNC_EXTRA_CONFIG_FIELDS
        )
        if unknown:
            raise ValueError(
                "unknown AFD async connector_extra_config field(s): "
                + ", ".join(unknown),
            )

        return cls(
            dynamic_quant=coerce_extra_int(
                raw.get("dynamicQuant", 0),
                field_name="dynamicQuant",
            ),
            attn_ranks_per_dp=coerce_extra_positive_int(
                raw.get("attn_ranks_per_dp", 1),
                field_name="attn_ranks_per_dp",
            ),
            async_moe_ubatching=coerce_extra_bool(
                raw.get("async_moe_ubatching", False),
                field_name="async_moe_ubatching",
            ),
            async_moe_num_ubatches=coerce_extra_positive_int(
                raw.get("async_moe_num_ubatches", 2),
                field_name="async_moe_num_ubatches",
            ),
            async_moe_split=coerce_extra_str(
                raw.get("async_moe_split", ASYNC_MOE_REQUEST_SPLIT),
                field_name="async_moe_split",
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "dynamicQuant": self.dynamic_quant,
            "attn_ranks_per_dp": self.attn_ranks_per_dp,
            "async_moe_ubatching": self.async_moe_ubatching,
            "async_moe_num_ubatches": self.async_moe_num_ubatches,
            "async_moe_split": self.async_moe_split,
        }


@dataclass(slots=True)
class AFDAsyncTransferState(AFDTransferState):
    """CAM-side metadata carried from dispatch recv to combine send."""

    batch_size: int = 1
    hidden_size: int = 1
    topk: int = 1
    layer_idx: int = 0
    token_nums_rankid_layeridx: Tensor | None = None
    topk_ids: Tensor | None = None
    topk_weights: Tensor | None = None
    expand_idx: Tensor | None = None
    expert_token_nums: Tensor | None = None
    expert_token_nums_shared: Tensor | None = None
    dynamic_scales: Tensor | None = None
    dynamic_scales_shared: Tensor | None = None
    atten_batch_size: Tensor | None = None
    x_active_mask: Tensor | None = None
    expand_x_shared: Tensor | None = None


@dataclass(slots=True)
class AFDAsyncFFNWorkItem:
    """Normalized FFN-side work item produced by async CAM dispatch recv."""

    hidden_states: Tensor
    metadata: AFDTransferMetadata
    recv_output: AFDA2FTransferPayload
    layer_idx: int
    stage_idx: int
    num_tokens: int
    total_num_tokens: int
    shared_num_tokens: int


@dataclass(frozen=True, slots=True)
class AFDAsyncTopology:
    """Role-local and HCCL-world rank information for one CAM participant."""

    role: str
    role_rank: int
    world_rank: int
    attention_rank_size: int
    expert_rank_size: int
    expert_per_rank: int

    @property
    def world_size(self) -> int:
        """Return the total number of Attention and FFN ranks."""
        return self.attention_rank_size + self.expert_rank_size


class CAMAsyncAFDConnector(AFDConnectorBase):
    """CAM-backed asynchronous connector for Ascend NPU AFD.

    Attention ranks occupy the first part of the HCCL world and FFN ranks the
    second. CAM dispatch/combine operators own both the collective data motion
    and its routing metadata, so this connector has no DP metadata control
    plane (``control_plane`` stays ``None``) and FFN work is triggered directly
    by the connector receive loop.
    """

    control_plane = None

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> AFDAsyncExtraInfo:
        return AFDAsyncExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
    ) -> None:
        """Derive CAM topology, tensor dimensions, and connector state.

        Communication resources are created collectively by
        ``init_afd_connector``. ``afd_role_rank`` must already contain the
        process's role-local DP/PCP-derived offset; ``attn_ranks_per_dp`` is
        used as the CAM Attention TP width.
        """
        super().__init__(rank, local_rank, vllm_config, afd_config)
        async_extra_info = cast(AFDAsyncExtraInfo, self.extra_info)
        self.async_extra_info = async_extra_info
        self._initialized = False
        hf_config = vllm_config.model_config.hf_config
        self._role_rank = int(afd_config.afd_role_rank)
        self.hidden_size = int(hf_config.hidden_size)
        self.topk = max(1, int(hf_config.num_experts_per_tok))
        self.num_routed_experts = max(1, int(hf_config.n_routed_experts))
        self.dynamic_quant = async_extra_info.dynamic_quant
        self.group_name = ""
        self.max_seq_len = max(
            1,
            int(vllm_config.scheduler_config.max_num_batched_tokens),
        )
        self.comm_id = CAM_COMM_ID
        self.tp_size = async_extra_info.attn_ranks_per_dp
        self.cam_pg: ProcessGroup | None = None
        self.topology = build_async_topology(
            afd_config,
            self._role_rank,
            num_routed_experts=self.num_routed_experts,
        )
        self.world_rank = self.topology.world_rank
        self.attention_rank_size = self.topology.attention_rank_size
        self.expert_rank_size = self.topology.expert_rank_size
        self.expert_per_rank = self.topology.expert_per_rank
        self.comm_args: Tensor | None = None
        self._placeholder: Tensor | None = None
        self._pending_attention_payloads: dict[
            int,
            list[tuple[AFDTransferMetadata, Tensor, Tensor]],
        ] = {}

    @property
    def is_initialized(self) -> bool:
        """Return whether the CAM HCCL group and operator buffers are ready."""
        return self._initialized

    def init_afd_connector(self) -> None:
        """Collectively initialize the CAM HCCL world and operator buffers.

        All Attention and FFN ranks must call this method with identical
        rendezvous and topology settings. Missing/duplicate ranks or mismatched
        world sizes fail or time out during the 30-minute rendezvous.
        """
        if self._initialized:
            return

        ensure_cam_async_ops_available()
        self.cam_pg = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.topology.world_size,
            rank=self.world_rank,
            group_name=AFD_ASYNC_CAM_GROUP_NAME,
            timeout=timedelta(minutes=30),
        )
        self.group_name = _hccl_comm_name(self.cam_pg, self.world_rank)
        device = f"npu:{self.local_rank}"
        self.comm_args = torch.empty((1,), dtype=torch.float16, device=device)
        self._placeholder = torch.empty(
            (1,),
            dtype=torch.bfloat16,
            device=device,
        )
        self._initialized = True

    def close(self) -> None:
        """Destroy the HCCL process group and clear pending transfer state."""
        if self.cam_pg is not None:
            import torch.distributed as dist

            dist.destroy_process_group(self.cam_pg)
        self.cam_pg = None
        self.comm_args = None
        self._placeholder = None
        self._pending_attention_payloads.clear()
        self._initialized = False

    def select_experts(self, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Run the vLLM Ascend expert selector on the Attention side."""
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        if "global_num_experts" in kwargs:
            signature = inspect.signature(select_experts)
            if (
                "global_num_experts" not in signature.parameters
                and "num_experts" in signature.parameters
            ):
                kwargs["num_experts"] = kwargs.pop("global_num_experts")
        return select_experts(**kwargs)

    def _create_transfer_metadata(
        self, batch_size: int = 1, layer_idx: int = 0, stage_idx: int = 0
    ) -> AFDTransferMetadata:
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[max(1, batch_size)],
        )
        metadata.transfer_state = self._make_transfer_state(
            batch_size=max(1, batch_size),
            layer_idx=layer_idx,
        )
        return metadata

    def recv_ffn_work_item(
        self,
        *,
        stage_idx: int,
        max_num_tokens: int,
    ) -> AFDAsyncFFNWorkItem:
        """Receive and normalize one connector-driven FFN dispatch item.

        CAM metadata supplies the actual layer and routed/shared token counts;
        returned tensors are sliced from operator capacity to those counts.
        """
        recv_output = self.recv_attn_output(
            stage_idx=stage_idx,
            layer_idx=0,
            batch_size=_connector_driven_batch_size(self, max_num_tokens),
            ubatch_idx=stage_idx,
        )
        metadata = recv_output.metadata

        token_nums_rankid_layeridx = _cam_token_nums_rankid_layeridx(
            recv_output,
            metadata,
        )
        total_num_tokens = max(1, _cam_metadata_int(token_nums_rankid_layeridx, 0))
        shared_num_tokens = _cam_shared_token_count(recv_output, total_num_tokens)
        layer_idx = _cam_metadata_int(token_nums_rankid_layeridx, 2)
        num_tokens = _cam_routed_token_count(
            recv_output,
            fallback=max(0, total_num_tokens - shared_num_tokens),
        )

        metadata.layer_idx = layer_idx
        metadata.stage_idx = stage_idx
        metadata.seq_lens = [num_tokens]

        hidden_states = _slice_cam_payload_to_actual_tokens(
            recv_output.hidden_states,
            recv_output,
            num_tokens,
            shared_num_tokens=shared_num_tokens,
        )
        _sync_transfer_state_with_cam_metadata(metadata, layer_idx=layer_idx)

        return AFDAsyncFFNWorkItem(
            hidden_states=hidden_states,
            metadata=metadata,
            recv_output=recv_output,
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            num_tokens=num_tokens,
            total_num_tokens=total_num_tokens,
            shared_num_tokens=shared_num_tokens,
        )

    def send_ffn_work_item_output(
        self,
        work_item: AFDAsyncFFNWorkItem,
        ffn_output: Tensor | AFDF2ATransferPayload,
    ) -> Tensor | AFDF2ATransferPayload:
        """Return one FFN work item's routed/shared outputs through CAM.

        A one-token BF16 routed placeholder is used when a rank receives no
        routed tokens because CAM combine-send cannot consume the dynamic
        quantized dispatch buffer as an empty routed result.
        """
        if int(work_item.num_tokens) > 0:
            _send_ffn_output_payload(
                self,
                ffn_output,
                work_item.metadata,
                stage_idx=work_item.stage_idx,
            )
            return ffn_output

        # Temporary NPU workaround: CAM combine-send does not accept the int8
        # dispatch-recv buffer as routed output. When this FFN rank has no
        # routed tokens, send one fake bf16 routed token instead of reusing the
        # dynamicQuant int8 input buffer.
        fake_routed_output = torch.zeros(
            (1, int(work_item.recv_output.hidden_states.shape[-1])),
            dtype=torch.bfloat16,
            device=work_item.recv_output.hidden_states.device,
        )
        if isinstance(ffn_output, AFDF2ATransferPayload):
            ffn_output = AFDF2ATransferPayload(
                routed_output=fake_routed_output,
                shared_output=ffn_output.shared_output,
            )
        else:
            ffn_output = fake_routed_output
        work_item.metadata.seq_lens = [1]
        _send_ffn_output_payload(
            self,
            ffn_output,
            work_item.metadata,
            stage_idx=work_item.stage_idx,
        )
        return ffn_output

    def send_attn_output(
        self,
        hidden_states: Tensor,
        metadata: AFDTransferMetadata,
        **kwargs: Any,
    ) -> None:
        """Dispatch Attention activations and top-k routing to FFN ranks.

        The matching metadata and routing tensors are queued per async stage
        for ``recv_ffn_output``. The input token count and top-k tensor shapes
        must match the transfer metadata and model configuration.
        """
        self._require_initialized()
        if not metadata.validate_tensor_shape(tuple(hidden_states.shape)):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"AFD async metadata token count {metadata.total_tokens}",
            )
        data = self._metadata_data_or_default(metadata)
        topk_ids = kwargs.get("topk_ids")
        topk_weights = kwargs.get("topk_weights")
        if topk_ids is None or topk_weights is None:
            raise RuntimeError(
                "CAMAsyncAFDConnector send_attn_output requires topk_ids/topk_weights",
            )
        topk_ids = cast(Tensor, topk_ids)
        topk_weights = cast(Tensor, topk_weights)
        _validate_topk_payload(
            topk_ids,
            topk_weights,
            batch_size=data.batch_size,
            topk=data.topk,
        )
        data.topk_ids = topk_ids
        data.topk_weights = topk_weights
        self._queue_attention_payload(metadata, topk_ids, topk_weights)
        _log_cam_op_values(
            "async_dispatch_send",
            "inputs",
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            comm_args=self.comm_args,
            comm_id=self.comm_id,
            max_seq_len=self.max_seq_len,
            batch_size=data.batch_size,
            hidden_size=data.hidden_size,
            topk=data.topk,
            expert_rank_size=self.expert_rank_size,
            attention_rank_size=self.attention_rank_size,
            expert_per_rank=self.expert_per_rank,
            rank=self.world_rank,
            world_size=self.topology.world_size,
            layer_idx=data.layer_idx,
            tp_size=self.tp_size,
            dynamic_quant=self.dynamic_quant,
            group_name=self.group_name,
        )
        torch.ops.umdk_cam_op_lib.async_dispatch_send(
            hidden_states,
            topk_ids,
            self.comm_args,
            self.comm_id,
            self.max_seq_len,
            data.batch_size,
            data.hidden_size,
            data.topk,
            self.expert_rank_size,
            self.attention_rank_size,
            self.expert_per_rank,
            self.world_rank,
            self.topology.world_size,
            data.layer_idx,
            self.tp_size,
            self.dynamic_quant,
            self.group_name,
        )
        return None

    def recv_ffn_output(self, **kwargs: Any) -> Tensor:
        """Receive and combine expert output on an Attention rank.

        Routing tensors may be supplied explicitly or recovered FIFO from the
        matching stage's pending dispatch. ``ref_tensor`` selects the output
        device and dtype used to construct the CAM receive placeholder.
        """
        self._require_initialized()
        ref_tensor = cast(Tensor, kwargs["ref_tensor"])
        metadata = kwargs.get("metadata")
        topk_ids = kwargs.get("topk_ids")
        topk_weights = kwargs.get("topk_weights")
        if metadata is None or topk_ids is None or topk_weights is None:
            (
                pending_metadata,
                pending_topk_ids,
                pending_topk_weights,
            ) = self._pop_attention_payload(int(kwargs.get("ubatch_idx", 0) or 0))
            if metadata is None:
                metadata = pending_metadata
            if topk_ids is None:
                topk_ids = pending_topk_ids
            if topk_weights is None:
                topk_weights = pending_topk_weights
        metadata = cast(AFDTransferMetadata, metadata)
        topk_ids = cast(Tensor, topk_ids)
        topk_weights = cast(Tensor, topk_weights)
        data = self._metadata_data_or_default(metadata)
        _validate_topk_payload(
            topk_ids,
            topk_weights,
            batch_size=data.batch_size,
            topk=data.topk,
        )
        placeholder = ref_tensor.new_empty((1,))
        _log_cam_op_values(
            "async_combine_recv",
            "inputs",
            placeholder=placeholder,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            comm_args=self.comm_args,
            comm_id=self.comm_id,
            batch_size=data.batch_size,
            hidden_size=data.hidden_size,
            topk=data.topk,
            expert_rank_size=self.expert_rank_size,
            attention_rank_size=self.attention_rank_size,
            expert_per_rank=self.expert_per_rank,
            rank=self.world_rank,
            world_size=self.topology.world_size,
            group_name=self.group_name,
        )
        output = torch.ops.umdk_cam_op_lib.async_combine_recv(
            placeholder,
            topk_ids,
            topk_weights,
            self.comm_args,
            self.comm_id,
            data.batch_size,
            data.hidden_size,
            data.topk,
            self.expert_rank_size,
            self.attention_rank_size,
            self.expert_per_rank,
            self.world_rank,
            self.topology.world_size,
            self.group_name,
        )
        _log_cam_op_values("async_combine_recv", "outputs", output=output)
        return output

    def recv_attn_output(
        self,
        ubatch_idx: int | None = None,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Receive CAM-dispatched routed/shared activations on an FFN rank.

        The returned payload preserves dynamic-quant scales, per-expert token
        counts, shared-expert activations, and CAM token/rank/layer metadata so
        local expert execution and the subsequent combine-send use the same
        routing contract.
        """
        self._require_initialized()
        ubatch_idx = 0 if ubatch_idx is None else int(ubatch_idx)
        metadata = self._create_transfer_metadata(
            stage_idx=ubatch_idx,
            batch_size=int(kwargs.get("batch_size", self.max_seq_len) or 1),
            layer_idx=int(kwargs.get("layer_idx", 0) or 0),
        )
        data = self._metadata_data_or_default(metadata)
        placeholder = kwargs.get("placeholder", self._placeholder)
        _log_cam_op_values(
            "async_dispatch_recv",
            "inputs",
            placeholder=placeholder,
            comm_args=self.comm_args,
            comm_id=self.comm_id,
            batch_size=data.batch_size,
            hidden_size=data.hidden_size,
            topk=data.topk,
            expert_rank_size=self.expert_rank_size,
            attention_rank_size=self.attention_rank_size,
            expert_per_rank=self.expert_per_rank,
            rank=self.world_rank,
            world_size=self.topology.world_size,
            tp_size=self.tp_size,
            dynamic_quant=self.dynamic_quant,
            group_name=self.group_name,
        )
        outputs = torch.ops.umdk_cam_op_lib.async_dispatch_recv(
            placeholder,
            self.comm_args,
            self.comm_id,
            data.batch_size,
            data.hidden_size,
            data.topk,
            self.expert_rank_size,
            self.attention_rank_size,
            self.expert_per_rank,
            self.world_rank,
            self.topology.world_size,
            self.tp_size,
            self.dynamic_quant,
            self.group_name,
        )
        (
            hidden_states,
            expand_x_shared,
            dynamic_scales,
            dynamic_scales_shared,
            token_nums_rankid_layeridx,
            expert_token_nums,
            expert_token_nums_shared,
        ) = outputs
        _log_cam_op_values(
            "async_dispatch_recv",
            "outputs",
            hidden_states=hidden_states,
            expand_x_shared=expand_x_shared,
            dynamic_scales=dynamic_scales,
            dynamic_scales_shared=dynamic_scales_shared,
            token_nums_rankid_layeridx=token_nums_rankid_layeridx,
            expert_token_nums=expert_token_nums,
            expert_token_nums_shared=expert_token_nums_shared,
        )
        data.expand_x_shared = expand_x_shared
        data.dynamic_scales = dynamic_scales
        data.dynamic_scales_shared = dynamic_scales_shared
        data.token_nums_rankid_layeridx = token_nums_rankid_layeridx
        data.expert_token_nums = expert_token_nums
        data.expert_token_nums_shared = expert_token_nums_shared
        data.atten_batch_size = token_nums_rankid_layeridx
        payload = AFDA2FTransferPayload(
            hidden_states=hidden_states,
            metadata=metadata,
            group_list=expert_token_nums,
            dynamic_scales=dynamic_scales,
            expand_x_shared=expand_x_shared,
            dynamic_scales_shared=dynamic_scales_shared,
            atten_batch_size=token_nums_rankid_layeridx,
            ep_recv_counts=expert_token_nums,
            ep_recv_counts_shared=expert_token_nums_shared,
        )
        data = self._metadata_data_or_default(metadata)
        data.topk_ids = payload.topk_ids
        data.topk_weights = payload.topk_weights
        data.expand_idx = payload.expand_idx
        data.expert_token_nums = payload.ep_recv_counts
        if data.expert_token_nums is None:
            data.expert_token_nums = payload.group_list
        data.dynamic_scales = payload.dynamic_scales
        data.dynamic_scales_shared = payload.dynamic_scales_shared
        data.expand_x_shared = payload.expand_x_shared
        data.expert_token_nums_shared = payload.ep_recv_counts_shared
        data.atten_batch_size = payload.atten_batch_size
        data.token_nums_rankid_layeridx = payload.atten_batch_size
        data.x_active_mask = payload.x_active_mask
        return payload

    def send_ffn_output(
        self,
        ffn_output: Tensor,
        metadata: AFDTransferMetadata,
        **kwargs: Any,
    ) -> None:
        """Send routed and optional shared-expert outputs for CAM combination.

        ``TokenNums_Rankid_Layeridx`` from the matching dispatch-recv is
        mandatory because CAM uses it to return results to Attention ranks.
        """
        self._require_initialized()
        data = self._metadata_data_or_default(metadata)
        expand_x_shared = kwargs.get("expand_x_shared")
        if expand_x_shared is None:
            expand_x_shared = ffn_output
        token_nums_rankid_layeridx = data.token_nums_rankid_layeridx
        if token_nums_rankid_layeridx is None:
            token_nums_rankid_layeridx = kwargs.get("token_nums_rankid_layeridx")
        if token_nums_rankid_layeridx is None:
            raise RuntimeError(
                "AFD async CAM combine send requires "
                "TokenNums_Rankid_Layeridx from async_dispatch_recv",
            )
        _log_cam_op_values(
            "async_combine_send",
            "inputs",
            ffn_output=ffn_output,
            expand_x_shared=expand_x_shared,
            comm_args=self.comm_args,
            token_nums_rankid_layeridx=token_nums_rankid_layeridx,
            comm_id=self.comm_id,
            batch_size=data.batch_size,
            hidden_size=data.hidden_size,
            topk=data.topk,
            expert_rank_size=self.expert_rank_size,
            attention_rank_size=self.attention_rank_size,
            expert_per_rank=self.expert_per_rank,
            rank=self.world_rank,
            world_size=self.topology.world_size,
            tp_size=self.tp_size,
            group_name=self.group_name,
        )
        torch.ops.umdk_cam_op_lib.async_combine_send(
            ffn_output,
            expand_x_shared,
            self.comm_args,
            token_nums_rankid_layeridx,
            self.comm_id,
            data.batch_size,
            data.hidden_size,
            data.topk,
            self.expert_rank_size,
            self.attention_rank_size,
            self.expert_per_rank,
            self.world_rank,
            self.topology.world_size,
            self.tp_size,
            self.group_name,
        )

    def _metadata_data_or_default(
        self,
        metadata: AFDTransferMetadata,
    ) -> AFDAsyncTransferState:
        data = metadata.transfer_state
        if data is None:
            data = self._make_transfer_state(
                batch_size=metadata.total_tokens,
                layer_idx=int(metadata.layer_idx),
            )
            metadata.transfer_state = data
        if not isinstance(data, AFDAsyncTransferState):
            raise RuntimeError("AFD async metadata is missing transfer_state")
        return data

    def _make_transfer_state(
        self,
        *,
        batch_size: int,
        layer_idx: int,
    ) -> AFDAsyncTransferState:
        return AFDAsyncTransferState(
            batch_size=max(1, int(batch_size)),
            hidden_size=self.hidden_size,
            topk=self.topk,
            layer_idx=max(0, int(layer_idx)),
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("CAMAsyncAFDConnector is not initialized")

    def _queue_attention_payload(
        self,
        metadata: AFDTransferMetadata,
        topk_ids: Tensor,
        topk_weights: Tensor,
    ) -> None:
        self._pending_attention_payloads.setdefault(int(metadata.stage_idx), []).append(
            (metadata, topk_ids, topk_weights),
        )

    def _pop_attention_payload(
        self,
        stage_idx: int,
    ) -> tuple[AFDTransferMetadata, Tensor, Tensor]:
        payloads = self._pending_attention_payloads.get(int(stage_idx))
        if not payloads:
            raise RuntimeError(
                "CAMAsyncAFDConnector recv_ffn_output is missing pending "
                "Attention metadata",
            )
        payload = payloads.pop(0)
        if not payloads:
            self._pending_attention_payloads.pop(int(stage_idx), None)
        return payload


_CAM_LOG_SKIPPED_ARGS = frozenset({"comm_args", "comm_id", "group_name"})
_CAM_OP_IO_LOG_ENV = "AFD_CAM_OP_IO_LOG"


def _cam_op_io_logging_enabled() -> bool:
    return os.environ.get(_CAM_OP_IO_LOG_ENV, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tensor_first_values(value: Tensor, count: int = 5) -> object:
    try:
        return value.detach().flatten()[:count].cpu().tolist()
    except Exception as exc:  # pragma: no cover - defensive logging helper
        return f"<unavailable: {type(exc).__name__}>"


def _describe_cam_op_arg(name: str, value: object) -> str:
    if isinstance(value, Tensor):
        description = f"Tensor(dtype={value.dtype}, shape={tuple(value.shape)})"
        if name == "token_nums_rankid_layeridx":
            description = (
                f"{description}, first5={_tensor_first_values(value, count=5)!r}"
            )
        return description
    return repr(value)


def _log_cam_op_values(op_name: str, label: str, **kwargs: object) -> None:
    if not _cam_op_io_logging_enabled():
        return
    formatted_args = "\n".join(
        f"  {name}={_describe_cam_op_arg(name, value)}"
        for name, value in kwargs.items()
        if name not in _CAM_LOG_SKIPPED_ARGS
    )
    logger.warning("AFD CAM %s %s:\n%s", op_name, label, formatted_args)


def build_async_topology(
    afd_config: AFDConfig,
    role_rank: int | None = None,
    *,
    num_routed_experts: int | None = None,
) -> AFDAsyncTopology:
    """Validate role-local rank settings and derive the CAM HCCL world rank.

    The world is Attention-first: Attention role rank ``i`` maps to world rank
    ``i`` and FFN role rank ``j`` maps to
    ``num_attention_ranks + j``. Routed experts are distributed across FFN
    ranks using a ceiling division; production model layouts should keep the
    routed-expert count divisible by the FFN rank count.
    """
    attention_size = int(afd_config.num_attention_ranks)
    expert_rank_size = int(afd_config.num_ffn_ranks)
    role_rank = int(afd_config.afd_role_rank if role_rank is None else role_rank)
    if attention_size <= 0 or expert_rank_size <= 0:
        raise ValueError("AFD async topology sizes must be positive")
    if role_rank < 0:
        raise ValueError(f"AFD async role rank must be non-negative, got {role_rank}")

    if afd_config.role == "attention":
        if role_rank >= attention_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attention_size})",
            )
        world_rank = role_rank
    elif afd_config.role == "ffn":
        if role_rank >= expert_rank_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={expert_rank_size})",
            )
        world_rank = attention_size + role_rank
    else:
        raise ValueError(f"unknown AFD role {afd_config.role!r}")

    expert_count = int(num_routed_experts or 1)
    expert_per_rank = max(1, (expert_count + expert_rank_size - 1) // expert_rank_size)
    return AFDAsyncTopology(
        role=afd_config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        attention_rank_size=attention_size,
        expert_rank_size=expert_rank_size,
        expert_per_rank=expert_per_rank,
    )


def _connector_driven_batch_size(
    connector: CAMAsyncAFDConnector,
    fallback: int,
) -> int:
    return max(1, int(getattr(connector, "max_seq_len", fallback) or fallback))


def _cam_token_nums_rankid_layeridx(
    payload: AFDA2FTransferPayload,
    metadata: AFDTransferMetadata,
) -> Tensor:
    token_nums_rankid_layeridx = payload.atten_batch_size
    if token_nums_rankid_layeridx is None:
        transfer_state = metadata.transfer_state
        if transfer_state is not None:
            token_nums_rankid_layeridx = getattr(
                transfer_state,
                "token_nums_rankid_layeridx",
                None,
            )
    if token_nums_rankid_layeridx is None:
        raise RuntimeError(
            "AFD async CAM FFN work item requires "
            "TokenNums_Rankid_Layeridx from async_dispatch_recv",
        )
    return token_nums_rankid_layeridx


def _cam_metadata_int(token_nums_rankid_layeridx: Tensor, index: int) -> int:
    value = token_nums_rankid_layeridx[index]
    if isinstance(value, (int, float)):
        return int(value)
    return int(value.item())


def _cam_shared_token_count(payload: AFDA2FTransferPayload, fallback: int) -> int:
    expert_token_nums_shared = payload.ep_recv_counts_shared
    if expert_token_nums_shared is None:
        return max(1, int(fallback))
    return max(1, _cam_metadata_int(expert_token_nums_shared, 0))


def _cam_routed_token_count(payload: AFDA2FTransferPayload, fallback: int) -> int:
    expert_token_nums = payload.ep_recv_counts
    if expert_token_nums is None:
        expert_token_nums = payload.group_list
    if isinstance(expert_token_nums, Tensor):
        return max(0, int(expert_token_nums.to(torch.int64).sum().item()))
    if isinstance(expert_token_nums, (list, tuple)):
        return max(
            0,
            sum(
                _cam_metadata_int(expert_token_nums, index)
                for index in range(len(expert_token_nums))
            ),
        )
    return max(0, int(fallback))


def _slice_cam_payload_to_actual_tokens(
    hidden_states: Tensor,
    payload: AFDA2FTransferPayload,
    num_tokens: int,
    *,
    shared_num_tokens: int | None = None,
) -> Tensor:
    if shared_num_tokens is None:
        shared_num_tokens = num_tokens
    shared_slice_tokens = shared_num_tokens if shared_num_tokens > 0 else 100
    hidden_states = hidden_states[:num_tokens]
    if payload.expand_x_shared is not None:
        payload.expand_x_shared = payload.expand_x_shared[:shared_slice_tokens]
    if payload.dynamic_scales is not None:
        payload.dynamic_scales = payload.dynamic_scales[:num_tokens]
    if payload.dynamic_scales_shared is not None:
        payload.dynamic_scales_shared = payload.dynamic_scales_shared[
            :shared_slice_tokens
        ]
    if payload.x_active_mask is not None:
        payload.x_active_mask = payload.x_active_mask[:num_tokens]
    return hidden_states


def _sync_transfer_state_with_cam_metadata(
    metadata: AFDTransferMetadata,
    *,
    layer_idx: int,
) -> None:
    transfer_state = metadata.transfer_state
    if transfer_state is None:
        return
    transfer_state.layer_idx = int(layer_idx)


def _send_ffn_output_payload(
    connector: CAMAsyncAFDConnector,
    ffn_output: Tensor | AFDF2ATransferPayload,
    metadata: AFDTransferMetadata,
    *,
    stage_idx: int,
) -> None:
    if not isinstance(ffn_output, AFDF2ATransferPayload):
        connector.send_ffn_output(
            ffn_output,
            metadata,
            ubatch_idx=stage_idx,
        )
        return

    kwargs: dict[str, object] = {"ubatch_idx": stage_idx}
    if ffn_output.shared_output is not None:
        kwargs["expand_x_shared"] = ffn_output.shared_output
    connector.send_ffn_output(
        ffn_output.routed_output,
        metadata,
        **kwargs,
    )


def _validate_topk_payload(
    topk_ids: Tensor,
    topk_weights: Tensor | None,
    *,
    batch_size: int,
    topk: int,
    require_weights: bool = True,
) -> None:
    if tuple(topk_ids.shape) != (int(batch_size), int(topk)):
        raise ValueError(
            f"topk_ids shape must match ({batch_size}, {topk}), got {topk_ids.shape!r}",
        )
    if not require_weights and topk_weights is None:
        return
    if topk_weights is None:
        raise ValueError("topk_weights is required")
    if tuple(topk_weights.shape) != (int(batch_size), int(topk)):
        raise ValueError(
            "topk_weights shape must match "
            f"({batch_size}, {topk}), "
            f"got {topk_weights.shape!r}",
        )


def _hccl_comm_name(group: ProcessGroup, rank: int) -> str:
    backend = group._get_backend(torch.device("npu"))
    return str(backend.get_hccl_comm_name(int(rank)))


__all__ = [
    "AFD_ASYNC_CAM_GROUP_NAME",
    "CAMAsyncAFDConnector",
    "AFDAsyncTransferState",
    "AFDAsyncFFNWorkItem",
    "AFDAsyncTopology",
    "ATTN_RANKS_PER_DP_CONFIG_KEY",
    "CAM_COMM_ID",
    "build_async_topology",
]
