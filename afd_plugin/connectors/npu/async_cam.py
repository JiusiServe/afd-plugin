# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend CAM async-DP connector skeleton for NPU AFD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from afd_plugin.compat.ascend.cam_stub_ops import (
    ensure_cam_ops_available,
    is_cam_stub_ops_enabled,
)
from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import AFDConnectorMetadata, AFDRecvOutput

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = object

DPMetadataMap: TypeAlias = dict[int, object]


@dataclass(slots=True)
class AFDAsyncConnectorData:
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
    atten_batch_size: Tensor | None = None
    x_active_mask: Tensor | None = None
    expand_x_shared: Tensor | None = None


@dataclass(frozen=True, slots=True)
class AFDAsyncTopology:
    role: str
    role_rank: int
    world_rank: int
    attention_rank_size: int
    expert_rank_size: int
    expert_per_rank: int

    @property
    def world_size(self) -> int:
        return self.attention_rank_size + self.expert_rank_size


class AFDAsyncConnector(AFDConnectorBase):
    """CAM-backed async-DP connector for Ascend NPU AFD.

    Phase 1 owns only the connector contract, CAM op call shape, and metadata
    control-plane bypass. Attention-side top-k and FFN loop integration are
    added in later phases.
    """

    uses_dp_metadata_control_plane = False
    ffn_step_trigger = "connector"
    requires_eager = True
    required_platform = "ascend"

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: object,
        afd_config: AFDConfig,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config)
        self._initialized = False
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        if int(parallel_config.data_parallel_size) > 1:
            self._role_rank = int(parallel_config.data_parallel_rank)
        else:
            self._role_rank = int(afd_config.afd_server_rank)
        self.hidden_size = int(hf_config.hidden_size)
        self.topk = max(1, int(hf_config.num_experts_per_tok))
        self.num_routed_experts = max(1, int(hf_config.n_routed_experts))
        dynamic_quant = afd_config.extra_config.get(
            "dynamicQuant",
            afd_config.extra_config.get("quant_mode", 0),
        )
        self.dynamic_quant = 1 if int(dynamic_quant or 0) == 1 else 0
        self.max_seq_len = max(
            1,
            int(vllm_config.scheduler_config.max_num_batched_tokens),
        )
        self.comm_id = int(afd_config.extra_config.get("comm_id", 0) or 0)
        self.tp_size = max(1, int(parallel_config.tensor_parallel_size))
        self.use_stub_cam_ops = is_cam_stub_ops_enabled(afd_config)
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
            list[tuple[AFDConnectorMetadata, Tensor, Tensor]],
        ] = {}

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def init_afd_connector(self) -> None:
        if self._initialized:
            return

        import torch

        ensure_cam_ops_available(self.afd_config)
        device = f"npu:{self.local_rank}"
        self.comm_args = torch.empty((1,), dtype=torch.int64, device=device)
        self._placeholder = torch.empty(
            (self.max_seq_len, self.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        self._initialized = True

    def close(self) -> None:
        self.comm_args = None
        self._placeholder = None
        self._pending_attention_payloads.clear()
        self._initialized = False

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list: DPMetadataMap,
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        del dp_metadata_list, is_graph_capturing, is_warmup

    def send_dp_metadata_list(
        self,
        dp_metadata_list: DPMetadataMap,
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        del dp_metadata_list, is_graph_capturing, is_warmup

    def recv_dp_metadata_list(
        self,
        timeout_ms: int | None = None,
    ) -> tuple[DPMetadataMap, bool, bool]:
        del timeout_ms
        raise RuntimeError(
            "AFDAsyncConnector does not use the DP metadata control plane",
        )

    def configure_metadata(
        self,
        metadata: AFDConnectorMetadata,
        **kwargs: object,
    ) -> None:
        metadata.connector_data = self._make_connector_data(
            batch_size=int(kwargs.get("batch_size", metadata.total_tokens)),
            layer_idx=int(metadata.layer_idx),
        )

    def create_recv_metadata(self, **kwargs: object) -> AFDConnectorMetadata:
        batch_size = int(kwargs.get("batch_size", kwargs.get("max_num_tokens", 1)) or 1)
        layer_idx = int(kwargs.get("layer_idx", 0) or 0)
        stage_idx = int(kwargs.get("ubatch_idx", 0) or 0)
        metadata = AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[max(1, batch_size)],
        )
        metadata.connector_data = self._make_connector_data(
            batch_size=max(1, batch_size),
            layer_idx=layer_idx,
        )
        return metadata

    def update_metadata(
        self,
        metadata: AFDConnectorMetadata,
        recv_output: AFDRecvOutput,
    ) -> None:
        data = _ensure_connector_data(metadata)
        data.topk_ids = recv_output.topk_ids
        data.topk_weights = recv_output.topk_weights
        data.expand_idx = recv_output.expand_idx
        data.expert_token_nums = recv_output.ep_recv_counts
        data.atten_batch_size = recv_output.atten_batch_size
        data.x_active_mask = recv_output.x_active_mask

    def send_attn_output(
        self,
        hidden_states: Tensor,
        metadata: AFDConnectorMetadata,
        **kwargs: object,
    ) -> Tensor:
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
            generated_topk_ids, generated_topk_weights = self._build_topk_payload(
                hidden_states,
                data,
            )
            if topk_ids is None:
                topk_ids = generated_topk_ids
            if topk_weights is None:
                topk_weights = generated_topk_weights
        import torch

        _validate_topk_payload(
            topk_ids,
            topk_weights,
            batch_size=data.batch_size,
            topk=data.topk,
        )
        data.topk_ids = topk_ids
        data.topk_weights = topk_weights
        self._queue_attention_payload(metadata, topk_ids, topk_weights)
        if self.use_stub_cam_ops:
            return hidden_states
        return torch.ops.cam.cam_dispatch_send(
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
        )

    def recv_ffn_output(self, handle: object = None, **kwargs: object) -> Tensor:
        del handle
        self._require_initialized()
        ref_tensor = kwargs["ref_tensor"]
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
        if metadata is None:
            metadata = self.create_recv_metadata(
                batch_size=int(ref_tensor.shape[0]),
                layer_idx=int(kwargs.get("layer_idx", 0) or 0),
            )
        data = self._metadata_data_or_default(metadata)
        import torch

        _validate_topk_payload(
            topk_ids,
            topk_weights,
            batch_size=data.batch_size,
            topk=data.topk,
        )
        if self.use_stub_cam_ops:
            return ref_tensor
        return torch.ops.cam.cam_combine_recv(
            ref_tensor,
            topk_ids,
            topk_weights,
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
        )

    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
        ubatch_idx: int | None = None,
        **kwargs: object,
    ) -> AFDRecvOutput:
        del timeout_ms
        self._require_initialized()
        ubatch_idx = 0 if ubatch_idx is None else int(ubatch_idx)
        metadata = kwargs.get("metadata")
        if metadata is None:
            metadata = self.create_recv_metadata(
                ubatch_idx=ubatch_idx,
                batch_size=int(kwargs.get("batch_size", self.max_seq_len) or 1),
                layer_idx=int(kwargs.get("layer_idx", 0) or 0),
            )
        data = _ensure_connector_data(metadata)
        placeholder = kwargs.get("placeholder", self._placeholder)
        if self.use_stub_cam_ops:
            return self._make_stub_recv_output(metadata, data, placeholder)
        import torch

        outputs = torch.ops.cam.cam_dispatch_recv(
            placeholder,
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
        )
        (
            hidden_states,
            topk_ids,
            topk_weights,
            expand_idx,
            expert_token_nums,
            atten_batch_size,
            x_active_mask,
        ) = outputs
        data.topk_ids = topk_ids
        data.topk_weights = topk_weights
        data.expand_idx = expand_idx
        data.expert_token_nums = expert_token_nums
        data.atten_batch_size = atten_batch_size
        data.x_active_mask = x_active_mask
        return AFDRecvOutput(
            hidden_states=hidden_states,
            metadata=metadata,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            x_active_mask=x_active_mask,
            atten_batch_size=atten_batch_size,
            expand_idx=expand_idx,
            ep_recv_counts=expert_token_nums,
        )

    def send_ffn_output(
        self,
        ffn_output: Tensor,
        metadata: AFDConnectorMetadata,
        **kwargs: object,
    ) -> None:
        self._require_initialized()
        data = _ensure_connector_data(metadata)
        import torch

        expand_x_shared = kwargs.get("expand_x_shared")
        if expand_x_shared is None:
            expand_x_shared = ffn_output
        expert_token_nums = data.expert_token_nums
        if expert_token_nums is None:
            expert_token_nums = kwargs["expert_token_nums"]
        if self.use_stub_cam_ops:
            return
        torch.ops.cam.cam_combine_send(
            ffn_output,
            expand_x_shared,
            self.comm_args,
            expert_token_nums,
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
        )

    def _metadata_data_or_default(
        self,
        metadata: AFDConnectorMetadata,
    ) -> AFDAsyncConnectorData:
        data = metadata.connector_data
        if data is None:
            data = self._make_connector_data(
                batch_size=metadata.total_tokens,
                layer_idx=int(metadata.layer_idx),
            )
            metadata.connector_data = data
        if not isinstance(data, AFDAsyncConnectorData):
            raise RuntimeError("AFD async metadata is missing connector_data")
        return data

    def _make_connector_data(
        self,
        *,
        batch_size: int,
        layer_idx: int,
    ) -> AFDAsyncConnectorData:
        return AFDAsyncConnectorData(
            batch_size=max(1, int(batch_size)),
            hidden_size=self.hidden_size,
            topk=self.topk,
            layer_idx=max(0, int(layer_idx)),
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("AFDAsyncConnector is not initialized")

    def _queue_attention_payload(
        self,
        metadata: AFDConnectorMetadata,
        topk_ids: Tensor,
        topk_weights: Tensor,
    ) -> None:
        self._pending_attention_payloads.setdefault(int(metadata.stage_idx), []).append(
            (metadata, topk_ids, topk_weights),
        )

    def _pop_attention_payload(
        self,
        stage_idx: int,
    ) -> tuple[AFDConnectorMetadata, Tensor, Tensor]:
        payloads = self._pending_attention_payloads.get(int(stage_idx))
        if not payloads:
            raise RuntimeError(
                "AFDAsyncConnector recv_ffn_output is missing pending "
                "Attention metadata",
            )
        payload = payloads.pop(0)
        if not payloads:
            self._pending_attention_payloads.pop(int(stage_idx), None)
        return payload

    def _build_topk_payload(
        self,
        hidden_states: Tensor,
        data: AFDAsyncConnectorData,
    ) -> tuple[Tensor, Tensor]:
        extra_config = self.afd_config.extra_config or {}
        if not _truthy(extra_config.get("use_stub_topk")):
            raise RuntimeError(
                "AFDAsyncConnector requires topk_ids/topk_weights unless "
                "use_stub_topk=true",
            )
        import torch

        expert_ids = torch.arange(
            data.topk,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        expert_ids = expert_ids.remainder(max(1, self.num_routed_experts))
        topk_ids = expert_ids.unsqueeze(0).expand(data.batch_size, data.topk)
        topk_ids = topk_ids.contiguous()
        topk_weights = torch.full(
            (data.batch_size, data.topk),
            1.0 / float(data.topk),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        return topk_ids, topk_weights

    def _make_stub_recv_output(
        self,
        metadata: AFDConnectorMetadata,
        data: AFDAsyncConnectorData,
        placeholder: object,
    ) -> AFDRecvOutput:
        if placeholder is None:
            raise RuntimeError("AFDAsyncConnector stub recv requires a placeholder")
        import torch

        hidden_states = placeholder.new_zeros((data.batch_size, data.hidden_size))
        expert_ids = torch.arange(
            data.topk,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        expert_ids = expert_ids.remainder(max(1, self.num_routed_experts))
        topk_ids = expert_ids.unsqueeze(0).expand(data.batch_size, data.topk)
        topk_ids = topk_ids.contiguous()
        topk_weights = torch.full(
            (data.batch_size, data.topk),
            1.0 / float(data.topk),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        expand_idx = torch.arange(
            data.batch_size * data.topk,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        expert_token_nums = torch.zeros(
            (self.expert_rank_size,),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        atten_batch_size = torch.zeros(
            (self.attention_rank_size,),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        x_active_mask = torch.ones(
            (data.batch_size,),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        data.topk_ids = topk_ids
        data.topk_weights = topk_weights
        data.expand_idx = expand_idx
        data.expert_token_nums = expert_token_nums
        data.atten_batch_size = atten_batch_size
        data.x_active_mask = x_active_mask
        return AFDRecvOutput(
            hidden_states=hidden_states,
            metadata=metadata,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            x_active_mask=x_active_mask,
            atten_batch_size=atten_batch_size,
            expand_idx=expand_idx,
            ep_recv_counts=expert_token_nums,
        )


def build_async_topology(
    afd_config: AFDConfig,
    role_rank: int | None = None,
    *,
    num_routed_experts: int | None = None,
) -> AFDAsyncTopology:
    attention_size = int(afd_config.num_attention_servers)
    expert_rank_size = int(afd_config.num_ffn_servers)
    role_rank = int(afd_config.afd_server_rank if role_rank is None else role_rank)
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

    expert_count = int(
        num_routed_experts or afd_config.extra_config.get("num_experts", 1),
    )
    expert_per_rank = max(1, (expert_count + expert_rank_size - 1) // expert_rank_size)
    return AFDAsyncTopology(
        role=afd_config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        attention_rank_size=attention_size,
        expert_rank_size=expert_rank_size,
        expert_per_rank=expert_per_rank,
    )


def _ensure_connector_data(metadata: AFDConnectorMetadata) -> AFDAsyncConnectorData:
    data = metadata.connector_data
    if not isinstance(data, AFDAsyncConnectorData):
        raise RuntimeError("AFD async metadata is missing connector_data")
    return data


def _validate_topk_payload(
    topk_ids: object,
    topk_weights: object | None,
    *,
    batch_size: int,
    topk: int,
    require_weights: bool = True,
) -> None:
    if tuple(getattr(topk_ids, "shape", ())) != (int(batch_size), int(topk)):
        raise ValueError(
            "topk_ids shape must match "
            f"({batch_size}, {topk}), got {getattr(topk_ids, 'shape', None)!r}",
        )
    if not require_weights and topk_weights is None:
        return
    if tuple(getattr(topk_weights, "shape", ())) != (int(batch_size), int(topk)):
        raise ValueError(
            "topk_weights shape must match "
            f"({batch_size}, {topk}), "
            f"got {getattr(topk_weights, 'shape', None)!r}",
        )


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = [
    "AFDAsyncConnector",
    "AFDAsyncConnectorData",
    "AFDAsyncTopology",
    "build_async_topology",
]
