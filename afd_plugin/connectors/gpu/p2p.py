# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NCCL-backed P2P AFD connector."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch.distributed.distributed_c10d import ProcessGroup, _get_default_group
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import (
    AFDConnectorMetadata,
    AFDDPMetadata,
    AFDDPMetadataPayload,
    AFDRecvOutput,
    recv_dp_metadata_payload,
    send_dp_metadata_payload,
)
from afd_plugin.distributed import (
    DefaultProcessGroupSwitcher,
    build_rank_mapping,
    init_afd_process_group,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_AFD_COMMUNICATORS: dict[int, Any] = {}
_AFD_COMM_ID_COUNTER = 0
_AFD_CUSTOM_OPS_REGISTERED = False


class _TensorMetadata(NamedTuple):
    device: torch.device
    dtype: torch.dtype
    size: torch.Size


class P2PAFDConnector(AFDConnectorBase):
    """NCCL-backed Attention <-> FFN connector.

    The P2P topology places FFN ranks before Attention ranks in the AFD world,
    and each FFN rank owns a subgroup with one or more consecutive Attention
    ranks.
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config)
        self._initialized = False
        # afd_role_rank already carries the dp/pcp/tp-derived offset (the
        # runners apply _with_dp_derived_afd_rank before create_connector);
        # re-deriving from data_parallel_rank here would collapse TP peers
        # onto the same role rank.
        self.mapping = build_rank_mapping(
            afd_config,
            role_rank=int(afd_config.afd_role_rank),
        )
        self.world_rank = self.mapping.world_rank
        self.p2p_rank = self.mapping.p2p_rank
        self.attn_size = self.mapping.attention_size
        self.ffn_size = self.mapping.ffn_size
        self.min_size = self.mapping.min_size
        self.ratio = self.mapping.ratio
        self.group_size = len(self.mapping.subgroup_ranks)
        self.dst_list = list(self.mapping.dp_metadata_destinations)
        self.num_hidden_layers = int(
            vllm_config.model_config.hf_config.num_hidden_layers,
        )
        self.hidden_size = int(vllm_config.model_config.hf_config.hidden_size)
        self.dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] = {}
        self.is_graph_capturing = False
        self.is_warmup = False
        self._tensor_metadata_list: dict[int, _TensorMetadata] = {}
        self._recv_attn_tensor_metadata_list: dict[
            tuple[int, int],
            _TensorMetadata,
        ] = {}
        self._recv_attn_buffers: dict[
            tuple[int, int, tuple[int, ...]],
            torch.Tensor,
        ] = {}
        self.a2e_group: StatelessProcessGroup | None = None
        self.e2a_group: StatelessProcessGroup | None = None
        self.p2p_pg: ProcessGroup | None = None
        self.a2e_pynccl: PyNcclCommunicator | None = None
        self.e2a_pynccl: PyNcclCommunicator | None = None
        self.a2e_comm_id: int | None = None
        self.e2a_comm_id: int | None = None

    def close(self) -> None:
        for comm_id_name in ("a2e_comm_id", "e2a_comm_id"):
            comm_id = getattr(self, comm_id_name, None)
            if comm_id is not None:
                _AFD_COMMUNICATORS.pop(comm_id, None)
                setattr(self, comm_id_name, None)
        for communicator_name in ("a2e_pynccl", "e2a_pynccl"):
            communicator = getattr(self, communicator_name, None)
            shutdown = getattr(communicator, "shutdown", None)
            if callable(shutdown):
                shutdown()
            setattr(self, communicator_name, None)
        self._initialized = False

    def init_afd_connector(self) -> None:
        if self._initialized:
            return

        _register_p2p_custom_ops()

        afd_pg = init_afd_process_group(
            backend="nccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.ffn_size + self.attn_size,
            rank=self.world_rank,
            group_name="afd",
            timeout=timedelta(minutes=2),
        )

        with DefaultProcessGroupSwitcher(_get_default_group(), afd_pg):
            base_port = self.afd_config.port
            self.a2e_group = StatelessProcessGroup.create(
                host=self.afd_config.host,
                port=base_port + self.mapping.subgroup_index + 1,
                rank=self.mapping.rank_in_subgroup,
                world_size=len(self.mapping.subgroup_ranks),
            )
            self.e2a_group = self.a2e_group
            self.a2e_pynccl = PyNcclCommunicator(
                group=self.a2e_group,
                device=self.local_rank,
            )
            self.a2e_comm_id = _register_comm(self.a2e_pynccl)
            self.e2a_pynccl = PyNcclCommunicator(
                group=self.e2a_group,
                device=self.local_rank,
            )
            self.e2a_comm_id = _register_comm(self.e2a_pynccl)

        if self.mapping.participates_in_dp_metadata_group:
            self.p2p_pg = init_afd_process_group(
                backend="nccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size + self.min_size,
                rank=self.p2p_rank,
                group_name="p2p",
                timeout=timedelta(minutes=30),
            )

        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def update_state_from_dp_metadata(
        self,
        payload: AFDDPMetadataPayload,
    ) -> None:
        self.dp_metadata_list = payload.dp_metadata_list
        self.is_graph_capturing = payload.is_graph_capturing
        self.is_warmup = payload.is_warmup
        self._tensor_metadata_list = {}
        self._recv_attn_tensor_metadata_list = {}
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = self.vllm_config.model_config.dtype
        for stage_idx, dp_metadata in payload.dp_metadata_list.items():
            stage_idx = int(stage_idx)
            if self.afd_config.role == "ffn":
                peer_metadata: list[_TensorMetadata] = []
                for src_rank in range(1, self.group_size):
                    attention_rank = self._attention_rank_for_subgroup_rank(src_rank)
                    tensor_metadata = _TensorMetadata(
                        device,
                        dtype,
                        torch.Size(
                            [
                                _num_tokens_for_attention_rank(
                                    dp_metadata,
                                    attention_rank=attention_rank,
                                    attention_size=self.attn_size,
                                ),
                                self.hidden_size,
                            ],
                        ),
                    )
                    self._recv_attn_tensor_metadata_list[(stage_idx, src_rank)] = (
                        tensor_metadata
                    )
                    peer_metadata.append(tensor_metadata)
                num_tokens = sum(
                    int(tensor_metadata.size[0]) for tensor_metadata in peer_metadata
                )
            else:
                num_tokens = _num_tokens_for_attention_rank(
                    dp_metadata,
                    attention_rank=self.mapping.role_rank,
                    attention_size=self.attn_size,
                )
            self._tensor_metadata_list[stage_idx] = _TensorMetadata(
                device,
                dtype,
                torch.Size([max(1, num_tokens), self.hidden_size]),
            )

        if (
            self.afd_config.role == "ffn"
            and not self.vllm_config.model_config.enforce_eager
        ):
            for (
                stage_idx,
                src_rank,
            ), tensor_metadata in self._recv_attn_tensor_metadata_list.items():
                buffer_key = (stage_idx, src_rank, tuple(tensor_metadata.size))
                existing = self._recv_attn_buffers.get(buffer_key)
                if existing is not None:
                    continue
                self._recv_attn_buffers[buffer_key] = torch.empty(
                    tuple(tensor_metadata.size),
                    dtype=tensor_metadata.dtype,
                    device=tensor_metadata.device,
                )

    def send_dp_metadata_list(
        self,
        payload: AFDDPMetadataPayload,
    ) -> None:
        if self.p2p_pg is None:
            return
        if not (self.ffn_size <= self.world_rank < self.ffn_size + self.min_size):
            return
        # NCCL transport requires the wire tensors to live on the CUDA device.
        device = torch.device(f"cuda:{self.local_rank}")
        send_dp_metadata_payload(
            payload,
            dst=self.dst_list,
            group=self.p2p_pg,
            device=device,
        )

    def recv_dp_metadata_list(
        self,
        timeout_ms: int | None = None,
    ) -> AFDDPMetadataPayload:
        if self.p2p_pg is None:
            raise RuntimeError("P2P DP metadata process group is not initialized")

        src = self.p2p_rank % self.min_size + self.ffn_size
        device = torch.device(f"cuda:{self.local_rank}")
        return recv_dp_metadata_payload(src=src, group=self.p2p_pg, device=device)

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
        **kwargs: Any,
    ) -> None:
        if not _torch_is_compiling() and not metadata.validate_tensor_shape(
            tuple(hidden_states.shape),
        ):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"AFD metadata token count {metadata.total_tokens}",
            )
        self._send_hidden_states(
            hidden_states,
            0,
            self.a2e_group,
            self.a2e_pynccl,
        )

    def recv_ffn_output(self, **kwargs: Any) -> torch.Tensor:
        ref_tensor = kwargs.get("ref_tensor")
        ubatch_idx = kwargs.get("ubatch_idx")
        if ubatch_idx is None:
            ubatch_idx = self._current_ubatch_idx()
        output = self._recv_hidden_states(
            0,
            self.e2a_group,
            self.e2a_pynccl,
            self._tensor_metadata_list[int(ubatch_idx)],
            ref_tensor=ref_tensor,
        )
        if output is None:
            raise RuntimeError(
                "P2P recv_ffn_output requires ref_tensor when no receive is performed",
            )
        return output

    def recv_attn_output(
        self,
        ubatch_idx: int | None = None,
        **kwargs: Any,
    ) -> AFDRecvOutput:
        ubatch_idx = 0 if ubatch_idx is None else int(ubatch_idx)
        hidden_states_list: list[torch.Tensor] = []

        for src in range(1, self.group_size):
            tensor_metadata = self._recv_attn_tensor_metadata_list.get(
                (ubatch_idx, src),
                self._tensor_metadata_list[ubatch_idx],
            )
            ref_tensor = None
            if not self.vllm_config.model_config.enforce_eager:
                ref_tensor = self._recv_attn_buffers.get(
                    (ubatch_idx, src, tuple(tensor_metadata.size)),
                )
            hidden_states_list.append(
                self._recv_hidden_states(
                    src,
                    self.a2e_group,
                    self.a2e_pynccl,
                    tensor_metadata,
                    ref_tensor=ref_tensor,
                ),
            )

        if not hidden_states_list:
            raise RuntimeError("P2P FFN rank has no Attention peers")
        hidden_states = (
            torch.cat(hidden_states_list, dim=0)
            if len(hidden_states_list) > 1
            else hidden_states_list[0]
        )
        metadata = AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=0,
            stage_idx=ubatch_idx,
            seq_lens=[int(tensor.shape[0]) for tensor in hidden_states_list],
        )
        return AFDRecvOutput(hidden_states=hidden_states, metadata=metadata)

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        metadata: AFDConnectorMetadata,
        **kwargs: Any,
    ) -> None:
        if not _torch_is_compiling() and not metadata.validate_tensor_shape(
            tuple(ffn_output.shape),
        ):
            raise ValueError(
                f"ffn_output shape {ffn_output.shape!r} does not match metadata",
            )
        if self.ratio == 1:
            self._send_hidden_states(ffn_output, 1, self.e2a_group, self.e2a_pynccl)
            return

        split_sizes = metadata.seq_lens
        if len(split_sizes) != self.ratio:
            total_tokens = int(ffn_output.shape[0])
            if total_tokens % self.ratio != 0:
                raise ValueError(
                    "cannot evenly split FFN output across Attention peers: "
                    f"tokens={total_tokens}, ratio={self.ratio}",
                )
            tokens_per_attention = total_tokens // self.ratio
            split_sizes = [tokens_per_attention] * self.ratio

        start = 0
        for dst, token_count in zip(
            range(1, self.group_size),
            split_sizes,
            strict=False,
        ):
            end = start + token_count
            self._send_hidden_states(
                ffn_output[start:end],
                dst,
                self.e2a_group,
                self.e2a_pynccl,
            )
            start = end

    def _send_hidden_states(
        self,
        hidden_states: torch.Tensor,
        dst: int,
        process_group: StatelessProcessGroup | None,
        communicator: PyNcclCommunicator | None,
    ) -> None:
        if process_group is None or communicator is None:
            raise RuntimeError("P2P connector is not initialized")
        if process_group.world_size == 1:
            return
        if dst >= process_group.world_size:
            raise ValueError(f"invalid P2P destination rank {dst}")
        if getattr(hidden_states, "is_cpu", False):
            raise ValueError("P2P hidden states must be on GPU")

        comm_id = self._comm_id_for_communicator(communicator)
        torch.ops.vllm.afd_p2p_send(
            hidden_states,
            int(dst),
            int(comm_id),
        )
        return

    def _recv_hidden_states(
        self,
        src: int,
        process_group: StatelessProcessGroup | None,
        communicator: PyNcclCommunicator | None,
        tensor_metadata: _TensorMetadata,
        *,
        ref_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if process_group is None or communicator is None:
            raise RuntimeError("P2P connector is not initialized")
        if process_group.world_size == 1:
            if ref_tensor is None:
                raise RuntimeError("single-rank P2P recv requires a reference tensor")
            return ref_tensor
        if src >= process_group.world_size:
            raise ValueError(f"invalid P2P source rank {src}")

        size = list(tensor_metadata.size)
        if ref_tensor is not None:
            size[0] = ref_tensor.shape[0]

        if (
            ref_tensor is not None
            and ref_tensor.shape == tuple(size)
            and ref_tensor.dtype == tensor_metadata.dtype
            and ref_tensor.device == tensor_metadata.device
        ):
            hidden_states = ref_tensor
        else:
            hidden_states = torch.empty(
                tuple(size),
                dtype=tensor_metadata.dtype,
                device=tensor_metadata.device,
            )
        comm_id = self._comm_id_for_communicator(communicator)
        torch.ops.vllm.afd_p2p_recv(hidden_states, int(src), int(comm_id))
        return hidden_states

    def _attention_rank_for_subgroup_rank(self, subgroup_rank: int) -> int:
        if subgroup_rank <= 0 or subgroup_rank >= self.group_size:
            raise ValueError(f"invalid Attention subgroup rank {subgroup_rank}")
        return self.mapping.subgroup_index * self.ratio + (int(subgroup_rank) - 1)

    def _comm_id_for_communicator(self, communicator: PyNcclCommunicator) -> int:
        if communicator is self.a2e_pynccl and self.a2e_comm_id is not None:
            return self.a2e_comm_id
        if communicator is self.e2a_pynccl and self.e2a_comm_id is not None:
            return self.e2a_comm_id
        raise RuntimeError("P2P communicator is not registered for AFD custom ops")

    @staticmethod
    def _current_ubatch_idx() -> int:
        try:
            forward_context = get_forward_context()
            afd_metadata = forward_context.additional_kwargs["afd_metadata"]
            return int(afd_metadata.ubatch_idx)
        except Exception:
            return 0


def _torch_is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def _to_int_list(value: object) -> list[int]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    elif hasattr(value, "item"):
        value = [value.item()]
    elif isinstance(value, (int, float)):
        value = [value]
    return [int(item) for item in value]


def _num_tokens_for_attention_rank(
    dp_metadata: DPMetadata | AFDDPMetadata,
    *,
    attention_rank: int,
    attention_size: int,
    fallback: int = 1,
) -> int:
    counts = _to_int_list(dp_metadata.num_tokens_across_dp_cpu)
    if not counts:
        return max(1, int(fallback))
    attention_size = int(attention_size)
    if len(counts) < attention_size and attention_size % len(counts) == 0:
        tp_size = attention_size // len(counts)
        counts = [counts[idx // tp_size] for idx in range(attention_size)]
    attention_rank = int(attention_rank)
    if 0 <= attention_rank < len(counts):
        return max(1, int(counts[attention_rank]))
    return max(1, int(fallback))


def _register_comm(communicator: Any) -> int:
    global _AFD_COMM_ID_COUNTER

    comm_id = _AFD_COMM_ID_COUNTER
    _AFD_COMMUNICATORS[comm_id] = communicator
    _AFD_COMM_ID_COUNTER += 1
    return comm_id


def _register_p2p_custom_ops() -> None:
    global _AFD_CUSTOM_OPS_REGISTERED

    if _AFD_CUSTOM_OPS_REGISTERED:
        return

    def afd_p2p_send_impl(
        tensor: torch.Tensor,
        dst: int,
        comm_id: int,
    ) -> None:
        communicator = _AFD_COMMUNICATORS.get(int(comm_id))
        if communicator is None:
            raise RuntimeError(f"AFD communicator id {comm_id} is not registered")
        communicator.send(
            tensor,
            int(dst),
            stream=torch.cuda.current_stream(tensor.device),
        )
        return None

    def afd_p2p_send_fake(
        tensor: torch.Tensor,
        dst: int,
        comm_id: int,
    ) -> None:
        pass

    def afd_p2p_recv_impl(out: torch.Tensor, src: int, comm_id: int) -> None:
        communicator = _AFD_COMMUNICATORS.get(int(comm_id))
        if communicator is None:
            raise RuntimeError(f"AFD communicator id {comm_id} is not registered")
        communicator.recv(
            out,
            int(src),
            stream=torch.cuda.current_stream(out.device),
        )

    def afd_p2p_recv_fake(out: torch.Tensor, src: int, comm_id: int) -> None:
        pass

    def register_one(**kwargs: Any) -> None:
        try:
            direct_register_custom_op(**kwargs)
        except RuntimeError as exc:
            # The op may already be registered in the vLLM namespace by another
            # connector instance in this process. Keep this module's
            # communicator registry local and reuse the existing op.
            text = str(exc).lower()
            if not any(
                marker in text
                for marker in ("already", "duplicate", "same name", "defined")
            ):
                raise

    register_one(
        op_name="afd_p2p_send",
        op_func=afd_p2p_send_impl,
        mutates_args=["tensor"],
        fake_impl=afd_p2p_send_fake,
    )
    register_one(
        op_name="afd_p2p_recv",
        op_func=afd_p2p_recv_impl,
        mutates_args=["out"],
        fake_impl=afd_p2p_recv_fake,
    )

    _AFD_CUSTOM_OPS_REGISTERED = True


__all__ = ["P2PAFDConnector"]
