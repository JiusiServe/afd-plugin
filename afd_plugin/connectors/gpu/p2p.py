# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""P2P AFD connector migrated from the original in-tree AFD branch."""

import json
from datetime import timedelta
from typing import Any, NamedTuple

import torch
from torch.distributed.distributed_c10d import _get_default_group
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import (
    AFDConnectorMetadata,
    AFDDPMetadata,
    AFDRecvOutput,
)
from afd_plugin.distributed import (
    DefaultProcessGroupSwitcher,
    build_rank_mapping,
    init_afd_process_group,
)

_AFD_COMMUNICATORS: dict[int, Any] = {}
_AFD_COMM_ID_COUNTER = 0
_AFD_CUSTOM_OPS_REGISTERED = False


class _TensorMetadata(NamedTuple):
    device: Any
    dtype: Any
    size: Any


class P2PAFDConnector(AFDConnectorBase):
    """NCCL-backed Attention <-> FFN connector for Phase 4.

    The first supported topology matches the original AFD branch: FFN ranks are
    placed before Attention ranks in the AFD world, and each FFN rank owns a
    subgroup with one or more consecutive Attention ranks.
    """

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
        if parallel_config.data_parallel_size > 1:
            role_rank = int(parallel_config.data_parallel_rank)
        else:
            role_rank = int(afd_config.afd_server_rank)
        self.mapping = build_rank_mapping(
            afd_config,
            role_rank=role_rank,
        )
        self.world_rank = self.mapping.world_rank
        self.p2p_rank = self.mapping.p2p_rank
        self.attn_size = self.mapping.attention_size
        self.ffn_size = self.mapping.ffn_size
        self.min_size = self.mapping.min_size
        self.ratio = self.mapping.ratio
        self.dst_list = list(self.mapping.dp_metadata_destinations)
        self.num_hidden_layers = int(
            vllm_config.model_config.hf_config.num_hidden_layers,
        )
        self.hidden_size = int(vllm_config.model_config.hf_config.hidden_size)
        self.dp_metadata_list: dict[int, Any] = {}
        self.is_graph_capturing = False
        self.is_warmup = False
        self._tensor_metadata_list: dict[int, _TensorMetadata] = {}
        self._recv_attn_buffers: dict[tuple[int, int, tuple[int, ...]], Any] = {}
        self.a2e_group: Any | None = None
        self.e2a_group: Any | None = None
        self.p2p_pg: Any | None = None
        self.a2e_pynccl: Any | None = None
        self.e2a_pynccl: Any | None = None
        self.a2e_comm_id: int | None = None
        self.e2a_comm_id: int | None = None

    def close(self) -> None:
        for comm_id_name in ("a2e_comm_id", "e2a_comm_id"):
            comm_id = getattr(self, comm_id_name, None)
            if comm_id is not None:
                _unregister_comm(comm_id)
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
            self.rank_in_group = self.mapping.rank_in_subgroup
            self.group_size = len(self.mapping.subgroup_ranks)
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
        dp_metadata_list: dict[int, Any],
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        self.dp_metadata_list = dp_metadata_list
        self.is_graph_capturing = is_graph_capturing
        self.is_warmup = is_warmup
        self._tensor_metadata_list = {}
        device = torch.device(f"cuda:{self.local_rank}")
        dtype = self.vllm_config.model_config.dtype
        dp_rank = int(self.vllm_config.parallel_config.data_parallel_rank)
        for stage_idx, dp_metadata in dp_metadata_list.items():
            num_tokens = _num_tokens_for_dp_rank(dp_metadata, dp_rank)
            self._tensor_metadata_list[int(stage_idx)] = _TensorMetadata(
                device,
                dtype,
                torch.Size([num_tokens, self.hidden_size]),
            )

        if (
            self.afd_config.role == "ffn"
            and not self.vllm_config.model_config.enforce_eager
        ):
            for stage_idx, tensor_metadata in self._tensor_metadata_list.items():
                for src_rank in range(1, self.group_size):
                    buffer_key = (stage_idx, src_rank, tuple(tensor_metadata.size))
                    existing = self._recv_attn_buffers.get(buffer_key)
                    if _matches_tensor_metadata(existing, tensor_metadata):
                        continue
                    self._recv_attn_buffers[buffer_key] = torch.empty(
                        tuple(tensor_metadata.size),
                        dtype=tensor_metadata.dtype,
                        device=tensor_metadata.device,
                    )

    def is_attn_top_min_size_rank(self, world_rank: int) -> bool:
        return self.ffn_size <= world_rank < self.ffn_size + self.min_size

    def send_dp_metadata_list(
        self,
        dp_metadata_list: dict[int, Any],
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        if self.p2p_pg is None:
            return
        device = torch.device(f"cuda:{self.local_rank}")
        object_bytes = _encode_dp_metadata_payload(
            dp_metadata_list,
            is_graph_capturing=is_graph_capturing,
            is_warmup=is_warmup,
        )
        object_tensor_cpu = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)
        object_tensor = object_tensor_cpu.to(device)
        size_tensor = torch.tensor(
            [object_tensor.numel()],
            dtype=torch.long,
            device=device,
        )

        for dst in self.dst_list:
            torch.distributed.send(size_tensor, dst=dst, group=self.p2p_pg)
            torch.distributed.send(object_tensor, dst=dst, group=self.p2p_pg)

    def recv_dp_metadata_list(
        self,
        timeout_ms: int | None = None,
    ) -> tuple[dict[int, Any], bool, bool]:
        del timeout_ms

        if self.p2p_pg is None:
            raise RuntimeError("P2P DP metadata process group is not initialized")

        src = self.p2p_rank % self.min_size + self.ffn_size
        device = torch.device(f"cuda:{self.local_rank}")
        size_tensor = torch.empty(1, dtype=torch.long, device=device)
        rank_size = torch.distributed.recv(size_tensor, src=src, group=self.p2p_pg)
        object_tensor = torch.empty(
            int(size_tensor.item()),
            dtype=torch.uint8,
            device=device,
        )
        rank_object = torch.distributed.recv(object_tensor, src=src, group=self.p2p_pg)
        if rank_object != rank_size:
            raise RuntimeError("received AFD metadata fragments from different ranks")

        return _decode_dp_metadata_payload(object_tensor.cpu().numpy().tobytes())

    def send_attn_output(
        self,
        hidden_states: Any,
        metadata: AFDConnectorMetadata,
        **kwargs: Any,
    ) -> None:
        del kwargs
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

    def recv_ffn_output(self, handle: Any = None, **kwargs: Any) -> Any:
        del handle
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
        return output

    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
        ubatch_idx: int | None = None,
        **kwargs: Any,
    ) -> AFDRecvOutput:
        del timeout_ms, kwargs

        ubatch_idx = 0 if ubatch_idx is None else int(ubatch_idx)
        tensor_metadata = self._tensor_metadata_list[ubatch_idx]
        hidden_states_list: list[Any] = []

        for src in range(1, self.group_size):
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
        ffn_output: Any,
        metadata: AFDConnectorMetadata,
        **kwargs: Any,
    ) -> None:
        del kwargs
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
        hidden_states: Any,
        dst: int,
        process_group: Any,
        communicator: Any,
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
        process_group: Any,
        communicator: Any,
        tensor_metadata: _TensorMetadata,
        *,
        ref_tensor: Any | None = None,
    ) -> Any:
        if process_group is None or communicator is None:
            raise RuntimeError("P2P connector is not initialized")
        if process_group.world_size == 1:
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

    def _comm_id_for_communicator(self, communicator: Any) -> int:
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


def _matches_tensor_metadata(value: Any, tensor_metadata: _TensorMetadata) -> bool:
    if value is None:
        return False
    return (
        getattr(value, "shape", None) == tensor_metadata.size
        and getattr(value, "dtype", None) == tensor_metadata.dtype
        and getattr(value, "device", None) == tensor_metadata.device
    )


def _torch_is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def _num_tokens_for_dp_rank(dp_metadata: Any, dp_rank: int) -> int:
    token_count = dp_metadata.num_tokens_across_dp_cpu[dp_rank]
    item = getattr(token_count, "item", None)
    return int(item() if callable(item) else token_count)


def _to_int_list(value: Any) -> list[int]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    elif hasattr(value, "item"):
        value = [value.item()]
    elif isinstance(value, (int, float)):
        value = [value]
    return [int(item) for item in value]


def _to_int(value: Any) -> int:
    item = getattr(value, "item", None)
    return int(item() if callable(item) else value)


def _encode_dp_metadata_payload(
    dp_metadata_list: dict[int, Any],
    *,
    is_graph_capturing: bool,
    is_warmup: bool,
) -> bytes:
    metadata_payload: dict[str, dict[str, Any]] = {}
    for stage_idx, dp_metadata in dp_metadata_list.items():
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

    payload = {
        "dp_metadata_list": metadata_payload,
        "is_graph_capturing": bool(is_graph_capturing),
        "is_warmup": bool(is_warmup),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decode_dp_metadata_payload(
    payload_bytes: bytes,
) -> tuple[dict[int, Any], bool, bool]:
    payload = json.loads(payload_bytes.decode("utf-8"))
    dp_metadata_list = {
        int(stage_idx): AFDDPMetadata(
            num_tokens_across_dp_cpu=[
                int(value) for value in metadata["num_tokens_across_dp_cpu"]
            ],
            max_tokens_across_dp_cpu=int(metadata["max_tokens_across_dp_cpu"]),
        )
        for stage_idx, metadata in payload["dp_metadata_list"].items()
    }
    return (
        dp_metadata_list,
        bool(payload.get("is_graph_capturing", False)),
        bool(payload.get("is_warmup", False)),
    )


def _register_comm(communicator: Any) -> int:
    global _AFD_COMM_ID_COUNTER

    comm_id = _AFD_COMM_ID_COUNTER
    _AFD_COMMUNICATORS[comm_id] = communicator
    _AFD_COMM_ID_COUNTER += 1
    return comm_id


def _unregister_comm(comm_id: int) -> None:
    _AFD_COMMUNICATORS.pop(comm_id, None)


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
        del tensor, dst, comm_id
        return None

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
        del out, src, comm_id
        return None

    def register_one(**kwargs: Any) -> None:
        try:
            direct_register_custom_op(**kwargs)
        except RuntimeError as exc:
            # The op may already exist if another AFD connector instance or the
            # in-tree reference implementation registered it first in this
            # process. Keep this module's communicator registry local and reuse
            # the existing vLLM namespace op.
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
