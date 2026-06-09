# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CAM op availability and stateless stub registration for AFD async NPU."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast

from afd_plugin.config import AFDConfig

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = object

CAM_OP_NAMESPACE: Final[str] = "umdk_cam_op_lib"
CAM_DISPATCH_SEND: Final[str] = "async_dispatch_send"
CAM_DISPATCH_RECV: Final[str] = "async_dispatch_recv"
CAM_COMBINE_SEND: Final[str] = "async_combine_send"
CAM_COMBINE_RECV: Final[str] = "async_combine_recv"
_CAM_OP_NAMES: Final[tuple[str, ...]] = (
    CAM_DISPATCH_SEND,
    CAM_DISPATCH_RECV,
    CAM_COMBINE_SEND,
    CAM_COMBINE_RECV,
)

_CAM_STUB_OPS_REGISTERED = False
_CAM_STUB_LIBRARIES: list[_TorchLibraryProtocol] = []


class _TorchLibraryProtocol(Protocol):
    def define(self, schema: str) -> None: ...

    def impl(self, op_name: str, fn: object, dispatch_key: str = "") -> None: ...


class _TorchLibraryFactoryProtocol(Protocol):
    def __call__(self, ns: str, kind: str) -> _TorchLibraryProtocol: ...


class _TorchOpsRootProtocol(Protocol):
    umdk_cam_op_lib: object


class _TorchModuleProtocol(Protocol):
    Tensor: type[object]
    float32: object
    int32: object
    int64: object
    library: object
    ops: _TorchOpsRootProtocol

    def empty(
        self,
        shape: tuple[int, ...],
        *,
        dtype: object,
        device: object,
    ) -> Tensor: ...


def ensure_cam_ops_available(afd_config: AFDConfig) -> None:
    """Ensure ``torch.ops.umdk_cam_op_lib`` exposes async CAM entry points."""

    torch = _torch()
    if _has_all_cam_ops(torch):
        return
    if is_cam_stub_ops_enabled(afd_config):
        register_cam_stub_ops()
        if _has_all_cam_ops(torch):
            return
    raise RuntimeError(
        "AFDAsyncConnector requires torch.ops.umdk_cam_op_lib CAM ops. "
        "Set extra_config['use_stub_cam_ops']=true to register stateless "
        "parameter-checking stubs while the real CAM ops are unavailable.",
    )


def is_cam_stub_ops_enabled(afd_config: AFDConfig) -> bool:
    return _truthy(afd_config.extra_config.get("use_stub_cam_ops", False))


def register_cam_stub_ops() -> None:
    """Register stateless CAM stub ops that validate inputs and make outputs."""

    global _CAM_STUB_OPS_REGISTERED
    if _CAM_STUB_OPS_REGISTERED:
        return

    torch = _torch()
    library_factory = cast(_TorchLibraryFactoryProtocol, torch.library.Library)
    lib = library_factory(CAM_OP_NAMESPACE, "DEF")
    _define_stub_schema(lib)

    lib.impl(CAM_DISPATCH_SEND, _cam_dispatch_send_impl, "CompositeExplicitAutograd")
    lib.impl(CAM_DISPATCH_RECV, _cam_dispatch_recv_impl, "CompositeExplicitAutograd")
    lib.impl(CAM_COMBINE_SEND, _cam_combine_send_impl, "CompositeExplicitAutograd")
    lib.impl(CAM_COMBINE_RECV, _cam_combine_recv_impl, "CompositeExplicitAutograd")

    _CAM_STUB_LIBRARIES.append(lib)
    _CAM_STUB_OPS_REGISTERED = True


def _define_stub_schema(lib: _TorchLibraryProtocol) -> None:
    lib.define(
        "async_dispatch_send(Tensor x, Tensor expert_ids, Tensor comm_args, "
        "int comm_id, int max_seq_len, int batch_size, int hidden_size, int topk, "
        "int expert_rank_size, int attention_rank_size, int expert_per_rank, "
        "int rank, int world_size, int layer_index, int tp_size, "
        "int dynamic_quant, str group_name) -> Tensor",
    )
    lib.define(
        "async_dispatch_recv(Tensor x, Tensor comm_args, int comm_id, "
        "int batch_size, int hidden_size, int topk, "
        "int expert_rank_size, int attention_rank_size, int expert_per_rank, "
        "int rank, int world_size, int tp_size, int dynamic_quant, "
        "str group_name) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
    )
    lib.define(
        "async_combine_send(Tensor expand_x, Tensor expand_x_shared, "
        "Tensor comm_args, Tensor expert_token_nums, int comm_id, "
        "int batch_size, int hidden_size, int topk, "
        "int expert_rank_size, int attention_rank_size, int expert_per_rank, "
        "int rank, int world_size, int tp_size, str group_name) -> Tensor",
    )
    lib.define(
        "async_combine_recv(Tensor expand_x, Tensor expert_ids, "
        "Tensor expert_scales, Tensor comm_args, int comm_id, "
        "int batch_size, int hidden_size, int topk, int expert_rank_size, "
        "int attention_rank_size, int expert_per_rank, int rank, int world_size, "
        "str group_name) -> Tensor",
    )


def _cam_dispatch_send_impl(
    x: Tensor,
    expert_ids: Tensor,
    comm_args: Tensor,
    comm_id: int,
    max_seq_len: int,
    batch_size: int,
    hidden_size: int,
    topk: int,
    expert_rank_size: int,
    attention_rank_size: int,
    expert_per_rank: int,
    rank: int,
    world_size: int,
    layer_index: int,
    tp_size: int,
    dynamic_quant: int,
    group_name: str,
) -> Tensor:
    _validate_common(
        x,
        comm_args,
        comm_id=comm_id,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        hidden_size=hidden_size,
        topk=topk,
        expert_rank_size=expert_rank_size,
        attention_rank_size=attention_rank_size,
        expert_per_rank=expert_per_rank,
        rank=rank,
        world_size=world_size,
        layer_index=layer_index,
        tp_size=tp_size,
        dynamic_quant=dynamic_quant,
    )
    _validate_topk_ids(expert_ids, batch_size=batch_size, topk=topk)
    _validate_group_name(group_name)
    return x


def _cam_dispatch_recv_impl(
    x: Tensor,
    comm_args: Tensor,
    comm_id: int,
    batch_size: int,
    hidden_size: int,
    topk: int,
    expert_rank_size: int,
    attention_rank_size: int,
    expert_per_rank: int,
    rank: int,
    world_size: int,
    tp_size: int,
    dynamic_quant: int,
    group_name: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    _validate_dispatch_recv_common(
        x,
        comm_args,
        comm_id=comm_id,
        batch_size=batch_size,
        hidden_size=hidden_size,
        topk=topk,
        expert_rank_size=expert_rank_size,
        attention_rank_size=attention_rank_size,
        expert_per_rank=expert_per_rank,
        rank=rank,
        world_size=world_size,
        tp_size=tp_size,
        dynamic_quant=dynamic_quant,
        group_name=group_name,
    )
    torch = _torch()
    expand_x_out = torch.empty(
        (batch_size, hidden_size),
        dtype=x.dtype,
        device=x.device,
    )
    shared_rows = max(1, batch_size // max(1, expert_rank_size))
    expand_x_out_shared = torch.empty(
        (shared_rows, hidden_size),
        dtype=x.dtype,
        device=x.device,
    )
    dynamic_scales = torch.empty(
        (batch_size,),
        dtype=torch.float32,
        device=x.device,
    )
    dynamic_scales_shared = torch.empty(
        (shared_rows,),
        dtype=torch.float32,
        device=x.device,
    )
    token_nums_rankid_layeridx = torch.empty(
        (5 + tp_size * (2 + expert_per_rank),),
        dtype=torch.int64,
        device=x.device,
    )
    expert_tokens = torch.empty(
        (expert_per_rank,),
        dtype=torch.int64,
        device=x.device,
    )
    expert_tokens_shared = torch.empty(
        (1,),
        dtype=torch.int64,
        device=x.device,
    )
    return (
        expand_x_out,
        expand_x_out_shared,
        dynamic_scales,
        dynamic_scales_shared,
        token_nums_rankid_layeridx,
        expert_tokens,
        expert_tokens_shared,
    )


def _cam_combine_send_impl(
    expand_x: Tensor,
    expand_x_shared: Tensor,
    comm_args: Tensor,
    expert_token_nums: Tensor,
    comm_id: int,
    batch_size: int,
    hidden_size: int,
    topk: int,
    expert_rank_size: int,
    attention_rank_size: int,
    expert_per_rank: int,
    rank: int,
    world_size: int,
    tp_size: int,
    group_name: str,
) -> Tensor:
    _validate_combine_common(
        expand_x,
        comm_args,
        comm_id=comm_id,
        batch_size=batch_size,
        hidden_size=hidden_size,
        topk=topk,
        expert_rank_size=expert_rank_size,
        attention_rank_size=attention_rank_size,
        expert_per_rank=expert_per_rank,
        rank=rank,
        world_size=world_size,
        tp_size=tp_size,
        group_name=group_name,
    )
    _validate_tensor(expand_x_shared, "expand_x_shared")
    _validate_tensor(expert_token_nums, "expert_token_nums")
    return expand_x


def _cam_combine_recv_impl(
    expand_x: Tensor,
    expert_ids: Tensor,
    expert_scales: Tensor,
    comm_args: Tensor,
    comm_id: int,
    batch_size: int,
    hidden_size: int,
    topk: int,
    expert_rank_size: int,
    attention_rank_size: int,
    expert_per_rank: int,
    rank: int,
    world_size: int,
    group_name: str,
) -> Tensor:
    _validate_combine_common(
        expand_x,
        comm_args,
        comm_id=comm_id,
        batch_size=batch_size,
        hidden_size=hidden_size,
        topk=topk,
        expert_rank_size=expert_rank_size,
        attention_rank_size=attention_rank_size,
        expert_per_rank=expert_per_rank,
        rank=rank,
        world_size=world_size,
        tp_size=1,
        group_name=group_name,
        validate_matrix=False,
    )
    _validate_topk_ids(expert_ids, batch_size=batch_size, topk=topk)
    _validate_tensor(expert_scales, "expert_scales")
    torch = _torch()
    return torch.empty(
        (batch_size, hidden_size),
        dtype=expand_x.dtype,
        device=expand_x.device,
    )


def _validate_dispatch_recv_common(
    tensor: Tensor,
    comm_args: Tensor,
    *,
    comm_id: int,
    batch_size: int,
    hidden_size: int,
    topk: int,
    expert_rank_size: int,
    attention_rank_size: int,
    expert_per_rank: int,
    rank: int,
    world_size: int,
    tp_size: int,
    dynamic_quant: int,
    group_name: str,
) -> None:
    _validate_combine_common(
        tensor,
        comm_args,
        comm_id=comm_id,
        batch_size=batch_size,
        hidden_size=hidden_size,
        topk=topk,
        expert_rank_size=expert_rank_size,
        attention_rank_size=attention_rank_size,
        expert_per_rank=expert_per_rank,
        rank=rank,
        world_size=world_size,
        tp_size=tp_size,
        group_name=group_name,
        validate_matrix=False,
    )
    if int(dynamic_quant) not in (0, 1):
        raise ValueError(f"dynamic_quant must be 0 or 1, got {dynamic_quant}")


def _validate_combine_common(
    tensor: Tensor,
    comm_args: Tensor,
    *,
    comm_id: int,
    batch_size: int,
    hidden_size: int,
    topk: int,
    expert_rank_size: int,
    attention_rank_size: int,
    expert_per_rank: int,
    rank: int,
    world_size: int,
    tp_size: int,
    group_name: str,
    validate_matrix: bool = True,
) -> None:
    _validate_tensor(tensor, "x")
    _validate_tensor(comm_args, "comm_args")
    _validate_non_negative_int("comm_id", comm_id)
    for name, value in (
        ("batch_size", batch_size),
        ("hidden_size", hidden_size),
        ("topk", topk),
        ("expert_rank_size", expert_rank_size),
        ("attention_rank_size", attention_rank_size),
        ("expert_per_rank", expert_per_rank),
        ("world_size", world_size),
        ("tp_size", tp_size),
    ):
        _validate_positive_int(name, value)
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(f"rank must be in [0, world_size), got {rank}")
    _validate_group_name(group_name)
    if validate_matrix:
        _validate_matrix(tensor, name="x", rows=1, columns=hidden_size)


def _validate_group_name(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("group_name must be a str")


def _validate_common(
    tensor: Tensor,
    comm_args: Tensor,
    *,
    comm_id: int,
    max_seq_len: int,
    batch_size: int,
    hidden_size: int,
    topk: int,
    expert_rank_size: int,
    attention_rank_size: int,
    expert_per_rank: int,
    rank: int,
    world_size: int,
    layer_index: int,
    tp_size: int,
    dynamic_quant: int,
) -> None:
    _validate_tensor(tensor, "x")
    _validate_tensor(comm_args, "comm_args")
    _validate_non_negative_int("comm_id", comm_id)
    for name, value in (
        ("max_seq_len", max_seq_len),
        ("batch_size", batch_size),
        ("hidden_size", hidden_size),
        ("topk", topk),
        ("expert_rank_size", expert_rank_size),
        ("attention_rank_size", attention_rank_size),
        ("expert_per_rank", expert_per_rank),
        ("world_size", world_size),
        ("tp_size", tp_size),
    ):
        _validate_positive_int(name, value)
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(f"rank must be in [0, world_size), got {rank}")
    if int(layer_index) < 0:
        raise ValueError(f"layer_index must be non-negative, got {layer_index}")
    if int(dynamic_quant) not in (0, 1):
        raise ValueError(f"dynamic_quant must be 0 or 1, got {dynamic_quant}")
    _validate_matrix(tensor, name="x", rows=batch_size, columns=hidden_size)


def _validate_tensor(value: object, name: str) -> None:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _validate_matrix(value: Tensor, *, name: str, rows: int, columns: int) -> None:
    if value.dim() != 2:
        raise ValueError(f"{name} must be a 2D tensor")
    if int(value.shape[0]) < int(rows) or int(value.shape[1]) != int(columns):
        raise ValueError(
            f"{name} shape must be at least ({rows}, {columns}), "
            f"got {tuple(value.shape)!r}",
        )


def _validate_topk_ids(value: Tensor, *, batch_size: int, topk: int) -> None:
    _validate_tensor(value, "expert_ids")
    if value.dim() != 2:
        raise ValueError("expert_ids must be a 2D tensor")
    if int(value.shape[0]) != int(batch_size) or int(value.shape[1]) != int(topk):
        raise ValueError(
            "expert_ids shape must match "
            f"({batch_size}, {topk}), got {tuple(value.shape)!r}",
        )


def _validate_positive_int(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative_int(name: str, value: int) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _has_all_cam_ops(torch: _TorchModuleProtocol) -> bool:
    ops = torch.ops
    if not hasattr(ops, CAM_OP_NAMESPACE):
        return False
    namespace = ops.umdk_cam_op_lib
    return all(hasattr(namespace, name) for name in _CAM_OP_NAMES)


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _torch() -> _TorchModuleProtocol:
    import torch

    return cast(_TorchModuleProtocol, torch)


__all__ = [
    "CAM_COMBINE_RECV",
    "CAM_COMBINE_SEND",
    "CAM_DISPATCH_RECV",
    "CAM_DISPATCH_SEND",
    "CAM_OP_NAMESPACE",
    "ensure_cam_ops_available",
    "is_cam_stub_ops_enabled",
    "register_cam_stub_ops",
]
