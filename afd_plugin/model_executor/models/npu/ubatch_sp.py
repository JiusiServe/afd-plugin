# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Sequence-parallel stage layout helpers for async MoE ubatching.

Ascend sequence parallelism stores the full-batch tensor as one contiguous
rank-major shard per TP rank. Async MoE stages instead need a contiguous
global stage reconstructed by the attention operator's TP all-gather. The
helpers below explicitly transpose between those two layouts:

* full-batch rank shards -> global full batch -> per-stage rank shards;
* per-stage rank shards -> global stages -> original full-batch rank shards.

Slicing each full-batch rank shard independently cannot implement this
transpose and silently pairs hidden states with another stage's positions.
"""

from __future__ import annotations

from typing import overload

import torch
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices


def build_async_moe_stage_inputs(
    *,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
    ubatch_slices: UBatchSlices,
    use_sp_stage_resharding: bool,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor | None],
    list[torch.Tensor],
    list[torch.Tensor | None],
    UBatchSlices,
]:
    """Build stage inputs in the layout expected by Ascend attention."""
    if not use_sp_stage_resharding:
        return _build_non_sp_stage_inputs(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            llama_4_scaling=llama_4_scaling,
            ubatch_slices=ubatch_slices,
        )

    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    global_num_tokens = sum(
        int(ubatch_slice.num_tokens) for ubatch_slice in ubatch_slices
    )
    if tp_size <= 1 or int(hidden_states.shape[0]) == global_num_tokens:
        return _build_non_sp_stage_inputs(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            llama_4_scaling=llama_4_scaling,
            ubatch_slices=ubatch_slices,
        )
    if global_num_tokens % tp_size != 0:
        raise ValueError(
            "Async MoE SP full-batch token count must be divisible by TP size; "
            f"got num_tokens={global_num_tokens}, tp_size={tp_size}",
        )
    expected_local_tokens = global_num_tokens // tp_size
    if int(hidden_states.shape[0]) != expected_local_tokens:
        raise ValueError(
            "Async MoE SP hidden-state layout mismatch: expected "
            f"{expected_local_tokens} local rows for {global_num_tokens} "
            f"global tokens at TP={tp_size}, got {int(hidden_states.shape[0])}",
        )

    # This transpose is paid once before and once after the complete MoE
    # pipeline, not once per layer. A future optimization can replace the
    # temporary global tensors with variable-split TP all-to-all after vLLM
    # exposes a stable backend-neutral API for it.
    global_hidden_states = _gather_sp_sequence_tensor(
        hidden_states,
        token_dim=0,
        expected_global_tokens=global_num_tokens,
    )
    global_residual = (
        None
        if residual is None
        else _gather_sp_sequence_tensor(
            residual,
            token_dim=0,
            expected_global_tokens=global_num_tokens,
        )
    )
    global_positions, positions_token_dim = _to_global_sequence_tensor(
        positions,
        local_num_tokens=expected_local_tokens,
        global_num_tokens=global_num_tokens,
    )
    if llama_4_scaling is None:
        global_llama_4_scaling = None
        scaling_token_dim = None
    else:
        global_llama_4_scaling, scaling_token_dim = _to_global_sequence_tensor(
            llama_4_scaling,
            local_num_tokens=expected_local_tokens,
            global_num_tokens=global_num_tokens,
        )

    stage_hidden_states: list[torch.Tensor] = []
    stage_residual: list[torch.Tensor | None] = []
    stage_positions: list[torch.Tensor] = []
    stage_llama_4_scaling: list[torch.Tensor | None] = []
    sp_local_stage_slices: UBatchSlices = []
    local_stage_start = 0

    for ubatch_slice in ubatch_slices:
        stage_tokens = int(ubatch_slice.num_tokens)
        if stage_tokens % tp_size != 0:
            raise ValueError(
                "Async MoE SP stage token count must be divisible by TP size; "
                f"got token_slice={ubatch_slice.token_slice}, tp_size={tp_size}",
            )
        local_stage_tokens = stage_tokens // tp_size
        global_stage_start = int(ubatch_slice.token_slice.start)
        local_global_start = global_stage_start + tp_rank * local_stage_tokens
        local_global_stop = local_global_start + local_stage_tokens
        local_global_slice = slice(local_global_start, local_global_stop)

        stage_hidden_states.append(global_hidden_states[local_global_slice])
        stage_residual.append(
            _slice_optional_first_dim(global_residual, local_global_slice),
        )
        stage_positions.append(
            _slice_sequence_dim(
                global_positions,
                positions_token_dim,
                local_global_slice,
            ),
        )
        stage_llama_4_scaling.append(
            None
            if global_llama_4_scaling is None or scaling_token_dim is None
            else _slice_sequence_dim(
                global_llama_4_scaling,
                scaling_token_dim,
                local_global_slice,
            )
        )

        local_stage_stop = local_stage_start + local_stage_tokens
        sp_local_stage_slices.append(
            UBatchSlice(
                ubatch_slice.request_slice,
                slice(local_stage_start, local_stage_stop),
            ),
        )
        local_stage_start = local_stage_stop

    return (
        stage_hidden_states,
        stage_residual,
        stage_positions,
        stage_llama_4_scaling,
        sp_local_stage_slices,
    )


def restore_async_moe_stage_outputs(
    stage_outputs: list[torch.Tensor],
    ubatch_slices: UBatchSlices,
    *,
    use_sp_stage_resharding: bool,
) -> torch.Tensor:
    """Restore stage outputs to the model's original full-batch SP layout."""
    if not use_sp_stage_resharding:
        return torch.cat(stage_outputs, dim=0)

    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    global_num_tokens = sum(
        int(ubatch_slice.num_tokens) for ubatch_slice in ubatch_slices
    )
    if len(stage_outputs) != len(ubatch_slices):
        raise ValueError(
            "Async MoE stage output count does not match ubatch metadata; "
            f"got {len(stage_outputs)} outputs and {len(ubatch_slices)} slices",
        )

    global_stage_outputs: list[torch.Tensor] = []
    for stage_output, ubatch_slice in zip(
        stage_outputs,
        ubatch_slices,
        strict=True,
    ):
        global_stage_outputs.append(
            _gather_sp_sequence_tensor(
                stage_output,
                token_dim=0,
                expected_global_tokens=int(ubatch_slice.num_tokens),
            ),
        )
    global_output = torch.cat(global_stage_outputs, dim=0)
    local_tokens = global_num_tokens // tp_size
    local_start = tp_rank * local_tokens
    return global_output[local_start : local_start + local_tokens]


def sp_local_actual_token_count(
    *,
    stage_actual_tokens: int,
    stage_input_tokens: int,
) -> int:
    """Return real token rows in the current rank's stage-local SP shard."""
    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    local_stage_tokens = int(stage_input_tokens) // tp_size
    local_start = tp_rank * local_stage_tokens
    local_stop = local_start + local_stage_tokens
    return max(0, min(int(stage_actual_tokens), local_stop) - local_start)


def _build_non_sp_stage_inputs(
    *,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
    ubatch_slices: UBatchSlices,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor | None],
    list[torch.Tensor],
    list[torch.Tensor | None],
    UBatchSlices,
]:
    num_tokens = int(hidden_states.shape[0])
    return (
        [
            _slice_and_pad_first_dim(hidden_states, ubatch_slice.token_slice)
            for ubatch_slice in ubatch_slices
        ],
        [
            _slice_and_pad_first_dim(residual, ubatch_slice.token_slice)
            for ubatch_slice in ubatch_slices
        ],
        [
            _slice_positions(positions, ubatch_slice.token_slice)
            for ubatch_slice in ubatch_slices
        ],
        [
            _slice_llama_4_scaling(
                llama_4_scaling,
                ubatch_slice.token_slice,
                num_tokens=num_tokens,
            )
            for ubatch_slice in ubatch_slices
        ],
        ubatch_slices,
    )


def _gather_sp_sequence_tensor(
    tensor: torch.Tensor,
    *,
    token_dim: int,
    expected_global_tokens: int,
) -> torch.Tensor:
    gathered = tensor_model_parallel_all_gather(tensor.contiguous(), token_dim)
    if int(gathered.shape[token_dim]) != int(expected_global_tokens):
        raise RuntimeError(
            "Async MoE SP all-gather returned an unexpected token count: "
            f"expected {expected_global_tokens}, got "
            f"{int(gathered.shape[token_dim])}",
        )
    return gathered


def _to_global_sequence_tensor(
    tensor: torch.Tensor,
    *,
    local_num_tokens: int,
    global_num_tokens: int,
) -> tuple[torch.Tensor, int]:
    global_token_dim = _sequence_tensor_token_dim(tensor, global_num_tokens)
    if global_token_dim is not None:
        return tensor, global_token_dim

    local_token_dim = _sequence_tensor_token_dim(tensor, local_num_tokens)
    if local_token_dim is None:
        raise ValueError(
            "Sequence tensor token dimension must be on axis 0 or 1; "
            f"got tensor shape {tuple(tensor.shape)} with "
            f"local_num_tokens={local_num_tokens}, "
            f"global_num_tokens={global_num_tokens}",
        )
    return (
        _gather_sp_sequence_tensor(
            tensor,
            token_dim=local_token_dim,
            expected_global_tokens=global_num_tokens,
        ),
        local_token_dim,
    )


def _sequence_tensor_token_dim(tensor: torch.Tensor, num_tokens: int) -> int | None:
    if tensor.dim() > 0 and int(tensor.shape[0]) == int(num_tokens):
        return 0
    if tensor.dim() > 1 and int(tensor.shape[1]) == int(num_tokens):
        return 1
    return None


def _slice_sequence_dim(
    tensor: torch.Tensor,
    token_dim: int,
    token_slice: slice,
) -> torch.Tensor:
    if token_dim == 0:
        return tensor[token_slice]
    return tensor[:, token_slice]


@overload
def _slice_optional_first_dim(
    tensor: torch.Tensor,
    token_slice: slice,
) -> torch.Tensor: ...


@overload
def _slice_optional_first_dim(
    tensor: None,
    token_slice: slice,
) -> None: ...


def _slice_optional_first_dim(
    tensor: torch.Tensor | None,
    token_slice: slice,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[token_slice]


@overload
def _slice_and_pad_first_dim(
    tensor: torch.Tensor,
    token_slice: slice,
) -> torch.Tensor: ...


@overload
def _slice_and_pad_first_dim(
    tensor: None,
    token_slice: slice,
) -> None: ...


def _slice_and_pad_first_dim(
    tensor: torch.Tensor | None,
    token_slice: slice,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    stage_tensor = tensor[token_slice]
    expected_tokens = int(token_slice.stop) - int(token_slice.start)
    missing_tokens = expected_tokens - int(stage_tensor.shape[0])
    if missing_tokens <= 0:
        return stage_tensor
    pad_shape = (missing_tokens, *tensor.shape[1:])
    return torch.cat([stage_tensor, tensor.new_zeros(pad_shape)], dim=0)


def _slice_positions(positions: torch.Tensor, token_slice: slice) -> torch.Tensor:
    if positions.dim() <= 1:
        return positions[token_slice]
    return positions[..., token_slice]


def _slice_llama_4_scaling(
    llama_4_scaling: torch.Tensor | None,
    token_slice: slice,
    *,
    num_tokens: int,
) -> torch.Tensor | None:
    if llama_4_scaling is None:
        return None
    if int(llama_4_scaling.shape[0]) == num_tokens:
        return llama_4_scaling[token_slice]
    if llama_4_scaling.dim() > 1 and int(llama_4_scaling.shape[1]) == num_tokens:
        return llama_4_scaling[:, token_slice]
    return llama_4_scaling


__all__ = [
    "build_async_moe_stage_inputs",
    "restore_async_moe_stage_outputs",
    "sp_local_actual_token_count",
]
