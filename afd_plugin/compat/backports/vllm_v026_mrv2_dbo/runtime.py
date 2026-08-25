# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Runtime pieces missing from vLLM 0.26 ModelRunnerV2 eager DBO.

The behavior is adapted from ``specture724/vllm`` branch
``feat/v2/dbo-fullcg`` at ``626fee7831``. The target ABI is vLLM
``568afb3a13``. Keep this module self-contained so it can be deleted once the
pinned vLLM release provides native ModelRunnerV2 DBO.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode, ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    UBatchSlices,
    check_ubatch_thresholds,
    maybe_create_ubatch_slices,
)
from vllm.v1.worker.utils import AttentionGroup


@dataclass(frozen=True)
class AFDBatchExecutionDescriptor(BatchExecutionDescriptor):
    """v0.26 batch descriptor extended only with the DBO execution count."""

    num_ubatches: int = 1


def assert_backport_required() -> None:
    """Fail when the pinned vLLM ABI no longer needs this backport."""

    descriptor_fields = {field.name for field in fields(BatchExecutionDescriptor)}
    if "num_ubatches" in descriptor_fields:
        raise RuntimeError(
            "vLLM already provides ModelRunnerV2 DBO descriptors; remove the "
            "temporary afd-plugin v0.26 backport",
        )


def dispatch_afd_dbo_and_sync_dp(
    *,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    parallel_config: ParallelConfig,
    decode_query_len: int,
    allow_ubatching: bool,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """Select eager DBO consistently across all DP ranks.

    The minimum rank token count controls the threshold, while all ranks run
    the maximum token count when DBO is selected. This matches the final
    upstream inference rule and avoids a separate per-rank DBO vote.
    """

    if dp_size == 1:
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
            ),
            None,
        )

    tensor = torch.zeros(3, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = uniform_token_count or 0
    tensor[2][dp_rank] = int(allow_ubatching)
    dist.all_reduce(tensor, group=get_dp_group().cpu_group)

    num_tokens_across_dp = tensor[0]
    if torch.all(num_tokens_across_dp == 0).item():
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=0,
                num_reqs=0,
            ),
            None,
        )

    uniform_decode = bool(
        torch.all(tensor[1] == int(decode_query_len)).item()
    )
    should_ubatch = (
        bool(torch.all(tensor[2] == 1).item())
        and int(num_tokens_across_dp.max().item())
        >= int(parallel_config.num_ubatches)
        and check_ubatch_thresholds(
            parallel_config,
            int(num_tokens_across_dp.min().item()),
            uniform_decode=uniform_decode,
        )
    )
    if not should_ubatch:
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
            ),
            num_tokens_across_dp,
        )

    padded_tokens = int(num_tokens_across_dp.max().item())
    num_tokens_across_dp.fill_(padded_tokens)
    return (
        AFDBatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=padded_tokens,
            num_reqs=num_reqs,
            num_ubatches=int(parallel_config.num_ubatches),
        ),
        num_tokens_across_dp,
    )


def create_ubatch_slices(input_batch: InputBatch, num_ubatches: int) -> UBatchSlices:
    """Split a DP-padded v0.26 InputBatch into microbatch views."""

    _, padded_slices = maybe_create_ubatch_slices(
        True,
        input_batch.num_scheduled_tokens,
        input_batch.num_tokens_after_padding,
        input_batch.num_reqs_after_padding,
        num_ubatches,
    )
    assert padded_slices is not None
    return [
        UBatchSlice(
            slice(
                min(stage.request_slice.start, input_batch.num_reqs - 1),
                min(stage.request_slice.stop, input_batch.num_reqs_after_padding),
            ),
            stage.token_slice,
        )
        for stage in padded_slices
    ]


def slice_input_batch(
    input_batch: InputBatch,
    stage: UBatchSlice,
    query_start_loc_buffer: torch.Tensor,
    seq_lens_buffer: torch.Tensor,
) -> InputBatch:
    """Build one microbatch while retaining the v0.26 InputBatch ABI."""

    assert not stage.is_empty(), f"Ubatch slice {stage} is empty"
    req_start = stage.request_slice.start
    req_stop = stage.request_slice.stop
    tok_start = stage.token_slice.start
    tok_stop = stage.token_slice.stop

    num_reqs_after_padding = req_stop - req_start
    num_tokens_after_padding = tok_stop - tok_start
    num_reqs = max(0, min(req_stop, input_batch.num_reqs) - req_start)
    num_tokens = max(0, min(tok_stop, input_batch.num_tokens) - tok_start)

    query_start_loc = query_start_loc_buffer[: num_reqs_after_padding + 1]
    torch.sub(
        input_batch.query_start_loc[req_start : req_stop + 1],
        tok_start,
        out=query_start_loc,
    )
    query_start_loc.clamp_(0, num_tokens_after_padding)
    query_start_loc_np = np.clip(
        input_batch.query_start_loc_np[req_start : req_stop + 1] - tok_start,
        0,
        num_tokens_after_padding,
    ).astype(np.int32)

    seq_lens = seq_lens_buffer[:num_reqs_after_padding]
    seq_lens.copy_(input_batch.seq_lens[req_start:req_stop])
    last = num_reqs_after_padding - 1
    seq_lens[last] -= (
        input_batch.query_start_loc[req_stop] - tok_stop
    ).clamp_(min=0)

    seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound[
        req_start:req_stop
    ].clone()
    truncated = max(0, int(input_batch.query_start_loc_np[req_stop]) - tok_stop)
    if truncated:
        seq_lens_cpu_upper_bound[-1] -= truncated

    dcp_local_seq_lens = input_batch.dcp_local_seq_lens
    if dcp_local_seq_lens is not None:
        dcp_local_seq_lens = dcp_local_seq_lens[req_start:req_stop]

    return replace(
        input_batch,
        req_ids=input_batch.req_ids[req_start : min(req_stop, input_batch.num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs_after_padding,
        idx_mapping=input_batch.idx_mapping[req_start:req_stop],
        idx_mapping_np=input_batch.idx_mapping_np[req_start:req_stop],
        num_scheduled_tokens=np.diff(query_start_loc_np)[:num_reqs],
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens_after_padding,
        query_start_loc=query_start_loc,
        query_start_loc_np=query_start_loc_np,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        dcp_local_seq_lens=dcp_local_seq_lens,
        num_computed_tokens_np=input_batch.num_computed_tokens_np[
            req_start:req_stop
        ],
        prefill_len_np=input_batch.prefill_len_np[req_start:req_stop],
        num_computed_prefill_tokens_np=input_batch.num_computed_prefill_tokens_np[
            req_start:req_stop
        ],
        is_prefilling_np=input_batch.is_prefilling_np[req_start:req_stop],
        max_seq_len_np=(
            None
            if input_batch.max_seq_len_np is None
            else input_batch.max_seq_len_np[req_start:req_stop]
        ),
        input_ids=input_batch.input_ids[tok_start:tok_stop],
        positions=input_batch.positions[tok_start:tok_stop],
        is_padding=input_batch.is_padding[tok_start:tok_stop],
        prompt_lens=(
            None
            if input_batch.prompt_lens is None
            else input_batch.prompt_lens[req_start:req_stop]
        ),
    )


def slice_model_inputs(
    model_inputs: dict[str, Any], token_slice: slice
) -> dict[str, Any]:
    """Narrow the model's per-token inputs to one microbatch."""

    sliced = dict(model_inputs)
    for key in ("input_ids", "inputs_embeds"):
        value = model_inputs.get(key)
        if value is not None:
            sliced[key] = value[token_slice]
    positions = model_inputs["positions"]
    sliced["positions"] = (
        positions[:, token_slice] if positions.ndim == 2 else positions[token_slice]
    )
    intermediate_tensors = model_inputs.get("intermediate_tensors")
    if intermediate_tensors is not None:
        sliced["intermediate_tensors"] = intermediate_tensors[token_slice]
    return sliced


def merge_ubatch_outputs(outputs: list[Any]) -> Any:
    """Reassemble the native ModelRunnerV2 output structure."""

    first = outputs[0]
    if isinstance(first, IntermediateTensors):
        return IntermediateTensors(
            {
                key: torch.cat([output.tensors[key] for output in outputs], dim=0)
                for key in first.tensors
            },
        )
    if isinstance(first, tuple):
        hidden_states = torch.cat([output[0] for output in outputs], dim=0)
        auxiliary = [
            torch.cat([output[1][index] for output in outputs], dim=0)
            for index in range(len(first[1]))
        ]
        return hidden_states, auxiliary
    return torch.cat(outputs, dim=0)


@contextmanager
def use_two_metadata_builders() -> Iterator[None]:
    """Create two builders during one AFD DBO KV-cache initialization."""

    original = AttentionGroup.create_metadata_builders

    def create_metadata_builders(
        group: AttentionGroup,
        vllm_config,
        device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ) -> None:
        del num_metadata_builders
        original(
            group,
            vllm_config,
            device,
            kernel_block_size,
            num_metadata_builders=2,
        )

    try:
        AttentionGroup.create_metadata_builders = create_metadata_builders
        yield
    finally:
        AttentionGroup.create_metadata_builders = original


def share_metadata_builder_workspaces(
    attn_groups: list[list[AttentionGroup]],
) -> None:
    """Share the backend workspace while retaining independent builders."""

    workspace = None
    for groups in attn_groups:
        for group in groups:
            for builder in group.metadata_builders:
                if workspace is None and hasattr(builder, "_get_workspace_buffer"):
                    workspace = builder._get_workspace_buffer()
                elif workspace is not None and hasattr(builder, "set_workspace_buffer"):
                    builder.set_workspace_buffer(workspace)


def prepare_attn_for_ubatch(
    model_state: ModelState,
    input_batch: InputBatch,
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: torch.Tensor,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    ubatch_index: int,
) -> dict[str, Any]:
    """Select one of the two builders without changing upstream signatures."""

    if ubatch_index == 0:
        return model_state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
        )

    swapped: list[AttentionGroup] = []
    for groups in attn_groups:
        for group in groups:
            group.metadata_builders[0], group.metadata_builders[ubatch_index] = (
                group.metadata_builders[ubatch_index],
                group.metadata_builders[0],
            )
            swapped.append(group)
    try:
        return model_state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
        )
    finally:
        for group in reversed(swapped):
            group.metadata_builders[0], group.metadata_builders[ubatch_index] = (
                group.metadata_builders[ubatch_index],
                group.metadata_builders[0],
            )
