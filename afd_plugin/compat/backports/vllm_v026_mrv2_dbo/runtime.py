# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Runtime pieces missing from vLLM 0.26 ModelRunnerV2 DBO.

The behavior is adapted from ``specture724/vllm`` branch
``feat/v2/dbo-fullcg`` at ``626fee7831``. The target ABI is vLLM
``568afb3a13``. Keep this module self-contained so it can be deleted once the
pinned vLLM release provides native ModelRunnerV2 DBO.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

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
    is_last_ubatch_empty,
    maybe_create_ubatch_slices,
)
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.aclgraph_manager_v2 import (
        AFDModelAclGraphManagerV2,
    )


# ### PATCH START: AFD v0.26 DBO descriptor
@dataclass(frozen=True)
class AFDBatchExecutionDescriptor(BatchExecutionDescriptor):
    """v0.26 batch descriptor extended only with the DBO execution count."""

    num_ubatches: int = 1


# ### PATCH END: AFD v0.26 DBO descriptor


# Upstream source: ``vllm/v1/worker/gpu/dp_utils.py`` from
# ``specture724/vllm`` commit ``626fee7831``.
# Patch reason: pinned vLLM's descriptor and graph-manager dispatch do not carry
# ModelRunnerV2 microbatch state.
# Patch functionality: preserve upstream DP-wide threshold selection while
# routing DBO graph selection through the plugin-owned Ascend manager.
# Signature: standalone backport helper; graph-manager and eager-control
# parameters replace the newer upstream descriptor inputs.
# Removal/upstream plan: delete this helper with the v0.26 compatibility layer.
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
    cudagraph_manager: AFDModelAclGraphManagerV2 | None = None,
    need_eager: bool = False,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """Select DBO and FULL graph execution consistently across DP ranks.

    The minimum rank token count controls the threshold, while all ranks run
    the maximum token count when DBO is selected. This matches the final
    upstream inference rule and avoids a separate per-rank DBO vote.
    """

    # ### PATCH START: AFD v0.26 graph dispatch adapter
    def dispatch(
        tokens: int,
        ubatches: int,
        uniform_tokens: int | None = uniform_token_count,
        force_eager: bool = False,
    ) -> BatchExecutionDescriptor:
        if force_eager or need_eager or cudagraph_manager is None:
            base = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=tokens,
                num_reqs=num_reqs,
            )
        else:
            base = cudagraph_manager.dispatch(
                num_reqs,
                tokens,
                uniform_tokens,
                num_active_loras=0,
            )

        if ubatches == 1:
            return base
        if base.cg_mode == CUDAGraphMode.NONE:
            return AFDBatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=tokens,
                num_reqs=num_reqs,
                uniform_token_count=uniform_tokens,
                num_active_loras=base.num_active_loras,
                num_ubatches=ubatches,
            )

        assert cudagraph_manager is not None
        return cudagraph_manager.dispatch_ubatches(base, ubatches)

    # ### PATCH END: AFD v0.26 graph dispatch adapter

    if dp_size == 1:
        return dispatch(num_tokens, 1), None

    # ### PATCH START: AFD DBO graph-mode synchronization
    desired_single = dispatch(num_tokens, 1)
    tensor = torch.zeros(4, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = uniform_token_count or 0
    tensor[2][dp_rank] = int(allow_ubatching)
    tensor[3][dp_rank] = int(desired_single.cg_mode.value)
    # ### PATCH END: AFD DBO graph-mode synchronization
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

    uniform_decode = bool(torch.all(tensor[1] == int(decode_query_len)).item())
    # ### PATCH START: AFD v0.26 synchronized graph descriptor
    synced_uniform_token_count: int | None = int(tensor[1][0].item())
    if (
        synced_uniform_token_count == 0
        or not torch.all(tensor[1] == synced_uniform_token_count).item()
    ):
        synced_uniform_token_count = None
    synced_mode = CUDAGraphMode(int(tensor[3].min().item()))
    # ### PATCH END: AFD v0.26 synchronized graph descriptor
    should_ubatch = (
        bool(torch.all(tensor[2] == 1).item())
        and int(num_tokens_across_dp.max().item()) >= int(parallel_config.num_ubatches)
        and check_ubatch_thresholds(
            parallel_config,
            int(num_tokens_across_dp.min().item()),
            uniform_decode=uniform_decode,
        )
    )
    if not should_ubatch:
        if synced_mode == CUDAGraphMode.NONE:
            return (
                BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                ),
                num_tokens_across_dp,
            )
        padded_tokens = int(num_tokens_across_dp.max().item())
        synced = dispatch(padded_tokens, 1, synced_uniform_token_count)
        num_tokens_across_dp.fill_(synced.num_tokens)
        return synced, num_tokens_across_dp

    # ### PATCH START: AFD DBO graph dispatch and empty-stage fallback
    padded_tokens = int(num_tokens_across_dp.max().item())
    num_ubatches = int(parallel_config.num_ubatches)
    ubatch_desc = dispatch(
        padded_tokens,
        num_ubatches,
        synced_uniform_token_count,
        force_eager=synced_mode == CUDAGraphMode.NONE,
    )
    # Graph dispatch can round the DP maximum upward. Check the final shape so
    # every rank has real work in its last microbatch.
    if not is_last_ubatch_empty(
        int(num_tokens_across_dp.min().item()),
        ubatch_desc.num_tokens,
        num_ubatches,
    ):
        num_tokens_across_dp.fill_(ubatch_desc.num_tokens)
        return ubatch_desc, num_tokens_across_dp

    synced = dispatch(
        padded_tokens,
        1,
        synced_uniform_token_count,
        force_eager=synced_mode == CUDAGraphMode.NONE,
    )
    num_tokens_across_dp.fill_(synced.num_tokens)
    # ### PATCH END: AFD DBO graph dispatch and empty-stage fallback
    return synced, num_tokens_across_dp


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
    seq_lens[last] -= (input_batch.query_start_loc[req_stop] - tok_stop).clamp_(min=0)

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
        num_computed_tokens_np=input_batch.num_computed_tokens_np[req_start:req_stop],
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
    # ### PATCH START: Ascend auxiliary hidden-state output
    if isinstance(first, tuple):
        hidden_states = torch.cat([output[0] for output in outputs], dim=0)
        auxiliary = [
            torch.cat([output[1][index] for output in outputs], dim=0)
            for index in range(len(first[1]))
        ]
        return hidden_states, auxiliary
    # ### PATCH END: Ascend auxiliary hidden-state output
    return torch.cat(outputs, dim=0)


@contextmanager
def use_two_metadata_builders() -> Iterator[None]:
    """Create two builders during one AFD DBO KV-cache initialization."""

    original = AttentionGroup.create_metadata_builders

    # Upstream source: vllm/v1/worker/utils.py,
    # AttentionGroup.create_metadata_builders; commit 568afb3a13.
    # Patch reason: pinned vLLM initializes one metadata builder for MRV2.
    # Patch functionality: force exactly two builders only during AFD DBO
    # KV-cache initialization.
    # Signature: matches AttentionGroup.create_metadata_builders exactly.
    # Removal/upstream plan: remove this replacement when native MRV2 DBO
    # initializes one metadata builder per microbatch.
    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ):
        # ### PATCH START: AFD two metadata builders
        del num_metadata_builders
        original(
            self,
            vllm_config,
            device,
            kernel_block_size,
            num_metadata_builders=2,
        )
        # ### PATCH END: AFD two metadata builders

    try:
        AttentionGroup.create_metadata_builders = create_metadata_builders
        yield
    finally:
        AttentionGroup.create_metadata_builders = original


# Upstream source: vllm/v1/worker/gpu/model_states/interface.py,
# ModelState.prepare_attn; commit 568afb3a13.
# Patch reason: pinned vLLM has one active metadata builder and no ubatch index.
# Patch functionality: select the builder owned by the requested microbatch
# while invoking the native attention-preparation path unchanged.
# Signature: standalone adapter; ``ubatch_index`` is the only added input.
# Removal/upstream plan: call native prepare_attn directly when it accepts an
# ubatch index.
def prepare_attn_for_ubatch(
    model_state: ModelState,
    input_batch: InputBatch,
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: torch.Tensor,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    ubatch_index: int,
    cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    for_capture: bool = False,
) -> dict[str, Any]:
    """Select one of the two builders without changing upstream signatures."""

    if ubatch_index == 0:
        return model_state.prepare_attn(
            input_batch,
            cg_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
        )

    # ### PATCH START: AFD per-microbatch metadata builder
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
            cg_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
        )
    finally:
        for group in reversed(swapped):
            group.metadata_builders[0], group.metadata_builders[ubatch_index] = (
                group.metadata_builders[ubatch_index],
                group.metadata_builders[0],
            )
    # ### PATCH END: AFD per-microbatch metadata builder
