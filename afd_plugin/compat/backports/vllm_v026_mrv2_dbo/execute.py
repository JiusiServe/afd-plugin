# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""vLLM 0.26 ModelRunnerV2 execute path with the eager DBO seams added."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    set_forward_context,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import get_uniform_token_count
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import ExecuteModelState
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

from .runtime import (
    AFDBatchExecutionDescriptor,
    create_ubatch_slices,
    dispatch_afd_dbo_and_sync_dp,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput


def execute_model_v026_eager_dbo(
    runner: Any,
    scheduler_output: SchedulerOutput,
    intermediate_tensors: IntermediateTensors | None = None,
    *,
    dummy_run: bool = False,
    skip_attn_for_dummy_run: bool = False,
    is_profile: bool = False,
) -> ModelRunnerOutput | IntermediateTensors | None:
    """Execute the supported plain-decoder subset with eager DBO."""

    if not dummy_run:
        runner.update_pp_decode_requests()
        runner.finish_requests(scheduler_output)
        runner.free_states(scheduler_output)
        runner.add_requests(scheduler_output)
        runner.update_requests(scheduler_output)
        runner.block_tables.apply_staged_writes()
        if scheduler_output.total_num_scheduled_tokens == 0:
            return runner.kv_connector.no_forward(scheduler_output)

    num_reqs = len(scheduler_output.num_scheduled_tokens)
    num_tokens = int(scheduler_output.total_num_scheduled_tokens)
    max_query_len = max(scheduler_output.num_scheduled_tokens.values())
    uniform_token_count = get_uniform_token_count(
        num_reqs,
        num_tokens,
        max_query_len,
    )
    batch_desc, num_tokens_across_dp = dispatch_afd_dbo_and_sync_dp(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        uniform_token_count=uniform_token_count,
        dp_size=runner.dp_size,
        dp_rank=runner.dp_rank,
        parallel_config=runner.parallel_config,
        decode_query_len=runner.decode_query_len,
        allow_ubatching=not skip_attn_for_dummy_run,
    )
    if batch_desc.num_tokens == 0:
        return runner.kv_connector.no_forward(scheduler_output)

    num_ubatches = (
        batch_desc.num_ubatches
        if isinstance(batch_desc, AFDBatchExecutionDescriptor)
        else 1
    )
    if not dummy_run:
        runner.input_buffers.is_padding[:num_tokens].fill_(False)
        runner.input_buffers.is_padding[
            num_tokens : batch_desc.num_tokens
        ].fill_(True)
        input_batch = runner.prepare_inputs(scheduler_output, batch_desc)
        block_tables, slot_mappings = runner.prepare_attn(input_batch)
        runner.model_state.preprocess_state(
            input_batch,
            block_tables,
            runner.kv_cache_config,
            runner.req_states.num_computed_tokens.gpu,
        )
    else:
        dummy_batch_cls = AscendInputBatch if num_ubatches > 1 else InputBatch
        input_batch = dummy_batch_cls.make_dummy(
            batch_desc.num_reqs or num_reqs,
            batch_desc.num_tokens,
            runner.input_buffers,
        )
        if not skip_attn_for_dummy_run:
            block_tables, slot_mappings = runner.prepare_dummy_attn(input_batch)
        else:
            block_tables = None
            slot_mappings = None

    attn_metadata = None
    slot_mappings_by_layer = None
    ubatch_slices = None
    if num_ubatches > 1:
        assert runner.ubatch_runner is not None
        assert block_tables is not None and slot_mappings is not None
        ubatch_slices = create_ubatch_slices(input_batch, num_ubatches)
    elif not (dummy_run and skip_attn_for_dummy_run):
        assert block_tables is not None and slot_mappings is not None
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings,
            runner.kv_cache_config,
        )
        attn_metadata = runner.model_state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            block_tables,
            slot_mappings,
            runner.attn_groups,
            runner.kv_cache_config,
        )

    model_inputs = {
        "input_ids": input_batch.input_ids,
        "positions": input_batch.positions,
        "inputs_embeds": None,
        "intermediate_tensors": None,
        **runner.model_state.prepare_inputs(input_batch, runner.req_states),
    }
    runner.eplb.prepare_forward(
        runner.model_config,
        input_batch.num_tokens,
        ubatch_slices,
    )

    if ubatch_slices is not None:
        batch_descriptor = BatchDescriptor(
            num_tokens=input_batch.num_tokens_after_padding,
            has_lora=False,
            num_active_loras=0,
        )
        with set_forward_context(
            None,
            runner.vllm_config,
            num_tokens=input_batch.num_tokens_after_padding,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=batch_descriptor,
            ubatch_slices=ubatch_slices,
            is_padding=input_batch.is_padding,
        ):
            parent_context = get_forward_context()
            afd_metadata = parent_context.additional_kwargs["afd_metadata"]
            afd_metadata.tokens_unpadded_lens = [
                max(
                    0,
                    min(int(stage.token_slice.stop), int(input_batch.num_tokens))
                    - int(stage.token_slice.start),
                )
                for stage in ubatch_slices
            ]
            ubatch_state = runner.ubatch_runner.prepare(
                input_batch,
                block_tables,
                slot_mappings,
                ubatch_slices,
                parent_context,
            )
            runner.kv_connector.pre_forward(scheduler_output)
            model_output = runner.ubatch_runner.run(
                runner.model,
                model_inputs,
                ubatch_state,
            )
    else:
        batch_descriptor = BatchDescriptor(
            num_tokens=input_batch.num_tokens_after_padding,
            has_lora=False,
            num_active_loras=0,
        )
        with set_forward_context(
            attn_metadata,
            runner.vllm_config,
            num_tokens=input_batch.num_tokens_after_padding,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=batch_descriptor,
            slot_mapping=slot_mappings_by_layer,
            is_padding=input_batch.is_padding,
        ):
            runner.kv_connector.pre_forward(scheduler_output)
            model_output = runner.model(**model_inputs)

    assert runner.is_last_pp_rank
    assert isinstance(model_output, torch.Tensor)
    runner.execute_model_state = ExecuteModelState(
        input_batch=input_batch,
        attn_metadata=attn_metadata,
        slot_mappings_by_layer=slot_mappings_by_layer,
        hidden_states=model_output,
        aux_hidden_states=None,
        finished_req_ids=scheduler_output.finished_req_ids,
    )
    return None


__all__ = ["execute_model_v026_eager_dbo"]
