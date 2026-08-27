# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend eager and capture-time DBO runner for vLLM 0.26 ModelRunnerV2."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
import torch_npu  # noqa: F401
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    DPMetadata,
    ForwardContext,
    override_forward_context,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

from afd_plugin.compat.backports.vllm_v026_mrv2_dbo import (
    prepare_attn_for_ubatch,
    slice_input_batch,
    slice_model_inputs,
)
from afd_plugin.compat.backports.vllm_v026_mrv2_dbo.runtime import (
    merge_ubatch_outputs,
)
from afd_plugin.v1.worker.npu.forward_context import create_ascend_forward_context
from afd_plugin.v1.worker.npu.npu_ubatch_wrapper import _all_gather_ubatch_output
from afd_plugin.v1.worker.npu.ubatching import make_ubatch_contexts

AFD_NPU_MRV2_NUM_UBATCHES = 2


@dataclass
class AFDAscendUBatchState:
    """Prepared inputs for one eager or FULL graph DBO step."""

    slices: UBatchSlices
    attn_metadata: list[dict[str, Any]]
    forward_contexts: list[ForwardContext] | None
    real_token_counts: list[int]


class AFDAscendUBatchRunnerV2:
    """Run exactly two ModelRunnerV2 microbatches on Ascend."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        model_state: ModelState,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_num_reqs: int,
    ) -> None:
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        self.num_ubatches = int(self.parallel_config.num_ubatches)
        if self.num_ubatches != AFD_NPU_MRV2_NUM_UBATCHES:
            raise RuntimeError("AFD NPU ModelRunnerV2 requires exactly two ubatches")
        self.device = device
        self.model_state = model_state
        self.attn_groups = attn_groups
        self.kv_cache_config = kv_cache_config
        self.ready_barrier = threading.Barrier(self.num_ubatches + 1)
        self.capture_stream = torch.npu.Stream(device=device)
        self.query_start_loc_buffers = [
            torch.zeros(max_num_reqs + 2, dtype=torch.int32, device=device)
            for _ in range(self.num_ubatches)
        ]
        self.seq_lens_buffers = [
            torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
            for _ in range(self.num_ubatches)
        ]

    def prepare(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        slices: UBatchSlices,
        parent_context: ForwardContext | None,
        *,
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        context_cg_mode: CUDAGraphMode | None = None,
        for_capture: bool = False,
        mla_graph_params: tuple[Any, Any] | None = None,
    ) -> AFDAscendUBatchState:
        attn_metadata_list: list[dict[str, Any]] = []
        forward_contexts: list[ForwardContext] = []
        real_token_counts: list[int] = []
        if context_cg_mode is None:
            context_cg_mode = cg_mode
        dp_size = int(self.parallel_config.data_parallel_size)
        for stage_index, stage in enumerate(slices):
            child_batch = cast(
                AscendInputBatch,
                slice_input_batch(
                    input_batch,
                    stage,
                    self.query_start_loc_buffers[stage_index],
                    self.seq_lens_buffers[stage_index],
                ),
            )
            child_batch = self._with_ascend_fields(input_batch, child_batch, stage)
            stage_slot_mappings = slot_mappings[:, stage.token_slice]
            stage_block_tables = tuple(
                block_table[stage.request_slice] for block_table in block_tables
            )
            attn_metadata = prepare_attn_for_ubatch(
                self.model_state,
                child_batch,
                stage_block_tables,
                stage_slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
                stage_index,
                cg_mode=cg_mode,
                for_capture=for_capture,
            )
            attn_metadata_list.append(attn_metadata)
            real_token_counts.append(int(child_batch.num_tokens))
            if parent_context is None:
                continue
            stage_tokens = int(stage.num_tokens)
            counts = torch.full(
                (dp_size,), stage_tokens, dtype=torch.int32, device="cpu"
            )
            dp_metadata = DPMetadata.make(
                self.parallel_config,
                stage_tokens,
                counts,
            )
            context = create_ascend_forward_context(
                parent_context,
                attn_metadata,
                self.vllm_config,
                slices,
                ubatch_num=stage_index,
                dp_metadata=dp_metadata,
                cudagraph_runtime_mode=context_cg_mode,
                skip_compiled=parent_context.skip_compiled,
                mla_graph_params=(
                    None if mla_graph_params is None else mla_graph_params[stage_index]
                ),
            )
            context.slot_mapping = build_slot_mappings_by_layer(
                stage_slot_mappings,
                self.kv_cache_config,
            )
            forward_contexts.append(context)

        return AFDAscendUBatchState(
            slices=slices,
            attn_metadata=attn_metadata_list,
            forward_contexts=forward_contexts or None,
            real_token_counts=real_token_counts,
        )

    @staticmethod
    def _with_ascend_fields(
        parent_batch: AscendInputBatch,
        child_batch: AscendInputBatch,
        stage,
    ) -> AscendInputBatch:
        req_start = int(stage.request_slice.start)
        req_stop = int(stage.request_slice.stop)
        tok_stop = int(stage.token_slice.stop)
        seq_lens_np = parent_batch.seq_lens_np[req_start:req_stop].copy()
        truncated = max(0, int(parent_batch.query_start_loc_np[req_stop]) - tok_stop)
        if truncated:
            seq_lens_np[-1] -= truncated
        return replace(
            child_batch,
            seq_lens_np=seq_lens_np,
            attn_state=parent_batch.attn_state,
        )

    def run(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        state: AFDAscendUBatchState,
    ) -> Any:
        return self.begin_capturable_run(model, model_inputs, state)()

    def begin_capturable_run(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        state: AFDAscendUBatchState,
        *,
        for_capture: bool = False,
    ) -> Callable[[], Any]:
        """Start both stages outside capture and return a one-shot finisher."""

        forward_contexts = state.forward_contexts
        if forward_contexts is None:
            raise RuntimeError("uBatch execution requires prepared forward contexts")
        compute_stream = (
            self.capture_stream if for_capture else torch.npu.current_stream()
        )
        contexts = make_ubatch_contexts(
            self.num_ubatches,
            compute_stream,
            forward_contexts,
            self.ready_barrier,
        )
        outputs: dict[int, Any] = {}
        errors: dict[int, BaseException] = {}

        @torch.inference_mode()
        def run_stage(context, inputs: dict[str, Any]) -> None:
            try:
                torch.npu.set_device(self.device)
                with context:
                    outputs[context.id] = model(**inputs)
            except BaseException as error:  # noqa: BLE001
                errors[context.id] = error

        stack = ExitStack()
        stack.enter_context(override_forward_context(None))
        threads = []
        for context, stage in zip(contexts, state.slices, strict=True):
            thread = threading.Thread(
                target=run_stage,
                args=(context, slice_model_inputs(model_inputs, stage.token_slice)),
            )
            threads.append(thread)
            thread.start()
        self.ready_barrier.wait()
        finished = False

        def finish() -> Any:
            nonlocal finished
            if finished:
                raise RuntimeError("uBatch finisher may only be called once")
            finished = True
            try:
                contexts[0].cpu_wait_event.set()
                for thread in threads:
                    thread.join()
            finally:
                stack.close()

            if errors:
                failed_stage = min(errors)
                raise RuntimeError(
                    f"AFD NPU microbatch {failed_stage} failed",
                ) from errors[failed_stage]
            ordered_outputs = [outputs[index] for index in range(self.num_ubatches)]
            if forward_contexts[0].additional_kwargs["flash_comm_v1_enabled"]:
                ordered_outputs = [
                    _all_gather_ubatch_output(
                        output,
                        context.additional_kwargs["pad_size"],
                    )
                    for output, context in zip(
                        ordered_outputs,
                        forward_contexts,
                        strict=True,
                    )
                ]
            return merge_ubatch_outputs(ordered_outputs)

        return finish


__all__ = [
    "AFD_NPU_MRV2_NUM_UBATCHES",
    "AFDAscendUBatchRunnerV2",
    "AFDAscendUBatchState",
]
