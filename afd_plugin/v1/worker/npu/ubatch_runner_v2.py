# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend eager and capture-time DBO runner for vLLM 0.26 ModelRunnerV2.

Upstream source: ``vllm/v1/worker/gpu/ubatch_utils.py`` from
``specture724/vllm`` commit ``626fee7831``. PATCH markers identify the Ascend
and AFD adaptations to that runner.
"""

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
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.compilation.acl_graph import GraphParams
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
from afd_plugin.v1.worker.npu.npu_ubatch_wrapper import (
    AscendModelOutput,
    _all_gather_ubatch_output,
)
from afd_plugin.v1.worker.npu.ubatching import (
    AscendUBatchContext,
    make_ubatch_contexts,
)

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

    # Upstream source: vllm/v1/worker/gpu/ubatch_utils.py,
    # UBatchRunner.__init__; specture724 commit 626fee7831.
    # Patch reason: the GPU runner owns CUDA streams and supports a variable
    # microbatch count; the AFD Ascend handoff protocol has exactly two stages.
    # Patch functionality: allocate Ascend capture state and stage-local input
    # buffers without CUDA SM-control state.
    # Signature: matches UBatchRunner.__init__ exactly.
    # Removal/upstream plan: replace this class with native Ascend MRV2 DBO.
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
        self.num_ubatches = self.parallel_config.num_ubatches
        # ### PATCH START: AFD NPU two-stage constraint
        if self.num_ubatches != AFD_NPU_MRV2_NUM_UBATCHES:
            raise RuntimeError("AFD NPU ModelRunnerV2 requires exactly two ubatches")
        # ### PATCH END: AFD NPU two-stage constraint
        self.device = device
        self.model_state = model_state
        self.attn_groups = attn_groups
        self.kv_cache_config = kv_cache_config
        self.ready_barrier = threading.Barrier(self.num_ubatches + 1)
        # ### PATCH START: Ascend capture stream
        self.capture_stream = torch.npu.Stream(device=device)
        # ### PATCH END: Ascend capture stream
        # ### PATCH START: Ascend stage-local input buffers
        self.query_start_loc_buffers = [
            torch.zeros(max_num_reqs + 2, dtype=torch.int32, device=device)
            for _ in range(self.num_ubatches)
        ]
        self.seq_lens_buffers = [
            torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
            for _ in range(self.num_ubatches)
        ]
        # ### PATCH END: Ascend stage-local input buffers

    # Upstream source: vllm/v1/worker/gpu/ubatch_utils.py,
    # UBatchRunner.prepare; specture724 commit 626fee7831.
    # Patch reason: Ascend requires additional InputBatch fields, its own
    # ForwardContext values, and per-stage MLA graph parameters.
    # Patch functionality: prepare the supplied two slices for eager execution,
    # graph warmup, or graph capture.
    # Signature: adds ``slices``, ``parent_context``, ``context_cg_mode``, and
    # ``mla_graph_params``; the return type is the concrete Ascend state.
    # Removal/upstream plan: use native Ascend UBatchRunner.prepare when added.
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
        mla_graph_params: tuple[GraphParams, GraphParams] | None = None,
    ) -> AFDAscendUBatchState:
        attn_metadata_list: list[dict[str, Any]] = []
        forward_contexts: list[ForwardContext] = []
        real_token_counts: list[int] = []
        if context_cg_mode is None:
            context_cg_mode = cg_mode
        dp_size = self.parallel_config.data_parallel_size
        for stage_index, stage in enumerate(slices):
            # ### PATCH START: Ascend input batch fields
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
            # ### PATCH END: Ascend input batch fields
            stage_slot_mappings = slot_mappings[:, stage.token_slice]
            stage_block_tables = tuple(
                block_table[stage.request_slice] for block_table in block_tables
            )
            # ### PATCH START: AFD v0.26 metadata builder selection
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
            # ### PATCH END: AFD v0.26 metadata builder selection
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
            # ### PATCH START: Ascend microbatch forward context
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
            # ### PATCH END: Ascend microbatch forward context
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
        stage: UBatchSlice,
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

    # Upstream source: vllm/v1/worker/gpu/ubatch_utils.py, UBatchRunner.run;
    # specture724 commit 626fee7831.
    # Patch reason: this adapter consumes the concrete Ascend ubatch state.
    # Patch functionality: start and finish one two-stage Ascend execution.
    # Signature: narrows the state and return types to the Ascend contract.
    # Removal/upstream plan: use native Ascend UBatchRunner.run when added.
    def run(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        ubatch_state: AFDAscendUBatchState,
    ) -> AscendModelOutput:
        return self.begin_capturable_run(model, model_inputs, ubatch_state)()

    # Upstream source: vllm/v1/worker/gpu/ubatch_utils.py,
    # UBatchRunner.begin_capturable_run; specture724 commit 626fee7831.
    # Patch reason: GPU stream/SM control must be replaced by the existing
    # Ascend two-stage contexts and FlashComm output handling.
    # Patch functionality: launch both stages and return their finisher.
    # Signature: narrows the state/output types; ``for_capture`` is unchanged.
    # Removal/upstream plan: use native Ascend capturable DBO when available.
    def begin_capturable_run(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        ubatch_state: AFDAscendUBatchState,
        for_capture: bool = False,
    ) -> Callable[[], AscendModelOutput]:
        """Start both stages outside capture and return a one-shot finisher."""

        forward_contexts = ubatch_state.forward_contexts
        if forward_contexts is None:
            raise RuntimeError("uBatch execution requires prepared forward contexts")
        # ### PATCH START: Ascend microbatch contexts
        compute_stream = (
            self.capture_stream if for_capture else torch.npu.current_stream()
        )
        contexts = make_ubatch_contexts(
            self.num_ubatches,
            compute_stream,
            forward_contexts,
            self.ready_barrier,
        )
        cancellation_event = contexts[0].cancellation_event
        # ### PATCH END: Ascend microbatch contexts
        outputs: dict[int, AscendModelOutput] = {}
        errors: dict[int, BaseException] = {}

        # ### PATCH START: Failed worker release
        def cancel_execution() -> None:
            cancellation_event.set()
            # Stages can block on different ring events, so cancellation must
            # wake every event before joining any worker.
            for context in contexts:
                context.cpu_wait_event.set()
            self.ready_barrier.abort()

        @torch.inference_mode()
        def run_stage(
            context: AscendUBatchContext,
            inputs: dict[str, Any],
        ) -> None:
            try:
                # ### PATCH START: Ascend worker device
                torch.npu.set_device(self.device)
                # ### PATCH END: Ascend worker device
                with context:
                    outputs[context.id] = model(**inputs)
            except BaseException as error:  # noqa: BLE001
                if not cancellation_event.is_set():
                    errors[context.id] = error
                cancel_execution()

        # ### PATCH END: Failed worker release

        stack = ExitStack()
        stack.enter_context(override_forward_context(None))
        threads: list[threading.Thread] = []

        # ### PATCH START: Failed execution cleanup
        def close_execution() -> None:
            try:
                for thread in threads:
                    thread.join()
            finally:
                stack.close()
                if self.ready_barrier.broken:
                    self.ready_barrier.reset()

        def raise_stage_error() -> None:
            if errors:
                failed_stage = min(errors)
                raise RuntimeError(
                    f"AFD NPU microbatch {failed_stage} failed",
                ) from errors[failed_stage]

        try:
            for context, stage in zip(contexts, ubatch_state.slices, strict=True):
                thread = threading.Thread(
                    target=run_stage,
                    args=(
                        context,
                        slice_model_inputs(model_inputs, stage.token_slice),
                    ),
                )
                thread.start()
                threads.append(thread)
            self.ready_barrier.wait()
        except BaseException:  # noqa: BLE001
            cancel_execution()
            close_execution()
            raise_stage_error()
            raise
        # ### PATCH END: Failed execution cleanup

        def finish() -> AscendModelOutput:
            # ### PATCH START: Failed execution cleanup
            contexts[0].cpu_wait_event.set()
            close_execution()
            raise_stage_error()
            # ### PATCH END: Failed execution cleanup
            ordered_outputs = [outputs[index] for index in range(self.num_ubatches)]
            # ### PATCH START: Ascend FlashComm output gathering
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
            # ### PATCH END: Ascend FlashComm output gathering
            return merge_ubatch_outputs(ordered_outputs)

        return finish


__all__ = [
    "AFD_NPU_MRV2_NUM_UBATCHES",
    "AFDAscendUBatchRunnerV2",
    "AFDAscendUBatchState",
]
