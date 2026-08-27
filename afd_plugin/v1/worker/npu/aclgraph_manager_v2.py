# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Temporary MRV2 DBO FULL ACL graph manager for vLLM 0.26."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    set_forward_context,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.compilation.acl_graph import (
    get_graph_params,
    update_full_graph_params,
)
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager, ModelWithContext
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.utils import communicator_switch

from afd_plugin.compat.backports.vllm_v026_mrv2_dbo import (
    AFDBatchExecutionDescriptor,
    create_ubatch_slices,
)
from afd_plugin.v1.worker.npu.mla_graph import (
    merge_mla_graph_params,
    new_mla_graph_params,
    override_mla_graph_params,
)
from afd_plugin.v1.worker.npu.ubatch_runner_v2 import (
    AFDAscendUBatchRunnerV2,
    AFDAscendUBatchState,
)


@dataclass
class _AFDGraphEntry:
    graph: Any
    output: Any
    graph_params: tuple[Any, Any]
    workspace: torch.Tensor


@dataclass
class _PendingReplay:
    descriptor: AFDBatchExecutionDescriptor
    state: AFDAscendUBatchState


class AFDModelAclGraphManagerV2(ModelAclGraphManager):
    """Keep DBO graph descriptors and storage separate from upstream graphs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        model_runner: Any,
        ubatch_runner: AFDAscendUBatchRunnerV2,
        lora_capture_cases: list[int] | None = None,
    ) -> None:
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            model_runner,
            lora_capture_cases=lora_capture_cases,
        )
        self.ubatch_runner = ubatch_runner
        self._afd_twins = {
            desc: AFDBatchExecutionDescriptor(
                cg_mode=desc.cg_mode,
                num_tokens=desc.num_tokens,
                num_reqs=desc.num_reqs,
                uniform_token_count=desc.uniform_token_count,
                num_active_loras=desc.num_active_loras,
                num_ubatches=2,
            )
            for desc in self._capture_descs.get(CUDAGraphMode.FULL, [])
            if self._needs_ubatch_twin(desc)
        }
        self._afd_graphs: dict[AFDBatchExecutionDescriptor, _AFDGraphEntry] = {}
        self._afd_pending_replay: _PendingReplay | None = None

    def _needs_ubatch_twin(self, desc: BatchExecutionDescriptor) -> bool:
        if desc.num_tokens % 2 or desc.num_tokens < 2:
            return False
        parallel = self.vllm_config.parallel_config
        return any(
            check_ubatch_thresholds(parallel, desc.num_tokens, uniform_decode=value)
            for value in (True, False)
        )

    def dispatch_ubatches(
        self,
        base: BatchExecutionDescriptor,
        num_ubatches: int,
    ) -> BatchExecutionDescriptor:
        if num_ubatches != 2:
            raise RuntimeError("AFD NPU ModelRunnerV2 requires exactly two ubatches")
        twin = self._afd_twins.get(base)
        if twin is not None and twin in self._afd_graphs:
            return twin
        return AFDBatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=base.num_tokens,
            num_reqs=base.num_reqs,
            uniform_token_count=base.uniform_token_count,
            num_active_loras=base.num_active_loras,
            num_ubatches=2,
        )

    def stage_replay(
        self,
        descriptor: AFDBatchExecutionDescriptor,
        state: AFDAscendUBatchState,
    ) -> None:
        if descriptor not in self._afd_graphs:
            raise RuntimeError(f"No AFD DBO ACL graph for {descriptor}")
        if self._afd_pending_replay is not None:
            raise RuntimeError("An AFD DBO ACL graph replay is already pending")
        self._afd_pending_replay = _PendingReplay(descriptor, state)

    def clear_afd_graphs(self) -> None:
        """Release plugin-owned graphs and any staged replay state."""

        self._afd_pending_replay = None
        self._afd_graphs.clear()

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        super().capture(
            model,
            model_state,
            input_buffers,
            intermediate_tensors,
            block_tables,
            attn_groups,
            kv_cache_config,
            has_lora=has_lora,
            use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
            lora_capture_hook=lora_capture_hook,
            progress_bar_desc=progress_bar_desc,
        )
        if not self._afd_twins:
            return

        wrapped_model = ModelWithContext(model)
        try:
            with graph_capture(device=self.device), communicator_switch():
                for descriptor in self._afd_twins.values():
                    self._capture_afd_graph(
                        descriptor,
                        wrapped_model,
                        model_state,
                        input_buffers,
                        intermediate_tensors,
                        block_tables,
                    )
        except BaseException:
            self.clear_afd_graphs()
            raise

    def _capture_afd_graph(
        self,
        descriptor: AFDBatchExecutionDescriptor,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
    ) -> None:
        num_tokens = descriptor.num_tokens
        num_reqs = descriptor.num_reqs or min(num_tokens, self.max_num_reqs)
        stage_tokens = num_tokens // 2
        aggregate = get_graph_params()
        if aggregate is None or aggregate.workspaces.get(num_tokens) is None:
            raise RuntimeError(
                "MLA DBO FULL graph requires the upstream FIA workspace for "
                f"{num_tokens} tokens"
            )
        workspace = aggregate.workspaces[num_tokens]
        graph_params = (
            new_mla_graph_params(stage_tokens, workspace),
            new_mla_graph_params(stage_tokens, workspace),
        )
        input_batch = AscendInputBatch.make_dummy(
            num_reqs,
            num_tokens,
            input_buffers,
        )
        slices = create_ubatch_slices(input_batch, 2)
        model_inputs = {
            "input_ids": input_buffers.input_ids[:num_tokens],
            "positions": input_buffers.positions[:num_tokens],
            **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
        }
        if not self.is_first_pp_rank:
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None
            assert intermediate_tensors is not None
            model_inputs["intermediate_tensors"] = intermediate_tensors[:num_tokens]
        input_buffers.is_padding.fill_(True)
        dummy_tables = block_tables.get_dummy_block_tables(num_reqs)
        dummy_slots = block_tables.get_dummy_slot_mappings(num_tokens)

        warmup_state = self._prepare_capture_state(
            descriptor,
            input_batch,
            dummy_tables,
            dummy_slots,
            slices,
            None,
            is_warmup=True,
        )
        self.ubatch_runner.run(model, model_inputs, warmup_state)

        capture_state = self._prepare_capture_state(
            descriptor,
            input_batch,
            dummy_tables,
            dummy_slots,
            slices,
            graph_params,
            is_warmup=False,
        )
        finish = self.ubatch_runner.begin_capturable_run(
            model,
            model_inputs,
            capture_state,
            for_capture=True,
        )
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(
            graph,
            pool=self.pool,
            stream=self.ubatch_runner.capture_stream,
        ):
            output = finish()
        self._afd_graphs[descriptor] = _AFDGraphEntry(
            graph=graph,
            output=output,
            graph_params=graph_params,
            workspace=workspace,
        )

    def _prepare_capture_state(
        self,
        descriptor: AFDBatchExecutionDescriptor,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        slices,
        graph_params: tuple[Any, Any] | None,
        *,
        is_warmup: bool,
    ) -> AFDAscendUBatchState:
        runner = self.model_runner
        runner._is_warmup = is_warmup
        runner._afd_is_graph_capturing = not is_warmup
        runner._afd_pending_metadata = runner.build_afd_metadata(
            slices, descriptor.num_tokens
        )
        runner.send_dp_metadata(
            runner.build_capture_dp_metadata(descriptor.num_tokens), slices
        )
        runner._afd_suppress_metadata_send = True
        batch_descriptor = BatchDescriptor(
            num_tokens=descriptor.num_tokens,
            has_lora=False,
            num_active_loras=0,
        )
        counts = torch.full((self.dp_size,), descriptor.num_tokens, dtype=torch.int32)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=descriptor.num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=counts,
            batch_descriptor=batch_descriptor,
            ubatch_slices=slices,
            is_padding=input_batch.is_padding,
        ):
            return self.ubatch_runner.prepare(
                input_batch,
                block_tables,
                slot_mappings,
                slices,
                get_forward_context(),
                cg_mode=CUDAGraphMode.FULL,
                context_cg_mode=CUDAGraphMode.NONE,
                for_capture=True,
                mla_graph_params=graph_params,
            )

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> Any:
        if not isinstance(desc, AFDBatchExecutionDescriptor):
            return super().run_fullgraph(desc)
        pending = self._afd_pending_replay
        self._afd_pending_replay = None
        if pending is None or pending.descriptor != desc:
            raise RuntimeError(
                "AFD DBO ACL graph replay state is missing or mismatched"
            )
        entry = self._afd_graphs[desc]
        state = pending.state
        runner = self.model_runner
        runner._is_warmup = False
        runner._afd_is_graph_capturing = False
        runner._afd_pending_metadata = runner.build_afd_metadata(
            state.slices,
            sum(state.real_token_counts),
        )
        runner._afd_pending_metadata.tokens_unpadded_lens = state.real_token_counts
        runner._afd_suppress_metadata_send = True
        runner.send_dp_metadata(
            runner.build_capture_dp_metadata(desc.num_tokens), state.slices
        )

        assert self.update_stream is not None
        current_stream = torch.npu.current_stream()
        self.update_stream.wait_stream(current_stream)
        # This graph bypasses Ascend's ACLGraphWrapper, so preserve its FULL
        # replay fence before updating the captured FIA task-group handles.
        current_stream.synchronize()
        entry.graph.replay()
        stage_tokens = desc.num_tokens // 2
        merged_metadata, merged_params = merge_mla_graph_params(
            state.attn_metadata,
            entry.graph_params,
            stage_tokens,
        )
        counts = torch.full((self.dp_size,), desc.num_tokens, dtype=torch.int32)
        with set_forward_context(
            state.attn_metadata,
            self.vllm_config,
            num_tokens=desc.num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            num_tokens_across_dp=counts,
            batch_descriptor=None,
            slot_mapping=None,
        ):
            context = get_forward_context()
            with override_mla_graph_params(context, merged_metadata, merged_params):
                update_full_graph_params(
                    self.model_runner.attn_groups[0][0].backend,
                    self.update_stream,
                    context,
                    stage_tokens,
                    self.vllm_config,
                    self.model_runner.speculative_config,
                )
        return entry.output


__all__ = ["AFDModelAclGraphManagerV2"]
