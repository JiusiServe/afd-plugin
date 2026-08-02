# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD-owned Ascend ubatch wrapper.

Provides the Ascend DBO wrapper used by AFD NPU runtimes while keeping the
implementation plugin-owned.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch_npu  # noqa: F401
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import (
    DPMetadata,
    get_forward_context,
    override_forward_context,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu_ubatch_wrapper import UbatchMetadata, UBatchWrapper
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.utils import enable_sp

from afd_plugin.v1.worker.npu.forward_context import (
    create_ascend_forward_context,
)
from afd_plugin.v1.worker.npu.ubatching import (
    AscendUBatchContext,
    make_ubatch_contexts,
)

AFD_NPU_NUM_UBATCHES = 2
_READY_BARRIER_PARTIES = AFD_NPU_NUM_UBATCHES + 1
AscendLastRankOutput = torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
AscendModelOutput = AscendLastRankOutput | IntermediateTensors


def _cat_ubatch_outputs(
    sorted_results: list[AscendLastRankOutput],
) -> AscendLastRankOutput:
    """Preserve the current Ascend model-output structure across ubatches.

    Upstream source: vLLM commit 68b0c3135,
    ``gpu_ubatch_wrapper._cat_ubatch_outputs``. Ascend auxiliary hidden states
    use ``tuple[Tensor, list[Tensor]]`` rather than upstream's tuple of tensors,
    so this plugin-owned wrapper concatenates that concrete nested contract.
    """
    assert sorted_results
    first_result = sorted_results[0]
    # ### PATCH START: Ascend auxiliary hidden-state output
    if isinstance(first_result, tuple):
        tuple_results = cast(
            list[tuple[torch.Tensor, list[torch.Tensor]]],
            sorted_results,
        )
        num_aux_outputs = len(first_result[1])
        assert all(len(result[1]) == num_aux_outputs for result in tuple_results)
        return (
            torch.cat([result[0] for result in tuple_results], dim=0),
            [
                torch.cat(
                    [result[1][index] for result in tuple_results],
                    dim=0,
                )
                for index in range(num_aux_outputs)
            ],
        )
    # ### PATCH END: Ascend auxiliary hidden-state output
    return torch.cat(cast(list[torch.Tensor], sorted_results), dim=0)


def _all_gather_ubatch_output(
    output: AscendLastRankOutput,
    pad_size: int,
) -> AscendLastRankOutput:
    if isinstance(output, tuple):
        hidden_states, aux_hidden_states = output
        gathered_hidden_states = _all_gather_ubatch_output(hidden_states, pad_size)
        assert isinstance(gathered_hidden_states, torch.Tensor)
        gathered_aux_hidden_states = [
            _all_gather_ubatch_output(aux_hidden_state, pad_size)
            for aux_hidden_state in aux_hidden_states
        ]
        assert all(
            isinstance(aux_hidden_state, torch.Tensor)
            for aux_hidden_state in gathered_aux_hidden_states
        )
        return gathered_hidden_states, cast(
            list[torch.Tensor],
            gathered_aux_hidden_states,
        )
    output = tensor_model_parallel_all_gather(output, 0)
    return output[:-pad_size, :] if pad_size > 0 else output


@dataclass
class AscendUbatchMetadata(UbatchMetadata):
    context: AscendUBatchContext
    input_ids: torch.Tensor | None


@dataclass
class AscendNPUGraphMetaData:
    aclgraph: torch.npu.NPUGraph
    ubatch_metadata: list[AscendUbatchMetadata]
    outputs: AscendModelOutput | None = None


class AscendUBatchWrapper(UBatchWrapper):
    """Ascend microbatch wrapper used only by AFD NPU runtimes."""

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.device,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.npu.Stream(device=device)
        assert self.vllm_config.parallel_config.num_ubatches == AFD_NPU_NUM_UBATCHES
        self.ready_barrier = threading.Barrier(_READY_BARRIER_PARTIES)
        self.cudagraphs: dict[int, AscendNPUGraphMetaData] = {}
        self.cudagraph_wrapper = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = ACLGraphWrapper(
                runnable,
                vllm_config,
                runtime_mode=runtime_mode,
            )
        self.device = device

    @property
    def graph_pool(self):
        if self.cudagraph_wrapper is not None:
            return self.cudagraph_wrapper.graph_pool
        return None

    def clear_graphs(self) -> None:
        self.cudagraphs.clear()
        if self.cudagraph_wrapper is not None:
            self.cudagraph_wrapper.concrete_aclgraph_entries.clear()

    def __getattr__(self, key: str):
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(
            f"Attribute {key} not found in AscendUBatchWrapper runnable."
        )

    def unwrap(self) -> Callable:
        return self.runnable

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if ubatch_slices is None:
            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                assert batch_descriptor is not None
                if batch_descriptor.num_tokens in self.cudagraphs:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE
            if cudagraph_runtime_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            assert self.cudagraph_wrapper is not None
            return self.cudagraph_wrapper(*args, **kwargs)

        assert len(ubatch_slices) == AFD_NPU_NUM_UBATCHES

        attn_metadata = forward_context.attn_metadata
        num_tokens = sum(ubatch_slice.num_tokens for ubatch_slice in ubatch_slices)
        input_ids = kwargs["input_ids"]
        positions = kwargs["positions"]
        intermediate_tensors = kwargs["intermediate_tensors"]
        inputs_embeds = kwargs["inputs_embeds"]
        compute_stream = torch.npu.current_stream()

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        ubatch_dp_metadata = []
        for ubatch_slice in ubatch_slices:
            if dp_size > 1:
                ubatch_num_tokens_across_dp = torch.tensor(
                    [ubatch_slice.num_tokens] * dp_size,
                    device="cpu",
                    dtype=torch.int32,
                )
                ubatch_dp_metadata.append(
                    DPMetadata.make(
                        self.vllm_config.parallel_config,
                        ubatch_slice.num_tokens,
                        ubatch_num_tokens_across_dp,
                    )
                )
            else:
                ubatch_dp_metadata.append(None)

        if (
            num_tokens not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices,
                attn_metadata,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
                torch.npu.Stream(device=torch.npu.current_device()),
                ubatch_dp_metadata,
                batch_descriptor,
                CUDAGraphMode.NONE,
            )
            return self._capture_ubatches(ubatch_metadata, self.runnable)
        if (
            num_tokens in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            cudagraph_metadata = self.cudagraphs[num_tokens]
            cudagraph_metadata.aclgraph.replay()
            get_forward_context().dbo_enabled = True
            assert cudagraph_metadata.outputs is not None
            return cudagraph_metadata.outputs

        ubatch_metadata = self._make_ubatch_metadata(
            ubatch_slices,
            attn_metadata,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            compute_stream,
            ubatch_dp_metadata,
            batch_descriptor,
            CUDAGraphMode.NONE,
        )
        return self._run_ubatches(ubatch_metadata, self.runnable)

    def _make_ubatch_metadata(
        self,
        ubatch_slices,
        attn_metadata,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
        compute_stream,
        dp_metadata,
        batch_descriptor,
        cudagraph_runtime_mode,
    ) -> list[AscendUbatchMetadata]:
        cur_forward_context = get_forward_context()
        forward_contexts = []
        for i, _ubatch_slice in enumerate(ubatch_slices):
            forward_contexts.append(
                create_ascend_forward_context(
                    cur_forward_context,
                    attn_metadata=attn_metadata[i]
                    if attn_metadata is not None
                    else None,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata[i],
                    ubatch_slices=ubatch_slices,
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    ubatch_num=i,
                    skip_compiled=cur_forward_context.skip_compiled,
                )
            )

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        metadata_list: list[AscendUbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            metadata_list.append(
                AscendUbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.num_tokens,
                )
            )
        return metadata_list

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
    ):
        sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
        sliced_positions = (
            positions[:, tokens_slice]
            if positions.ndim == 2
            else positions[tokens_slice]
        )
        sliced_inputs_embeds = (
            inputs_embeds[tokens_slice] if inputs_embeds is not None else None
        )

        if intermediate_tensors is not None and enable_sp():
            tp_size = get_tensor_model_parallel_world_size()
            start = (tokens_slice.start + tp_size - 1) // tp_size
            if start != 0:
                stop = (
                    start
                    + (tokens_slice.stop - tokens_slice.start + tp_size - 1) // tp_size
                )
            else:
                stop = (tokens_slice.stop + tp_size - 1) // tp_size
            tokens_slice = slice(start, stop)
        sliced_intermediate_tensors = (
            intermediate_tensors[tokens_slice]
            if intermediate_tensors is not None
            else None
        )
        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )

    def _merge_intermediate_tensors(self, intermediate_tensor_list):
        assert len(intermediate_tensor_list) == 2
        result = {}
        for key in intermediate_tensor_list[0].tensors:
            result[key] = torch.cat(
                [
                    intermediate_tensor_list[0].tensors[key],
                    intermediate_tensor_list[1].tensors[key],
                ],
                dim=0,
            )
        return IntermediateTensors(result)

    def _merge_outputs(
        self,
        sorted_results: list[AscendModelOutput],
        ubatch_metadata: list[AscendUbatchMetadata],
    ) -> AscendModelOutput:
        if not get_pp_group().is_last_rank:
            return self._merge_intermediate_tensors(
                cast(list[IntermediateTensors], sorted_results),
            )

        last_rank_results = cast(list[AscendLastRankOutput], sorted_results)
        ubatch_forward_context = ubatch_metadata[0].context.forward_context
        if ubatch_forward_context.flash_comm_v1_enabled:
            for i, result in enumerate(last_rank_results):
                pad_size = ubatch_metadata[i].context.forward_context.pad_size
                last_rank_results[i] = _all_gather_ubatch_output(result, pad_size)
        return _cat_ubatch_outputs(last_rank_results)

    @torch.inference_mode()
    def _run_ubatch_thread(self, results, model, ubatch_metadata):
        with ubatch_metadata.context:
            model_output = model(
                input_ids=ubatch_metadata.input_ids,
                positions=ubatch_metadata.positions,
                intermediate_tensors=ubatch_metadata.intermediate_tensors,
                inputs_embeds=ubatch_metadata.inputs_embeds,
            )
        results.append((ubatch_metadata.context.id, model_output))

    def _run_ubatches(
        self,
        ubatch_metadata: list[AscendUbatchMetadata],
        model,
    ) -> AscendModelOutput:
        results: list[tuple[int, AscendModelOutput]] = []
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=self._run_ubatch_thread,
                    args=(results, model, metadata),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()

        sorted_results = [value for _, value in sorted(results)]
        get_forward_context().dbo_enabled = True
        return self._merge_outputs(sorted_results, ubatch_metadata)

    def _capture_ubatches(
        self,
        ubatch_metadata: list[AscendUbatchMetadata],
        model,
    ) -> AscendModelOutput:
        results: list[tuple[int, AscendModelOutput]] = []
        compute_stream = ubatch_metadata[0].context.compute_stream
        num_tokens = sum(metadata.num_tokens for metadata in ubatch_metadata)

        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=self._run_ubatch_thread,
                    args=(results, model, metadata),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()

            cudagraph_metadata = AscendNPUGraphMetaData(
                aclgraph=torch.npu.NPUGraph(),
                ubatch_metadata=ubatch_metadata,
            )
            with torch.npu.graph(
                cudagraph_metadata.aclgraph,
                stream=compute_stream,
                pool=self.graph_pool,
            ):
                ubatch_metadata[0].context.cpu_wait_event.set()
                for thread in ubatch_threads:
                    thread.join()
                sorted_results = [value for _, value in sorted(results)]
                cudagraph_metadata.outputs = self._merge_outputs(
                    sorted_results,
                    ubatch_metadata,
                )
            self.cudagraphs[num_tokens] = cudagraph_metadata
        get_forward_context().dbo_enabled = True
        assert cudagraph_metadata.outputs is not None
        return cudagraph_metadata.outputs


__all__ = [
    "AscendNPUGraphMetaData",
    "AscendModelOutput",
    "AscendUBatchWrapper",
    "AscendUbatchMetadata",
]
