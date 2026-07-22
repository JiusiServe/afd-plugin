# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Remaining vLLM-Ascend v0.19.1rc1 NPU ModelRunner adaptation."""

from __future__ import annotations

from vllm.sequence import IntermediateTensors
from vllm_ascend.utils import enable_sp

from afd_plugin.v1.worker.npu.ubatch_plan import get_ubatch_plan


class AscendDBOCompatV0191rc1:
    """One pinned seam not yet provided by the upstream NPU runner."""

    # Upstream source: vllm_ascend/worker/model_runner_v1.py.
    # Upstream ref: v0.19.1rc1 (da421afad7192dac64e39ae1d32305d57344f3cf).
    # Patch reason: upstream intermediate-tensor slicing is not uBatch aware.
    # Patch functionality: size the PP tensor from the active explicit plan.
    # Signature: matches vLLM-Ascend v0.19.1rc1.
    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        # ### PATCH START: Ascend uBatch intermediate-tensor slicing
        assert self.intermediate_tensors is not None
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        ubatch_slices = get_ubatch_plan(self).execution_slices

        if ubatch_slices is None:
            slice_len = (
                (num_tokens + tp_size - 1) // tp_size if enable_sp() else num_tokens
            )
        else:
            slice_len = (
                sum(
                    (ubatch_slice.num_tokens + tp_size - 1) // tp_size
                    for ubatch_slice in ubatch_slices
                )
                if enable_sp()
                else sum(ubatch_slice.num_tokens for ubatch_slice in ubatch_slices)
            )
            intermediate_tensor_size = next(
                iter(self.intermediate_tensors.tensors.values())
            ).size(0)
            if intermediate_tensor_size < slice_len:
                self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                    batch_size=slice_len,
                    dtype=self.dtype,
                    device=self.device,
                )

        if sync_self:
            assert intermediate_tensors is not None
            for key, value in intermediate_tensors.items():
                self.intermediate_tensors[key][:slice_len].copy_(
                    value[:slice_len],
                    non_blocking=True,
                )
        return IntermediateTensors(
            {key: value[:slice_len] for key, value in self.intermediate_tensors.items()}
        )
        # ### PATCH END: Ascend uBatch intermediate-tensor slicing


__all__ = ["AscendDBOCompatV0191rc1"]
