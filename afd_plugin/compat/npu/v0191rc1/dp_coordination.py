# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pinned DP coordination seam for the v0.19.1rc1 NPU DBO backport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import BatchDescriptor
from vllm_ascend.ascend_forward_context import select_moe_comm_method
from vllm_ascend.utils import should_skip_allreduce_across_dp_group
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from afd_plugin.v1.worker.npu.ubatch_plan import (
    NativeDBOPlanner,
    UbatchDecision,
    ensure_ubatch_plan_scope,
)
from afd_plugin.v1.worker.npu.ubatch_utils import (
    check_enable_ubatch,
    maybe_create_ubatch_slices,
)

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        AFDNPUAttentionModelRunner,
    )


@dataclass(frozen=True)
class DPCoordinationResult:
    should_ubatch: bool
    max_tokens: int
    num_tokens_after_padding: torch.Tensor | None
    cudagraph_mode: CUDAGraphMode


class NPUDBOCoordinatorV0191rc1:
    """Preserve the pinned CPU-group collective while exposing a narrow API."""

    # Upstream source: vllm_ascend/worker/model_runner_v1.py.
    # Upstream ref: v0.19.1rc1 (da421afad7192dac64e39ae1d32305d57344f3cf).
    # Patch reason: AFD DBO also coordinates unpadded tokens and its decision.
    # Patch functionality: one packed CPU-group all-reduce per batch decision.
    # Signature: AFD coordinator; unpadded/uniform inputs are explicit additions.
    @staticmethod
    def coordinate(
        runner: AFDNPUAttentionModelRunner,
        *,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
        uniform_decode: bool,
        is_draft_model: bool,
        cudagraph_mode: CUDAGraphMode,
        allow_dp_padding: bool,
    ) -> DPCoordinationResult:
        # ### PATCH START: pinned AFD DBO DP coordination
        if runner.dp_size == 1:
            moe_comm_type = select_moe_comm_method(
                num_tokens_padded,
                runner.vllm_config,
                is_draft_model,
            )
            should_ubatch = check_enable_ubatch(
                num_tokens_unpadded,
                num_tokens_padded,
                uniform_decode=uniform_decode,
                vllm_config=runner.vllm_config,
                moe_comm_type=moe_comm_type,
            )
            return DPCoordinationResult(
                should_ubatch,
                num_tokens_padded,
                None,
                cudagraph_mode,
            )

        if runner.connector.control_plane is None:
            num_tokens_after_padding = torch.tensor(
                [num_tokens_padded] * runner.dp_size,
                device="cpu",
                dtype=torch.int32,
            )
            moe_comm_type = select_moe_comm_method(
                num_tokens_padded,
                runner.vllm_config,
                is_draft_model,
            )
            should_ubatch = check_enable_ubatch(
                num_tokens_unpadded,
                num_tokens_padded,
                uniform_decode=uniform_decode,
                vllm_config=runner.vllm_config,
                moe_comm_type=moe_comm_type,
            )
            return DPCoordinationResult(
                should_ubatch,
                num_tokens_padded,
                num_tokens_after_padding,
                cudagraph_mode,
            )

        parallel_config = runner.vllm_config.parallel_config
        can_skip_dp_sync = should_skip_allreduce_across_dp_group(
            runner.vllm_config,
            is_draft_model,
        )
        may_ubatch = bool(parallel_config.enable_dbo and parallel_config.use_ubatching)
        if can_skip_dp_sync and not may_ubatch:
            num_tokens_after_padding = torch.tensor(
                [num_tokens_padded] * runner.dp_size,
                device="cpu",
                dtype=torch.int32,
            )
            moe_comm_type = select_moe_comm_method(
                num_tokens_padded,
                runner.vllm_config,
                is_draft_model,
            )
            should_ubatch = check_enable_ubatch(
                num_tokens_unpadded,
                num_tokens_padded,
                uniform_decode=uniform_decode,
                vllm_config=runner.vllm_config,
                moe_comm_type=moe_comm_type,
            )
            return DPCoordinationResult(
                should_ubatch,
                num_tokens_padded,
                num_tokens_after_padding,
                cudagraph_mode,
            )

        packed_tensor = torch.zeros(
            3,
            runner.dp_size,
            device="cpu",
            dtype=torch.int32,
        )
        packed_tensor[0][runner.dp_rank] = num_tokens_unpadded
        packed_tensor[1][runner.dp_rank] = num_tokens_padded
        packed_tensor[2][runner.dp_rank] = cudagraph_mode.value
        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)

        num_tokens_unpadded_across_dp = packed_tensor[0, :]
        num_tokens_padded_across_dp = packed_tensor[1, :]
        max_tokens_across_dp = int(num_tokens_padded_across_dp.max().item())
        min_tokens_across_dp = int(num_tokens_unpadded_across_dp.min().item())
        synced_cudagraph_mode = CUDAGraphMode(int(packed_tensor[-1, :].min().item()))
        moe_comm_type = select_moe_comm_method(
            max_tokens_across_dp,
            runner.vllm_config,
            is_draft_model,
        )
        should_ubatch = check_enable_ubatch(
            min_tokens_across_dp,
            max_tokens_across_dp,
            uniform_decode=uniform_decode,
            vllm_config=runner.vllm_config,
            moe_comm_type=moe_comm_type,
        )
        if allow_dp_padding or is_draft_model or should_ubatch:
            num_tokens_after_padding = torch.tensor(
                [max_tokens_across_dp] * runner.dp_size,
                device="cpu",
                dtype=torch.int32,
            )
        else:
            num_tokens_after_padding = num_tokens_padded_across_dp.cpu()
        return DPCoordinationResult(
            should_ubatch,
            max_tokens_across_dp,
            num_tokens_after_padding,
            synced_cudagraph_mode,
        )
        # ### PATCH END: pinned AFD DBO DP coordination


class NPUDBOBatchDecisionV0191rc1:
    """Narrow hook joining the pinned upstream decision to an AFD plan."""

    # Upstream source: vllm_ascend/worker/model_runner_v1.py.
    # Patch reason: the upstream DP hook receives only the padded token count.
    # Patch functionality: add scoped unpadded/uniform inputs to the coordinator.
    # Signature: mirrors the upstream hook after the explicit runner parameter.
    @staticmethod
    def sync_metadata_across_dp(
        runner: AFDNPUAttentionModelRunner,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        allow_dp_padding: bool = False,
    ) -> tuple[int, torch.Tensor | None, CUDAGraphMode]:
        # ### PATCH START: scoped AFD DP decision inputs
        scope = runner._afd_ubatch_plan_scope
        if scope is None or scope.decision is None:
            raise RuntimeError(
                "NPU DBO DP coordination requires an active batch decision scope"
            )
        coordination = NPUDBOCoordinatorV0191rc1.coordinate(
            runner,
            num_tokens_unpadded=scope.decision.num_tokens_unpadded,
            num_tokens_padded=num_tokens,
            uniform_decode=scope.decision.uniform_decode,
            is_draft_model=is_draft_model,
            cudagraph_mode=cudagraph_mode,
            allow_dp_padding=allow_dp_padding,
        )
        scope.decision.coordination = coordination
        # ### PATCH END: scoped AFD DP decision inputs
        return (
            coordination.max_tokens,
            coordination.num_tokens_after_padding,
            coordination.cudagraph_mode,
        )

    # Upstream source: vllm_ascend/worker/model_runner_v1.py.
    # Patch reason: the pinned upstream decision always returns should_ubatch=False.
    # Patch functionality: retain upstream graph/padding and finalize an AFD plan.
    # Signature: mirrors the AFD runner hook after the explicit runner parameter.
    @staticmethod
    def determine_batch_execution_and_padding(
        runner: AFDNPUAttentionModelRunner,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        # ### PATCH START: pinned Ascend native DBO decision
        with ensure_ubatch_plan_scope(runner) as scope:
            decision = UbatchDecision(
                num_tokens_unpadded=num_tokens,
                uniform_decode=NativeDBOPlanner.uniform_decode(
                    runner,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    force_uniform_decode=force_uniform_decode,
                ),
            )
            scope.decision = decision
            try:
                result = NPUModelRunner._determine_batch_execution_and_padding(
                    runner,
                    num_tokens,
                    num_reqs,
                    num_scheduled_tokens_np,
                    max_num_scheduled_tokens,
                    use_cascade_attn,
                    False,
                    force_eager,
                    force_uniform_decode,
                    force_has_lora,
                    force_num_active_loras,
                    num_encoder_reqs,
                )
            finally:
                scope.decision = None

            cudagraph_mode, batch_descriptor, _, num_tokens_across_dp, stats = result
            plan = NativeDBOPlanner.finalize(
                runner,
                decision=decision,
                batch_descriptor=batch_descriptor,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_mode=cudagraph_mode,
                allow_microbatching=allow_microbatching,
            )
            if plan.should_ubatch:
                num_reqs_padded = batch_descriptor.num_reqs or num_reqs
                ubatch_slices, padded_ubatch_slices = maybe_create_ubatch_slices(
                    True,
                    num_scheduled_tokens_np,
                    int(batch_descriptor.num_tokens),
                    int(num_reqs_padded),
                    runner.vllm_config,
                )
                plan = plan.with_slices(
                    ubatch_slices,
                    padded_ubatch_slices,
                )
            scope.set_plan(plan)
            # ### PATCH END: pinned Ascend native DBO decision
            return (
                cudagraph_mode,
                batch_descriptor,
                plan.should_ubatch,
                num_tokens_across_dp,
                stats,
            )


__all__ = [
    "DPCoordinationResult",
    "NPUDBOBatchDecisionV0191rc1",
    "NPUDBOCoordinatorV0191rc1",
]
