# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit, instance-scoped NPU uBatch execution plans."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, ForwardContext
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm_ascend.ascend_forward_context import select_moe_comm_method

from afd_plugin.v1.worker.npu.ubatch_utils import (
    check_enable_ubatch,
    create_request_boundary_ubatch_slices,
)

if TYPE_CHECKING:
    from afd_plugin.compat.npu.v0191rc1.dp_coordination import (
        DPCoordinationResult,
    )
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        AFDNPUAttentionModelRunner,
    )

UBATCH_PLAN_FORWARD_CONTEXT_KEY = "afd_npu_ubatch_plan"


class UbatchMode(str, Enum):
    NONE = "none"
    NATIVE_DBO = "native_dbo"
    ASYNC_MOE = "async_moe"


@dataclass(frozen=True)
class UbatchPlan:
    mode: UbatchMode = UbatchMode.NONE
    should_ubatch: bool = False
    num_tokens_unpadded: int = 0
    num_tokens_padded: int = 0
    uniform_decode: bool = False
    ubatch_slices: UBatchSlices | None = None
    padded_ubatch_slices: UBatchSlices | None = None
    num_tokens_across_dp: torch.Tensor | None = None
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE

    @property
    def execution_slices(self) -> UBatchSlices | None:
        if (
            self.cudagraph_mode is CUDAGraphMode.FULL
            and self.padded_ubatch_slices is not None
        ):
            return self.padded_ubatch_slices
        return self.ubatch_slices

    def with_slices(
        self,
        ubatch_slices: UBatchSlices | None,
        padded_ubatch_slices: UBatchSlices | None = None,
    ) -> UbatchPlan:
        return replace(
            self,
            ubatch_slices=ubatch_slices,
            padded_ubatch_slices=padded_ubatch_slices,
        )


@dataclass
class UbatchDecision:
    num_tokens_unpadded: int
    uniform_decode: bool
    coordination: DPCoordinationResult | None = None


class UbatchPlanScope:
    """Per-runner scope; always clears the active plan on exit."""

    def __init__(self, runner: AFDNPUAttentionModelRunner) -> None:
        self.runner = runner
        self.plan = UbatchPlan()
        self.decision: UbatchDecision | None = None
        self._previous: UbatchPlanScope | None = None

    def __enter__(self) -> UbatchPlanScope:
        self._previous = self.runner._afd_ubatch_plan_scope
        self.runner._afd_ubatch_plan_scope = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.decision = None
        self.plan = UbatchPlan()
        self.runner._afd_ubatch_plan_scope = self._previous
        return False

    def set_plan(self, plan: UbatchPlan) -> None:
        if (
            self.plan.mode is not UbatchMode.NONE
            and plan.mode is not UbatchMode.NONE
            and self.plan.mode is not plan.mode
        ):
            raise RuntimeError(
                "native DBO and async MoE uBatch plans are mutually exclusive"
            )
        self.plan = plan


@contextmanager
def ensure_ubatch_plan_scope(
    runner: AFDNPUAttentionModelRunner,
) -> Iterator[UbatchPlanScope]:
    current = runner._afd_ubatch_plan_scope
    if current is not None:
        yield current
        return
    with UbatchPlanScope(runner) as scope:
        yield scope


def get_ubatch_plan(runner: AFDNPUAttentionModelRunner) -> UbatchPlan:
    scope = runner._afd_ubatch_plan_scope
    return UbatchPlan() if scope is None else scope.plan


def install_ubatch_plan_on_forward_context(
    runner: AFDNPUAttentionModelRunner,
    forward_context: ForwardContext,
) -> UbatchPlan:
    plan = get_ubatch_plan(runner)
    if forward_context.additional_kwargs is None:
        forward_context.additional_kwargs = {}
    forward_context.additional_kwargs[UBATCH_PLAN_FORWARD_CONTEXT_KEY] = plan
    if plan.mode is UbatchMode.NATIVE_DBO:
        forward_context.ubatch_slices = plan.execution_slices
    return plan


def get_forward_context_ubatch_plan(forward_context: ForwardContext) -> UbatchPlan:
    additional_kwargs = forward_context.additional_kwargs or {}
    return additional_kwargs.get(
        UBATCH_PLAN_FORWARD_CONTEXT_KEY,
        UbatchPlan(),
    )


class NativeDBOPlanner:
    """Map pinned Ascend inputs/results to one explicit native DBO plan."""

    @staticmethod
    def uniform_decode(
        runner: AFDNPUAttentionModelRunner,
        *,
        num_tokens: int,
        num_reqs: int,
        max_num_scheduled_tokens: int,
        force_uniform_decode: bool | None,
    ) -> bool:
        if force_uniform_decode is not None:
            return force_uniform_decode
        is_all_decode = np.all(
            runner.input_batch.num_computed_tokens_cpu[:num_reqs] > 0
        )
        return bool(
            (is_all_decode if runner.speculative_config else True)
            and max_num_scheduled_tokens == runner.uniform_decode_query_len
            and num_tokens == max_num_scheduled_tokens * num_reqs
        )

    @staticmethod
    def finalize(
        runner: AFDNPUAttentionModelRunner,
        *,
        decision: UbatchDecision,
        batch_descriptor: BatchDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_mode: CUDAGraphMode,
        allow_microbatching: bool,
    ) -> UbatchPlan:
        coordination = decision.coordination
        if coordination is not None:
            should_ubatch = coordination.should_ubatch
        else:
            padded_tokens = int(batch_descriptor.num_tokens)
            moe_comm_type = select_moe_comm_method(
                padded_tokens,
                runner.vllm_config,
            )
            should_ubatch = check_enable_ubatch(
                decision.num_tokens_unpadded,
                padded_tokens,
                uniform_decode=decision.uniform_decode,
                vllm_config=runner.vllm_config,
                moe_comm_type=moe_comm_type,
            )
        should_ubatch = bool(should_ubatch and allow_microbatching)
        return UbatchPlan(
            mode=(UbatchMode.NATIVE_DBO if should_ubatch else UbatchMode.NONE),
            should_ubatch=should_ubatch,
            num_tokens_unpadded=decision.num_tokens_unpadded,
            num_tokens_padded=int(batch_descriptor.num_tokens),
            uniform_decode=decision.uniform_decode,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_mode=cudagraph_mode,
        )


class AsyncMoePlanner:
    """Build the request-boundary plan used by async MoE."""

    @staticmethod
    def plan(
        num_scheduled_tokens: np.ndarray,
        *,
        num_ubatches: int,
    ) -> UbatchPlan:
        ubatch_slices = create_request_boundary_ubatch_slices(
            num_scheduled_tokens,
            num_ubatches=num_ubatches,
        )
        if ubatch_slices is None:
            return UbatchPlan(
                num_tokens_unpadded=int(num_scheduled_tokens.sum()),
                num_tokens_padded=int(num_scheduled_tokens.sum()),
            )
        total_tokens = int(num_scheduled_tokens.sum())
        return UbatchPlan(
            mode=UbatchMode.ASYNC_MOE,
            should_ubatch=True,
            num_tokens_unpadded=total_tokens,
            num_tokens_padded=total_tokens,
            ubatch_slices=ubatch_slices,
        )


__all__ = [
    "AsyncMoePlanner",
    "ensure_ubatch_plan_scope",
    "get_forward_context_ubatch_plan",
    "get_ubatch_plan",
    "install_ubatch_plan_on_forward_context",
    "NativeDBOPlanner",
    "UBATCH_PLAN_FORWARD_CONTEXT_KEY",
    "UbatchDecision",
    "UbatchMode",
    "UbatchPlan",
    "UbatchPlanScope",
]
