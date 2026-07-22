# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Async MoE request-boundary uBatch planning and forward sidecar."""

from __future__ import annotations

from typing import Any

from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from afd_plugin.model_executor.models import ASYNC_MOE_UBATCH_METADATA_KEY
from afd_plugin.v1.worker.npu.ubatch_plan import (
    AsyncMoePlanner,
    ensure_ubatch_plan_scope,
)

logger = init_logger("afd_plugin.v1.worker.npu.attention_model_runner")


class AsyncMoeUbatch:
    """Unbound request-boundary planner and stage metadata sidecar."""

    # AFD source: pre-refactor NPU attention ModelRunner async MoE path.
    # Move reason: isolate request-boundary planning and its sidecar state.
    # Functionality: moved without changing split or metadata build ordering.
    # Signature: retained from the pre-refactor AFD implementation.
    def build_attention_metadata(
        self,
        values: dict[str, Any],
    ) -> Any:
        # ### PATCH START: async MoE request-boundary uBatch
        full_metadata = NPUModelRunner._build_attention_metadata(self, **values)
        self._afd_async_moe_ubatch_metadata = None
        self._afd_pending_metadata = self._build_afd_metadata(
            None,
            int(values.get("num_tokens", 0)),
        )

        num_scheduled_tokens_np = values.get("num_scheduled_tokens_np")
        if num_scheduled_tokens_np is None:
            return full_metadata

        with ensure_ubatch_plan_scope(self) as scope:
            plan = AsyncMoePlanner.plan(
                num_scheduled_tokens_np,
                num_ubatches=self.afd_async_extra_info.async_moe_num_ubatches,
            )
            scope.set_plan(plan)
        ubatch_slices = plan.ubatch_slices
        if ubatch_slices is None:
            return full_metadata

        logger.debug(
            "AFD NPU async MoE ubatch split; num_reqs=%s num_tokens=%s "
            "num_scheduled_tokens=%s request_slices=%s token_slices=%s "
            "stage_num_tokens=%s",
            len(num_scheduled_tokens_np),
            int(values.get("num_tokens", 0)),
            num_scheduled_tokens_np.tolist(),
            [
                (ubatch_slice.request_slice.start, ubatch_slice.request_slice.stop)
                for ubatch_slice in ubatch_slices
            ],
            [
                (ubatch_slice.token_slice.start, ubatch_slice.token_slice.stop)
                for ubatch_slice in ubatch_slices
            ],
            [int(ubatch_slice.num_tokens) for ubatch_slice in ubatch_slices],
        )

        stage_values = dict(values)
        stage_values["ubatch_slices"] = ubatch_slices
        stage_attn_metadata, _ = self._build_attention_metadata_with_ubatches(
            **stage_values,
        )
        self._afd_pending_metadata = self._build_afd_metadata(
            ubatch_slices,
            int(values.get("num_tokens", 0)),
        )
        self._afd_async_moe_ubatch_metadata = {
            "attn_metadata": stage_attn_metadata,
            "ubatch_slices": ubatch_slices,
        }
        return full_metadata
        # ### PATCH END: async MoE request-boundary uBatch

    # AFD source: pre-refactor NPU attention ModelRunner async MoE path.
    # Move reason: isolate request-boundary planning and its sidecar state.
    # Functionality: moved without changing split or metadata build ordering.
    # Signature: retained from the pre-refactor AFD implementation.
    def _install_async_moe_ubatch_metadata_on_forward_context(
        self,
        forward_context: ForwardContext,
    ) -> None:
        # ### PATCH START: async MoE request-boundary uBatch
        if self._afd_async_moe_ubatch_metadata is None:
            return
        if forward_context.additional_kwargs is None:
            forward_context.additional_kwargs = {}
        forward_context.additional_kwargs[ASYNC_MOE_UBATCH_METADATA_KEY] = (
            self._afd_async_moe_ubatch_metadata
        )
        # ### PATCH END: async MoE request-boundary uBatch


__all__ = ["AsyncMoeUbatch"]
