# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Scoped graph-manager injection for the temporary MRV2 DBO backport."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm_ascend.worker.v2 import model_runner as ascend_model_runner

from afd_plugin.compat.backports.vllm_v026_mrv2_dbo import (
    share_metadata_builder_workspaces,
)
from afd_plugin.v1.worker.npu.aclgraph_manager_v2 import (
    AFDModelAclGraphManagerV2,
)
from afd_plugin.v1.worker.npu.ubatch_runner_v2 import AFDAscendUBatchRunnerV2


@contextmanager
def use_afd_mrv2_dbo_graph_manager(model_runner) -> Iterator[None]:
    """Replace only Ascend's initialization-scoped manager wrapper."""

    original_wrapper = ascend_model_runner.graph_manager_wrapper

    @contextmanager
    def graph_manager_wrapper(runner) -> Iterator[None]:
        original_manager = vllm_model_runner.ModelCudaGraphManager

        def factory(
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            lora_capture_cases: list[int] | None = None,
        ) -> AFDModelAclGraphManagerV2:
            share_metadata_builder_workspaces(runner.attn_groups)
            ubatch_runner = AFDAscendUBatchRunnerV2(
                vllm_config,
                device,
                runner.model_state,
                runner.attn_groups,
                runner.kv_cache_config,
                runner.max_num_reqs,
            )
            runner.ubatch_runner = ubatch_runner
            return AFDModelAclGraphManagerV2(
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                runner,
                ubatch_runner,
                lora_capture_cases=lora_capture_cases,
            )

        try:
            vllm_model_runner.ModelCudaGraphManager = factory
            yield
        finally:
            vllm_model_runner.ModelCudaGraphManager = original_manager

    try:
        ascend_model_runner.graph_manager_wrapper = graph_manager_wrapper
        yield
    finally:
        ascend_model_runner.graph_manager_wrapper = original_wrapper


__all__ = ["use_afd_mrv2_dbo_graph_manager"]
