# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Scoped graph-manager injection for the temporary MRV2 DBO backport.

Upstream source: ``vllm_ascend/worker/v2/model_runner.py`` at commit
``d543ccee0``, function ``graph_manager_wrapper``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm_ascend.worker.v2 import model_runner as ascend_model_runner

from afd_plugin.v1.worker.npu.aclgraph_manager_v2 import (
    AFDModelAclGraphManagerV2,
)
from afd_plugin.v1.worker.npu.ubatch_runner_v2 import AFDAscendUBatchRunnerV2


# Patch reason: vLLM-Ascend's initialization wrapper always constructs the
# native single-batch ACL graph manager.
# Patch functionality: replace only that initialization scope with the AFD DBO
# manager and restore the upstream wrapper afterwards.
# Signature: plugin-owned context manager; ``model_runner`` is the AFD runner
# whose initialization is being scoped.
# Removal/upstream plan: delete this wrapper when vLLM-Ascend accepts a native
# ModelRunnerV2 DBO graph manager/runner pair.
@contextmanager
def use_afd_mrv2_dbo_graph_manager(model_runner) -> Iterator[None]:
    """Replace only Ascend's initialization-scoped manager wrapper."""

    original_wrapper = ascend_model_runner.graph_manager_wrapper

    # Patch reason: the upstream wrapper factory has no DBO runner dependency.
    # Patch functionality: construct the AFD two-stage runner and graph manager
    # while retaining upstream's temporary ModelCudaGraphManager substitution.
    # Signature: matches vLLM-Ascend graph_manager_wrapper exactly.
    @contextmanager
    def graph_manager_wrapper(model_runner):
        original_manager = vllm_model_runner.ModelCudaGraphManager

        # Upstream source: vllm_ascend/worker/v2/model_runner.py,
        # graph_manager_wrapper.factory; commit d543ccee0.
        # Patch reason: the native factory cannot receive an ubatch runner.
        # Patch functionality: construct the plugin-owned runner and manager.
        # Signature: matches the native nested factory exactly.
        # Removal/upstream plan: delete with this scoped wrapper.
        def factory(
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            lora_capture_cases: list[int] | None = None,
        ):
            # ### PATCH START: AFD MRV2 DBO graph manager
            ubatch_runner = AFDAscendUBatchRunnerV2(
                vllm_config,
                device,
                model_runner.model_state,
                model_runner.attn_groups,
                model_runner.kv_cache_config,
                model_runner.max_num_reqs,
            )
            model_runner.ubatch_runner = ubatch_runner
            return AFDModelAclGraphManagerV2(
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                model_runner,
                ubatch_runner,
                lora_capture_cases=lora_capture_cases,
            )
            # ### PATCH END: AFD MRV2 DBO graph manager

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
