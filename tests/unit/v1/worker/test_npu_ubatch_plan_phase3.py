# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Phase 3 gates for explicit NPU uBatch planning and DP coordination."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PLAN_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/ubatch_plan.py"
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/attention_model_runner.py"
COMPAT_PATH = REPO_ROOT / "afd_plugin/compat/npu/v0191rc1/model_runner.py"
COORDINATOR_PATH = REPO_ROOT / "afd_plugin/compat/npu/v0191rc1/dp_coordination.py"
ASYNC_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/async_moe_ubatch.py"
WRAPPER_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/npu_ubatch_wrapper.py"


class _CUDAGraphMode(IntEnum):
    NONE = 0
    FULL = 1


@pytest.fixture
def ubatch_plan_module(monkeypatch):
    modules = {
        "torch": types.ModuleType("torch"),
        "vllm": types.ModuleType("vllm"),
        "vllm.config": types.ModuleType("vllm.config"),
        "vllm.forward_context": types.ModuleType("vllm.forward_context"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.ubatch_utils": types.ModuleType("vllm.v1.worker.ubatch_utils"),
        "vllm_ascend": types.ModuleType("vllm_ascend"),
        "vllm_ascend.ascend_forward_context": types.ModuleType(
            "vllm_ascend.ascend_forward_context"
        ),
        "afd_plugin.v1.worker.npu.ubatch_utils": types.ModuleType(
            "afd_plugin.v1.worker.npu.ubatch_utils"
        ),
    }
    modules["torch"].Tensor = object
    modules["vllm.config"].CUDAGraphMode = _CUDAGraphMode
    modules["vllm.forward_context"].BatchDescriptor = object
    modules["vllm.forward_context"].ForwardContext = object
    modules["vllm.v1.worker.ubatch_utils"].UBatchSlices = list
    modules["vllm_ascend.ascend_forward_context"].select_moe_comm_method = (
        lambda *_args: "allgather"
    )
    modules["afd_plugin.v1.worker.npu.ubatch_utils"].check_enable_ubatch = (
        lambda unpadded, padded, **_kwargs: unpadded == padded and padded >= 8
    )
    modules[
        "afd_plugin.v1.worker.npu.ubatch_utils"
    ].create_request_boundary_ubatch_slices = lambda values, *, num_ubatches: (
        [SimpleNamespace(num_tokens=int(values.sum()), stages=num_ubatches)]
        if len(values) >= num_ubatches
        else None
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_afd_phase3_ubatch_plan"
    spec = importlib.util.spec_from_file_location(module_name, PLAN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_plan_scope_restores_previous_scope_after_exception(
    ubatch_plan_module,
) -> None:
    runner = SimpleNamespace(_afd_ubatch_plan_scope=None)
    outer = ubatch_plan_module.UbatchPlanScope(runner)

    with pytest.raises(ValueError, match="stop"), outer:
        outer.set_plan(
            ubatch_plan_module.UbatchPlan(
                mode=ubatch_plan_module.UbatchMode.NATIVE_DBO,
                should_ubatch=True,
            )
        )
        with ubatch_plan_module.UbatchPlanScope(runner):
            raise ValueError("stop")

    assert runner._afd_ubatch_plan_scope is None
    assert outer.plan.mode is ubatch_plan_module.UbatchMode.NONE


def test_native_and_async_plans_are_mutually_exclusive(ubatch_plan_module) -> None:
    runner = SimpleNamespace(_afd_ubatch_plan_scope=None)
    with ubatch_plan_module.UbatchPlanScope(runner) as scope:
        scope.set_plan(
            ubatch_plan_module.UbatchPlan(mode=ubatch_plan_module.UbatchMode.NATIVE_DBO)
        )
        with pytest.raises(RuntimeError, match="mutually exclusive"):
            scope.set_plan(
                ubatch_plan_module.UbatchPlan(
                    mode=ubatch_plan_module.UbatchMode.ASYNC_MOE
                )
            )


def test_async_planner_records_request_boundary_slices(ubatch_plan_module) -> None:
    plan = ubatch_plan_module.AsyncMoePlanner.plan(
        np.asarray([2, 3], dtype=np.int32),
        num_ubatches=2,
    )

    assert plan.mode is ubatch_plan_module.UbatchMode.ASYNC_MOE
    assert plan.should_ubatch is True
    assert plan.num_tokens_unpadded == 5
    assert plan.ubatch_slices[0].stages == 2


def test_phase3_production_path_has_one_plan_and_no_legacy_decision_copy() -> None:
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    compat_tree = ast.parse(COMPAT_PATH.read_text(encoding="utf-8"))
    compat = next(
        node
        for node in compat_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendDBOCompatV0191rc1"
    )
    compat_methods = {
        node.name for node in compat.body if isinstance(node, ast.FunctionDef)
    }
    coordinator_source = COORDINATOR_PATH.read_text(encoding="utf-8")

    assert "with UbatchPlanScope(self):" in runner_source
    assert "_sync_metadata_across_dp" not in compat_methods
    assert "_determine_batch_execution_and_padding" not in compat_methods
    assert coordinator_source.count("dist.all_reduce(") == 1
    assert "group=get_dp_group().cpu_group" in coordinator_source
    assert "AsyncMoePlanner.plan(" in ASYNC_PATH.read_text(encoding="utf-8")
    assert "get_forward_context_ubatch_plan(" in WRAPPER_PATH.read_text(
        encoding="utf-8"
    )
