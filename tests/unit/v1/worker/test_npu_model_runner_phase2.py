# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Phase 2 gates for upstream delegation and ForwardContext cloning."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/attention_model_runner.py"
FORWARD_CONTEXT_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/forward_context.py"
WRAPPER_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/npu_ubatch_wrapper.py"


def test_phase2_model_forward_is_an_around_super_hook() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert "return super()._model_forward(" in source
    assert "run_model = partial(" not in source
    assert "self._update_full_graph_params_if_needed(" not in source
    assert "def _all_gather_hidden_states_and_aux(hidden_states):" in source
    assert "if bool(forward_context.dbo_enabled):" in source


def test_phase2_reuses_upstream_aclgraph_wrapper() -> None:
    source = WRAPPER_PATH.read_text(encoding="utf-8")

    assert "existing_aclgraph_wrapper" in source
    assert "existing_aclgraph_wrapper.unwrap()" in source
    assert "self.cudagraph_wrapper = existing_aclgraph_wrapper" in source


@pytest.fixture
def forward_context_module(monkeypatch):
    modules = {
        "torch": types.ModuleType("torch"),
        "vllm": types.ModuleType("vllm"),
        "vllm.config": types.ModuleType("vllm.config"),
        "vllm.distributed": types.ModuleType("vllm.distributed"),
        "vllm.forward_context": types.ModuleType("vllm.forward_context"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.ubatch_utils": types.ModuleType("vllm.v1.worker.ubatch_utils"),
        "vllm_ascend": types.ModuleType("vllm_ascend"),
        "vllm_ascend.ops": types.ModuleType("vllm_ascend.ops"),
        "vllm_ascend.ops.fused_moe": types.ModuleType("vllm_ascend.ops.fused_moe"),
        "vllm_ascend.ops.fused_moe.moe_comm_method": types.ModuleType(
            "vllm_ascend.ops.fused_moe.moe_comm_method"
        ),
        "afd_plugin.v1.worker.ubatch_wrapper": types.ModuleType(
            "afd_plugin.v1.worker.ubatch_wrapper"
        ),
    }
    modules["vllm.config"].CUDAGraphMode = SimpleNamespace(NONE="none")
    modules["vllm.config"].VllmConfig = object
    modules["vllm.distributed"].get_dp_group = lambda: SimpleNamespace(world_size=1)
    modules["vllm.distributed"].get_tensor_model_parallel_world_size = lambda: 1
    modules["vllm.forward_context"].BatchDescriptor = object
    modules["vllm.forward_context"].DPMetadata = object
    modules["vllm.forward_context"].ForwardContext = object
    modules["vllm.v1.worker.ubatch_utils"].UBatchSlices = list
    modules["vllm_ascend.ops.fused_moe.moe_comm_method"].get_moe_comm_method = (
        lambda value: f"method:{value}"
    )
    modules["afd_plugin.v1.worker.ubatch_wrapper"].build_ubatch_afd_metadata = (
        lambda metadata, _slices, stage: (metadata, stage)
    )
    modules["afd_plugin.v1.worker.ubatch_wrapper"].build_ubatch_additional_kwargs = (
        lambda values, metadata: {**values, "afd_metadata": metadata}
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_afd_phase2_forward_context"
    spec = importlib.util.spec_from_file_location(module_name, FORWARD_CONTEXT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_phase2_child_context_clones_parent_and_mutable_fields(
    forward_context_module,
) -> None:
    parent = SimpleNamespace(
        additional_kwargs={"afd_metadata": "parent", "platform": []},
        no_compile_layers={"old": object()},
        all_moe_layers={"layer": object()},
        moe_comm_type="mc2",
        in_profile_run=False,
        capturing=False,
        mmrs_fusion=False,
        flash_comm_v1_enabled=False,
        flashcomm_v2_enabled=False,
        is_first_layer=True,
        layer_idx=0,
        prefetch_mlp_gate_up_proj=False,
        prefetch_mlp_down_proj=False,
        model_instance=None,
        is_draft_model=False,
        is_draft_model_prefill=False,
        draft_attn_metadatas=["draft"],
        max_tokens_across_pcp=None,
        mc2_mask=None,
        future_scalar_field="preserved",
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"new": object()})
    )
    slices = [
        SimpleNamespace(num_tokens=2),
        SimpleNamespace(num_tokens=3),
    ]

    child = forward_context_module.create_ascend_forward_context(
        parent,
        attn_metadata="stage-attention",
        vllm_config=config,
        ubatch_slices=slices,
        ubatch_num=1,
    )

    assert child.future_scalar_field == "preserved"
    assert child.additional_kwargs is not parent.additional_kwargs
    assert child.all_moe_layers is not parent.all_moe_layers
    assert child.draft_attn_metadatas is not parent.draft_attn_metadatas
    assert child.attn_metadata == "stage-attention"
    assert child.ubatch_idx == 1
    assert child.num_tokens == 3
