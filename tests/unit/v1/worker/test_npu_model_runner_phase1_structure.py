# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static Phase 1 gates for the thin NPU attention ModelRunner."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

from tools.compat.snapshot_npu_model_runner_contract import (
    RUNNER_METHODS,
    extract_method_contracts,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/attention_model_runner.py"
COMPAT_PATH = REPO_ROOT / "afd_plugin/compat/npu/v0191rc1/model_runner.py"
METADATA_ADAPTER_PATH = (
    REPO_ROOT / "afd_plugin/v1/worker/npu/attention_metadata_adapter.py"
)
ASYNC_MOE_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/async_moe_ubatch.py"
CONTRACT_PATH = REPO_ROOT / "tests/contracts/npu_model_runner_v0191rc1.json"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _class_node(path: Path, class_name: str) -> ast.ClassDef:
    tree = ast.parse(_source(path))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def test_phase1_runner_is_thin_and_exposes_lifecycle_hooks() -> None:
    source = _source(RUNNER_PATH)
    runner = _class_node(RUNNER_PATH, "AFDNPUAttentionModelRunner")
    methods = {
        node.name
        for node in runner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert len(source.splitlines()) <= 650
    assert {
        "execute_model",
        "_model_forward",
        "_build_attention_metadata",
        "_dummy_run",
        "_warmup_and_capture",
        "load_model",
        "initialize_attn_backend",
        "shutdown",
    } <= methods
    assert "dist.all_reduce" not in source
    assert "set_ascend_forward_context" not in source
    assert "snapshot_pcp_manager_state" not in source


def test_phase1_runner_uses_explicit_component_boundaries() -> None:
    source = _source(RUNNER_PATH)
    metadata_adapter_source = _source(METADATA_ADAPTER_PATH)

    assert "AscendDBOCompatV0191rc1" in source
    assert "AFDNPUAttentionMetadataAdapter" in source
    assert "AsyncMoeUbatch" in source
    assert "AFDNPUControlPlane" not in metadata_adapter_source
    assert "self.connector.control_plane" in metadata_adapter_source
    assert COMPAT_PATH.is_file()
    assert METADATA_ADAPTER_PATH.is_file()
    assert ASYNC_MOE_PATH.is_file()
    assert "PCP" not in _source(ASYNC_MOE_PATH)


def test_phase1_public_hooks_match_pinned_upstream_contract() -> None:
    baseline = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    upstream = {
        name: contract["upstream"]
        for name, contract in baseline["runner_methods"].items()
    }
    plugin = extract_method_contracts(
        _source(RUNNER_PATH),
        "AFDNPUAttentionModelRunner",
        RUNNER_METHODS,
    )

    assert plugin["_prepare_inputs"] is None
    for method_name in set(RUNNER_METHODS) - {
        "_prepare_inputs",
        "_determine_batch_execution_and_padding",
    }:
        assert plugin[method_name] == upstream[method_name], method_name

    expected_determine = copy.deepcopy(
        upstream["_determine_batch_execution_and_padding"]
    )
    assert expected_determine is not None
    allow_microbatching = next(
        parameter
        for parameter in expected_determine["parameters"]
        if parameter["name"] == "allow_microbatching"
    )
    allow_microbatching["default"] = "True"
    expected_determine["signature"] = expected_determine["signature"].replace(
        "allow_microbatching: bool = False",
        "allow_microbatching: bool = True",
    )
    assert plugin["_determine_batch_execution_and_padding"] == expected_determine


def test_phase1_copied_adapter_has_provenance_and_patch_markers() -> None:
    source = _source(COMPAT_PATH)
    adapter = _class_node(COMPAT_PATH, "AscendDBOCompatV0191rc1")
    methods = [
        node
        for node in adapter.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert methods
    assert "v0.19.1rc1" in source
    assert source.count("# ### PATCH START:") >= len(methods)
    assert source.count("# ### PATCH END:") >= len(methods)
