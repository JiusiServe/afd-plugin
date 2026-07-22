# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Phase 7 gates for the frozen current-version NPU ModelRunner design."""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/attention_model_runner.py"
COMPAT_DIR = REPO_ROOT / "afd_plugin/compat/npu/v0191rc1"
CONTRACT_PATH = REPO_ROOT / "tests/contracts/npu_model_runner_v0191rc1.json"


def _class_methods(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}


def test_phase7_runner_is_thin_and_has_no_high_frequency_upstream_copy() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert len(source.splitlines()) <= 600
    assert "return super().execute_model(" in source
    assert "return super()._model_forward(" in source
    assert "return super()._dummy_run(" in source
    assert "def _prepare_inputs(" not in source
    assert "def _dummy_run_with_ubatches(" not in source
    assert "def build_attention_metadata_legacy(" not in source
    assert "run_model = partial(" not in source
    assert "_get_block_table_and_slot_mapping" not in source


def test_phase7_version_specific_model_runner_seams_are_in_compat() -> None:
    assert not (REPO_ROOT / "afd_plugin/v1/worker/npu/dp_coordination.py").exists()
    assert not (
        REPO_ROOT / "afd_plugin/v1/worker/npu/attention_metadata_fanout.py"
    ).exists()
    assert {
        "attention_metadata_fanout.py",
        "dp_coordination.py",
        "model_runner.py",
    } <= {path.name for path in COMPAT_DIR.glob("*.py")}
    assert not (COMPAT_DIR / "pcp_stage.py").exists()
    assert (REPO_ROOT / "afd_plugin/compat/npu/version.py").is_file()
    assert _class_methods(
        COMPAT_DIR / "model_runner.py",
        "AscendDBOCompatV0191rc1",
    ) == {"sync_and_slice_intermediate_tensors"}


def test_phase7_has_one_production_path_and_no_pcp_adapter() -> None:
    production_paths = [RUNNER_PATH, *COMPAT_DIR.glob("*.py")]
    production_source = "\n".join(
        path.read_text(encoding="utf-8") for path in production_paths
    ).lower()

    assert "metadata_legacy" not in production_source
    assert "legacy planner" not in production_source
    assert "test-only" not in production_source
    assert "pcpstage" not in production_source
    assert not (REPO_ROOT / "afd_plugin/v1/worker/npu/pcp_debug.py").exists()


def test_phase7_frozen_contract_records_only_intentional_runner_differences() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    mismatches = {
        method_name
        for method_name, method in contract["runner_methods"].items()
        if not method["matches"]
    }

    assert contract["baseline"]["vllm"]["ref"] == "v0.19.1"
    assert contract["baseline"]["vllm_ascend"]["ref"] == "v0.19.1rc1"
    assert mismatches == {
        "_determine_batch_execution_and_padding",
        "_prepare_inputs",
    }
    assert len(contract["external_imports"]) <= 13
    assert "vllm_ascend.attention.utils" not in contract["external_imports"]
