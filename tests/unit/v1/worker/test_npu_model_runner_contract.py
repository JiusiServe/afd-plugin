# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Phase 0 contract gates for the pinned NPU ModelRunner baseline."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from tools.compat.snapshot_npu_model_runner_contract import (
    ATTENTION_BUILDER_METHODS,
    ATTENTION_BUILDERS,
    RUNNER_METHODS,
    extract_external_imports,
    extract_method_contracts,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = REPO_ROOT / "tests/contracts/npu_model_runner_v0191rc1.json"
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/attention_model_runner.py"
EXPECTED_VLLM_REVISION = "b1388b1fbf5aaef47937fabe98931211684666a6"
EXPECTED_ASCEND_REVISION = "da421afad7192dac64e39ae1d32305d57344f3cf"


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _runtime_parameter_contract(method) -> list[dict[str, object]]:
    parameters = []
    for parameter in inspect.signature(method).parameters.values():
        parameters.append(
            {
                "kind": parameter.kind.name.lower(),
                "name": parameter.name,
                "required": parameter.default is inspect.Parameter.empty
                and parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                },
            }
        )
    return parameters


def _snapshot_parameter_contract(method: dict) -> list[dict[str, object]]:
    return [
        {
            "kind": parameter["kind"],
            "name": parameter["name"],
            "required": parameter["required"],
        }
        for parameter in method["parameters"]
    ]


def test_checked_in_contract_identifies_exact_phase0_baseline() -> None:
    contract = _load_contract()

    assert contract["schema_version"] == 1
    assert contract["baseline"] == {
        "vllm": {
            "ref": "v0.19.1",
            "revision": EXPECTED_VLLM_REVISION,
        },
        "vllm_ascend": {
            "ref": "v0.19.1rc1",
            "revision": EXPECTED_ASCEND_REVISION,
        },
    }
    assert set(contract["runner_methods"]) == set(RUNNER_METHODS)
    assert set(contract["attention_builders"]) == {
        class_name for class_name, _ in ATTENTION_BUILDERS
    }
    for methods in contract["attention_builders"].values():
        assert set(methods) == set(ATTENTION_BUILDER_METHODS)


def test_checked_in_contract_contains_upgrade_sensitive_schemas() -> None:
    contract = _load_contract()
    schemas = contract["schemas"]

    assert set(schemas) == {
        "AscendCommonAttentionMetadata",
        "BatchDescriptor",
        "DPMetadata",
        "ForwardContext",
        "UBatchSlice",
    }
    assert [field["name"] for field in schemas["UBatchSlice"]] == [
        "request_slice",
        "token_slice",
    ]
    assert "ubatch_slices" in {field["name"] for field in schemas["ForwardContext"]}
    assert {
        "vllm.forward_context",
        "vllm.v1.worker.ubatch_utils",
        "vllm_ascend.worker.model_runner_v1",
    } <= set(contract["external_imports"])
    assert {
        method_name
        for method_name, method in contract["runner_methods"].items()
        if not method["matches"]
    } == {
        "_determine_batch_execution_and_padding",
        "_prepare_inputs",
    }


def test_checked_in_contract_matches_current_plugin_runner_source() -> None:
    contract = _load_contract()
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    actual_methods = extract_method_contracts(
        runner_source,
        "AFDNPUAttentionModelRunner",
        RUNNER_METHODS,
    )

    assert actual_methods == {
        method_name: method["plugin"]
        for method_name, method in contract["runner_methods"].items()
    }
    assert extract_external_imports(runner_source) == contract["external_imports"]


@pytest.mark.vllm_runtime
def test_runtime_runner_signatures_match_phase0_snapshot() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    from afd_plugin.compat.npu import assert_vllm_ascend_version_supported
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        AFDNPUAttentionModelRunner,
    )

    assert_vllm_ascend_version_supported()

    contract = _load_contract()
    for method_name, expected in contract["runner_methods"].items():
        for class_name, runner_class, contract_name in (
            ("upstream", NPUModelRunner, "upstream"),
            ("plugin", AFDNPUAttentionModelRunner, "plugin"),
        ):
            method_contract = expected[contract_name]
            if method_contract is None:
                assert class_name == "plugin"
                assert method_name not in runner_class.__dict__
                method_contract = expected["upstream"]
            assert method_contract is not None
            method = getattr(runner_class, method_name)
            assert _runtime_parameter_contract(method) == (
                _snapshot_parameter_contract(method_contract)
            )
            assert inspect.iscoroutinefunction(method) is method_contract["async"]


@pytest.mark.vllm_runtime
def test_runtime_metadata_fields_match_phase0_snapshot() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_ascend")
    from vllm.forward_context import BatchDescriptor, DPMetadata, ForwardContext
    from vllm.v1.worker.ubatch_utils import UBatchSlice
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

    runtime_classes = {
        "AscendCommonAttentionMetadata": AscendCommonAttentionMetadata,
        "BatchDescriptor": BatchDescriptor,
        "DPMetadata": DPMetadata,
        "ForwardContext": ForwardContext,
        "UBatchSlice": UBatchSlice,
    }
    for class_name, expected_fields in _load_contract()["schemas"].items():
        runtime_fields = runtime_classes[class_name].__annotations__
        assert list(runtime_fields) == [field["name"] for field in expected_fields], (
            class_name
        )


@pytest.mark.vllm_runtime
def test_runtime_attention_builder_signatures_match_phase0_snapshot() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionMetadataBuilder,
    )
    from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder

    runtime_builders = {
        "AscendAttentionMetadataBuilder": AscendAttentionMetadataBuilder,
        "AscendMLAMetadataBuilder": AscendMLAMetadataBuilder,
    }
    for class_name, methods in _load_contract()["attention_builders"].items():
        runtime_class = runtime_builders[class_name]
        for method_name, expected in methods.items():
            assert expected is not None
            method = getattr(runtime_class, method_name)
            assert _runtime_parameter_contract(method) == (
                _snapshot_parameter_contract(expected)
            )
