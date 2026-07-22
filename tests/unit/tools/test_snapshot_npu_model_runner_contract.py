# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Tests for the source-only NPU ModelRunner contract snapshotter."""

from __future__ import annotations

from tools.compat.snapshot_npu_model_runner_contract import (
    extract_class_fields,
    extract_external_imports,
    extract_method_contracts,
)


def test_extract_method_contracts_preserves_parameter_contract() -> None:
    source = """
class Runner:
    async def execute(self, value: int, /, size=1, *items: str,
                      enabled: bool = True, **kwargs) -> list[int]:
        return []
"""

    contract = extract_method_contracts(source, "Runner", ("execute",))["execute"]

    assert contract is not None
    assert contract["async"] is True
    assert contract["return_annotation"] == "list[int]"
    assert [parameter["name"] for parameter in contract["parameters"]] == [
        "self",
        "value",
        "size",
        "items",
        "enabled",
        "kwargs",
    ]
    assert [parameter["kind"] for parameter in contract["parameters"]] == [
        "positional_only",
        "positional_only",
        "positional_or_keyword",
        "var_positional",
        "keyword_only",
        "var_keyword",
    ]
    assert [parameter["required"] for parameter in contract["parameters"]] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_extract_method_contracts_records_missing_inherited_method() -> None:
    source = """
class Runner:
    def local(self):
        return None
"""

    contracts = extract_method_contracts(
        source,
        "Runner",
        ("local", "inherited"),
    )

    assert contracts["local"] is not None
    assert contracts["inherited"] is None


def test_extract_class_fields_only_records_direct_annotations() -> None:
    source = """
class Metadata:
    required: int
    optional: str | None = None

    def method(self):
        local: int = 1
"""

    assert extract_class_fields(source, "Metadata") == [
        {"annotation": "int", "default": None, "name": "required"},
        {
            "annotation": "str | None",
            "default": "None",
            "name": "optional",
        },
    ]


def test_extract_external_imports_filters_to_runtime_dependencies() -> None:
    source = """
import json
import torch
import vllm
from torch_npu import npu
from vllm_ascend.worker import model_runner_v1
from afd_plugin.config import AFDConfig
"""

    assert extract_external_imports(source) == [
        "torch",
        "torch_npu",
        "vllm",
        "vllm_ascend.worker",
    ]
