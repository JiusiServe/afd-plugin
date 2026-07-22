# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Phase 6 gates for upstream dummy-run and NPU graph capture convergence."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/attention_model_runner.py"
COMPAT_PATH = REPO_ROOT / "afd_plugin/compat/npu/v0191rc1/model_runner.py"
FANOUT_PATH = REPO_ROOT / "afd_plugin/compat/npu/v0191rc1/attention_metadata_fanout.py"
GRAPH_CAPTURE_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/graph_capture.py"
WRAPPER_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/npu_ubatch_wrapper.py"


class _CUDAGraphMode(IntEnum):
    NONE = 0
    FULL = 1


def _graph_capture_module(monkeypatch):
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.config": types.ModuleType("vllm.config"),
        "vllm.forward_context": types.ModuleType("vllm.forward_context"),
    }
    modules["vllm.config"].CUDAGraphMode = _CUDAGraphMode
    modules["vllm.forward_context"].BatchDescriptor = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_afd_phase6_graph_capture"
    spec = importlib.util.spec_from_file_location(module_name, GRAPH_CAPTURE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _CaptureRunner:
    def __init__(self) -> None:
        self.compilation_config = SimpleNamespace(cudagraph_num_of_warmups=1)
        self.connector = SimpleNamespace(control_plane=object())
        self._is_warmup = False
        self._afd_pending_metadata = "original"
        self._afd_suppress_metadata_send = False
        self._afd_is_graph_capturing = False
        self.dummy_calls = []
        self.metadata_sends = []

    def _dummy_run(self, num_tokens, **kwargs):
        self.dummy_calls.append(
            (
                num_tokens,
                kwargs["cudagraph_runtime_mode"],
                kwargs["allow_microbatching"],
                kwargs.get("is_graph_capturing", False),
                self._is_warmup,
            )
        )

    def _build_afd_metadata(self, slices, num_tokens):
        return (slices, num_tokens)

    def _build_capture_dp_metadata(self, num_tokens):
        return ("dp", num_tokens)

    def _send_dp_metadata(self, metadata, slices):
        self.metadata_sends.append((metadata, slices))


def test_graph_capture_separates_single_and_ubatch_keys(monkeypatch) -> None:
    module = _graph_capture_module(monkeypatch)
    runner = _CaptureRunner()
    desc = SimpleNamespace(num_tokens=16, uniform=True, num_active_loras=0)

    module.AFDNPUGraphCapture.warmup_and_capture(
        runner,
        desc,
        _CUDAGraphMode.FULL,
        allow_microbatching=True,
    )

    assert runner.dummy_calls == [
        (16, _CUDAGraphMode.NONE, False, False, True),
        (16, _CUDAGraphMode.FULL, False, True, False),
        (16, _CUDAGraphMode.NONE, True, False, True),
        (16, _CUDAGraphMode.FULL, True, True, False),
    ]
    assert runner.metadata_sends == [(("dp", 16), None)]
    assert runner._afd_pending_metadata == "original"
    assert runner._afd_suppress_metadata_send is False
    assert runner._afd_is_graph_capturing is False


def test_graph_capture_skips_send_without_connector_control_plane(
    monkeypatch,
) -> None:
    module = _graph_capture_module(monkeypatch)
    runner = _CaptureRunner()
    runner.connector.control_plane = None
    desc = SimpleNamespace(num_tokens=16, uniform=True, num_active_loras=0)

    module.AFDNPUGraphCapture.warmup_and_capture(
        runner,
        desc,
        _CUDAGraphMode.FULL,
    )

    assert runner.metadata_sends == []


def test_phase6_runner_delegates_dummy_and_removed_legacy_bodies() -> None:
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

    assert "return super()._dummy_run(" in runner_source
    assert "_dummy_run_with_ubatches" not in runner_source
    assert "_dummy_run_with_ubatches" not in compat_methods
    assert "build_attention_metadata_legacy" not in compat_methods
    assert len(COMPAT_PATH.read_text(encoding="utf-8").splitlines()) < 100


def test_phase6_graph_fanout_and_output_merge_contracts_are_present() -> None:
    fanout_source = FANOUT_PATH.read_text(encoding="utf-8")
    wrapper_source = WRAPPER_PATH.read_text(encoding="utf-8")

    assert "builder.build_for_cudagraph_capture(common)" in fanout_source
    assert "_capture_ubatches" in wrapper_source
    assert "IntermediateTensors" in wrapper_source
    assert "tensor_model_parallel_all_gather" in wrapper_source
    assert "get_forward_context().dbo_enabled = True" in wrapper_source
