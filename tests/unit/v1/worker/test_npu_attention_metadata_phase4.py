# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Phase 4 gates for eager attention metadata builder fanout."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FANOUT_PATH = REPO_ROOT / "afd_plugin/compat/npu/v0191rc1/attention_metadata_fanout.py"
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/attention_model_runner.py"


class _SupportedBuilder:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.calls = []

    def build(self, *, common_prefix_len, common_attn_metadata, **_kwargs):
        self.calls.append((common_prefix_len, common_attn_metadata))
        if self.fail:
            raise RuntimeError("builder failed")
        return SimpleNamespace(
            builder=self.name,
            common=common_attn_metadata,
            mm_prefix_range=None,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )


class _AttentionGroup:
    def __init__(self, builders) -> None:
        self.metadata_builders = builders
        self.layer_names = ["model.layers.0.self_attn"]

    def get_metadata_builder(self, stage):
        return self.metadata_builders[stage]


@pytest.fixture
def fanout_module(monkeypatch):
    class _NPUModelRunner:
        @staticmethod
        def _build_attention_metadata(runner, *_args):
            metadata = {}
            common = SimpleNamespace(future_invariant="preserved")
            for_cudagraph_capture = _args[8]
            for groups in runner.attn_groups:
                for group in groups:
                    builder = group.get_metadata_builder(0)
                    value = (
                        builder.build_for_cudagraph_capture(common)
                        if for_cudagraph_capture
                        else builder.build(
                            common_prefix_len=7,
                            common_attn_metadata=common,
                        )
                    )
                    value.mm_prefix_range = {"image": (1, 2)}
                    for layer_name in group.layer_names:
                        metadata[layer_name] = value
            return metadata, "spec-common"

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.attention": types.ModuleType("vllm.v1.attention"),
        "vllm.v1.attention.backends": types.ModuleType("vllm.v1.attention.backends"),
        "vllm.v1.attention.backends.utils": types.ModuleType(
            "vllm.v1.attention.backends.utils"
        ),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.ubatch_utils": types.ModuleType("vllm.v1.worker.ubatch_utils"),
        "vllm_ascend": types.ModuleType("vllm_ascend"),
        "vllm_ascend.attention": types.ModuleType("vllm_ascend.attention"),
        "vllm_ascend.attention.attention_v1": types.ModuleType(
            "vllm_ascend.attention.attention_v1"
        ),
        "vllm_ascend.attention.mla_v1": types.ModuleType(
            "vllm_ascend.attention.mla_v1"
        ),
        "vllm_ascend.worker": types.ModuleType("vllm_ascend.worker"),
        "vllm_ascend.worker.model_runner_v1": types.ModuleType(
            "vllm_ascend.worker.model_runner_v1"
        ),
        "afd_plugin.v1.worker.npu.ubatch_utils": types.ModuleType(
            "afd_plugin.v1.worker.npu.ubatch_utils"
        ),
    }
    modules["vllm.v1.attention.backends.utils"].CommonAttentionMetadata = object
    modules["vllm.v1.worker.ubatch_utils"].UBatchSlices = list
    modules[
        "vllm_ascend.attention.attention_v1"
    ].AscendAttentionMetadataBuilder = _SupportedBuilder
    modules["vllm_ascend.attention.mla_v1"].AscendMLAMetadataBuilder = _SupportedBuilder
    modules["vllm_ascend.worker.model_runner_v1"].NPUModelRunner = _NPUModelRunner
    modules["afd_plugin.v1.worker.npu.ubatch_utils"].split_attn_metadata = (
        lambda _slices, common, _max_tokens: [
            SimpleNamespace(stage=0, future_invariant=common.future_invariant),
            SimpleNamespace(stage=1, future_invariant=common.future_invariant),
        ]
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_afd_phase4_attention_fanout"
    spec = importlib.util.spec_from_file_location(module_name, FANOUT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _runner(builders):
    return SimpleNamespace(attn_groups=[[_AttentionGroup(builders)]])


def test_fanout_reuses_stage_builders_and_restores_interception(fanout_module) -> None:
    builders = [_SupportedBuilder("stage-0"), _SupportedBuilder("stage-1")]
    runner = _runner(builders)
    original_builder = runner.attn_groups[0][0].metadata_builders[0]

    metadata, spec_common = fanout_module.AttentionMetadataFanoutV0191rc1.build(
        runner,
        num_tokens=5,
        num_reqs=2,
        max_query_len=3,
        ubatch_slices=[object(), object()],
    )

    assert runner.attn_groups[0][0].metadata_builders[0] is original_builder
    assert [stage["model.layers.0.self_attn"].builder for stage in metadata] == [
        "stage-0",
        "stage-1",
    ]
    assert [len(builder.calls) for builder in builders] == [1, 1]
    assert all(
        stage["model.layers.0.self_attn"].common.future_invariant == "preserved"
        for stage in metadata
    )
    assert all(
        stage["model.layers.0.self_attn"].mm_prefix_range == {"image": (1, 2)}
        for stage in metadata
    )
    assert spec_common == "spec-common"


def test_fanout_restores_builder_when_a_stage_build_fails(fanout_module) -> None:
    builders = [
        _SupportedBuilder("stage-0"),
        _SupportedBuilder("stage-1", fail=True),
    ]
    runner = _runner(builders)
    original_builder = runner.attn_groups[0][0].metadata_builders[0]

    with pytest.raises(RuntimeError, match="builder failed"):
        fanout_module.AttentionMetadataFanoutV0191rc1.build(
            runner,
            num_tokens=5,
            num_reqs=2,
            max_query_len=3,
            ubatch_slices=[object(), object()],
        )

    assert runner.attn_groups[0][0].metadata_builders[0] is original_builder


def test_fanout_uses_stage_builders_during_graph_capture(fanout_module) -> None:
    builders = [_SupportedBuilder("stage-0"), _SupportedBuilder("stage-1")]
    runner = _runner(builders)

    metadata, _ = fanout_module.AttentionMetadataFanoutV0191rc1.build(
        runner,
        num_tokens=8,
        num_reqs=2,
        max_query_len=1,
        ubatch_slices=[object(), object()],
        for_cudagraph_capture=True,
    )

    assert [stage["model.layers.0.self_attn"].builder for stage in metadata] == [
        "stage-0",
        "stage-1",
    ]


def test_phase4_runner_uses_fanout_for_eager_execution() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert "return AttentionMetadataFanoutV0191rc1.build(" in source
    assert "build_attention_metadata_legacy" not in source
