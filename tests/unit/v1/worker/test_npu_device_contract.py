# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import ast
from pathlib import Path


def _method_parameters(source_path: Path, class_name: str, method_name: str) -> list[str]:
    module = ast.parse(source_path.read_text())
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return [
        argument.arg
        for argument in (
            method.args.posonlyargs + method.args.args + method.args.kwonlyargs
        )
    ]


def test_npu_runner_overrides_match_ascend_gdn_dummy_run_contract():
    afd_source = Path("afd_plugin/v1/worker/npu/attention_model_runner.py")
    ascend_source = Path(
        "../vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
    )

    for method_name in ("_dummy_run", "_build_attention_metadata"):
        assert _method_parameters(
            afd_source,
            "AFDNPUAttentionModelRunner",
            method_name,
        ) == _method_parameters(ascend_source, "NPUModelRunner", method_name)


def test_npu_connectors_use_model_device_index_for_dp_workers():
    for source_path in (
        Path("afd_plugin/v1/worker/npu/attention_model_runner.py"),
        Path("afd_plugin/v1/worker/npu/ffn_model_runner.py"),
    ):
        source = source_path.read_text()

        assert "rank, _ = _resolve_world_ranks()" in source
        assert "local_rank = int(device.index)" in source
        assert "rank, local_rank = _resolve_world_ranks()" not in source
