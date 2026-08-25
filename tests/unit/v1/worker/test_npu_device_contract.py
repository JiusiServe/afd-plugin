# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import ast
from pathlib import Path


def _method_parameters(
    source_path: Path, class_name: str, method_name: str
) -> list[str]:
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


def test_npu_v2_profile_state_reaches_ffn_ascend_context():
    metadata_source = Path("afd_plugin/connectors/metadata.py").read_text()
    attention_source = Path(
        "afd_plugin/v1/worker/npu/attention_model_runner_v2.py"
    ).read_text()
    ffn_worker_source = Path("afd_plugin/v1/worker/npu/ffn_worker.py").read_text()
    ffn_runner_source = Path("afd_plugin/v1/worker/npu/ffn_model_runner.py").read_text()
    forward_context_source = Path(
        "afd_plugin/compat/npu/forward_context.py"
    ).read_text()

    assert "is_profile: bool = False" in metadata_source
    assert 'payload.get("is_profile", False)' in metadata_source
    assert "self._afd_is_profile = bool(is_profile)" in attention_source
    assert "is_profile = payload.is_profile" in ffn_worker_source
    assert "in_profile_run=is_profile" in ffn_runner_source
    assert "if vllm_config.use_v2_model_runner:" in forward_context_source
    assert "override_mrv2_in_profile_run(in_profile_run)" in forward_context_source
    assert "set_forward_context(" in forward_context_source
    assert 'additional_kwargs["afd_metadata"]' in forward_context_source


def test_npu_connectors_use_model_device_index_for_dp_workers():
    for source_path in (
        Path("afd_plugin/v1/worker/npu/attention_model_runner.py"),
        Path("afd_plugin/v1/worker/npu/ffn_model_runner.py"),
    ):
        source = source_path.read_text()

        assert "rank, _ = _resolve_world_ranks()" in source
        assert "local_rank = int(device.index)" in source
        assert "rank, local_rank = _resolve_world_ranks()" not in source
