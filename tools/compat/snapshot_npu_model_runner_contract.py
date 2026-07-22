# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Snapshot the pinned NPU ModelRunner source contract without importing it."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_VLLM_REF = "v0.19.1"
DEFAULT_ASCEND_REF = "v0.19.1rc1"

RUNNER_METHODS = (
    "_sync_metadata_across_dp",
    "_prepare_inputs",
    "execute_model",
    "_model_forward",
    "sync_and_slice_intermediate_tensors",
    "_determine_batch_execution_and_padding",
    "_build_attention_metadata",
    "_dummy_run",
    "load_model",
)

ATTENTION_BUILDERS = (
    (
        "AscendAttentionMetadataBuilder",
        "vllm_ascend/attention/attention_v1.py",
    ),
    (
        "AscendMLAMetadataBuilder",
        "vllm_ascend/attention/mla_v1.py",
    ),
)

ATTENTION_BUILDER_METHODS = (
    "build",
    "build_for_graph_capture",
)


@dataclass(frozen=True)
class SourceTree:
    """Read source from a checkout or from an exact Git ref."""

    root: Path
    ref: str | None = None

    def read_text(self, relative_path: str) -> str:
        if self.ref is None:
            return (self.root / relative_path).read_text(encoding="utf-8")
        result = subprocess.run(
            ["git", "-C", str(self.root), "show", f"{self.ref}:{relative_path}"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def revision(self) -> str | None:
        if self.ref is None or not (self.root / ".git").exists():
            return None
        result = subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", self.ref],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()


def _unparse(node: ast.AST | None) -> str | None:
    return None if node is None else ast.unparse(node)


def _argument_contract(
    argument: ast.arg,
    *,
    kind: str,
    default: ast.expr | None,
) -> dict[str, Any]:
    return {
        "annotation": _unparse(argument.annotation),
        "default": _unparse(default),
        "kind": kind,
        "name": argument.arg,
        "required": default is None,
    }


def _function_contract(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, Any]:
    arguments = node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(arguments.defaults)
    ) + list(arguments.defaults)
    parameters: list[dict[str, Any]] = []
    posonly_count = len(arguments.posonlyargs)
    for index, (argument, default) in enumerate(zip(positional, defaults, strict=True)):
        kind = "positional_only" if index < posonly_count else "positional_or_keyword"
        parameters.append(
            _argument_contract(argument, kind=kind, default=default),
        )
    if arguments.vararg is not None:
        parameters.append(
            _argument_contract(
                arguments.vararg,
                kind="var_positional",
                default=None,
            ),
        )
        parameters[-1]["required"] = False
    for argument, default in zip(
        arguments.kwonlyargs,
        arguments.kw_defaults,
        strict=True,
    ):
        parameters.append(
            _argument_contract(
                argument,
                kind="keyword_only",
                default=default,
            ),
        )
    if arguments.kwarg is not None:
        parameters.append(
            _argument_contract(
                arguments.kwarg,
                kind="var_keyword",
                default=None,
            ),
        )
        parameters[-1]["required"] = False
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    rendered_parameters = []
    for parameter in parameters:
        rendered = parameter["name"]
        if parameter["annotation"] is not None:
            rendered += f": {parameter['annotation']}"
        if parameter["kind"] == "var_positional":
            rendered = f"*{rendered}"
        elif parameter["kind"] == "var_keyword":
            rendered = f"**{rendered}"
        elif parameter["default"] is not None:
            rendered += f" = {parameter['default']}"
        rendered_parameters.append(rendered)
    returns = _unparse(node.returns)
    signature = f"{prefix} {node.name}({', '.join(rendered_parameters)})"
    if returns is not None:
        signature += f" -> {returns}"
    return {
        "async": isinstance(node, ast.AsyncFunctionDef),
        "parameters": parameters,
        "return_annotation": returns,
        "signature": signature,
    }


def extract_method_contracts(
    source: str,
    class_name: str,
    method_names: Sequence[str],
) -> dict[str, dict[str, Any] | None]:
    tree = ast.parse(source)
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if class_node is None:
        raise ValueError(f"class {class_name!r} was not found")
    methods = {
        node.name: node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return {
        name: _function_contract(methods[name]) if name in methods else None
        for name in method_names
    }


def extract_class_fields(source: str, class_name: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if class_node is None:
        raise ValueError(f"class {class_name!r} was not found")
    fields = []
    for node in class_node.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        fields.append(
            {
                "annotation": ast.unparse(node.annotation),
                "default": _unparse(node.value),
                "name": node.target.id,
            },
        )
    return fields


def extract_external_imports(source: str) -> list[str]:
    imports = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith(("vllm", "torch"))
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith(("vllm", "torch"))
        ):
            imports.add(node.module)
    return sorted(imports)


def build_contract(
    *,
    plugin_tree: SourceTree,
    vllm_tree: SourceTree,
    ascend_tree: SourceTree,
) -> dict[str, Any]:
    plugin_runner_source = plugin_tree.read_text(
        "afd_plugin/v1/worker/npu/attention_model_runner.py",
    )
    upstream_runner_source = ascend_tree.read_text(
        "vllm_ascend/worker/model_runner_v1.py",
    )
    plugin_methods = extract_method_contracts(
        plugin_runner_source,
        "AFDNPUAttentionModelRunner",
        RUNNER_METHODS,
    )
    upstream_methods = extract_method_contracts(
        upstream_runner_source,
        "NPUModelRunner",
        RUNNER_METHODS,
    )
    method_contracts = {}
    for name in RUNNER_METHODS:
        plugin_contract = plugin_methods[name]
        upstream_contract = upstream_methods[name]
        method_contracts[name] = {
            "matches": plugin_contract == upstream_contract,
            "plugin": plugin_contract,
            "upstream": upstream_contract,
        }
    attention_source = ascend_tree.read_text("vllm_ascend/attention/utils.py")
    forward_context_source = vllm_tree.read_text("vllm/forward_context.py")
    ubatch_source = vllm_tree.read_text("vllm/v1/worker/ubatch_utils.py")
    return {
        "attention_builders": {
            class_name: extract_method_contracts(
                ascend_tree.read_text(relative_path),
                class_name,
                ATTENTION_BUILDER_METHODS,
            )
            for class_name, relative_path in ATTENTION_BUILDERS
        },
        "baseline": {
            "vllm": {
                "ref": vllm_tree.ref,
                "revision": vllm_tree.revision(),
            },
            "vllm_ascend": {
                "ref": ascend_tree.ref,
                "revision": ascend_tree.revision(),
            },
        },
        "external_imports": extract_external_imports(plugin_runner_source),
        "runner_methods": method_contracts,
        "schema_version": SCHEMA_VERSION,
        "schemas": {
            "AscendCommonAttentionMetadata": extract_class_fields(
                attention_source,
                "AscendCommonAttentionMetadata",
            ),
            "BatchDescriptor": extract_class_fields(
                forward_context_source,
                "BatchDescriptor",
            ),
            "DPMetadata": extract_class_fields(
                forward_context_source,
                "DPMetadata",
            ),
            "ForwardContext": extract_class_fields(
                forward_context_source,
                "ForwardContext",
            ),
            "UBatchSlice": extract_class_fields(
                ubatch_source,
                "UBatchSlice",
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print the pinned NPU ModelRunner source contract as JSON. "
            "The command is read-only and does not import vLLM or torch."
        ),
    )
    parser.add_argument("--plugin-root", type=Path, default=Path.cwd())
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--vllm-ref", default=DEFAULT_VLLM_REF)
    parser.add_argument("--vllm-ascend-root", type=Path, required=True)
    parser.add_argument("--vllm-ascend-ref", default=DEFAULT_ASCEND_REF)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = build_contract(
        plugin_tree=SourceTree(args.plugin_root),
        vllm_tree=SourceTree(args.vllm_root, args.vllm_ref),
        ascend_tree=SourceTree(args.vllm_ascend_root, args.vllm_ascend_ref),
    )
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
