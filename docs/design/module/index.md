---
title: AFD module design index
kind: module
status: draft
owners:
  - "@hsliuustc0106"
  - "@jiangkuaixue123"
primary_code_paths: []
related_code_paths:
  - "afd_plugin/**"
  - "csrc/**"
  - "setup.py"
  - "MANIFEST.in"
  - "pyproject.toml"
depends_on: []
validation_paths:
  - "tests/unit/**"
  - "tests/e2e/**"
upstream_refs:
  - "vLLM 0.19.1"
  - "vLLM-Ascend environment evidence recorded in the NPU guides"
verified_platform_refs:
  - "CUDA: tests/e2e tests marked gpu; no canonical image is recorded"
  - "Ascend: quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler (test evidence only)"
related_issues:
  - "#129"
last_reviewed: 2026-07-20
---

# AFD module design

This directory is the routing and ownership layer for AFD module design. The
documents are `draft` until their owners verify boundaries, invariants, and
validation evidence. The role and platform content from the former GPU/NPU
runtime documents was migrated here in Phase 2 and the former files were
removed. Phase 3 now records the current plugin, connector, model, and patch
contracts with direct source/test evidence; open interfaces remain explicitly
draft.

## Migration progress

| Phase | Documentation result | Status |
| --- | --- | --- |
| 1 | Eight module documents, metadata, ownership, and unique production-path routing. | Complete |
| 2 | Unified role runtime and platform design; former GPU/NPU runtime files removed after migration. | Complete |
| 3 | Plugin boundary, connector lifecycle/payloads, model integration, and full compatibility/Patch inventory linked to implementation and tests. | Complete; owner review still required before any document becomes `normative` |

## Reading order

1. [Plugin boundary](plugin_boundary.md) for registration, configuration, and
   supported upstream boundaries.
2. [Attention runtime](attention_runtime.md) or
   [FFN runtime](ffn_runtime.md) for role lifecycle and execution flow.
3. [Connector contracts](connector_contracts.md) and
   [model integration](model_integration.md) for the handoff between roles.
4. [Execution platforms](execution_platforms.md) for CUDA/NPU mechanisms.
5. [Compatibility and patches](compatibility_and_patches.md) before modifying
   upstream compatibility behavior.

## Dependency direction

Dependencies flow from role and integration modules toward shared boundaries;
platform and compatibility modules adapt those boundaries to upstream
runtimes. A lower-level document must not depend on a role implementation.

```text
attention_runtime ----+----> connector_contracts ----> plugin_boundary
                      |                |
ffn_runtime ----------+                +----> execution_platforms
                      |
model_integration ----+----> execution_platforms

attention_runtime ----+
ffn_runtime ----------+----> compatibility_and_patches ----> plugin_boundary
```

## Production path routing

Every shipped Python path, native build path, and packaging file has one
primary document. Related documents may discuss a path but do not own its
contract. File-level entries deliberately resolve mixed directories.

| Primary document | Production paths |
| --- | --- |
| [Plugin boundary](plugin_boundary.md) | `afd_plugin/__init__.py`, `afd_plugin/config.py`, `afd_plugin/config_utils.py`, `afd_plugin/envs.py`, `afd_plugin/validation.py`, `afd_plugin/py.typed`, `afd_plugin/v1/__init__.py`, `afd_plugin/v1/worker/__init__.py`, `afd_plugin/v1/worker/npu/__init__.py`, `pyproject.toml` |
| [Attention runtime](attention_runtime.md) | `afd_plugin/v1/worker/attention_model_runner.py`, `afd_plugin/v1/worker/attention_worker.py`, `afd_plugin/v1/worker/ubatch_wrapper.py`, `afd_plugin/v1/worker/npu/attention_model_runner.py`, `afd_plugin/v1/worker/npu/attention_worker.py` |
| [FFN runtime](ffn_runtime.md) | `afd_plugin/v1/worker/ffn_model_runner.py`, `afd_plugin/v1/worker/ffn_worker.py`, `afd_plugin/v1/worker/npu/ffn_model_runner.py`, `afd_plugin/v1/worker/npu/ffn_worker.py` |
| [Connector contracts](connector_contracts.md) | `afd_plugin/connectors/**/*.py`, `afd_plugin/connectors/npu/bin/**`, `afd_plugin/distributed/**/*.py` |
| [Model integration](model_integration.md) | `afd_plugin/model_executor/**/*.py` |
| [Execution platforms](execution_platforms.md) | `afd_plugin/compat/profiler.py`, `afd_plugin/compat/npu/forward_context.py`, `afd_plugin/compat/npu/ops.py`, `afd_plugin/compat/npu/profiler.py`, `afd_plugin/v1/worker/cuda_graph.py`, `afd_plugin/v1/worker/dbo.py`, `afd_plugin/v1/worker/npu/forward_context.py`, `afd_plugin/v1/worker/npu/npu_ubatch_wrapper.py`, `afd_plugin/v1/worker/npu/pcp_debug.py`, `afd_plugin/v1/worker/npu/ubatch_utils.py`, `afd_plugin/v1/worker/npu/ubatching.py`, `csrc/**`, `setup.py`, `MANIFEST.in` |
| [Compatibility and patches](compatibility_and_patches.md) | `afd_plugin/compat/__init__.py`, `afd_plugin/compat/vllm.py`, `afd_plugin/compat/npu/__init__.py`, `afd_plugin/compat/npu/feature_validation.py`, `afd_plugin/compat/npu/runtime.py`, `afd_plugin/compat/npu/runtime_config.py`, `afd_plugin/compat/patches/**/*.py` |

The routing inventory covers runtime and package code under `afd_plugin/**`,
native sources under `csrc/**`, and packaging files that affect shipped
artifacts. Tests, development tools, recipes, generated files, user guides,
and design documents are outside the production ownership inventory.

## Document status

| Status | Meaning |
| --- | --- |
| `draft` | Boundaries or evidence still require owner review. Statements describe current intent and are not stable contracts. |
| `normative` | Owners have approved the boundary, invariants, upstream references, and enforcement evidence. |

Only identified invariant blocks in a `normative` document may use `MUST`,
`MUST NOT`, or `SHOULD` as contract terms.

## User and operational guides

Operational guides remain separate from normative module design:

- [NCCL P2P connector guide](../../gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md)
- [CAMP2P connector guide](../../npu/CAMP2P_CONNECTOR_USER_GUIDE.md)
- [CAM async connector guide](../../npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md)
- [Ascend NPU installation](../../../README.md#ascend-npu-installation)
- [Connector overview](../../../afd_plugin/connectors/README.md)
- [Deployment recipes](../../../recipe/README.md)

## Phase 1 completion criteria

- all eight draft documents exist;
- every production path has one primary document;
- owners, validation paths, upstream references, and related issues are
  recorded;
- disputed interfaces remain draft and link to their open issues.

## Phase 2 completion criteria

- GPU and NPU Attention behavior is consolidated in `attention_runtime.md`;
- GPU and NPU FFN behavior is consolidated in `ffn_runtime.md`;
- cross-cutting CUDA and NPU mechanisms live in
  `execution_platforms.md`;
- the four former role-by-platform runtime documents are removed after their
  content and ownership move to the module documents.

## Phase 3 completion criteria

- plugin registration, configuration, validation, and public launch class paths
  are specified from current source behavior;
- connector factory, capabilities, lifecycle, payload/control-plane ownership,
  topology, cleanup, and failure behavior are recorded;
- model registration, role-aware construction/loading, forward-context handoff,
  and DeepSeek-specific execution paths are recorded;
- every production monkey patch has an upstream symbol, AFD delta, application
  guard, non-AFD expectation, focused validation, and removal/upstream plan;
- behavior descriptions link to source and tests instead of copying
  implementation bodies;
- the configuration decisions implemented from
  [#89](https://github.com/JiusiServe/afd-plugin/issues/89) are reflected in
  the plugin and connector documents;
- interfaces still disputed in
  [#88](https://github.com/JiusiServe/afd-plugin/issues/88),
  [#105](https://github.com/JiusiServe/afd-plugin/issues/105), and
  [#107](https://github.com/JiusiServe/afd-plugin/issues/107) remain `draft` and
  link to the corresponding issue.
