---
title: E2E testing
kind: module
status: normative
owners:
  - "@yujuancao07"
primary_code_paths:
  - "tests/e2e/runner.py"
  - "tests/e2e/process_utils.py"
  - "tests/e2e/accuracy/**"
  - "tests/e2e/models/**"
related_code_paths:
  - "tests/e2e/README.md"
  - ".agents/skills/run-e2e/SKILL.md"
depends_on:
  - "plugin_boundary.md"
  - "attention_runtime.md"
  - "ffn_runtime.md"
  - "connector_contracts.md"
  - "model_integration.md"
  - "execution_platforms.md"
  - "compatibility_and_patches.md"
validation_paths:
  - "tests/unit/test_e2e_runner.py"
  - "tests/unit/test_e2e_process_utils.py"
  - "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py"
  - "tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py"
  - "tests/e2e/models/qwen3_moe/test_qwen3_moe.py"
  - "tests/e2e/models/qwen3_6/test_qwen3_6.py"
upstream_refs:
  - "vLLM 0.26.0 serving and shutdown interfaces"
  - "lm-evaluation-harness GSM8K task and local-completions API"
  - "pytest parameterized test IDs"
verified_platform_refs:
  - "CUDA DeepSeek-V2-Lite"
  - "Ascend NPU DeepSeek-V2-Lite"
  - "CUDA Qwen3 MoE"
  - "CUDA Qwen3.5/3.6 MoE"
related_issues: []
last_reviewed: 2026-08-13
---

# E2E testing

## Scope

An E2E case starts real vLLM services, runs a real workload, validates the
result, and cleans up every process it started.

This contract is independent of GitHub Actions and Buildkite. CI selects pytest
node IDs and supplies hardware paths; it does not redefine a case. Unit tests,
operator tests, benchmarks, and performance tests are outside this scope.

## Structure

```text
tests/e2e/
├── models/<model>/test_*.py  # Test entries and case lists
├── accuracy/<task>.py       # Accuracy tool and result parsing
├── runner.py                # Service startup, evaluation, and cleanup
└── process_utils.py         # Process-group termination and reaping
```

New model files do not copy service startup, lm-eval, signal handling, or
cleanup. Production code does not depend on the E2E harness.

## Normative invariants

- `E2E-INV-001` — A case **MUST** have a stable lower-kebab-case ID and cover
  behavior not already covered by an existing case.
- `E2E-INV-002` — A PR case **MUST NOT** use more than four unique devices.
  Standard AFD cases **MUST** use 2 Attention ranks and 1 FFN rank.
- `E2E-INV-003` — Cases sharing devices or ports **MUST** run sequentially,
  remain order-independent, and release owned process groups before the next
  case.
- `E2E-INV-004` — Gate cases **MUST** fail on missing setup, service failure,
  incomplete results, failed validation, or failed cleanup. They **MUST NOT**
  use `skip`, `xfail`, or success-on-empty behavior.
- `E2E-INV-005` — The harness **MUST** check service liveness before and after
  evaluation. Accuracy cases **MUST** also check evaluator exit status, sample
  count, `NaN`, and accuracy.
- `E2E-INV-006` — Child processes **MUST** use owned process groups.
  Cancellation **MUST** send `SIGTERM`, use a bounded grace period, then reap
  every leader. A harness `SIGKILL` escalation **MUST** fail the case.
- `E2E-INV-007` — CI **MUST NOT** lower the sample count, accuracy threshold,
  or required case set. It **MUST** select cases by pytest node ID.
- `E2E-INV-008` — A shared case **MUST** pass without skips on every platform
  where CI selects it. New behavior **MUST** have focused unit tests and real
  hardware evidence.

## Required coverage

| Case | Runtime | Devices | Purpose |
| --- | --- | ---: | --- |
| `afd-eager` | AFD eager, 2A1F | 3 | Lifecycle and eager smoke test. |
| `afd-graph` | AFD graph, 2A1F | 3 | Primary graph path. |
| `afd-graph-dbo` | AFD graph + DBO, 2A1F | 3 | Graph path with DBO. |
| `baseline-graph` | Native vLLM graph, DP4/TP1/EP4 | 4 | Non-AFD control. |

Target runtime is about 20 minutes per platform for PRs and at most 30 minutes
for merge validation. Put slower coverage in a scheduled job.

Prefer graph coverage. Keep one eager smoke test unless a feature cannot run in
graph mode.

`afd-eager-async-cam` is a separate NPU-only smoke test. It uses four devices
for Attention DP1/TP2 and FFN DP2/TP1/EP2. It is not part of the PR gate above.

`afd-graph-dbo-2a2f` is a separate GPU/NPU accuracy case. It uses four devices
for Attention DP2/TP1 and FFN DP2/TP1/EP2 and runs GSM8K-7. The default gate
selects only the four default cases by pytest node ID.

The Qwen3.5/3.6 CUDA lane is text-only and uses the native Qwen3.5/3.6 model
boundary with `--language-model-only`. Its default suite uses
`Qwen/Qwen3.6-35B-A3B`, native DP4/TP1/EP4 for `baseline-graph`, and
synchronous AFD 2A1F for `afd-eager`, `afd-graph`, and `afd-graph-dbo`.
Multimodal, NPU, `compute_gate_on_attention=true`, pipeline-parallel,
asynchronous, and multi-node execution are outside this case; quantization is
unverified.

## Accuracy gate

| Setting | PR | Weekly |
| --- | ---: | ---: |
| Task | GSM8K | GSM8K |
| Few-shot examples | 8 | 8 |
| Generated-token limit | 512 | 512 |
| Samples | first 7 | all 1319 |
| Metric | GSM8K exact match | GSM8K exact match |
| Minimum accuracy | 0.27 | 0.27 |
| Cases | all four | `afd-graph-dbo` only |

An accuracy of `0.27` requires at least 2 correct answers out of 7, or 357 out
of 1319.

- PR CI leaves `AFD_GSM8K_LIMIT` unset.
- Weekly CI sets `AFD_GSM8K_LIMIT=all`.
- Other limits are for local debugging, not CI gates.
- CI leaves `AFD_GSM8K_THRESHOLD` unset or raises it.
- Use the official GSM8K task, `HF_HOME`, and `results_*.json`. Do not commit a
  seven-row dataset or custom task YAML.

## Adding a case

1. State the coverage gap and why an existing case cannot catch it.
2. Choose one stable case ID and one fixed configuration.
3. Confirm platforms, devices, ports, CI tier, and runtime budget.
4. Reuse `runner.py`; add a task adapter only for a new evaluation task.
5. Add focused unit tests for new commands, topology, flags, and failure paths.
6. Run the case on every selected platform. Record versions, duration, result.
7. Update `tests/e2e/README.md`. Update the `run-e2e` skill only when commands,
   variables, prerequisites, or the default suite change.

Model entrypoints reuse the subprocess and download helpers in
`tests/conftest.py`.
