---
name: upgrade-npu
description: Upgrade and align the AFD Plugin NPU backend across pinned vLLM and vLLM-Ascend tags or commits. Use when Codex must analyze or implement an NPU runtime upgrade, rebase compatibility patches, adapt NPU models/workers/model runners/connectors, resolve version-driven NPU regressions, or complete the full NPU E2E and accuracy qualification for a new runtime pair. Do not use for GPU-only upgrades, model-only adaptation, ordinary NPU bug fixes unrelated to an upstream upgrade, or E2E-only execution.
---

# Upgrade the AFD NPU backend

Run this workflow as a gated state machine. Resolve both runtime histories,
finish architectural analysis, and pass the architecture gate before editing
code. Preserve AFD behavior on the target upstream architecture; do not preserve
obsolete implementation shapes merely because they existed at the old version.

Read the repository `AGENTS.md` before acting. Read
[`references/upgrade-workbook.md`](references/upgrade-workbook.md) for the
required evidence tables, commands, and report templates. Read
[`references/v0191-to-v026-lessons.md`](references/v0191-to-v026-lessons.md)
when the upgrade affects MoE construction, W8A8, PCP, DBO/uBatch, CAM,
rank/device mapping, or when choosing commit boundaries.

## Hard boundaries

- Treat the current and target runtime as four immutable revisions: current
  vLLM, current vLLM-Ascend, target vLLM, and target vLLM-Ascend. Resolve every
  tag or branch to a full commit SHA before diffing or editing.
- Require the user to provide both target refs and the local source repository
  directories for vLLM and vLLM-Ascend. Infer missing current refs from
  repository evidence. If either target ref or source directory is absent, ask
  for it and stop.
- Do not modify, switch, or check out another revision in either user-provided
  source directory. Resolve its HEAD before comparison. If it is already at the
  target SHA, use it as the target source tree. Otherwise create a separate
  detached worktree from that repository at the target SHA. Preserve a provided
  checkout at the current SHA as the current source tree; if its HEAD is neither
  the current nor target SHA, leave it untouched and read the current revision
  with `git show` while using the new worktree for the target revision.
- Do not edit AFD code until the architecture gate passes. If a major
  architecture change is found, stop this upgrade and report it; do not silently
  remove a supported feature or invent a large compatibility layer.
- Preserve unrelated dirty changes. Before editing, record branch, HEAD, and
  status. Stop if an existing user change overlaps the planned files. Never
  reset, discard, or include unrelated changes in an upgrade commit.
- Use the repository's patch, inheritance, and composition rules. Never modify
  vLLM or vLLM-Ascend source to make AFD tests pass.
- Do not add reflection-heavy compatibility. Access known upstream members
  directly, avoid `Any` and `object` unless necessary, and allow upstream errors
  to reveal incompatible contracts as required by `AGENTS.md`.
- Make one local signed-off commit for each coherent adaptation or test-failure
  root cause. Include code and its focused tests together. Do not push, publish,
  merge, or open a PR unless the user separately asks.
- Run all non-accuracy NPU E2E gates before accuracy. Never use a limited
  accuracy run as final qualification.

## Subagent orchestration

Use distinct subagents when available to separate analysis, implementation,
testing, and review. The primary agent owns the phase state, architecture-gate
decision, task graph, integration, staged diff, commits, and final report.
Subagent conclusions are evidence, not automatic gate decisions.

- **Analysis agents**: keep them read-only. Split work by upstream history or
  AFD surface, such as vLLM, vLLM-Ascend, patches, models, NPU runtime, and
  connectors. Require exact paths, symbols, SHAs, classifications, affected
  invariants, and proposed tests. Allow independent analysis agents to run in
  parallel.
- **Implementation agents**: dispatch them only after the architecture gate
  passes. Give each agent one coherent root cause and an explicit, non-overlapping
  AFD file set. They may edit only those files and their focused tests. Prohibit
  staging, committing, pushing, changing branches, or modifying either upstream
  source tree or target worktree. Serialize implementation in a shared AFD
  checkout. Parallelize only in primary-created, fixed-base AFD worktrees with
  independent file ownership; return each raw diff to the primary agent for
  one-at-a-time integration and validation in the main upgrade checkout.
- **Test agents**: prohibit changes to tracked or source files. Allow writes only
  to handoff-designated temporary, log, cache, and artifact paths. Assign exact
  commands and validation cells, require complete logs, counts, first root
  traceback, skips, and cleanup, and return failures to the primary agent for
  classification. Make agents that run NPU E2E read `../run-e2e/SKILL.md`.
  Serialize jobs unless device IDs, port ranges, writable cache paths, temporary
  and log directories, process/service ownership, and cleanup ownership are all
  explicitly isolated. Permit shared read-only model access.
- **Review agents**: keep them read-only and independent from the implementation
  task. Give them the raw diff plus exact target upstream trees and require
  review first for minimal and simple implementation, compliance with
  `AGENTS.md` and established local style, and duplicated, obsolete, dead, or
  unnecessary compatibility code. Then review exact upstream signatures, patch
  markers, architecture, state ownership, execution order, and test coverage.
  Require actionable findings with priority and file/line evidence; do not ask
  the reviewer to fix its findings.

Before dispatching any subagent, record its role, objective, current and target
SHAs, upstream source/worktree paths, assigned AFD checkout/worktree and fixed
base, allowed AFD files, allowed commands, prohibited actions, dependencies, and
required output using the handoff template in the workbook. Because agents share
one workspace, record the starting diff and never give concurrent implementation
agents the same checkout or overlapping paths. Require every edit, test, and
review command to run in the assigned AFD path. Never run testing or review in a
workspace while an implementation agent is writing it. Freeze the diff first,
then inspect every returned diff and command result directly.

Use this loop for every coherent adaptation or test-failure root cause:

```text
ANALYZE -> PRIMARY GATE DECISION -> IMPLEMENT -> FOCUSED TEST
        -> INDEPENDENT REVIEW -> PRIMARY STAGE/COMMIT -> REGRESSION GATE
```

If review or testing finds a new major architecture change, return to the
architecture gate and stop production edits. If it finds a local defect, assign
one bounded implementation task, repeat focused validation and independent
review, then let only the primary agent create the signed-off commit.

## Required phase record

At each phase boundary record:

```text
phase: <name>
status: PASS | FAIL | BLOCKED | STOPPED | SKIPPED
evidence: <resolved SHAs, files, commands, tests, or logs>
blocker: <none or exact blocker>
next_allowed_action: <one action>
```

Do not enter a later phase when an earlier required phase is `FAIL`, `BLOCKED`,
or `STOPPED`.

```text
FREEZE_IDENTITY
  -> INVENTORY_AFD_CONTRACTS
  -> DIFF_BOTH_UPSTREAMS
  -> ARCHITECTURE_GATE
  -> IMPLEMENT_ATOMIC_ADAPTATIONS
  -> STATIC_AND_UNIT_GATE
  -> TARGET_NATIVE_NPU_CONTROL
  -> NPU_BASIC_E2E_GATE
  -> NPU_ACCURACY_GATE
  -> DOCUMENTATION_REFRESH_GATE
  -> AUDIT_AND_REPORT
```

## Phase 0 — Freeze identity

Record the AFD checkout, branch, HEAD, dirty status, Python environment, and
available NPU hardware. Validate the user-provided vLLM and vLLM-Ascend source
directories as Git repositories; record their paths, HEADs, branches, and dirty
status. Then build the four-revision ledger and create any required detached
target worktrees without changing the provided checkouts.

For a missing current ref, use evidence in this order:

1. exact dependency pins and lockfiles in AFD;
2. version constants, patch source annotations, support matrices, installation
   docs, recipes, containers, and CI configuration in AFD;
3. the current vLLM-Ascend revision's
   `.github/vllm-main-verified.commit` or
   `.github/vllm-release-tag.commit`;
4. for older vLLM-Ascend revisions, `VLLM_TAG` or `VLLM_COMMIT` in that
   revision's Dockerfile, then its compatibility matrix;
5. installed package versions and adjacent checkout state as corroboration
   only.

Do not assume equal version strings imply compatibility. Resolve annotated tags
to commits and record both the user-facing ref and SHA. If authoritative
evidence conflicts, mark identity `BLOCKED` and ask the user which current pair
is authoritative.

Validate the target pair against vLLM-Ascend's commit locks, release matrix, and
complete software tuple: Python, PyTorch, torch_npu, CANN, Triton Ascend, ABI,
and any required communication/operator package. If the target refs are not a
supported pair, stop before code changes.

## Phase 1 — Inventory AFD contracts

Build an inventory before reading upstream diffs. Start with these high-risk
surfaces:

- `afd_plugin/compat/patches/**`, especially `patches/npu/**`;
- `afd_plugin/compat/npu/**`;
- `afd_plugin/model_executor/models/**`, including NPU overlays, native model
  factories, role construction, forward paths, and weight loading;
- `afd_plugin/v1/worker/npu/**` and the shared workers/model runners they
  extend;
- forward context, Attention metadata, graph capture, DBO/uBatch, scheduler
  output, input batch, sampler, speculative decode, and device selection;
- NPU connectors, payloads, process topology, rank/group/device mapping, CAM
  packages, configuration validation, and plugin initialization order;
- dependency pins, tests, recipes, support matrices, and versioned docs.

For every copied or patched function, record its AFD target, current upstream
source revision and symbol, exact signature, AFD marker blocks, behavioral
invariant, state/lifecycle ownership, direct dependencies, and focused tests.
Use source grep as authority; documentation is supporting evidence.

Also freeze the supported feature matrix. Include every discovered combination
of runner generation, model, eager/graph, DBO/uBatch, PCP, sync/async CAM,
Attention/FFN gate placement, TP/DP/EP, quantization/W8A8/EPLB, and
speculative/MTP behavior. Do not exclude a feature merely because its path is
hard to upgrade.

## Phase 2 — Diff both upstreams

Analyze both histories independently:

- current vLLM SHA to target vLLM SHA;
- current vLLM-Ascend SHA to target vLLM-Ascend SHA.

Start with name-status, rename-aware stats, first-parent logs, and focused
function-context diffs. Map every relevant upstream change to one or more AFD
contracts. Classify each item:

- `MECHANICAL`: import move, rename, or signature change with equivalent
  behavior;
- `BEHAVIORAL`: defaults, metadata schema, state lifetime, call order, device
  mapping, or ownership changed while an equivalent AFD invariant still exists;
- `ARCHITECTURAL`: an execution generation, feature, stable seam, ownership
  boundary, or cross-component protocol was removed or conceptually replaced.

Use a method-level reconstruction for copied upstream logic:

1. take the target pinned upstream function as the new skeleton;
2. take current AFD patch markers and tests as the local-difference ledger;
3. take current AFD model/runtime behavior as the semantic authority;
4. replay only still-required AFD differences into the new skeleton;
5. verify call order, state ownership, and behavior, not only imports.

Do not mechanically apply the old-to-new upstream diff to an AFD copy.

## Phase 3 — Architecture gate

Stop without editing production code if any of these is true:

- the target vLLM and vLLM-Ascend refs or software stack are incompatible;
- the NPU worker/model-runner base, execution generation, or lifecycle is
  replaced and AFD would require a redesign rather than local adaptation;
- an AFD patch seam is removed with no equivalent stable target, or the patch
  can no longer be expressed as target upstream logic plus small marked AFD
  differences;
- upstream removes or explicitly disables an AFD-supported feature, such as a
  runner-generation PCP path or DBO path, and continuing would require deleting
  the feature, bypassing upstream validation, or owning a new execution engine;
- model split, remote-experts ownership, connector payload, rank/group topology,
  scheduler/KV-cache/request-output contract, or core state lifecycle changes
  across multiple subsystems without a one-to-one invariant mapping;
- preserving behavior requires broad core rewrites, new public abstractions,
  extensive unknown-version copies, exception swallowing, or reflection-based
  version probing;
- the required Python/PyTorch/torch_npu/CANN/operator ABI transition cannot be
  built and tested as one supported target environment.

A moved implementation is not automatically a stop. Continue only when the old
and new inputs, outputs, lifecycle, and invariant map cleanly and the adaptation
remains narrow. For example, a patch moving from a layer to its quantization
method requires architectural review, but may pass if ownership and behavior
are still provably equivalent.

When stopping, report the old and new architecture, exact removed or replaced
interfaces, affected AFD files/features, why a local compatibility patch is not
safe, and the smallest design decision needed from maintainers. Leave the
worktree unchanged.

## Phase 4 — Implement atomic adaptations

Create or use a dedicated upgrade branch. Before each edit batch, write a small
ledger of files, upstream cause, preserved invariant, focused tests, and patch
removal/upstream plan.

Implement in this order unless the dependency graph proves another order:

1. dependency/version contract and CPU-safe imports;
2. compatibility patches and initialization order;
3. model construction, factory binding, forward, and weight policy;
4. NPU worker/model runner, forward context, graph, DBO/uBatch, and device
   mapping;
5. connectors and payload/topology integration;
6. focused tests and E2E scenarios required to qualify the adaptations.

Defer repository-wide recipes, support matrices, and documentation refreshes to
Phase 9 so their claims are based on completed NPU validation evidence. Update a
focused test fixture or narrowly coupled comment earlier only when the code
adaptation requires it.

For each patch function:

- copy it from the exact target pinned source revision;
- match the target signature exactly, including defaults and return annotation;
- add comments immediately above it stating source revision, reason, changed
  behavior, and whether parameters differ;
- surround only AFD-specific differences with paired
  `# ### PATCH START: ...` and `# ### PATCH END: ...` markers;
- re-evaluate whether upstream absorbed the workaround; remove obsolete patches
  instead of rebasing them indefinitely;
- add tests for enabled and disabled paths, initialization order, config
  lifetime, state ownership, topology, and device remapping as applicable.

Use inheritance or composition when the target exposes a stable seam. Do not
change common runtime code to conceal an incomplete NPU adaptation. Preserve
observable execution order where state is consumed or mutated.

After each coherent adaptation passes its focused static/unit tests, inspect the
exact staged diff and create a signed-off local commit. Do not combine separate
root causes. Use the commit template in the workbook and include the old/target
SHAs when the change is caused by vLLM or vLLM-Ascend.

## Phase 5 — Static and unit gate

Before NPU execution, run the repository's actual commands for:

- `git diff --check`, Ruff check and format check, and compile/import checks;
- target-runtime import and plugin registration;
- exact patch signature and marker checks;
- affected focused unit tests;
- the complete CPU/unit suite;
- `python -m pip check` in the target runtime environment.

Treat collection/import failure as a version or environment failure until the
four-revision tuple and installed packages are proven. Do not call import
success, construction, dummy execution, or server startup full validation.

## Phase 6 — Target native NPU control

Before AFD E2E, prove that the exact target vLLM/vLLM-Ascend runtime can load the
selected checkpoint and serve one real NPU request without AFD. Record runtime
SHAs, package versions, model, quantization, topology, eager/graph mode, command,
readiness, output, and cleanup.

If the native control fails, preserve the root traceback and stop AFD parity
testing. Diagnose native runtime, environment, model, and operator-package
failures separately from AFD compatibility. A fallback mode is a new validation
cell and must not silently replace the requested target.

## Phase 7 — NPU basic E2E gate

Read and follow `../run-e2e/SKILL.md` for hardware detection, provisioning,
marker selection, live output, and cleanup. Its tests-only boundary governs this
execution phase; return to this skill only after the test result is captured.

Run the complete marker-based NPU non-accuracy suite before accuracy:

1. features;
2. models, including every configured sync/async connector scenario;
3. any repository-owned NPU upgrade scenario not covered by those categories.

Use enough NPUs to avoid hardware-capacity skips when claiming a full upgrade.
Every skip must map to a pre-existing documented unsupported combination or an
explicitly out-of-scope test in `run-e2e`; no new or unexplained skip is a pass.
Send real requests and validate cleanup of processes, ports, and NPU allocation.

For each failure:

1. save the exact command, environment, runtime SHAs, failing test ID, first
   useful root traceback, logs, resource state, and cleanup result;
2. classify it as environment, native runtime, AFD patch/model/runtime,
   connector/topology, accuracy, or unknown;
3. trace it to the smallest upstream contract change and reproduce it with the
   narrowest focused test;
4. if it reveals a major architecture change, return to the architecture gate
   and stop;
5. otherwise implement one minimal root-cause fix with its regression test;
6. run focused static/unit validation, create one signed-off commit with the
   failure and upstream cause in the body, then rerun the failed E2E and the
   complete non-accuracy gate.

Do not commit environment-only changes as AFD compatibility fixes. Do not change
multiple runtime knobs and code in one retry.

## Phase 8 — NPU accuracy gate

Enter only after the complete non-accuracy gate passes. Use `run-e2e` to run all
NPU accuracy cases, including eager and graph. Leave `AFD_GSM8K_LIMIT` unset for
final qualification so the full dataset runs. A limited run may be used only as
a post-basic diagnostic and must be labeled non-final.

On an accuracy failure, preserve metrics and artifacts, compare the exact native
control and AFD validation cells, make only a proven minimal fix, add a focused
regression test, create one root-cause commit, rerun the complete non-accuracy
gate, and then rerun full accuracy. Never lower the threshold or widen tolerance
to make an upgrade pass unless the user explicitly approves a separately
justified policy change.

## Phase 9 — Documentation refresh gate

Enter only after the complete NPU basic and full accuracy gates pass. Before
editing any upgrade-related documentation in this phase, read
[`references/documentation-refresh.md`](references/documentation-refresh.md)
completely and follow its evidence, coverage, history-preservation, consistency,
and completion requirements.

Refresh the public runtime baseline, connector support statements and guides,
target recipes and runnable scripts, internal design and patch contracts,
contributor templates, and applicable generated metadata. Base every support
claim on the exact target validation ledger. Distinguish hardware-validated,
unit-only, implemented-but-unvalidated, and unsupported combinations. Do not
mechanically replace intentionally historical version references or relabel old
benchmark results as target evidence.

Run the reference's repository-wide consistency searches and inspect every hit
in context. Create one or more focused signed-off documentation commits after
the documentation checks pass. Record changed documents, intentionally
preserved history, reconciled claims, validation commands, and commit SHAs in
the phase evidence.

Do not enter final audit while documentation claims conflict, target runnable
examples still use an obsolete stack, or a validated/unsupported status lacks
evidence.

## Phase 10 — Audit and report

Audit all version pins, source annotations, support matrices, generated metadata,
recipes, container/operator packages, historical-branch notes, and docs. Do not
replace historical version references that remain intentionally scoped to old
branches. Confirm every changed patch has a focused test and an upstream/removal
plan.

Report:

- all four refs and resolved SHAs plus the target software stack;
- the upstream-to-AFD impact matrix and architecture-gate decision;
- changed components, preserved invariants, patch inventory, and feature matrix;
- every created commit SHA/title and the root cause it isolates;
- static/unit, native control, basic E2E, and full accuracy commands/results;
- pass/fail/error/skip counts, skip reasons, metrics, logs, and cleanup status;
- every unvalidated, unsupported, removed, or explicitly excluded combination;
- whether the branch is only local and that nothing was pushed.

Declare the upgrade complete only when the target pair is immutable and
compatible, the architecture gate passed before implementation, all code changes
are committed atomically, the complete non-accuracy NPU gate passed with no new
unexplained skips, full accuracy passed as the final test gate, the documentation
refresh gate passed, and the final version/documentation audit is clean.
