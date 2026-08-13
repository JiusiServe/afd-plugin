# GPU upgrade workbook

Use these ledgers during any GPU vLLM upgrade. Keep filled working tables and
completion reports in task notes or designated evidence storage; do not add
generated reports to the repository unless the user requests it. Historical
lesson files are evidence records, not target-version recipes.

## Contents

1. Identity ledger
2. Source resolution and diff commands
3. Native build and ABI evidence
4. Subagent handoff ledger
5. Patch inventory
6. Feature inventory
7. Upstream-to-AFD impact matrix
8. Architecture stop report
9. Staged upgrade plan
10. Commit discipline
11. Validation ledger
12. Upgrade lesson template
13. Completion report

## Identity ledger

```text
Task:
  mode: planning-audit | implementation | final-validation
  active RFC or issue:
  requested stopping point:

AFD:
  path:
  branch:
  HEAD:
  immutable pre-upgrade base revision:
  base revision evidence: issue/RFC or proven branch ancestry
  dirty paths:
  target published package/compatibility version:
  mirrored-version policy or NOT_APPLICABLE evidence:

vLLM:
  source directory: path, HEAD, branch, dirty status
  current ref:
  current resolved full SHA:
  current identity evidence:
  target ref:
  target resolved full SHA:
  target worktree:

Target software stack:
  Python:
  PyTorch:
  PyTorch build CUDA:
  PyTorch C++11 ABI:
  vLLM package/build identity:
  vLLM native module/artifact path:
  CUDA toolkit/runtime:
  NVIDIA driver:
  nvcc:
  host compiler:
  NCCL:
  other native/communication packages:

Target validation environment:
  host/image:
  GPU model and compute capability:
  native artifact support for target compute capability:
  GPU count:
  interconnect/network topology:
  scheduler reservation requirement:
  model/checkpoint and quantization:
  model paths:
  runner generation:
```

Record where every value came from and resolve conflicts explicitly. Package
pins and exact source revisions outrank installed-package guesses. An installed
version is corroboration until it is tied to the intended source/build.

## Source resolution and diff commands

Replace placeholders with verified values from the identity ledger. Do not
check out or edit refs in user-provided vLLM directories. Reuse a checkout only
when its HEAD is the needed immutable SHA. Prefer `git show` in a read-only
planning audit; add a detached worktree only when workspace writes are in
scope.

```bash
git -C <vllm-repo> rev-parse --verify '<ref>^{commit}'
git -C <vllm-repo> rev-parse HEAD
git -C <vllm-repo> status --short --branch
git -C <vllm-repo> worktree add --detach <worktree-path> <sha>
git -C <vllm-repo> show '<sha>:<path>'
git -C <vllm-repo> diff --find-renames --name-status <old-sha>..<new-sha>
git -C <vllm-repo> diff --find-renames --stat <old-sha>..<new-sha>
git -C <vllm-repo> log --first-parent --reverse --oneline <old-sha>..<new-sha>
git -C <vllm-repo> diff --function-context <old-sha>..<new-sha> -- <paths>
```

Use repository discovery before assuming the affected surface:

```bash
rg -n '^(from|import) vllm' afd_plugin tests
rg -n 'PATCH START|PATCH END|Upstream source|Upstream:' afd_plugin tests
rg -n 'vllm|TARGET_VLLM_VERSION|CUDA|NCCL' \
  pyproject.toml uv.lock README.md docs recipe .github afd_plugin tests
rg -n 'GPUModelRunner|gpu_worker|EngineCore|CUDA Graph|cudagraph|DBO|ubatch' \
  afd_plugin tests docs recipe
rg -n 'TP|DP|EP|PP|process.group|PyNccl|profiler|weight|load_weights' \
  afd_plugin tests docs recipe
```

Adapt searches to renamed target symbols. For every copied method, inspect the
full current AFD method and both upstream definitions. Compare decorator,
method type, parameters, ordering, defaults, return annotation, body order,
state ownership, resource lifetime, and every AFD marker block.

For a final-validation environment, record output from the available canonical
tools rather than assuming intended settings were effective. Typical evidence
includes Python and package versions, `nvidia-smi`, topology output, PyTorch
CUDA/NCCL information, `nvcc`, and the host compiler. Follow `AGENTS.md` for the
actual machine and scheduler commands.

## Native build and ABI evidence

Inspect `setup.py`, `pyproject.toml`, `csrc/gpu/**`, package manifests, and the
exact target vLLM build configuration before deciding whether AFD owns a
compiled GPU extension. Record:

```text
PyTorch version/build CUDA/C++11 ABI:
vLLM install provenance and native module path:
CUDA runtime/toolkit, driver, nvcc, and host compiler:
target GPU model and compute capability:
vLLM artifact-declared compute-capability support or UNKNOWN:
target-native GPU kernel-smoke result:
AFD plugin-owned compiled GPU sources: paths | NOT_APPLICABLE with evidence
AFD clean package build and extension import: result | NOT_APPLICABLE
AFD compiled GPU operator smoke: result | NOT_APPLICABLE
```

Use the target source to discover the current native module name and build
metadata; do not assume a permanent module path. Final validation always
requires a real target-native kernel smoke on the recorded GPU. If published
metadata does not enumerate compiled architectures, record that field `UNKNOWN`;
the smoke proves execution without binary fingerprinting or speculative
inspection. Do not rebuild vLLM when `AGENTS.md` requires the installed runtime;
perform a source build only when the approved upgrade scope explicitly includes
build qualification.

When AFD has plugin-owned compiled GPU sources, build the package cleanly in the
target stack, import the produced extension from that environment, and execute
its smallest real operator on a reserved target GPU. When it has none, record
the inspected paths and `NOT_APPLICABLE`; do not treat Python-registered custom
ops as compiled AFD extensions.

## Subagent handoff ledger

Record each delegated task before dispatch. Do not assign overlapping files to
concurrent implementation agents.

| Agent/task | Role | Objective/root cause | Allowed files/commands | Dependencies | Status/evidence |
| --- | --- | --- | --- | --- | --- |
| | analysis / implementation / test / review | | | | |

Use this template when the task needs a detailed handoff:

```text
Role:
Objective or root cause:
Task mode:
Current and target vLLM refs/full SHAs:
vLLM source/worktree paths:
AFD checkout/worktree and fixed base HEAD:
Starting diff:
Allowed AFD files:
Allowed commands:
Required invariants and focused tests:
Primary review checks:
Prohibited actions:
Dependencies:
Exclusive GPU reservation/device IDs:
Exclusive port range:
Read-only model path:
Exclusive cache/temp/log/artifact paths:
Owned process/service instances:
Cleanup owner and command:
Required output:
```

Default prohibited actions are staging, committing, pushing, changing branches,
editing upstream trees, and resetting user changes. Analysis and review agents
are read-only. Test agents must not edit tracked files. Implementation agents
may edit only assigned AFD files and their focused tests.

Serialize implementation in a shared checkout. For safe parallel work, use
primary-created fixed-base worktrees with non-overlapping ownership and integrate
returned raw diffs one at a time. Serialize GPU tests unless every exclusive
resource field is isolated.

## Patch inventory

Inventory all copied and patched upstream behavior, not only files under
`compat/patches`.

| AFD file/symbol | Upstream path/symbol | Current SHA/signature | Target SHA/signature | AFD invariant/markers | Owner/lifetime | Tests | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | remove / delegate-inherit / exact adapter / propose upstream / stop |

Disposition rules:

- `remove`: prove target upstream absorbed the AFD behavior and delete
  registration plus now-obsolete tests only after replacement coverage exists;
- `delegate-inherit`: identify the target extension point and prove ownership,
  ordering, resources, performance, and non-AFD behavior remain correct;
- `exact adapter`: copy the exact target skeleton, replay only required marked
  AFD deltas, and record a removal/upstream plan;
- `propose upstream`: retain only the smallest temporary exact adapter and state
  the concrete upstream extension needed;
- `stop`: use when no narrow invariant-preserving adaptation exists.

## Feature inventory

Expand this seed table from repository source, tests, recipes, and docs. Do not
assume every future release has the same features or test count.

| Feature cell | Current evidence | Target upstream impact | Planned adaptation | Required test/evidence | Status |
| --- | --- | --- | --- | --- | --- |
| package/version/plugin isolation | | | | | |
| GPU runner generation | | | | | |
| Attention worker/model runner | | | | | |
| FFN worker/model runner/daemon | | | | | |
| model construction and loading | | | | | |
| gate/expert ownership | | | | | |
| connector/payload/ordering | | | | | |
| eager / CUDA Graph | | | | | |
| DBO / uBatch | | | | | |
| TP / DP / EP / PP topology | | | | | |
| profiling / warmup / dummy runs | | | | | |
| quantization / EPLB | | | | | |
| serving / request output / sampling | | | | | |
| accuracy | | | | | |
| startup / failure / shutdown / cleanup | | | | | |
| shared-code NPU impact | | | | | |

An omitted feature is `UNKNOWN`. Record unsupported or unvalidated cells
explicitly instead of dropping them from the upgrade scope.

## Upstream-to-AFD impact matrix

| Upstream SHA/path/symbol | Old contract | Target contract | Class | AFD consumers | Preserved invariant | Adaptation | Test | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | mechanical / behavioral / architectural | | | | | |

For hot or distributed methods, record call order and resource ownership. Check
ForwardContext installation, scheduler/request state, KV-cache ownership,
graph capture/replay, DBO stage metadata, process groups, physical device
mapping, P2P ordering, sampling/output ownership, error propagation, and
shutdown where applicable.

## Architecture stop report

```text
Decision: STOPPED — major architecture change

Identity:
- AFD HEAD/base:
- current vLLM ref/full SHA:
- target vLLM ref/full SHA:
- requested CUDA/PyTorch/toolchain tuple:

Change:
- old architecture/interface:
- target architecture/interface:
- exact upstream evidence:

AFD impact:
- affected files/features:
- invariant that no longer maps:
- why target-upstream-plus-small-AFD-diff is impossible:
- why feature deletion, validation bypass, reflection, or broad copying is unsafe:

Decision required:
- smallest maintainer/design choice:
- candidate directions, without implementation:

Worktree:
- production edits made: none
- unrelated dirty paths preserved:
```

## Staged upgrade plan

Use this table only after the architecture gate passes. When an earlier gate is
blocked, replace it with a short blocker-resolution plan listing the missing
identity/source/environment evidence and next allowed actions; do not populate
implementation rows from historical lessons.

| Stage | Upstream cause | Expected AFD files | Preserved invariant | Patch disposition | Tests | Performance/distributed risk | Docs/removal plan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

Order stages by actual dependencies. Keep mechanical target-source refreshes
reviewable separately from AFD behavior changes when practical. Each stage must
name evidence that would prove it complete.

## Commit discipline

Create local commits only in implementation mode when commits are in scope.
Keep one upstream cause or validated failure root cause with its focused tests.
Stage explicit paths, inspect the staged diff, and never include unrelated work.

Suggested form after checking repository history:

```bash
git commit -s -m 'fix(gpu-upgrade): adapt <component> to <target>' \
  -m 'Upgrade requirement or failure:
- <contract change, test, or runtime symptom>

Upstream cause:
- vLLM <old-sha>..<target-sha>: <exact change>

Why this change:
- <correct AFD adaptation layer>
- <preserved invariant and why the patch is minimal>

Validation:
- <exact focused command and result>
- <regression command and result>'
```

Use the appropriate `feat`, `fix`, `refactor`, `test`, `docs`, or `chore` type.
Do not commit speculative code that still fails its focused gate, and do not
amend unrelated earlier commits to hide later failures.

## Validation ledger

```text
Validation cell:
  purpose:
  AFD commit/tree:
  vLLM ref/full SHA/package build:
  Python/PyTorch/build CUDA/C++11 ABI:
  vLLM native artifact path/provenance:
  CUDA runtime/toolkit/driver/compiler/NCCL:
  host/image and GPU topology:
  target compute capability and native kernel smoke:
  AFD compiled GPU extension build/import/smoke or NOT_APPLICABLE:
  scheduler reservation:
  model/checkpoint/quantization:
  runner generation:
  connector and Attention/FFN topology:
  TP/DP/EP/PP:
  eager/graph/DBO/uBatch/profiling:
  prompt/sampling/accuracy inputs:
  command and relevant environment:

Result:
  status: PASS | FAIL | BLOCKED | SKIPPED
  collected/passed/failed/error/skipped counts:
  metrics/oracle:
  first useful root traceback:
  log/artifact paths:
  skip reasons:
  process/port/GPU/IPC/scheduler cleanup:
```

Use this sequence:

1. static, format, compile, import, package, native build/ABI, and exact patch
   checks;
2. affected focused tests, then complete CPU/unit suite;
3. exact target native vLLM GPU control;
4. complete GPU feature E2E;
5. complete GPU model E2E;
6. every full GPU accuracy case with no final dataset limit;
7. documentation refresh and claim reconciliation;
8. final version/source/docs/recipe/metadata audit;
9. version-specific lesson when implementation completion is in scope.

After a code failure, preserve the failing artifacts, rerun the narrow
reproduction after the fix, then rerun the complete earlier gates. A passing
retry does not erase the original failed cell.

## Upgrade lesson template

After a completed implementation upgrade, create
`references/<YYYYMMDD>-<current>-to-<target>-lessons.md`. Sanitize refs, use a
short resolved SHA when no immutable tag exists, and add a suffix instead of
overwriting an existing file.

```markdown
# Lessons from the <current> to <target> GPU upgrade

Use this file as historical evidence, not as a recipe. Re-derive every fact from
the exact revisions of the next upgrade.

## Runtime identity and scope

- AFD before/after commits:
- current and target vLLM refs/full SHAs:
- Python/PyTorch/CUDA/driver/compiler/NCCL/GPU topology:
- models, connectors, modes, parallelism, and exclusions:

## Upstream changes that mattered

- changed contract and evidence:
- affected AFD invariant:
- mechanical, behavioral, or architectural classification:

## Adaptations and patch lifecycle

- smallest correct adaptation seam:
- patches retained, delegated, removed, ported, or proposed upstream:
- commits and focused tests:

## Failures and root causes

- symptom and first useful traceback:
- upstream or AFD root cause:
- misleading attempts or assumptions:
- final fix and proof:

## Validation and environment lessons

- checks that exposed real issues:
- CUDA, scheduler, topology, model, port, cache, or cleanup pitfalls:
- gaps between unit, native, E2E, accuracy, and performance evidence:

## Documentation and process lessons

- stale or conflicting claims:
- workflow, review, or commit practices that helped or failed:

## Reusable principles

- lessons likely to apply beyond this version pair:

## Version-specific facts

- facts that must not be generalized without revalidation:

## Remaining debt and next-upgrade checklist

- unsupported or unvalidated cells:
- temporary adapters and removal triggers:
- symbols, commands, and invariants to inspect first next time:
```

Keep lessons concise and evidence-backed. Link commits, tests, logs, or source
symbols rather than embedding large outputs. Exclude credentials, private
endpoints, and machine-specific secrets. Review the frozen lesson diff before a
signed-off documentation commit.

## Completion report

```text
Task mode and requested stopping point:

Runtime identity:
- AFD path/branch/HEAD/base/target version:
- current and target vLLM refs/full SHAs:
- target source/worktree paths:

Target environment:
- Python/PyTorch/CUDA/driver/compiler/NCCL:
- host/image/GPU topology/scheduler:
- model paths and validation scope:

Architecture gate:
Impact summary and staged plan:
Patch inventory and removal/upstream plans:
Feature matrix:

Changes and commits:
- <sha> <title> — <upstream cause/root cause>

CPU-safe validation:
- commands/results/skips:

GPU validation:
- native control:
- features:
- models:
- full accuracy:
- actual counts, metrics, logs, skips, and cleanup:

Documentation refresh and claim reconciliation:
Intentionally preserved historical references:
Final version/source/docs audit:
Upgrade lesson path/commit or proposed lesson notes:

Unsupported/unvalidated/excluded:
Remaining private seams and performance observations:
Release blockers:
Readiness: GPU-validated | implemented-GPU-unverified | planning-only | blocked
Shared-code and NPU claim status:
Branch/publish status:
```
