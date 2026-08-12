# NPU upgrade workbook

Use these tables during an upgrade. Keep them in task notes or the final report;
do not add generated reports to the repository unless the user requests one.

## Contents

1. Identity ledger
2. Source resolution and diff commands
3. Patch and feature inventories
4. Impact matrix
5. Architecture stop report
6. Commit discipline
7. Validation ledger
8. Completion report

## Identity ledger

```text
AFD:
  path:
  branch:
  HEAD:
  dirty paths:

Runtime revisions:
  vllm_source_directory: path, HEAD, branch, dirty status
  vllm_ascend_source_directory: path, HEAD, branch, dirty status
  current_vllm: input/ref, resolved SHA, evidence
  current_vllm_ascend: input/ref, resolved SHA, evidence
  target_vllm: input/ref, resolved SHA, evidence
  target_vllm_ascend: input/ref, resolved SHA, evidence
  target_vllm_worktree: reused source directory or detached worktree path
  target_vllm_ascend_worktree: reused source directory or detached worktree path

Target stack:
  Python:
  PyTorch:
  torch_npu:
  CANN:
  Triton Ascend:
  CAM/operator packages:
  model/checkpoint and quantization:
  available NPU count/type:
```

Record every version source and resolve conflicts explicitly. A useful evidence
order is dependency pin/lockfile, source annotation, vLLM-Ascend commit lock,
Dockerfile, compatibility matrix, installed package, adjacent checkout.

## Source resolution and diff commands

Use placeholders literally; do not check out refs in user-provided source
directories. Resolve the provided checkout identity first. Reuse it only when
its HEAD matches the required target SHA; otherwise create a separate detached
target worktree from that repository.

```bash
git -C <repo> rev-parse --verify '<ref>^{commit}'
git -C <repo> rev-parse HEAD
git -C <repo> status --short --branch
git -C <repo> worktree add --detach <target-worktree-path> <target-sha>
git -C <repo> show '<ref>:<path>'
git -C <repo> diff --find-renames --name-status <old-sha>..<new-sha>
git -C <repo> diff --find-renames --stat <old-sha>..<new-sha>
git -C <repo> log --first-parent --reverse --oneline <old-sha>..<new-sha>
git -C <repo> diff --function-context <old-sha>..<new-sha> -- <focus-paths>
```

For vLLM-Ascend pairing evidence:

```bash
git -C <vllm-ascend> show <ascend-ref>:.github/vllm-main-verified.commit
git -C <vllm-ascend> show <ascend-ref>:.github/vllm-release-tag.commit
git -C <vllm-ascend> show <ascend-ref>:Dockerfile
```

For AFD source discovery:

```bash
rg -n '^(from|import) (vllm|vllm_ascend)' afd_plugin tests
rg -n 'PATCH START|PATCH END|Upstream source|Upstream:' afd_plugin tests
rg -n 'vllm|vllm-ascend|VLLM_TAG|VLLM_COMMIT' \
  pyproject.toml uv.lock README.md docs recipe .github afd_plugin tests
rg -n 'NPUWorker|NPUModelRunner|forward_context|ACLGraph|ubatch|DBO|PCP|CAM' \
  afd_plugin tests
```

Compare an AFD copied function with both old and target upstream definitions.
Check the full signature, defaults, return annotation, decorator, method type,
body order, state owner, and every local marker block.

## Patch inventory

| AFD file/symbol | Upstream repo/path/symbol | Current SHA/signature | Target SHA/signature | AFD invariant and markers | State owner/lifetime | Tests | Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | keep / rebase / replace / remove / stop |

For `remove`, prove the target upstream absorbed the behavior. For `replace`,
prove the target exposes an equivalent stable seam. Otherwise use `stop`.

## Feature inventory

| Feature cell | Current support evidence | Target upstream impact | Planned adaptation | Required NPU test | Status |
| --- | --- | --- | --- | --- | --- |
| runner generation | | | | | |
| eager / graph | | | | | |
| DBO / uBatch | | | | | |
| PCP | | | | | |
| sync CAM / async CAM | | | | | |
| Attention / FFN gate | | | | | |
| TP / DP / EP topology | | | | | |
| W8A8 / EPLB / force balance | | | | | |
| speculative / MTP | | | | | |

Expand the table with every repository-owned scenario. An omitted feature is an
unknown, not an implicit exclusion.

## Upstream-to-AFD impact matrix

| Upstream | Old contract | Target contract | Classification | AFD consumers | Preserved invariant | Adaptation | Test | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vLLM / Ascend + SHA/path/symbol | | | mechanical / behavioral / architectural | | | | | |

For execution methods, note ordering constraints. Examples include consuming
finished-request state before removal, installing forward context before model
execution, updating graph parameters at the correct phase, and using the model
runner-selected physical NPU rather than assuming rank equals device.

## Architecture stop report

```text
Decision: STOPPED — major architecture change

Runtime pair:
- current vLLM / vLLM-Ascend SHAs:
- target vLLM / vLLM-Ascend SHAs:

Change:
- old architecture and interface:
- target architecture and interface:
- upstream evidence:

AFD impact:
- affected files and features:
- invariant that no longer maps:
- why target-upstream-plus-small-AFD-diff is impossible:
- why deletion, validation bypass, reflection, or broad copying is unsafe:

Decision required:
- smallest maintainer/design choice:
- candidate directions, without implementing them:

Worktree:
- production edits made: none
- unrelated dirty paths preserved:
```

## Commit discipline

Create a commit only for a coherent adaptation or a validated code-failure fix.
Keep its source change and focused regression tests together. Stage explicit
paths and inspect the staged diff before committing. Never stage unrelated
files.

Suggested command after verifying repository convention:

```bash
git commit -s -m 'fix(npu-upgrade): adapt <component> to <target>' \
  -m 'Failure:
- <test or analysis symptom>

Upstream cause:
- vLLM <old-sha>..<target-sha>: <contract change or none>
- vLLM-Ascend <old-sha>..<target-sha>: <contract change or none>

Why this change:
- <why this AFD layer is the correct adaptation point>
- <behavioral invariant preserved and why the patch is minimal>

Validation:
- <exact focused command and result>
- <regression command and result>'
```

For an analysis-planned adaptation that precedes a test failure, replace
`Failure` with `Upgrade requirement`. Use `feat`, `fix`, `refactor`, `test`,
`docs`, or `chore` according to repository history; do not force every commit
into `fix`.

Do not commit a speculative change that still fails its focused check. Continue
working within the same root cause, then commit the complete minimal fix. Do not
amend an earlier, unrelated upgrade commit to hide a later failure.

## Validation ledger

```text
Validation cell:
  AFD commit:
  vLLM SHA/version:
  vLLM-Ascend SHA/version:
  Python/PyTorch/torch_npu/CANN:
  model/checkpoint/quantization:
  topology and physical devices:
  connector:
  eager/graph/DBO/uBatch:
  command/environment:

Result:
  status: PASS | FAIL | BLOCKED | SKIPPED
  counts/metrics:
  first root traceback:
  log/artifact paths:
  skip reasons:
  cleanup:
```

Use this sequence:

1. static, format, compile, import, and exact signature/marker checks;
2. affected unit tests, then full CPU/unit suite;
3. target native NPU real request;
4. all NPU feature E2E;
5. all NPU model/connector E2E;
6. full NPU accuracy eager and graph with no GSM8K limit;
7. documentation refresh using `documentation-refresh.md`;
8. final version/docs/generated-metadata consistency audit.

After a code failure, rerun the narrow reproduction, the complete basic gate,
then any later gates. Preserve failure logs instead of overwriting them with a
passing retry.

## Completion report

```text
Runtime identity:
Target software stack:
Architecture gate:

Impact summary:
Patch inventory and removal plans:
Feature matrix:

Commits:
- <sha> <title> — <root cause>

Validation:
- static/unit:
- native NPU control:
- NPU features:
- NPU models/connectors:
- NPU full accuracy eager/graph:
- skips and reasons:
- cleanup:

Documentation refresh:
- evidence and reconciled support claims:
- public, recipe, design, template, and metadata changes:
- intentionally preserved historical references:
- documentation commit SHAs:

Final version/docs audit:
Unsupported/unvalidated/excluded:
Branch/publish status:
```
