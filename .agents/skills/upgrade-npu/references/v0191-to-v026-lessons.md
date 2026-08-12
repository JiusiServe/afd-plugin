# Lessons from the v0.19.1rc1 to v0.26.0 upgrade

Use this history as a source of failure patterns, not as a target-version recipe.
Always re-derive facts from the exact current and target revisions.

## Scope of the historical work

PR [#183](https://github.com/vllm-project/afd-plugin/pull/183) was the first NPU
compatibility slice, not the entire upgrade. Its final commit `e6e49417` changed
five files and covered NPU runner API drift, DeepSeek MoE factory binding, and
the W8A8 force-load-balance patch. The complete NPU transition also required:

- PR [#184](https://github.com/vllm-project/afd-plugin/pull/184): remove the PCP
  path no longer supported by v0.26 ModelRunner V1;
- PR [#185](https://github.com/vllm-project/afd-plugin/pull/185): restore AFD
  DBO/two-ubatch behavior;
- a later CAM async adaptation;
- PR [#186](https://github.com/vllm-project/afd-plugin/pull/186): integrate the
  upgrade, refresh packages/docs, and address review findings.

The recorded target was vLLM `0.26.0` at commit `568afb3a1` and vLLM-Ascend at
commit `80d8c194f`. Do not reuse this pair for a future upgrade unless it is the
requested target.

## Reusable failure patterns

### API removal is not proof that behavior is obsolete

The Attention runner removed retired KV-compression and argsort hooks and
aligned changed method signatures. Before deleting an old hook, verify whether
target upstream removed, absorbed, or conceptually replaced its behavior. An
import error alone is insufficient evidence.

### Plugin initialization order is part of the contract

The native DeepSeek module could bind `FusedMoE` before vLLM-Ascend installed
its factory patch. The NPU FFN path then constructed the wrong MoE runner even
though all symbols imported. The adaptation refreshed the factory before FFN
construction and added a test that observed the factory actually used by the
native MoE constructor.

Always test module import order, plugin registration order, factory binding,
and object construction—not only symbol availability.

### Patch ownership and state lifetime can move

The force-load-balance patch moved from `AscendFusedMoE` construction to
`AscendW8A8DynamicFusedMoEMethod` construction/apply. The successful rebase:

- recopied the exact target functions and signatures;
- captured AFD configuration while `get_current_vllm_config()` was valid;
- delayed device-dependent buffer creation until apply;
- kept state on the new quant-method owner;
- avoided rereading invalid config context or mutating temporary layer topology;
- tested disabled passthrough, lazy initialization, growth, determinism, and
  balance across EP ranks.

When a patch target moves, establish the new owner, construction lifetime,
forward lifetime, and device lifetime before replaying AFD differences.

### Feature removal or disabling is an architecture decision

PCP support disappeared from the target ModelRunner V1 contract. DBO was reset
or unsupported by the target Ascend platform and required an AFD-owned runtime
path. Future upgrades must stop at analysis when an equivalent situation is
found. Do not silently delete a supported feature or bypass an upstream
validation decision.

### Excluded connector paths become upgrade debt

PR #183 explicitly excluded CAM async, which needed a later adaptation. Build a
complete connector/feature inventory before editing and list every exclusion.
An excluded path prevents a claim of full NPU upgrade unless the user accepts a
narrower scope.

### Rank is not necessarily the physical device

After placement and visible-device remapping changes, world-group local rank was
not a reliable CAM device ID. Use the physical device selected by the model
runner and test asymmetric topologies. Equal Attention/FFN rank counts can hide
aggregation and subgroup mapping errors.

### Exact signature means exact

Review found that a return annotation difference alone violated the repository
patch rule. Compare parameters, ordering, defaults, decorators, and return
annotations against the exact pinned target source.

### Upgrade documentation is part of compatibility

The upgrade needed support-matrix, recipe, installation, patch-inventory, and
generated-metadata cleanup. Search the whole repository for stale version
strings, but preserve references intentionally scoped to historical branches or
recipes.

## Historical validation sequence

The useful pattern was:

1. Ruff, format/compile, `git diff --check`, and focused unit tests;
2. marker-based non-accuracy NPU E2E;
3. limited eager/graph GSM8K for diagnosis;
4. full GSM8K eager evidence;
5. explicit cleanup and explicit exclusions.

The new skill strengthens this: run the complete non-accuracy gate, then all
full accuracy cases, including eager and graph. Limited accuracy never replaces
final full qualification.

## Historical commit lesson

PR #183 compressed three root causes into one commit and placed most rationale
in the PR body. Prefer smaller commits such as patch ownership, feature removal,
DBO runtime restoration, and CAM async adaptation. Put the failure, upstream
cause, adaptation rationale, and exact validation in every commit body so the
history remains useful without the PR discussion.
