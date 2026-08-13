# Lessons from the v0.19.1 to v0.26.0 GPU upgrade

Use this history as a source of failure patterns, not as a target-version
recipe. Re-derive every applicable fact from the exact current and target vLLM
revisions, AFD contracts, tests, and validation environment of the next upgrade.

## Scope and evidence

The transition was broader than a version-pin edit:

- [RFC #167](https://github.com/vllm-project/afd-plugin/issues/167) defined the
  GPU migration and release-gate expectations.
- [PR #176](https://github.com/vllm-project/afd-plugin/pull/176) developed and
  simplified the DeepSeek remote-experts and GPU runtime adaptation.
- [PR #182](https://github.com/vllm-project/afd-plugin/pull/182) carried the
  reviewed GPU slice onto the v0.26 integration branch and recorded exact-head
  CPU and GPU validation.
- [PR #186](https://github.com/vllm-project/afd-plugin/pull/186) integrated GPU
  and NPU work, refreshed package/docs contracts, and exposed final review
  issues around signatures, topology, inventory, and support wording.

The target recorded by that work was vLLM `0.26.0`. GPU hardware evidence used
NVIDIA L20X and DeepSeek-V2-Lite with ModelRunner V1. These are historical
version facts, not defaults for a future upgrade.

## Reusable failure patterns

### A version pin is only the identity change

The real migration crossed model construction and loading, Attention and FFN
workers/model runners, EngineCore lifecycle, P2P metadata, graph capture, DBO,
parallelism, profiling, package isolation, recipes, and documentation. Build an
affected-surface and feature inventory before editing; otherwise a passing
import can hide unported runtime paths.

### Prove a new public interface is necessary

The early PR #176 design added experts-specific connector methods, headers,
transfer IDs, grouped operations, and a second protocol. Review showed the
existing Attention-to-FFN and FFN-to-Attention data path could carry the needed
ordered router logits. Removing the extra abstractions substantially reduced
the patch while preserving the native MoE boundary.

For future upgrades, treat new public connector, protocol, coordinator, or
runner abstractions as an architecture-gate decision. First prove that the
target native contract and existing AFD interface cannot express the required
invariant.

### Preserve native lifecycle instead of copying churn

The successful model boundary retained native model, decoder-layer, and MoE
forward behavior while AFD owned role-aware construction, remote expert
handoff, and weight filtering. Routed/shared experts remained on FFN and
request/KV/sampling ownership remained on Attention.

When target upstream changes ordinary forward, loading, compile, graph, or
parallel behavior, prefer delegation through the new native lifecycle. Copy
only the constructor or method for which no stable injection point exists.

### Ordered transport includes metadata semantics

Adding an optional tensor to an existing P2P path is not merely a send/receive
change. The upgrade had to preserve operation order, graph replay, DBO stage
identity, gate placement, shape/dtype ownership, and FFN-side consumption.
Review the producer, wire allocation, consumer, control plane, dummy/warmup
path, and cleanup as one contract.

### Patch inventory extends beyond the patch directory

Patched or copied upstream methods also lived in model and worker modules. Final
review found that even one return annotation different from target upstream
violated the exact-signature requirement. Compare decorators, method type,
parameters, defaults, and return annotation, not just callable behavior.

Keep design patch inventories synchronized with actual assignments and state
owners. A stale inventory causes the next upgrade to audit the wrong symbol.

### Equal-rank topology can hide aggregation bugs

Final review identified an FFN DP metadata issue for asymmetric Attention:FFN
layouts: P2P fan-in aggregated token tensors while native MoE could still see
raw per-Attention counts. Symmetric 1A1F or 2A2F tests did not prove 4A2F-style
subgroup aggregation.

Inventory topology formulas and test at least one asymmetric valid layout when
the supported contract allows it. Distinguish wire-allocation counts from the
metadata consumed by target native compute.

### Focused tests do not complete the exact-head gate

PR #176 had focused eager, graph, parallel, DBO, and limited accuracy evidence,
but explicitly had not rerun the complete repository GPU matrix on its final
head. That head was therefore an implementation candidate, not final release
evidence. PR #182 later recorded complete marker-based execution for the frozen
integration revision.

Tie every result to the exact AFD tree and runtime identity. Evidence from an
earlier commit remains historical after a behaviorally relevant change until
the affected and required regression gates are rerun.

### Configuration is not proof of execution

Graph and DBO support required evidence that graph capture/replay occurred and
that both ubatch stages executed, not only that flags were present. Parallelism
requires collective and topology behavior, not server startup. Profiling
requires usable traces from both roles where claimed.

Record observable execution evidence for the feature being claimed. Do not
promote launch success into graph, DBO, TP/DP/EP, accuracy, or performance
support.

### Accuracy scope must remain explicit

The recorded GPU integration used limited GSM8K samples for eager and graph.
That was useful diagnostic evidence, but it was not a full-dataset final
qualification. Report dataset size, threshold, metric, mode, and exact cell.
Never let a limited run become the generic accuracy claim for the release.

### Native control separates runtime failure from AFD failure

A target model, wheel, CUDA stack, or kernel may fail before AFD participates.
Establish an exact native eager control before interpreting AFD failures. A
fallback mode is a separate cell; do not silently turn it into the requested
target or patch AFD around a native runtime defect.

### Cleanup is part of correctness evidence

The useful GPU evidence included process, port, and selected-device cleanup.
Distributed success followed by leaked workers, communicators, IPC state, or
scheduler reservations is not a clean pass. Preserve cleanup output with the
run and investigate post-response shutdown errors separately from request
correctness.

### Documentation claims require their own audit

The integration review found stale generated metadata, patch descriptions, and
support wording. Current documentation and intentionally historical recipes
needed different treatment: active target claims were updated, while old
branch-specific material stayed historical.

Search broadly, inspect each result in context, and distinguish GPU-validated,
shared-code, unit-only, unvalidated, and unsupported claims. A GPU upgrade does
not prove NPU support merely because shared code changed.

## Historical validation sequence

The useful sequence was:

1. diff, lint, format, compile/import, package, and focused contract tests;
2. complete CPU/unit coverage with expected backend skips identified;
3. focused GPU eager cases for both gate placements;
4. graph, TP/EP, DP/EP, and representative DBO cells with execution evidence;
5. the complete repository GPU marker suite on the frozen integration head;
6. limited eager/graph GSM8K reported as limited scope;
7. process, port, device, and scheduler cleanup;
8. integration review of signatures, topology, docs, and support claims.

Future upgrades must discover the current repository tests and run the full
accuracy scope required by the current skill. Do not preserve the historical
test count or limited dataset as a permanent gate.

## Reusable principles

- Derive architecture from exact source before implementation.
- Prefer target native behavior plus small AFD differences.
- Keep public interface expansion behind an explicit architecture decision.
- Freeze supported feature and topology matrices before editing.
- Revalidate behaviorally affected evidence on the exact final tree.
- Separate functional, accuracy, performance, cleanup, and documentation claims.
- Preserve historical evidence without turning it into current support.

## Version-specific facts

The following facts belong only to this historical transition and must be
revalidated for another target:

- target vLLM was `0.26.0`;
- the selected GPU path used ModelRunner V1;
- hardware evidence cited NVIDIA L20X and DeepSeek-V2-Lite;
- CUDA Graph claims were scoped to the modes implemented at that revision;
- the repository GPU suite had a particular collected count at that time;
- final GPU accuracy evidence in the integration record was limited-sample, not
  a permanent full-qualification policy;
- quantized GPU MoE, EPLB, and ModelRunner V2 were excluded from that slice.

## Historical process lesson

The work benefited from a draft for architecture simplification, a separate
exact-source GPU integration commit, and a final combined integration review.
Future upgrades need not reproduce those branch names or PR boundaries, but
should keep architectural decisions, coherent adaptations, exact-head hardware
evidence, and final claim reconciliation independently reviewable.
