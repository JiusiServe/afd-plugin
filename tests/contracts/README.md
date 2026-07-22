# NPU ModelRunner contract snapshots

`npu_model_runner_v0191rc1.json` freezes the source-level compatibility
surface used by the Phase 0 baseline:

- vLLM `v0.19.1`
- vLLM-Ascend `v0.19.1rc1`
- the AFD NPU attention ModelRunner in the current checkout

It records selected ModelRunner method signatures, upgrade-sensitive
metadata fields, and external runtime imports. It does not compare against
`main`; candidate-version comparison starts only after the current-version
refactor passes its release gate.

Regenerate the snapshot from exact local Git refs:

```bash
python -m tools.compat.snapshot_npu_model_runner_contract \
  --vllm-root ../vllm \
  --vllm-ref v0.19.1 \
  --vllm-ascend-root ../vllm-ascend \
  --vllm-ascend-ref v0.19.1rc1 \
  > tests/contracts/npu_model_runner_v0191rc1.json
```

Review the JSON diff before accepting any regenerated snapshot. A changed
snapshot is an explicit compatibility decision, not an automatic update.
