# End-to-End Tests

These tests validate DeepSeek-V2-Lite on real GPU or Ascend NPU hardware.
They run four scenarios:

- `baseline-graph`
- `afd-eager`
- `afd-graph`
- `afd-graph-dbo`

Each scenario evaluates the first 7 GSM8K samples. AFD uses three devices:
the first two for Attention and the third for FFN. Tests run sequentially and
must not skip. Every GSM8K evaluation uses 8 few-shot examples.

See the [E2E testing design](../../docs/design/module/e2e_testing.md) before
adding a model or case.

## Run

Run from the repository root. The environment needs `vllm`, `pytest`,
`afd_plugin`, and `lm_eval`. NPU also needs `torch_npu`.

Set the cache location and prepare GSM8K once:

```bash
export HF_HOME=/path/to/huggingface
python -c 'from datasets import load_dataset; load_dataset("openai/gsm8k", "main")'
```

GPU:

```bash
export AFD_E2E_BACKEND=gpu
export AFD_E2E_DEVICES=0,1,2
export AFD_GPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
```

NPU:

```bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
```

Then run:

```bash
python -m pytest -q -s \
  tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py
```

Success means 4 passed and 0 skipped.

For the weekly full GSM8K test, run only `afd-graph-dbo`:

```bash
export AFD_GSM8K_LIMIT=all
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo]"
```

This evaluates all 1319 GSM8K test samples. Without `AFD_GSM8K_LIMIT`, each
scenario evaluates the first 7 samples.

## Run with the Codex skill

The repository includes the [`run-e2e`](../../.agents/skills/run-e2e/SKILL.md)
skill. Open the repository in Codex and ask, for example:

```text
Use run-e2e to run the GPU E2E tests with model /models/DeepSeek-V2-Lite,
devices 0,1,2, and HF_HOME /data/huggingface.
```

For either backend, provide the model path, three device IDs, and `HF_HOME`.
The skill checks prerequisites, runs the same four tests, and reports failures
and process cleanup.
