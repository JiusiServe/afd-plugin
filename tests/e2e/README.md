# End-to-End Tests

These tests validate DeepSeek-V2-Lite on real GPU or Ascend NPU hardware.
The default gate runs four scenarios:

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
`afd_plugin`, `lm_eval`, `datasets`, and `huggingface_hub`. NPU also needs
`torch_npu`.

The test downloads/caches `openai/gsm8k` and
`deepseek-ai/DeepSeek-V2-Lite` when the backend model env var is unset. Point
`HF_HOME` at a persistent cache if you want to reuse downloads across runs.

GPU:

```bash
export HF_HOME=/path/to/huggingface
export AFD_E2E_BACKEND=gpu
export AFD_E2E_DEVICES=0,1,2
# Optional if the model is already local:
# export AFD_GPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
```

NPU:

```bash
export HF_HOME=/path/to/huggingface
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2
# Optional if the model is already local:
# export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
```

Then run:

```bash
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-eager]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[baseline-graph]"
```

Success means 4 passed and 0 skipped.

### Graph + DBO 2A2F

This separate GPU/NPU case uses four devices: the first two for Attention
DP=2/TP=1 and the last two for FFN DP=2/TP=1/EP=2. It runs GSM8K-7.

```bash
export AFD_E2E_DEVICES=0,1,2,3
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo-2a2f]"
```

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
Use run-e2e to run the GPU E2E tests with devices 0,1,2 and
HF_HOME /data/huggingface.
```

For either backend, provide three device IDs and `HF_HOME`. The model path is
optional when Hugging Face download is available. The skill checks
prerequisites, runs the same four tests, and reports failures and process
cleanup.

## NPU async CAM smoke test

This separate test uses four NPUs: the first two for Attention TP=2, and the
last two for FFN DP=2/TP=1/EP=2. It sends one prompt and requests 32 tokens. It
does not run GSM8K.

```bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2,3
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
python -m pytest -q -s \
  tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py
```

The CAM/CANN runtime and custom operators must already be installed. Missing
model configuration or a device list other than four unique IDs fails the
test.
