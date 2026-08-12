---
name: run-e2e
description: Use when the user asks to run, validate, or diagnose the AFD plugin's DeepSeek-V2-Lite end-to-end tests on GPU or Ascend NPU hardware, including PR-gate E2E, GSM8K-7 accuracy, graph, eager, or DBO scenarios.
---

# Run AFD E2E Tests

## Scope

Run the four tests in
tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py:

- baseline-graph
- afd-eager
- afd-graph
- afd-graph-dbo

Each scenario evaluates the first 7 GSM8K samples with 2 Attention ranks and 1
FFN rank. Do not use removed markers, categories, TP, or 2A2F workflows. This
skill does not cover unit tests or source edits.

## Workflow

### 1. Select the backend

Honor an explicit backend. Otherwise inspect nvidia-smi -L and npu-smi info.
If both are available, ask which to use. If neither is available, stop.

### 2. Validate prerequisites

Before starting pytest, confirm:

- AFD_E2E_DEVICES contains the device IDs required by test cases.
- The backend model variable points to an available model.
- The selected vllm command runs.
- pytest, afd_plugin, and lm_eval are importable.
- HF_HOME points to the Hugging Face cache used for GSM8K.
- GPU: the selected devices are visible to CUDA.
- NPU: torch_npu and the Ascend runtime work.

Install missing lm_eval only in the runner environment, never in pyproject.toml
or uv.lock.

Fail before pytest when a prerequisite is missing; never turn it into a skip.

Set HF_HOME before every run. If GSM8K is not cached, prepare it before
starting the cluster:

~~~bash
export HF_HOME=/path/to/huggingface
python -c 'from datasets import load_dataset; load_dataset("openai/gsm8k", "main")'
~~~

### 3. Configure the run

For GPU:

~~~bash
export AFD_E2E_BACKEND=gpu
export AFD_E2E_DEVICES=0,1,2
export AFD_GPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
# Optional: export AFD_GPU_E2E_VLLM_BIN=/path/to/vllm
~~~

For NPU:

~~~bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
# Optional: export AFD_NPU_E2E_VLLM_BIN=/path/to/vllm
~~~

Device order defines roles: the first two devices run Attention and the third
runs FFN. baseline-graph uses only the first device.

### 4. Run

From the repository root, stream output in the foreground:

~~~bash
python -m pytest -q -s \
  tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py
~~~

Do not add backend markers or run scenarios in parallel; they share devices.

On cancellation, forward SIGTERM and allow over 90 seconds for cleanup.

### 5. Report

Success means 4 passed and 0 skipped. Report the failed scenario, first
actionable error, and cleanup status. Any skip is a gate failure.

## Environment reference

| Variable | Backend | Required |
|---|---|---|
| AFD_E2E_BACKEND | both | yes: gpu or npu |
| AFD_E2E_DEVICES | both | yes: exactly 3 unique IDs |
| AFD_GPU_E2E_MODEL | GPU | yes |
| AFD_GPU_E2E_VLLM_BIN | GPU | no; defaults to vllm |
| AFD_NPU_E2E_MODEL | NPU | yes |
| AFD_NPU_E2E_VLLM_BIN | NPU | no; defaults to vllm |
| HF_HOME | both | yes; GSM8K cache location |
