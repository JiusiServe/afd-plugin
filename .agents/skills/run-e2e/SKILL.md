---
name: run-e2e
description: Use when the user asks to run, validate, or diagnose the AFD plugin's DeepSeek-V2-Lite GPU/NPU, Qwen3 MoE GPU, Qwen3.6 MoE CUDA, or opt-in Qwen3.5-122B CUDA end-to-end tests, including PR-gate E2E, GSM8K accuracy, graph, eager, DBO, or AFD topology scenarios.
---

# Run AFD E2E Tests

## Scope

Run one of the model suites:

- `tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py` on GPU or NPU
- `tests/e2e/models/qwen3_moe/test_qwen3_moe.py` on GPU
- `tests/e2e/models/qwen3_6/test_qwen3_6.py` on CUDA (text-only Qwen3.6
  evidence for the Qwen3.5/3.6 adapter family)
- `tests/e2e/models/qwen3_5/test_qwen3_5_122b.py` on eight CUDA devices
  (manual, eager-only large-model profile)

Each suite contains four gate scenarios:

- baseline-graph
- afd-eager-2a2f
- afd-graph-2a2f
- afd-graph-dbo-2a2f

Each gate scenario evaluates the first 7 GSM8K samples. The AFD gate scenarios
use 2 Attention ranks and 2 FFN ranks. `baseline-graph` uses native
DP4/TP1/EP4. The 2A1F cases (`afd-eager-2a1f`, `afd-graph-2a1f`,
`afd-graph-dbo-2a1f`) are local-only scenarios.

## Workflow

### 1. Select the backend

Honor an explicit backend. Otherwise inspect nvidia-smi -L and npu-smi info.
If both are available, ask which to use. If neither is available, stop.

### 2. Validate prerequisites

Before starting pytest, confirm:

- AFD_E2E_DEVICES contains the device IDs required by test cases.
- The backend model variable is set to a local path, or the environment can
  download the selected suite's checkpoint via huggingface_hub.
- The selected vllm command runs.
- pytest, afd_plugin, lm_eval, datasets, and huggingface_hub are importable.
- HF_HOME points to the Hugging Face cache used for GSM8K and model weights.
- GPU: the selected devices are visible to CUDA.
- NPU: torch_npu and the Ascend runtime work.

Install missing lm_eval only in the runner environment, never in pyproject.toml
or uv.lock.

Fail before pytest when a prerequisite is missing; never turn it into a skip.

Set HF_HOME before every run. Default pytest entrypoints download/cache GSM8K
and the model when the backend model env var is unset. The Qwen3.5-122B
profile only caches GSM8K and never downloads its checkpoint.

### 3. Configure the run

For GPU:

~~~bash
export HF_HOME=/path/to/huggingface
export AFD_E2E_BACKEND=gpu
export AFD_E2E_DEVICES=0,1,2,3
# Optional if the model is already local:
# export AFD_GPU_E2E_MODEL=/path/to/model
# Optional: export AFD_GPU_E2E_VLLM_BIN=/path/to/vllm
~~~

For NPU:

~~~bash
export HF_HOME=/path/to/huggingface
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2,3
# Optional if the model is already local:
# export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
# Optional: export AFD_NPU_E2E_VLLM_BIN=/path/to/vllm
~~~

Device order defines roles: the 2A2F AFD scenarios use the first two devices
for Attention DP2/TP1 and the last two for FFN DP2/TP1/EP2. `baseline-graph`
uses the first four for native DP4/TP1/EP4. The local 2A1F scenarios use the
first two for Attention and the third for FFN.

### 4. Run

From the repository root, stream output in the foreground:

~~~bash
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[baseline-graph]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-eager-2a2f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-2a2f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo-2a2f]"
~~~

For Qwen3 MoE on GPU, run:

~~~bash
python -m pytest -q -s \
  tests/e2e/models/qwen3_moe/test_qwen3_moe.py
~~~

For Qwen3.6 MoE on CUDA, run:

~~~bash
export AFD_E2E_BACKEND=gpu
export AFD_E2E_DEVICES=0,1,2,3
export AFD_GPU_E2E_MODEL=/path/to/Qwen3.6-35B-A3B
python -m pytest -q -s \
  tests/e2e/models/qwen3_6/test_qwen3_6.py
~~~

The Qwen3.6 suite uses `Qwen/Qwen3.6-35B-A3B`, vLLM V1, and passes
`--language-model-only` to every runner invocation. It is CUDA-only and
text-only: `baseline-graph` is native DP4/TP1/EP4, while the AFD scenarios are
synchronous 2A1F with Attention on the first two devices and FFN on the third.
Multimodal, NPU, `compute_gate_on_attention=true`, asynchronous,
pipeline-parallel, and multi-node execution are not covered; quantization is
unverified.

For the Qwen3.5-122B large-model profile, run:

~~~bash
export AFD_E2E_BACKEND=gpu
export AFD_E2E_LARGE_MODEL=1
export AFD_GPU_E2E_MODEL=/path/to/Qwen3.5-122B-A10B
export AFD_E2E_DEVICES=0,2,4,6,1,3,5,7
python -m pytest -q -s \
  tests/e2e/models/qwen3_5/test_qwen3_5_122b.py
~~~

Verify the model path and eight unique devices before pytest. Device order is
part of the profile contract: the first four run native DP4 or AFD Attention
DP4, and the last four run AFD FFN DP4/EP4. Both cases are BF16, text-only,
vLLM V1, eager, and natural-routing GSM8K-7 checks. Do not enable graph, DBO,
or benchmark-only forced routing. The profile disables the FlashInfer sampler
because vLLM 0.26.0 rejects Blackwell SM12 during its capability check. Report
`baseline-eager` and `afd-eager-4a4f` separately, including cleanup and
released GPU memory.

Do not add backend markers or run scenarios in parallel; they share devices.

For the local DeepSeek-V2-Lite 2A1F cases, run the same pytest entrypoint with
`[afd-eager-2a1f]`, `[afd-graph-2a1f]`, or `[afd-graph-dbo-2a1f]`.

On cancellation, forward SIGTERM and allow over 90 seconds for cleanup.

### 5. Report

Success means a default suite reports 4 passed and 0 skipped; the Qwen3.5-122B
profile reports 2 passed and 0 skipped. Report the failed scenario, first
actionable error, and cleanup status. Any skip is a gate failure.

## Environment reference

| Variable | Backend | Required |
|---|---|---|
| AFD_E2E_BACKEND | both | yes: gpu or npu |
| AFD_E2E_DEVICES | both | yes: four unique IDs for the default suite |
| AFD_E2E_LARGE_MODEL | GPU | Qwen3.5-122B only: must equal 1 |
| AFD_GPU_E2E_MODEL | GPU | Qwen3.5-122B: yes; default suites download when unset |
| AFD_GPU_E2E_VLLM_BIN | GPU | no; defaults to vllm |
| AFD_NPU_E2E_MODEL | NPU | no; downloads the selected suite's model when unset |
| AFD_NPU_E2E_VLLM_BIN | NPU | no; defaults to vllm |
| HF_HOME | both | recommended; HF dataset/model cache |
