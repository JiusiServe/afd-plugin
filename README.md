# vllm-afd-plugin

**vllm-afd-plugin** is a [vLLM](https://github.com/vllm-project/vllm)
external plugin for **Attention-FFN Disaggregation (AFD)**. It provides
plugin-owned worker classes, model runners, model wrappers, connectors,
configuration validation, compatibility shims, and hardware-gated integration
tests for GPU and Ascend NPU deployments.

The target runtime is **vLLM `v0.19.1`**. The plugin does not modify the vLLM
source tree. AFD behavior is installed through the `vllm.general_plugins` entry
point, explicit `--worker-cls` class paths, `--additional-config`, plugin-owned
model wrappers, and narrow version-scoped compatibility shims.

## Architecture

![vLLM AFD plugin architecture](docs/assets/vllm-afd-plugin-architecture.svg)

## Current Status

Core runtime support:

- Python package metadata for the `vllm-afd-plugin` distribution.
- `vllm.general_plugins` entry point named `afd`, implemented by
  `afd_plugin:register_afd`.
- CPU-safe package import, config parsing, stack validation, class-path
  resolution, and compatibility-patch tests.
- Plugin-owned `AFDConfig`, parsed from vLLM
  `additional_config["afd"]`.
- GPU Attention and FFN workers:
  `afd_plugin.v1.worker.AFDAttentionWorker` and
  `afd_plugin.v1.worker.AFDFFNWorker`.
- NPU Attention and FFN workers:
  `afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker` and
  `afd_plugin.v1.worker.ascend.AFDNPUFFNWorker`.
- Plugin-owned DeepSeekV2-family wrappers and forward-context helpers.
- Connector-driven FFN daemon loops for GPU and NPU.
- GPU P2P connector and Ascend CAMP2P connector.
- GPU `FULL_DECODE_ONLY` CUDA graph support for the current Attention/FFN
  runtime shape, including FFN graph-keyed capture/replay.
- NPU ACL graph plumbing in the FFN runner, with eager and graph/capture paths
  driven by metadata from the Attention side.
- GPU and NPU profiler helpers controlled by plugin-owned environment variables.

Model support:

| Model family | Registered architectures | Status | Notes |
| --- | --- | --- | --- |
| DeepSeekV2 / DeepSeekV3 / GLM MoE DSA | `DeepseekForCausalLM`, `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, `GlmMoeDsaForCausalLM` | Supported for AFD smoke and E2E validation | Uses `afd_plugin.model_executor.models.deepseek_v2` wrappers. Attention and FFN sides currently load full model weights. |
| Other model families | Not registered by this plugin | Not supported | Add a plugin-owned model wrapper before using AFD-specific model forward behavior. |

Connector support:

| Connector | Platform | Status | Notes |
| --- | --- | --- | --- |
| `p2pconnector` | CUDA | Supported | FFN ranks are ordered before Attention ranks. `num_attention_ranks` must be greater than or equal to `num_ffn_ranks` and divisible by it. |
| `camp2pconnector` | Ascend NPU | Supported | Uses HCCL/CAMP2P custom ops. Ascend ops build by default; set `AFD_BUILD_ASCEND_OPS=0` to skip them. |

Connector implementations are grouped by backend package:
`afd_plugin.connectors.gpu` for GPU-only connectors,
`afd_plugin.connectors.npu` for NPU-only connectors, and
`afd_plugin.connectors` for shared contracts and metadata.

Known gaps:

- vLLM versions other than `0.19.1` are not claimed as supported.
- vLLM/vLLM-Ascend model runner v2 is not supported.
- Role-based weight pruning is not implemented; Attention and FFN sides still
  load full DeepSeekV2-family weights.
- GPU and NPU E2E tests are opt-in and require real hardware plus model weights.
- GPU CUDA graph support is limited to `FULL_DECODE_ONLY`.
- GPU DBO plus CUDA graph is limited to exactly two ubatches.
- NPU runtime rejects `compute_gate_on_attention=true`, `quant_mode != 0`, and
  multistream communication.

## Install

Requires Python **3.10-3.13** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/vllm-project/afd-plugin.git
cd afd-plugin
uv sync --group dev
```

`vllm` is an optional runtime extra so CPU-only or macOS development
environments can still run import/config tests without a CUDA wheel:

```bash
# Linux / CUDA-capable environments
uv sync --group dev --extra vllm
```

The optional extra pins `vllm==0.19.1`.

## Using the Plugin

Install or sync the distribution as `vllm-afd-plugin`. Python imports and CLI
class paths use the `afd_plugin` package name.

For GPU:

```bash
export VLLM_PLUGINS=afd
unset VLLM_USE_V2_MODEL_RUNNER
```

For NPU, load the Ascend plugin before AFD:

```bash
export VLLM_PLUGINS=ascend,afd
unset VLLM_USE_V2_MODEL_RUNNER
```

AFD is configured through vLLM `--additional-config`. There is no separate
`--afd-config` flag.

GPU Attention-side shape:

```bash
vllm serve /path/to/DeepSeek-V2-Lite \
  --worker-cls afd_plugin.v1.worker.AFDAttentionWorker \
  --served-model-name deepseek-v2-lite-afd-attention \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 18000 \
  --additional-config '{"afd":{"enabled":true,"role":"attention","connector":"p2pconnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

GPU FFN-side shape:

```bash
vllm serve /path/to/DeepSeek-V2-Lite \
  --worker-cls afd_plugin.v1.worker.AFDFFNWorker \
  --served-model-name deepseek-v2-lite-afd-ffn \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 18001 \
  --additional-config '{"afd":{"enabled":true,"role":"ffn","connector":"p2pconnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

NPU uses the same config channel with Ascend class paths and
`camp2pconnector`:

```bash
vllm serve /path/to/DeepSeek-V2-Lite \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker \
  --served-model-name deepseek-v2-lite-afd-attention \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 18000 \
  --additional-config '{"afd":{"enabled":true,"role":"attention","connector":"camp2pconnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

Start the FFN side first, then start the Attention side and send requests to
the Attention API server. FFN workers are connector-driven; scheduler-driven
FFN `execute_model()` calls fail fast.

For repeatable local smoke testing, prefer the bundled runner:

```bash
uv run python tests/e2e/runner.py \
  --model /path/to/DeepSeek-V2-Lite \
  --device-backend gpu \
  --num-attention-ranks 1 \
  --num-ffn-ranks 1 \
  --attention-gpus 0 \
  --ffn-gpus 1 \
  --api-port-base 18000 \
  --afd-port 6239 \
  --common-vllm-arg=--trust-remote-code
```

For NPU, use `--device-backend npu`; the runner maps the same device arguments
to `ASCEND_RT_VISIBLE_DEVICES` and selects `camp2pconnector`.

## AFD Config

The canonical config shape is:

```json
{
  "afd": {
    "enabled": true,
    "role": "attention",
    "connector": "p2pconnector",
    "host": "127.0.0.1",
    "port": 1239,
    "num_attention_ranks": 2,
    "num_ffn_ranks": 1,
    "afd_role_rank": 0,
    "compute_gate_on_attention": false,
    "extra_config": {}
  }
}
```

`role` must be `attention` or `ffn`. `connector` must be `p2pconnector` or
`camp2pconnector`. The plugin also accepts selected compatibility aliases such
as `afd_role`, `afd_connector`, `afd_host`, `afd_port`, and `afd_extra_config`.

## Development

Run the default CPU-safe checks:

```bash
uv run pytest
uv run ruff check .
```

Native C/C++ sources are grouped by backend under `csrc/`: Ascend/CANN sources
live in `csrc/npu`, including the `a2e` and `e2a` ACLNN operators, and
`csrc/gpu` is reserved for GPU native sources.

Opt-in GPU E2E tests require a CUDA-capable vLLM environment and a DeepSeekV2
Lite model path:

```bash
AFD_GPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite uv run pytest -q -m gpu
```

Opt-in NPU E2E tests require vLLM-Ascend, torch-npu, CANN, built AFD Ascend
custom ops, and a DeepSeekV2 Lite model path:

```bash
AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite uv run pytest -q -m npu
```

## Docs

- [docs/gpu/ATTENTION_RUNTIME_DESIGN.md](docs/gpu/ATTENTION_RUNTIME_DESIGN.md)
  - GPU Attention worker and model-runner design.
- [docs/gpu/FFN_RUNTIME_DESIGN.md](docs/gpu/FFN_RUNTIME_DESIGN.md) - GPU FFN
  worker, daemon loop, and connector-driven execution design.
- [docs/npu/NPU_ATTENTION_RUNTIME_DESIGN.md](docs/npu/NPU_ATTENTION_RUNTIME_DESIGN.md)
  - Ascend NPU Attention worker and model-runner design.
- [docs/npu/NPU_FFN_RUNTIME_DESIGN.md](docs/npu/NPU_FFN_RUNTIME_DESIGN.md) -
  Ascend NPU FFN worker, daemon loop, CAMP2P, and ACL graph design.

## License

Apache License 2.0 - see [LICENSE](LICENSE).
