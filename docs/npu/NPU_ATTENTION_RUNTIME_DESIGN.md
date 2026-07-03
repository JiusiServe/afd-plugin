# NPU Attention Runtime Design

This document describes the current Ascend NPU Attention-side runtime in
`afd_plugin.v1.worker.ascend`.

## Entry Point

NPU Attention is selected with an explicit vLLM-Ascend worker class:

```bash
VLLM_PLUGINS=ascend,afd vllm serve <model> \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker \
  --additional-config '{"afd":{"enabled":true,"role":"attention","connector":"camp2pconnector","host":"127.0.0.1","port":1239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

NPU runtime modules intentionally import real vLLM-Ascend dependencies. The
top-level package and validation/config modules remain CPU-safe.

## Class Boundary

GPU and NPU runtimes use separate public class paths:

```text
GPU:
  afd_plugin.v1.worker.AFDAttentionWorker
  afd_plugin.v1.worker.AFDAttentionModelRunner

NPU:
  afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker
  afd_plugin.v1.worker.ascend.AFDNPUAttentionModelRunner
```

The NPU classes inherit vLLM-Ascend classes directly. They do not inherit the
GPU AFD worker/model runner, which keeps CUDA graph and Ascend graph assumptions
separate.

## Worker

`AFDNPUAttentionWorker` inherits `vllm_ascend.worker.worker.NPUWorker`.

Current behavior:

- verifies that vLLM-Ascend is importable;
- applies plugin-owned Ascend patches through
  `apply_afd_ascend_patches_if_needed`;
- validates AFD config, role, and worker class path with
  `assert_compatible_afd_stack`;
- rejects unsupported NPU AFD features with
  `fail_if_unsupported_npu_afd_features`;
- fixes the all-to-all backend for AFD through `fix_all2all_backend_for_afd`;
- rejects vLLM-Ascend model runner v2;
- calls `self._init_device()` from `NPUWorker`;
- initializes the vLLM workspace manager for one or two ubatches;
- creates `AFDNPUAttentionModelRunner` directly.

The worker keeps vLLM-Ascend-owned lifecycle behavior for load, KV cache,
profiling, sleep/wake, and request execution.

## Model Runner

`AFDNPUAttentionModelRunner` inherits
`vllm_ascend.worker.model_runner_v1.NPUModelRunner`.

Current behavior:

- parses `AFDConfig` with expected role `attention`;
- installs a read-only `vllm_config.afd_config` compatibility proxy for
  vLLM-Ascend code that still reads that attribute;
- validates unsupported NPU feature flags;
- derives `afd_role_rank` from DP/TP ranks;
- creates and initializes `camp2pconnector`;
- injects AFD metadata into Ascend/vLLM forward context;
- sends DP metadata to FFN ranks before model forward;
- supports NPU DBO metadata splitting through plugin-owned ubatch utilities;
- handles vLLM-Ascend graph parameter updates without capturing connector
  control-plane sends into the model graph;
- steps/stops the plugin-owned NPU profiler.

## Forward Path

```text
OpenAI request
  -> vLLM scheduler
  -> AFDNPUAttentionWorker.execute_model(...)
  -> AFDNPUAttentionModelRunner.execute_model(...)
  -> vLLM-Ascend builds scheduler/input/attention metadata
  -> AFD runner installs AFD metadata
  -> AFD runner sends DP metadata through camp2pconnector
  -> model forward under Ascend forward context
  -> plugin-owned model wrapper sends Attention output
  -> NPU FFN side computes and sends FFN output
  -> plugin-owned model wrapper receives FFN output
  -> native vLLM-Ascend sampling/output path
```

## Metadata

The canonical metadata location remains:

```python
forward_context.additional_kwargs["afd_metadata"]
```

NPU also mirrors metadata to `forward_context.afd_metadata` through
`mirror_afd_metadata_on_forward_context`, because parts of vLLM-Ascend and the
ported model path read that attribute directly.

DP metadata follows the same semantics as GPU:

```text
forward_context.dp_metadata
  -> dp_metadata_list
  -> connector.update_state_from_dp_metadata(...)
  -> connector.send_dp_metadata_list(...)
```

When DP size is 1 and vLLM does not provide `DPMetadata`, the runner can build
the plugin-owned fallback `AFDDPMetadata`.

## Connector

NPU Attention uses `camp2pconnector`, implemented by
`afd_plugin.connectors.npu.camp2p`. The connector initializes HCCL/Gloo process
groups and loads plugin-owned Ascend custom ops lazily when
`init_afd_connector()` runs.

The custom ops are optional at package import time, but NPU AFD data path
requires an Ascend ops build. This build is enabled by default; set
`AFD_BUILD_ASCEND_OPS=0` only when intentionally skipping the NPU extension.

```bash
AFD_BUILD_ASCEND_OPS=0
```

## Supported And Rejected Features

Supported:

- vLLM `0.19.1` runtime stack with vLLM-Ascend model runner v1;
- `--additional-config '{"afd": ...}'`;
- `camp2pconnector`;
- eager Attention path;
- DBO with exactly two ubatches;
- full model weight loading.

Rejected by validation:

- vLLM-Ascend model runner v2;
- `compute_gate_on_attention=true`;
- `quant_mode != 0`;
- attention/FFN multistream communication;
- DBO with a ubatch count other than two;
- role-based weight pruning.
