# GPU Attention Runtime Design

This document describes the current CUDA Attention-side runtime in
`afd_plugin.v1.worker`.

## Entry Point

GPU Attention is selected with an explicit worker class:

```bash
vllm serve <model> \
  --worker-cls afd_plugin.v1.worker.AFDAttentionWorker \
  --additional-config '{"afd":{"enabled":true,"role":"attention","connector":"p2pconnector","host":"127.0.0.1","port":1239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

The public config channel is vLLM `additional_config["afd"]`; the plugin does
not add a separate CLI flag.

## Worker

`AFDAttentionWorker` inherits vLLM v1 `Worker`.

Current behavior:

- validates `additional_config["afd"]`, role, connector, and `--worker-cls`
  through `assert_compatible_afd_stack`;
- rejects vLLM model runner v2;
- rejects unsupported ubatching shapes through
  `fail_if_unsupported_ubatching`;
- calls native `Worker.init_device()`;
- replaces the native model runner with `AFDAttentionModelRunner`;
- clears accelerator cache after the replacement.

The worker intentionally keeps vLLM-owned lifecycle behavior for distributed
initialization, device setup, model loading, KV cache management, memory
profiling, sleep/wake, and shutdown.

## Model Runner

`AFDAttentionModelRunner` inherits vLLM v1 `GPUModelRunner`.

Current behavior:

- parses and validates `AFDConfig` with expected role `attention`;
- derives `afd_role_rank` from DP/TP ranks when DP or TP is enabled;
- validates CUDA graph mode with `validate_cuda_graph_mode`;
- creates and initializes the configured connector through
  `AFDConnectorFactory`;
- installs AFD metadata on `ForwardContext.additional_kwargs["afd_metadata"]`;
- sends DP metadata to FFN ranks before model forward;
- supports DP=1 fallback metadata when vLLM does not provide `DPMetadata`;
- wraps vLLM's ubatch wrapper with `AFDUBatchWrapper` when DBO is enabled;
- marks warmup and graph-capture metadata for FFN graph capture/replay;
- closes the connector and profiler on shutdown.

## Forward Path

```text
OpenAI request
  -> vLLM scheduler
  -> AFDAttentionWorker.execute_model(...)
  -> AFDAttentionModelRunner.execute_model(...)
  -> build attention metadata and AFD metadata
  -> send DP metadata through p2pconnector
  -> model forward
  -> plugin-owned model wrapper sends Attention output
  -> FFN side computes and sends FFN output
  -> plugin-owned model wrapper receives FFN output
  -> native vLLM sampling/output path
```

Attention still owns KV cache and normal request scheduling.

## Metadata

The canonical metadata location is:

```python
forward_context.additional_kwargs["afd_metadata"]
```

The metadata object is `AFDMetadata`. It carries token slices, request slices,
stage information, transaction ids, and the connector reference used by
plugin-owned model wrappers.

For DBO, `AFDUBatchWrapper` builds per-ubatch metadata and
`build_ubatch_dp_metadata_list()` sends one DP metadata entry per stage. The
current GPU runtime supports exactly two ubatches when DBO is enabled.

## CUDA Graph

GPU Attention supports the current AFD graph path only for vLLM
`FULL_DECODE_ONLY` semantics. DP metadata transfer is treated as a control-plane
side effect and is sent before formal CUDA graph capture. FFN receives warmup
and capture flags through `send_dp_metadata_list()`.

Unsupported graph modes fail fast in `validate_cuda_graph_mode`.

## Connector

GPU Attention uses `p2pconnector`, implemented by
`afd_plugin.connectors.gpu.p2p`. The connector is created during runner
initialization and remains owned by the model runner. Rank topology is validated
from `AFDConfig`; FFN ranks are ordered before Attention ranks.

`num_attention_ranks` must be greater than or equal to `num_ffn_ranks` and
divisible by it.

## Current Limits

- Only vLLM `0.19.1` and model runner v1 are supported.
- Runtime modules import real `torch` and `vllm` dependencies at module import
  time.
- DBO requires exactly two ubatches.
- Role-based weight pruning is not implemented.
- The only CUDA connector is `p2pconnector`.
