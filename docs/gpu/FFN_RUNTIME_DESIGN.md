# GPU FFN Runtime Design

This document describes the current CUDA FFN-side runtime in
`afd_plugin.v1.worker`.

## Entry Point

GPU FFN is launched as a normal `vllm serve` process with an explicit worker
class:

```bash
vllm serve <model> \
  --worker-cls afd_plugin.v1.worker.AFDFFNWorker \
  --additional-config '{"afd":{"enabled":true,"role":"ffn","connector":"p2pconnector","host":"127.0.0.1","port":1239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

The FFN process is not request-driven. Start the FFN process first, then start
the Attention process. Requests should be sent only to the Attention API port.

## Worker

`AFDFFNWorker` inherits vLLM v1 `Worker`.

Current behavior:

- validates the AFD stack with expected role `ffn`;
- rejects vLLM model runner v2;
- rejects unsupported ubatching shapes;
- calls native `Worker.init_device()`;
- replaces the native model runner with `GPUFFNModelRunner`;
- returns an empty KV cache spec;
- skips normal warmup by returning `0.0` from `compile_or_warm_up_model`;
- starts and stops a connector-driven background FFN loop;
- fails fast if vLLM scheduler calls `execute_model()`;
- propagates loop exceptions through `raise_ffn_loop_error_if_any()`.

The empty KV cache path is paired with the plugin's `EngineCore` compatibility
patch in `afd_plugin.compat.patches.engine_core`, which keeps FFN daemon mode
out of vLLM's normal request/KV-cache assumptions.

## Model Runner

`GPUFFNModelRunner` inherits `LoRAModelRunnerMixin` and implements the minimal
runner surface needed by vLLM worker/executor lifecycle plus AFD connector
execution.

Current behavior:

- parses and validates `AFDConfig` with expected role `ffn`;
- derives `afd_role_rank` from DP/TP ranks when needed;
- validates CUDA graph mode;
- creates the configured connector through `AFDConnectorFactory`;
- loads the model through vLLM's model loader;
- returns empty KV cache specs and no-ops KV initialization;
- rejects sampling and LoRA mutation APIs that are not meaningful for FFN;
- receives DP metadata, Attention hidden states, and connector payloads;
- establishes a minimal vLLM forward context;
- calls `model.compute_ffn_output(hidden_states, layer_idx)` when available;
- sends FFN output back through the connector;
- owns FFN CUDA graph cache keyed by DP metadata shape;
- closes connector/profiler resources on shutdown.

## Daemon Loop

```text
AFDFFNWorker.initialize_from_config(...)
  -> model_runner.initialize_kv_cache(...)
  -> model_runner.initialize_afd_connector()
  -> start_ffn_server_loop()

background loop:
  -> recv_dp_metadata_list(timeout_ms=100)
  -> if graph warmup/capture: model_runner.capture_model(...)
  -> else: model_runner.execute_model(dp_metadata_list=...)
  -> torch.cuda.synchronize()
```

The loop is connector-driven. `AFDFFNWorker.execute_model()` intentionally
raises if the native scheduler attempts to execute a normal vLLM request on the
FFN process.

## FFN Forward

For each layer and stage, `GPUFFNModelRunner`:

1. updates connector state from DP metadata;
2. receives Attention output with `connector.recv_attn_output()`;
3. normalizes the payload to hidden states and `AFDConnectorMetadata`;
4. installs `afd_metadata` in the current forward context;
5. waits for async receive handles if present;
6. calls `compute_ffn_output()` when the model wrapper provides it;
7. sends the FFN output with `connector.send_ffn_output()`.

If the model does not expose `compute_ffn_output`, the runner passes hidden
states through. Production AFD model paths are expected to use plugin-owned
model wrappers that implement the FFN computation contract.

## CUDA Graph

`GPUFFNModelRunner` supports graph-keyed capture/replay for the current
`FULL_DECODE_ONLY` AFD path. DP metadata update is performed before capture so
control-plane connector side effects are not captured as replayable graph work.

Warmup and capture are driven by flags received from the Attention side through
`recv_dp_metadata_list()`.

## Current Limits

- Only vLLM `0.19.1` and model runner v1 are supported.
- FFN workers are connector-driven only; scheduler-driven request execution is
  rejected.
- The GPU connector is `p2pconnector`, implemented by
  `afd_plugin.connectors.gpu.p2p`.
- DBO requires exactly two ubatches.
- Role-based weight pruning is not implemented.
