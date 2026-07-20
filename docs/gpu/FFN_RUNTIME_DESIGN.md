# GPU FFN Runtime Design

This document describes the current CUDA FFN-side runtime in
`afd_plugin.v1.worker`.

## Entry Point

GPU FFN is launched as a normal `vllm serve` process with an explicit worker
class:

```bash
vllm serve <model> \
  --worker-cls afd_plugin.v1.worker.AFDFFNWorker \
  --additional-config '{"afd":{"role":"ffn","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":1239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
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
- starts and stops a background FFN daemon loop;
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
- derives `afd_role_rank` from DP/PCP/TP ranks when needed;
- validates CUDA graph mode;
- creates the configured connector through `AFDConnectorFactory`;
- loads the model through vLLM's model loader; the plugin-owned DeepSeek
  wrapper constructs and loads FFN MLP/expert components plus shared model
  components required by the vLLM lifecycle, without Attention modules;
- returns empty KV cache specs and no-ops KV initialization;
- rejects sampling and LoRA mutation APIs that are not meaningful for FFN;
- receives DP metadata, Attention hidden states, and connector payloads;
- establishes a minimal vLLM forward context;
- calls `model.compute_ffn_output(hidden_states, layer_idx)` when available;
- sends FFN output back through the connector;
- owns the FFN CUDA graph cache keyed by DP metadata shape;
- closes connector/profiler resources on shutdown.

## Daemon Loop

```text
AFDFFNWorker.initialize_from_config(...)
  -> model_runner.initialize_kv_cache(...)
  -> model_runner.initialize_afd_connector()
  -> start_ffn_server_loop()

background loop:
  -> if connector.ffn_step_trigger is Trigger.CONNECTOR:
       -> model_runner.execute_connector_driven_step()
       -> torch.cuda.synchronize(); continue
  -> control_plane.recv_dp_metadata_list()
  -> if graph warmup/capture: model_runner.capture_model(...)
  -> else: model_runner.execute_model(dp_metadata_list=...)
  -> torch.cuda.synchronize()
```

The loop dispatches on `connector.ffn_step_trigger`:

- `Trigger.DP_METADATA` (current `P2pNcclAFDConnector` behavior): each step is
  driven by the arrival of a control-plane payload from
  `connector.control_plane.recv_dp_metadata_list()`, which also carries the
  warmup/graph-capture flags.
- `Trigger.CONNECTOR`: the connector has no control plane
  (`control_plane is None`) and drives FFN work itself; the loop calls
  `model_runner.execute_connector_driven_step()`, which blocks inside the
  connector's own receive path.

`AFDFFNWorker.execute_model()` intentionally raises if the native scheduler
attempts to execute a normal vLLM request on the FFN process.

## FFN Forward (DP-Metadata Trigger)

For each layer and stage, `GPUFFNModelRunner`:

1. updates connector state from DP metadata through
   `control_plane.update_state_from_dp_metadata()`;
2. receives Attention output with `connector.recv_attn_output(ubatch_idx=...)`;
3. reads hidden states and `AFDTransferMetadata` from the
   `AFDA2FTransferPayload` returned by the connector;
4. installs per-stage DP metadata and `afd_metadata` in the current forward
   context;
5. calls `compute_ffn_output()` when the model wrapper provides it;
6. sends the FFN output with `connector.send_ffn_output()`.

If the model does not expose `compute_ffn_output`, the runner passes hidden
states through. Production AFD model paths are expected to use plugin-owned
model wrappers that implement the FFN computation contract.

## FFN Forward (Connector Trigger)

`execute_connector_driven_step()` rejects connectors that have a control plane,
then runs `_ffn_forward_connector_driven()`: one eager FFN pass driven purely
by the base-contract data path. There is no DP metadata to apply; for each
layer the runner receives a payload with `connector.recv_attn_output()`, takes
the layer index from the payload's `AFDTransferMetadata`, installs the metadata
on the forward context with `dp_metadata = None`, computes, and sends the
result back with `connector.send_ffn_output()`.

This path is eager-only and does not use the CUDA graph cache; graph keys are
derived from DP metadata, which does not exist without a control plane.

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
- The only CUDA connector is `P2pNcclAFDConnector`, implemented by
  `afd_plugin.connectors.gpu.p2p`; no `Trigger.CONNECTOR` GPU connector exists
  yet, although the daemon loop and runner support one.
- DBO requires exactly two ubatches.
- Role-aware model construction and weight loading currently depend on the
  plugin-owned DeepSeek model wrappers.
