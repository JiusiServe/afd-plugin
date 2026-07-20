# GPU Attention Runtime Design

This document describes the current CUDA Attention-side runtime in
`afd_plugin.v1.worker`.

## Entry Point

GPU Attention is selected with an explicit worker class:

```bash
vllm serve <model> \
  --worker-cls afd_plugin.v1.worker.AFDAttentionWorker \
  --additional-config '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":1239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

The public config channel is vLLM `additional_config["afd"]`; the plugin does
not add a separate CLI flag. Connector-owned settings go into the nested
`connector_extra_config` mapping and are parsed by the selected connector's
`parse_extra_config()` into a typed `ConnectorExtraInfo`.

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
- derives `afd_role_rank` from DP/PCP/TP ranks when any of those parallel
  sizes is greater than one;
- validates CUDA graph mode with `validate_cuda_graph_mode`;
- creates and initializes the configured connector through
  `AFDConnectorFactory`;
- constructs and loads the Attention-side DeepSeek components instead of the
  FFN MLP/expert components, while retaining shared model components required
  by the vLLM lifecycle;
- installs AFD metadata on `ForwardContext.additional_kwargs["afd_metadata"]`;
- sends DP metadata to FFN ranks through the connector control plane before
  model forward; GPU requires a control-plane connector, so the runner asserts
  `connector.control_plane is not None`;
- supports DP=1 fallback metadata when vLLM does not provide `DPMetadata`;
- wraps vLLM's ubatch wrapper with `AFDUBatchWrapper` when DBO is enabled;
- marks warmup and graph-capture metadata for FFN graph capture/replay;
- closes the connector and profiler on shutdown.

## Connector Contract

The runner talks to the FFN side through an `AFDConnectorBase` implementation
with two planes:

- **Data plane**: `send_attn_output()` / `recv_ffn_output()` move hidden
  states, driven by the plugin-owned model wrappers via `afd_metadata`.
- **Control plane**: `connector.control_plane` is an `AFDControlPlane`. The
  runner applies and sends per-stage DP metadata payloads
  (`update_state_from_dp_metadata()` + `send_dp_metadata_list()`) before model
  forward so the FFN side can derive wire tensor shapes and warmup/capture
  flags.

GPU only supports control-plane-driven connectors, so both planes are always
present. The runner asserts `connector.control_plane is not None` before every
control-plane send. FFN steps are driven by control-plane payload arrival on
the FFN side. The current GPU connector, `P2pNcclAFDConnector`, exposes
`P2pNcclAFDControlPlane` as its control plane. (The base contract also allows
control-plane-less connectors that drive FFN steps from their own receive loop,
but that mode is used only by NPU connectors, not on GPU.)

## Forward Path

```text
OpenAI request
  -> vLLM scheduler
  -> AFDAttentionWorker.execute_model(...)
  -> AFDAttentionModelRunner.execute_model(...)
  -> build attention metadata and AFD metadata
  -> send DP metadata through the connector control plane
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

The metadata object is `AFDForwardContextMetadata`. It carries token slices,
request slices, stage information, transaction ids, and the connector reference
used by plugin-owned model wrappers.

For DBO, `AFDUBatchWrapper` builds per-ubatch metadata and
`build_ubatch_dp_metadata_list()` sends one DP metadata entry per stage. The
current GPU runtime supports exactly two ubatches when DBO is enabled.

## Ubatching Decision

vLLM hardcodes `should_ubatch=False` for DP=1, so the runner replicates the
DP-coordinated decision rank-locally in `_should_ubatch_single_rank()`:
preconditions, thresholds, then the empty-ubatch guard against padded token
counts. For DP>1, the runner additionally aborts ubatching when the token
count is smaller than the ubatch count, which would empty the first ubatch.

## CUDA Graph

GPU Attention supports the current AFD graph path only for vLLM
`FULL_DECODE_ONLY` semantics. DP metadata transfer is treated as a control-plane
side effect and is sent before formal CUDA graph capture, so the capture
contains only replayable model/data-plane work. FFN receives warmup and capture
flags through `send_dp_metadata_list()`.

Unsupported graph modes fail fast in `validate_cuda_graph_mode`.

## Connector

GPU Attention uses `P2pNcclAFDConnector`, implemented by
`afd_plugin.connectors.gpu.p2p`. The connector is created during runner
initialization and remains owned by the model runner. Rank topology is validated
from `AFDConfig`; FFN ranks are ordered before Attention ranks
(`[F0, F1, ..., A0, A1, ...]`).

`num_attention_ranks` must be greater than or equal to `num_ffn_ranks` and
divisible by it.

## Current Limits

- Only vLLM `0.19.1` and model runner v1 are supported.
- Runtime modules import real `torch` and `vllm` dependencies at module import
  time.
- DBO requires exactly two ubatches.
- Role-aware model construction and weight loading currently depend on the
  plugin-owned DeepSeek model wrappers.
- The only CUDA connector is `P2pNcclAFDConnector`. GPU supports only
  control-plane-driven connectors; a connector without a control plane
  (`control_plane is None`) is not supported and the runner asserts against it.
