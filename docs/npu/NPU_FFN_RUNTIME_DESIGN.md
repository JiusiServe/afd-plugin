# NPU FFN Runtime Design

This document describes the current Ascend NPU FFN-side runtime in
`afd_plugin.v1.worker.npu`.

## Entry Point

NPU FFN is launched as a normal `vllm serve` process with an explicit
vLLM-Ascend worker class:

```bash
VLLM_PLUGINS=ascend,afd vllm serve <model> \
  --worker-cls afd_plugin.v1.worker.npu.AFDNPUFFNWorker \
  --additional-config '{"afd":{"role":"ffn","connector":"CAMP2pAFDConnector","host":"127.0.0.1","port":1239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

The FFN process is connector-driven. It should not receive OpenAI/vLLM
requests; requests go to the Attention API process.

## Class Boundary

GPU and NPU runtimes use separate public class paths:

```text
GPU:
  afd_plugin.v1.worker.AFDFFNWorker
  afd_plugin.v1.worker.GPUFFNModelRunner

NPU:
  afd_plugin.v1.worker.npu.AFDNPUFFNWorker
  afd_plugin.v1.worker.npu.AFDNPUFFNModelRunner
```

`AFDNPUFFNModelRunner` inherits vLLM-Ascend `NPUModelRunner` directly instead
of inheriting the GPU `GPUFFNModelRunner`. Shared AFD semantics are kept in
config, connector, metadata, validation, and small helper functions rather than
through a cross-device inheritance chain.

## Worker

`AFDNPUFFNWorker` inherits `vllm_ascend.worker.worker.NPUWorker`.

Current behavior:

- verifies that vLLM-Ascend is importable;
- applies plugin-owned Ascend patches;
- validates AFD config, role, and worker class path;
- rejects unsupported NPU AFD feature flags;
- fixes the all-to-all backend for AFD;
- rejects vLLM-Ascend model runner v2;
- initializes the NPU device with `self._init_device()`;
- initializes the vLLM workspace manager for one or two ubatches;
- creates `AFDNPUFFNModelRunner`;
- returns an empty KV cache spec;
- starts/stops the FFN daemon loop from `initialize_from_config()`;
- returns `0.0` from `compile_or_warm_up_model()`;
- rejects scheduler-driven `execute_model()`;
- propagates daemon-loop failures back to caller.

## Model Runner

`AFDNPUFFNModelRunner` inherits
`vllm_ascend.worker.model_runner_v1.NPUModelRunner`.

Current behavior:

- parses `AFDConfig` with expected role `ffn`;
- installs a vLLM-Ascend `vllm_config.afd_config` compatibility proxy;
- validates unsupported NPU AFD features;
- derives `afd_role_rank` from DP/PCP/TP ranks;
- creates the configured connector (`CAMP2pAFDConnector` or
  `CAMAsyncAFDConnector`);
- constructs and loads the DeepSeek FFN MLP/expert components plus shared model
  components required by the vLLM lifecycle, without Attention modules;
- returns empty KV cache specs and no-ops KV initialization;
- executes DP-metadata-triggered steps through `execute_ffn_step()` and
  connector-driven steps through `execute_connector_driven_step()`;
- builds a minimal Ascend forward context for each FFN step;
- mirrors AFD metadata into `additional_kwargs["afd_metadata"]` and
  `forward_context.afd_metadata`;
- calls `model.compute_ffn_output(...)` with CAM payload fields;
- sends FFN output back through the connector;
- supports ACL graph warmup/capture/replay keyed by DP metadata shape;
- rejects token sampling;
- closes connector/profiler resources on shutdown.

## Daemon Loop

```text
AFDNPUFFNWorker.initialize_from_config(...)
  -> model_runner.initialize_kv_cache(...)
  -> model_runner.initialize_afd_connector()
  -> start_ffn_server_loop()

background loop:
  -> torch.npu.set_device(...)
  -> if connector.control_plane is None:
       -> model_runner.execute_connector_driven_step()
       -> torch.npu.synchronize(); continue
  -> control_plane.recv_dp_metadata_list()
  -> model_runner.execute_ffn_step(...)
  -> torch.npu.synchronize()
```

The loop dispatches on whether `connector.control_plane` is set:

- control plane present (`CAMP2pAFDConnector`): each step starts when a
  control-plane payload arrives through
  `connector.control_plane.recv_dp_metadata_list()`. `execute_ffn_step()`
  routes warmup/capture metadata to `capture_model()` when ACL graph is
  active; otherwise it calls `execute_model()` with the received
  `dp_metadata_list`.
- `control_plane is None` (`CAMAsyncAFDConnector`): the connector has no control
  plane and drives FFN work itself.
  `execute_connector_driven_step()` rejects connectors that do have a control
  plane and blocks inside the connector's own receive path.

## FFN Forward (DP-Metadata Trigger)

For each layer and stage, `AFDNPUFFNModelRunner`:

1. updates connector state from DP metadata through
   `control_plane.update_state_from_dp_metadata()`;
2. receives an `AFDA2FTransferPayload` from `connector.recv_attn_output(...)`;
3. installs per-stage DP and AFD metadata on the Ascend forward context;
4. calls `model.compute_ffn_output(...)`;
5. sends the FFN output through `connector.send_ffn_output(...)`.

FFN-level token counts are derived from Attention DP metadata and projected
back to DP-level counts for vLLM's forward context when TP is enabled.

The current compute call forwards these payload fields when available:

- `hidden_states`;
- `group_list`;
- `dynamic_scales`;
- `expand_x_shared` / `dynamic_scales_shared`;
- `topk_weights`;
- `topk_ids`;
- `router_logits`;
- `row_idx`;
- `x_active_mask`;
- `cam_p2p_ep_name`.

## FFN Forward (Connector Trigger)

`_ffn_forward_connector_driven()` requires the async connector work-item APIs.
For each layer, the runner:

1. receives a normalized `AFDAsyncFFNWorkItem` from
   `connector.recv_ffn_work_item(...)`; CAM metadata supplies the actual layer
   index plus routed/shared token counts, and the connector slices tensors
   from operator capacity down to those counts;
2. builds a single-stage Ascend forward context sized to the work item's token
   count, with `dp_metadata = None`;
3. installs the work item's `AFDTransferMetadata` as `afd_metadata`;
4. calls `model.compute_ffn_output(...)` with the work item's payload fields;
5. returns the routed/shared outputs through
   `connector.send_ffn_work_item_output(...)`, which also handles the
   zero-routed-token placeholder required by CAM combine-send.

This path is eager-only; the ACL graph cache is keyed by DP metadata, which
does not exist without a control plane.

## Connectors

`CAMP2pAFDConnector`, implemented by `afd_plugin.connectors.npu.camp2p`, owns
NPU topology, HCCL/Gloo process groups, custom-op loading, receive metadata
construction, and FFN/Attention payload transfer. DP metadata exchange lives on
its `CAMP2pAFDControlPlane`, exposed as `connector.control_plane`. The
connector supports non-equal A/F topologies where
`num_attention_ranks >= num_ffn_ranks` and the ratio is integral.

`CAMAsyncAFDConnector`, implemented by `afd_plugin.connectors.npu.async_cam`,
uses CAM dispatch/combine operators that own both the collective data motion
and its routing metadata; there is no control plane. Its behavior is
configured through `connector_extra_config`, parsed into `AFDAsyncExtraInfo`.

## ACL Graph

The NPU FFN runner can use ACL graph when vLLM-Ascend has graph mode enabled
and the connector is DP-metadata triggered. Graph cache keys are built from DP
metadata shape plus A/F topology. Capture updates connector state before
entering the NPU graph context, so connector control-plane state is not
repeatedly recomputed as part of normal replay.

If no captured graph exists for a key, the runner falls back to eager
execution.

## Supported And Rejected Features

Supported:

- vLLM `0.19.1` runtime stack with vLLM-Ascend model runner v1;
- `--additional-config '{"afd": ...}'` with connector-owned
  `connector_extra_config`;
- `CAMP2pAFDConnector` (DP-metadata trigger) and `CAMAsyncAFDConnector`
  (connector trigger);
- connector-driven FFN daemon loop;
- empty KV cache;
- eager FFN execution;
- ACL graph warmup/capture/replay path for the DP-metadata trigger;
- DBO with exactly two ubatches;
- role-aware DeepSeek model construction and FFN-side weight loading.

Rejected by validation:

- vLLM-Ascend model runner v2;
- scheduler-driven FFN requests;
- `compute_gate_on_attention=true`;
- `quant_mode != 0`;
- DBO with a ubatch count other than two;
- `connector_extra_config` keys unknown to the selected connector.
