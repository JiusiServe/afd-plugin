# NPU Attention Runtime Design

This document describes the current Ascend NPU Attention-side runtime in
`afd_plugin.v1.worker.npu`.

## Entry Point

NPU Attention is selected with an explicit vLLM-Ascend worker class:

```bash
VLLM_PLUGINS=ascend,afd vllm serve <model> \
  --worker-cls afd_plugin.v1.worker.npu.AFDNPUAttentionWorker \
  --additional-config '{"afd":{"role":"attention","connector":"CAMP2pAFDConnector","host":"127.0.0.1","port":1239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

NPU runtime modules intentionally import real vLLM-Ascend dependencies. The
top-level package and validation/config modules remain CPU-safe.

Connector-owned settings go into the nested `connector_extra_config` mapping
inside `additional_config["afd"]` and are parsed by the selected connector's
`parse_extra_config()` into a typed `ConnectorExtraInfo`. For
`CAMAsyncAFDConnector` this is `AFDAsyncExtraInfo` (`dynamic_quant`,
`attn_ranks_per_dp`, `async_moe_ubatching`, `async_moe_num_ubatches`,
`async_moe_split`).

## Class Boundary

GPU and NPU runtimes use separate public class paths:

```text
GPU:
  afd_plugin.v1.worker.AFDAttentionWorker
  afd_plugin.v1.worker.AFDAttentionModelRunner

NPU:
  afd_plugin.v1.worker.npu.AFDNPUAttentionWorker
  afd_plugin.v1.worker.npu.AFDNPUAttentionModelRunner
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
- derives `afd_role_rank` from DP/PCP/TP ranks;
- creates and initializes the configured connector
  (`CAMP2pAFDConnector` or `CAMAsyncAFDConnector`);
- constructs and loads the Attention-side DeepSeek components instead of the
  FFN MLP/expert components, while retaining shared model components required
  by the vLLM lifecycle;
- injects AFD metadata into Ascend/vLLM forward context;
- sends DP metadata to FFN ranks before model forward when the connector has a
  control plane; connector-driven connectors (`control_plane is None`) skip
  every control-plane send;
- supports NPU DBO metadata splitting through plugin-owned ubatch utilities;
- supports request-boundary async MoE ubatching for the async CAM connector,
  building per-ubatch attention metadata and installing it on the forward
  context under a dedicated `additional_kwargs` key;
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
  -> AFD runner sends DP metadata through the connector control plane
     (skipped for connector-driven connectors)
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

DP metadata follows the same semantics as GPU when a control plane exists:

```text
forward_context.dp_metadata
  -> dp_metadata_list
  -> connector.control_plane.update_state_from_dp_metadata(...)
  -> connector.control_plane.send_dp_metadata_list(...)
```

When DP size is 1 and vLLM does not provide `DPMetadata`, the runner can build
the plugin-owned fallback `AFDDPMetadata`.

For connector-driven connectors there is no DP metadata exchange. With DP > 1
the runner also skips the DP token-count synchronization and pads every rank to
its own padded token count, since CAM dispatch/combine carries the routing
metadata inside the data plane.

## Connectors

Both NPU connectors are implemented under `afd_plugin.connectors.npu` and are
created through `AFDConnectorFactory`:

- `CAMP2pAFDConnector` (`camp2p`): synchronous CAM point-to-point transfer.
  Exposes `CAMP2pAFDControlPlane` as `connector.control_plane` for DP metadata
  exchange, so FFN steps are driven by control-plane payloads. Initializes
  HCCL/Gloo process groups and loads plugin-owned Ascend custom ops lazily when
  `init_afd_connector()` runs.
- `CAMAsyncAFDConnector` (`async_cam`): asynchronous CAM dispatch/combine.
  CAM operators own both the collective data motion and its routing metadata,
  so the connector has no control plane (`control_plane` stays `None`) and
  drives FFN steps from its own receive loop. Attention ranks occupy the
  first part of the HCCL world and FFN ranks the second.

The custom ops are optional at package import time, but the NPU AFD data path
requires an Ascend ops build. This build is enabled by default; set
`AFD_BUILD_ASCEND_OPS=0` only when intentionally skipping the NPU extension.

See `docs/npu/CAMP2P_CONNECTOR_USER_GUIDE.md` and
`docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md` for configuration contracts and
launch examples.

## Supported And Rejected Features

Supported:

- vLLM `0.19.1` runtime stack with vLLM-Ascend model runner v1;
- `--additional-config '{"afd": ...}'` with connector-owned
  `connector_extra_config`;
- `CAMP2pAFDConnector` and `CAMAsyncAFDConnector`;
- eager Attention path;
- DBO with exactly two ubatches;
- async MoE ubatching with the async CAM connector;
- role-aware DeepSeek model construction and Attention-side weight loading.

Rejected by validation:

- vLLM-Ascend model runner v2;
- `compute_gate_on_attention=true`;
- `quant_mode != 0`;
- DBO with a ubatch count other than two;
- `connector_extra_config` keys unknown to the selected connector.
