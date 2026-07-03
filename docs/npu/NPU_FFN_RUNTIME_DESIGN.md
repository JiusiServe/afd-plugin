# NPU FFN Runtime Design

This document describes the current Ascend NPU FFN-side runtime in
`afd_plugin.v1.worker.ascend`.

## Entry Point

NPU FFN is launched as a normal `vllm serve` process with an explicit
vLLM-Ascend worker class:

```bash
VLLM_PLUGINS=ascend,afd vllm serve <model> \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUFFNWorker \
  --additional-config '{"afd":{"enabled":true,"role":"ffn","connector":"camp2pconnector","host":"127.0.0.1","port":1239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
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
  afd_plugin.v1.worker.ascend.AFDNPUFFNWorker
  afd_plugin.v1.worker.ascend.AFDNPUFFNModelRunner
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
- derives `afd_role_rank` from DP/TP ranks;
- creates `camp2pconnector`;
- returns empty KV cache specs and no-ops KV initialization;
- receives DP metadata and Attention outputs from the connector;
- builds a minimal Ascend forward context for connector-driven FFN steps;
- mirrors AFD metadata into `additional_kwargs["afd_metadata"]` and
  `forward_context.afd_metadata`;
- calls `model.compute_ffn_output(...)` with CAMP2P payload fields;
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
  -> recv_dp_metadata_list(timeout_ms=100)
  -> model_runner.execute_ffn_step(...)
  -> torch.npu.synchronize()
```

`execute_ffn_step()` routes warmup/capture metadata to `capture_model()` when
ACL graph is active. Otherwise it calls connector-driven `execute_model()` with
the received `dp_metadata_list`.

## FFN Forward

For each layer and stage, `AFDNPUFFNModelRunner`:

1. updates connector state from DP metadata;
2. creates receive metadata with `connector.create_recv_metadata(...)`;
3. receives an `AFDRecvOutput` from `connector.recv_attn_output(...)`;
4. updates connector metadata from the received payload;
5. installs DP and AFD metadata on Ascend forward context;
6. waits for async receive handles when present;
7. calls `model.compute_ffn_output(...)`;
8. sends the FFN output through `connector.send_ffn_output(...)`.

The current compute call forwards these payload fields when available:

- `hidden_states`;
- `group_list`;
- `dynamic_scales`;
- `topk_weights`;
- `topk_ids`;
- `router_logits`;
- `row_idx`;
- `x_active_mask`;
- `cam_p2p_ep_name`.

## CAMP2P

`camp2pconnector`, implemented by `afd_plugin.connectors.npu.camp2p`, owns NPU
topology, HCCL/Gloo process groups, custom-op loading, DP metadata exchange,
receive metadata construction, and FFN/Attention payload transfer.

The connector supports non-equal A/F topologies where
`num_attention_ranks >= num_ffn_ranks` and the ratio is integral. FFN token
counts are derived from Attention DP metadata and projected back to DP-level
counts for vLLM's forward context when TP is enabled.

## ACL Graph

The NPU FFN runner can use ACL graph when vLLM-Ascend has graph mode enabled.
Graph cache keys are built from DP metadata shape plus A/F topology. Capture
updates connector state before entering the NPU graph context, so connector
control-plane state is not repeatedly recomputed as part of normal replay.

If no captured graph exists for a key, the runner falls back to eager execution.

## Supported And Rejected Features

Supported:

- vLLM `0.19.1` runtime stack with vLLM-Ascend model runner v1;
- `--additional-config '{"afd": ...}'`;
- `camp2pconnector`;
- connector-driven FFN daemon loop;
- empty KV cache;
- eager FFN execution;
- ACL graph warmup/capture/replay path;
- DBO with exactly two ubatches;
- full model weight loading.

Rejected by validation:

- vLLM-Ascend model runner v2;
- scheduler-driven FFN requests;
- `compute_gate_on_attention=true`;
- `quant_mode != 0`;
- attention/FFN multistream communication;
- DBO with a ubatch count other than two;
- role-based weight pruning.
