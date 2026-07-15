# CAM Async Connector User Guide

`CAMAsyncAFDConnector` is the Ascend CAM-backed asynchronous connector for AFD
Attention/FFN disaggregation. It lets Attention workers compute MoE routing and
exchange routed and shared-expert activations with independent FFN expert ranks
through CAM async dispatch/combine operators.

This guide describes the supported deployment shape, configuration contract,
rank mapping, data flow, startup requirements, and current limitations. The
[DeepSeek-V3.2 recipe](../../recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md)
contains the complete validated multi-node launch commands.

## When to use this connector

Use `CAMAsyncAFDConnector` for the currently supported asynchronous Ascend NPU
prefill path when all of the following are true:

- CAM operator packages are installed on every node;
- Attention performs MoE gating before dispatch to FFN ranks;
- execution is eager and AFD async-DP is enabled with `async=true`;
- the service is the prefill stage of a prefill/decode-disaggregated deployment;
- optional MoE ubatching is managed by AFD as two request-boundary stages.

For CUDA deployments use `P2pNcclAFDConnector`. For synchronous Ascend P2P use
`CAMP2pAFDConnector`. CAM async currently does not support decode, ACL graph
execution, vLLM native DBO, or multistream.

## CAM async data flow

One MoE layer follows this sequence:

1. Attention computes top-k expert IDs and weights.
2. `async_dispatch_send` sends hidden states and routing IDs into the CAM group.
3. Each FFN rank calls `async_dispatch_recv` and receives its routed expert
   tokens, shared-expert tokens, token counts, and optional dynamic-quant scales.
4. The FFN worker executes its local routed and shared experts.
5. `async_combine_send` returns those outputs with the dispatch metadata.
6. Attention calls `async_combine_recv`; CAM routes, weights, and combines the
   expert results for the original tokens.

CAM dispatch payloads carry the token-count and routing metadata. Consequently,
this connector does not use the separate Gloo DP-metadata control plane used by
the synchronous connectors.

## Topology and rank derivation

The connector creates one HCCL world with all Attention ranks first and all FFN
ranks second:

```text
world rank:  0    1   ...  A-1   A    A+1  ...  A+F-1
member:      A0   A1  ...  A_    F0   F1   ...  F_
```

For `A = num_attention_ranks` and `F = num_ffn_ranks`:

- Attention role rank `i` has world rank `i`;
- FFN role rank `j` has world rank `A + j`;
- world size is `A + F`;
- each role rank must be unique and within its role's configured rank count.

Attention ranks are normally `DP x PCP`. `attn_ranks_per_dp` is the PCP width
and is also passed to CAM as its Attention TP width. For an Attention process
whose first data-parallel rank is `d`, use:

```text
afd_role_rank = d * attn_ranks_per_dp
```

The DeepSeek-V3.2 recipe uses `DP3PCP8 + EP8`:

```text
num_attention_ranks = 3 * 8 = 24
num_ffn_ranks = 8
attn_ranks_per_dp = 8

Attention node 0, DP start 0: afd_role_rank = 0 * 8 = 0
Attention node 1, DP start 2: afd_role_rank = 2 * 8 = 16
FFN EP8 process:              afd_role_rank = 0

CAM world ranks: A0..A23 = 0..23, F0..F7 = 24..31
```

FFN ranks follow expert parallel placement. The runtime derives experts per rank
from the model routed-expert count and `num_ffn_ranks`; use a model/topology in
which routed experts divide evenly across FFN ranks. All roles must use the same
model, routed-expert layout, quantization, HCCL address, rank counts, CAM
settings, and ubatching settings.

## AFD configuration

Pass AFD configuration through vLLM's `--additional-config` under the `afd`
key. There is no separate `--afd-config` option.

```jsonc
{
  "afd": {
    "enabled": true,
    "role": "attention",
    "connector": "CAMAsyncAFDConnector",
    "async": true,
    "host": "10.0.0.1",
    "port": 6239,
    "num_attention_ranks": 24,
    "num_ffn_ranks": 8,
    "afd_role_rank": 0,
    "compute_gate_on_attention": true,
    "extra_config": {
      "quant_mode": 0,
      "dynamicQuant": 1,
      "attn_ranks_per_dp": 8,
      "async_moe_ubatching": true,
      "async_moe_num_ubatches": 2,
      "async_moe_split": "request"
    }
  }
}
```

### Common fields

| Field | Type | Default | Meaning and constraint |
| --- | --- | --- | --- |
| `enabled` | `bool` | `false` | Must be `true` to enable the AFD runtime. |
| `role` | `"attention" \| "ffn"` | `"attention"` | Role owned by this process. |
| `connector` | `str` | `"P2pNcclAFDConnector"` | Must be `CAMAsyncAFDConnector`. |
| `async` / `async_dp` | `bool` | `false` | Must be `true`. `async` is the accepted compatibility alias for canonical `async_dp`. |
| `host` | `str` | `"127.0.0.1"` | HCCL rendezvous host, reachable with the same value from every rank. In the multi-node recipe it is the node owning Attention rank 0. |
| `port` | `int` | `1239` | HCCL rendezvous port in `1..65535`; it must be free and reachable. |
| `num_attention_ranks` | `int` | `1` | Total Attention ranks, including all DP/PCP-derived ranks. |
| `num_ffn_ranks` | `int` | `1` | Total FFN expert ranks. |
| `afd_role_rank` | `int` | `0` | Role-local starting rank. Account for `attn_ranks_per_dp` on Attention. |
| `compute_gate_on_attention` | `bool` | `false` | Runs MoE routing on Attention. Required when async MoE ubatching is enabled. |
| `extra_config` | `dict` | `{}` | Connector-specific settings. Unknown top-level AFD fields are rejected. |

Compatibility aliases `afd_role`, `afd_connector`, `afd_host`, `afd_port`, and
`afd_extra_config` are also accepted. New configurations should use the
canonical names shown above, except `async`, which is retained as the documented
compatibility spelling used by the recipes.

### CAM async `extra_config`

| Field | Type | Default | Meaning and constraint |
| --- | --- | --- | --- |
| `dynamicQuant` | `int` | `0` | Enables CAM dispatch/combine dynamic-quant metadata. Only `0` and `1` are accepted. With `1`, FFN receives quantized routed activations plus scale tensors and must return output compatible with combine-send. |
| `quant_mode` | `int` | `0` | Recipe-level Ascend/CAM quantization setting for the surrounding model/operator path; the connector itself does not read it. Only `0` is verified by the checked-in recipe. This is separate from `dynamicQuant`. |
| `attn_ranks_per_dp` | `int` | `1` | Positive Attention rank count per DP replica, normally the PCP width. It affects Attention role-rank derivation and CAM TP size. |
| `async_moe_ubatching` | `bool` | `false` | Enables AFD-managed asynchronous MoE-only ubatching. |
| `async_moe_num_ubatches` | `int` | `2` | Number of asynchronous MoE stages. Only `2` is supported. |
| `async_moe_split` | `str` | `"request"` | Stage split policy. Only request-boundary splitting is supported. |
| `is_multistream` | `bool` | `false` | Must remain disabled. |
| `is_attn_multistream` | `bool` | `false` | Must remain disabled. |
| `is_ffn_multistream` | `bool` | `false` | Must remain disabled. |
| `multistream_info` | mapping | unset | Any enabled global, Attention, or FFN multistream entry is rejected. |

## Native DBO and async MoE ubatching are different

### vLLM native DBO

Do not pass any of these options to a CAM async process:

```bash
--enable-dbo
--dbo-decode-token-threshold <N>
--dbo-prefill-token-threshold <N>
```

They enable vLLM's native dual-batch overlap/ubatching. Runtime validation
rejects native DBO with `CAMAsyncAFDConnector`; those flags belong to supported
synchronous connector deployments.

### AFD-managed asynchronous MoE ubatching

`async_moe_ubatching` pipelines only the MoE portion of CAM async execution.
Requests are divided at request boundaries into exactly two stages. Each stage
keeps its own pending Attention routing metadata so dispatch and combine remain
paired while Attention and FFN work overlap. It does not enable vLLM native DBO
and does not use the DBO threshold flags.

When `async_moe_ubatching=true`, all roles must set:

```json
{
  "compute_gate_on_attention": true,
  "extra_config": {
    "async_moe_ubatching": true,
    "async_moe_num_ubatches": 2,
    "async_moe_split": "request"
  }
}
```

Decode context parallel size greater than one is also rejected because the
current async MoE metadata path does not support it.

## Requirements

The checked-in recipe has been verified with:

- Ascend 910C;
- `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler`;
- the included `CAM_ascend910_93_openEuler_aarch64.run` installer;
- `umdk_cam_op_lib-208.1.0b1-cp311-cp311-linux_aarch64.whl`;
- DeepSeek-V3.2 W8A8, using the reduced 10-layer experiment model described in
  the recipe;
- two nodes with `DP3PCP8` Attention and `EP8` FFN.

Install the CAM packages from the repository root inside the container:

```bash
bash afd_plugin/connectors/npu/bin/CAM_ascend910_93_openEuler_aarch64.run
pip install afd_plugin/connectors/npu/bin/umdk_cam_op_lib-208.1.0b1-cp311-cp311-linux_aarch64.whl
```

Every CAM async process needs the CAM operator library on its loader path and
the Ascend plugin enabled. The complete recipe includes all tuning variables;
the essential setup is:

```bash
export VLLM_PLUGINS=ascend,afd
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

At initialization, the runtime verifies that `torch`, `torch_npu`,
`umdk_cam_op_lib`, and the four real `torch.ops.umdk_cam_op_lib` operators are
available: `async_dispatch_send`, `async_dispatch_recv`,
`async_combine_send`, and `async_combine_recv`.

## Launch checklist

Use the three complete commands in the
[DeepSeek-V3.2 recipe](../../recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md#afd-cam-async):

1. Start the Attention rank-0 node first; it hosts the vLLM API and the HCCL
   rendezvous address.
2. Start remaining headless Attention nodes with their DP-derived
   `afd_role_rank` values.
3. Start the FFN process with the same model, topology, quantization, async MoE,
   `host`, and `port` settings.
4. Send prefill requests only to the non-headless Attention API endpoint (port
   `8000` in the recipe). The FFN port is not a request endpoint.

All CAM async commands must use the Ascend worker classes, `--enforce-eager`,
`--quantization ascend`, `--enable-expert-parallel`, and no native DBO flags:

```text
Attention: afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker
FFN:       afd_plugin.v1.worker.ascend.AFDNPUFFNWorker
```

HCCL initialization is collective. Missing ranks, duplicate role ranks,
unreachable rendezvous addresses, or inconsistent world sizes/configuration can
fail or wait until the connector's initialization timeout.

## Current limitations

- Eager execution only; ACL graph mode is unsupported.
- Prefill stage only in a prefill/decode-disaggregated deployment.
- vLLM native DBO/ubatching is unsupported.
- AFD-managed MoE ubatching supports exactly two request-boundary stages.
- Global, Attention, and FFN multistream modes are unsupported.
- `dynamicQuant` accepts only `0` or `1`; only `quant_mode=0` is verified.
- Decode context parallel metadata is unsupported with async MoE ubatching.
- Routed experts should divide evenly across FFN ranks.
- Other Ascend hardware, full unmodified DeepSeek-V3.2, different model
  families, CAM/CANN/container versions, cross-version combinations, and
  topologies other than the recipe should be treated as unverified.
- There is no automatic transport fallback. Select a synchronous connector
  explicitly if CAM async does not match the deployment.
