# NPU AFDAsyncConnector MoE-only 双 batch 流水设计

本文档记录 `afdasyncconnector` 在 Ascend NPU、PD 分离 prefill 阶段、
`--enforce-eager` 模式下支持双 batch 流水的增量方案。

当前 Phase 1-4 已在插件内实现。后续实现仍应遵守本仓库 `AGENTS.md` 的
external plugin 边界：不修改 `../vllm` 或 `../vllm-ascend`，所有行为由插件包、
显式 class path、connector 注册、兼容 shim 或插件内测试提供。

## 背景

当前 `AFDAsyncConnector` 面向 vLLM `async_dp` 场景，使用 CAM 四个异步算子承载
Attention 与 FFN/MoE 间的数据流：

```text
Attention side:
  compute_attn_output(...)
  -> async_dispatch_send(...)
  -> async_combine_recv(...)
```

当前基础设计中 `DBO / ubatching` 被列为非目标，因为 vLLM 原生 `UbatchWrapper`
会把整段模型 forward 切成 micro-batch，并带来多线程、全模型输入切片和 graph
相关语义。

新的目标不是打开 vLLM 原生 DBO，而是在 `afdasyncconnector` 内部增加一个
**MoE-only、request-boundary、eager-only** 的轻量 stage 流水：

```text
Dense / Attention layer:
  使用完整 batch

MoE layer:
  -> 按 request boundary 切成两个 stage
  -> stage0 / stage1 分别切换 stage attention metadata
  -> stage0 / stage1 分别 recv 上一层 FFN 输出（如果有 pending）
  -> stage0 / stage1 分别 compute attention / gate
  -> stage0 / stage1 分别通过 async connector dispatch
  -> stitch 回 full batch
```

## 目标

- 仅面向 `afdasyncconnector`。
- 仅面向 Ascend NPU。
- 仅支持 `--enforce-eager`，不进入 ACL graph capture / replay。
- 主要面向 PD 分离中的 prefill 阶段。
- 使用两个 stage，第一版固定 `num_stages == 2`。
- 使用 request-boundary 切分，不做 token-level 切分。
- 不使用 vLLM `UbatchWrapper` 或插件当前 `AscendUBatchWrapper`。
- Dense 层、非 MoE 层和 attention 计算继续使用完整 batch。
- 只有 MoE 层的 Attention->FFN->Attention payload 使用双 stage。
- 在 MoE stage 运行期间临时切换 `forward_context` 中与 stage 相关的 metadata，
  退出 stage 后恢复 full-batch context。

## 当前实现状态

- Phase 1：配置开关与 validation 已实现。
- Phase 2：request-boundary 双 stage 切分 helper 已实现。
- Phase 3：Attention runner full metadata + async MoE sidecar metadata 已实现。
- Phase 4：DeepSeek MoE 层按 stage send/recv 并 stitch 回 full batch 已实现。
- Phase 5：PCP / DSA-CP 的 stage-local metadata 仍未实现，当前 validation
  直接拒绝 context parallel。

## 非目标

第一版不覆盖：

- vLLM 原生 DBO/ubatching；
- 通过 `parallel_config.use_ubatching` 触发 `UbatchWrapper`；
- token-level request 内切分；
- decode 阶段双 batch；
- ACL graph；
- NPU multistream；
- 多于两个 stage；
- arbitrary attention backend 的 PCP stage metadata 自动支持。

如果某些场景不能构造正确 stage-local metadata，应 fail fast，而不是退化为隐式共用
full-batch metadata。

## 关键设计结论

### 不使用 `UbatchWrapper`

`UbatchWrapper` 的语义是把模型输入、forward context、attention metadata 和
intermediate tensor 都按 ubatch 切开，然后以多线程或 graph replay 方式执行整个模型
子图。这和本方案目标冲突：

- dense 层也会被切分；
- model forward 会进入多线程；
- eager async connector 只需要 MoE 传输流水，不需要整模型 micro-batch；
- PCP / DSA-CP 的 metadata 切分应只服务 MoE stage，而不是改变全局 attention
  execution。

因此，本方案只复用 `UBatchSlice` 这类轻量数据结构和少量 metadata helper 思路，
不复用 `UbatchWrapper.__call__()` 的执行模型。

### Full metadata 与 sidecar metadata 分离

Attention runner 仍应返回 full-batch attention metadata：

```text
forward_context.attn_metadata = full_batch_attn_metadata
```

双 batch 所需的 stage metadata 作为 sidecar 存在：

```python
forward_context.additional_kwargs["afd_async_moe_ubatch_metadata"] = {
    "ubatch_slices": ubatch_slices,
    "attn_metadata": stage_attn_metadata,
}
```

默认情况下，Attention 侧模型层看到的是 full-batch context。只有进入 MoE stage
send/recv 的短窗口，才临时覆盖：

```text
forward_context.attn_metadata
forward_context.additional_kwargs["afd_metadata"]
forward_context.ubatch_idx
forward_context.num_ubatches
forward_context.num_tokens
```

### Request-boundary 切分

第一版切分必须落在 request boundary 上，避免一个 prefill request 被两个 stage
分别处理。这样 PCP metadata 可以按 request 集合重建，不需要处理“后半段 query 把前半
段 query 当作 context”的复杂语义。

基础切法可以是按 request 数对半：

```text
req_split = ceil(num_reqs / 2)

stage0:
  request_slice = [0, req_split)
  token_slice   = [0, query_start_loc_cpu[req_split])

stage1:
  request_slice = [req_split, num_reqs)
  token_slice   = [query_start_loc_cpu[req_split], num_tokens)
```

后续可以升级为“在 request boundary 上寻找最接近 token 一半的位置”：

```text
token_prefix = query_start_loc_cpu
target = num_tokens / 2
req_split = argmin_i abs(token_prefix[i] - target), 1 <= i < num_reqs
```

如果 `num_reqs < 2`，或任一 stage token 数为 0，则关闭双 batch，回退单 stage。

## End-to-end 数据流

### Attention side

```text
execute_model()
  -> vLLM-Ascend build full attention metadata
  -> plugin 生成 async MoE ubatch sidecar
  -> model forward(full batch)

model.forward_with_afd_v3()
  for layer in layers:
    if dense:
      if has pending MoE output:
        recv all stages and stitch
      layer(full batch)
      continue

    if moe:
      for stage in [0, 1]:
        with async_moe_stage_context(stage):
          if has pending MoE output:
            hidden[stage] = recv_ffn_output(stage)
          attn_out[stage], residual[stage], topk[stage] =
              compute_attn_output(hidden[stage])
          send_attn_output(attn_out[stage], topk[stage], metadata[stage])
      hidden = stitch(attn_out[0], attn_out[1])
      residual = stitch(residual[0], residual[1])
```

Dense 层不切。MoE 层会在 stage context 内切换 attention metadata，并对 stage-local
hidden / residual / positions 执行 `compute_attn_output()`。

## 配置建议

建议不要复用 `parallel_config.use_ubatching` 作为此功能开关。该字段属于 vLLM 原生
ubatching，会导致当前 NPU runner 安装 `AscendUBatchWrapper`。

建议放在 AFD `extra_config`：

```json
{
  "afd": {
    "enabled": true,
    "connector": "afdasyncconnector",
    "role": "attention",
    "extra_config": {
      "async_moe_ubatching": true,
      "async_moe_num_ubatches": 2,
      "async_moe_split": "request"
    }
  }
}
```

Validation 规则：

- `connector == "afdasyncconnector"`；
- `model_config.enforce_eager == True`；
- `parallel_config.async_dp == True`；
- `compute_gate_on_attention == True`；
- `async_moe_num_ubatches == 2`；
- `parallel_config.use_ubatching == False`，避免安装 `UbatchWrapper`；
- `prefill_context_parallel_size == 1` 且 `decode_context_parallel_size == 1`；
- graph / multistream 继续禁用；
- 非 prefill 或不支持的 attention backend 可以先 fail fast。

## Attention metadata sidecar

### Full-batch metadata 保持不变

`AFDNPUAttentionModelRunner._build_attention_metadata()` 的返回值仍应是 full-batch
metadata，不能在 async MoE ubatching 开启时返回 list。

建议新增内部 helper：

```python
def _maybe_build_async_moe_ubatch_sidecar(...):
    if not enabled:
        return None
    ubatch_slices = create_request_boundary_ubatch_slices(...)
    stage_attn_metadata = build_stage_attn_metadata(ubatch_slices, full_common_metadata)
    stage_afd_metadata = build_stage_afd_metadata(...)
    return AFDAsyncMoEUbatchMetadata(...)
```

sidecar 可以挂在 runner 的 pending field，再由 `_install_afd_metadata_on_forward_context`
写入 `forward_context.additional_kwargs`。

### Stage common metadata

每个 stage 的 `AscendCommonAttentionMetadata` 应按 request/token slice 重建：

- `query_start_loc` / `query_start_loc_cpu` 从 0 rebase；
- `seq_lens` / `seq_lens_cpu` 按 request slice 切；
- `num_computed_tokens_cpu` 按 request slice 切；
- `block_table_tensor` 按 request slice 切；
- `slot_mapping` 按 token slice 切；
- `positions` 按 token slice 切；
- `num_reqs`、`num_actual_tokens`、`num_input_tokens`、`max_query_len`、`max_seq_len`
  按 stage 重新计算。

因为切分落在 request boundary 上，不需要处理 `splits_first_request` 和
`splits_last_request` 的 token 修正。

## PCP metadata 处理

开启 PCP 后，stage metadata 不能复用 full-batch
`prefill_context_parallel_metadata`。它应按 stage 重新建立：

```text
stage common metadata
  -> stage-local prefill_context_parallel_metadata
  -> CP-aware builder.build(stage common metadata)
  -> final stage attention metadata
```

原因是 `prefill_context_parallel_metadata` 内有多套坐标系：

- Q token 坐标；
- PCP-expanded KV 坐标；
- all-gather restore 坐标；
- head/tail attention 坐标；
- linear-attn enter/exit restore/scatter 坐标。

这些字段不能简单从 full-batch metadata 上 slice。request-boundary 切分只降低复杂度，
不改变“必须 stage-local”的要求。

建议将 PCPManager 中生成 long-seq metadata 的核心逻辑抽成可复用 helper，显式接收
stage-local 输入：

```python
query_lens
num_scheduled_tokens
num_computed_tokens_cpu
block_table_tensor
num_reqs
num_reqs_padded
total_num_scheduled_tokens
```

生成的字段必须从 stage-local 0 坐标开始，包括：

- `q_head_idx_tensor`
- `q_tail_idx_tensor`
- `q_full_idx`
- `pcp_allgather_restore_idx`
- `pcp_unpad_mask`
- `pcp_fa_query_idx`
- `pcp_enter_fa_restore_idx`
- `pcp_exit_fa_scatter_idx`
- `num_computed_tokens_of_pcp_dcp`
- `query_lens_pcp_full_cpu`
- `num_actual_tokens_pcp_padded`

### SFA CP builder 例子

SFA CP builder 的 `build()` 先调用普通 SFA builder 生成 `AscendSFAMetadata`，再通过
`build_cp_metadata()` 构造 `sfa_cp_metadata`。其中 `AscendPCPMetadata` 主要来自：

```text
common_attn_metadata.prefill_context_parallel_metadata
  -> q_head_idx_tensor
  -> q_tail_idx_tensor
  -> q_full_idx
  -> pcp_allgather_restore_idx

common_attn_metadata.num_computed_tokens_cpu + seq_lens
  -> head_attn_nomask_seqlens
  -> tail_attn_nomask_seqlens
```

因此 PCP + async MoE ubatching 下，传给 SFA CP builder 的
`common_attn_metadata.prefill_context_parallel_metadata` 必须已经是 stage-local。
否则 `index_select`、all-gather restore 和 q head/tail 恢复都会使用 full-batch 坐标。

### DSA-CP

`DSACPContext` 不是 PCP metadata 的同一个对象。它由普通 SFA builder 根据当前
`common_attn_metadata` 计算：

- `num_tokens` / `num_tokens_pad`；
- TP rank 的 `local_start` / `local_end_with_pad`；
- `slot_mapping_cp`；
- `actual_seq_lengths_query`；
- `actual_seq_lengths_key`。

如果每个 stage 都重新走 SFA builder，并且 stage common metadata 已经是 stage-local，
`DSACPContext` 会自然按 stage 重建。不要手工 slice full-batch `DSACPContext`。

## Forward context 切换

建议新增一个很小的 context manager，仅用于 MoE stage：

```python
with use_async_moe_stage_forward_context(stage_idx):
    connector.send_attn_output(...)
```

它负责保存并恢复 parent context 中的字段：

```text
attn_metadata
additional_kwargs
afd_metadata
ubatch_idx
num_ubatches
num_tokens
```

stage context 中：

- `attn_metadata = sidecar.stage_attn_metadata[stage_idx]`；
- `additional_kwargs["afd_metadata"] = stage-local AFDMetadata`；
- `ubatch_idx = stage_idx`；
- `num_ubatches = 2`；
- `num_tokens = stage_num_tokens`；
- 不改变 `dbo_enabled`。

这里不建议把 `dbo_enabled=True` 作为主语义，因为这不是 vLLM 原生 DBO，也没有
`UbatchWrapper` 的 yield/thread 语义。

## Attention side 模型改动点

新增 `AFDDeepseekV2Model.forward_with_afd_v3()` 承载 async MoE 双流水。
`forward_with_afd_v2()` 保持 compute-gate 的单 stage 路径。

未开启 async MoE ubatching 时，每个 MoE 层仍是单 stage：

```text
compute_attn_output(single stage)
send_attn_output(single stage)
pending_ffn_recv = True
下一层前 recv_ffn_output(single stage)
```

新逻辑在 async MoE ubatching 开启时变为：

```text
for stage in stages:
  switch forward_context.attn_metadata to stage metadata
  if previous MoE output is pending:
    hidden_states[token_slice] = recv_ffn_output(stage)
  attn_out, residual, topk = compute_attn_output(stage-local tensors)
  send_attn_output(attn_out, topk, ...)

hidden_states = stitch(stage attention outputs)
residual = stitch(stage residuals)
```

如果下一层是 dense 层，必须先把 pending MoE output 按 stage 收回并 stitch 成完整
hidden states；如果下一层仍是 MoE 层，可以在每个 stage 内先 recv 再计算当前层
attention。

后续若要跨 MoE 层或跨 dense 层进一步 overlap，需要额外证明残差、dense 输入和
connector ordering 都安全，第一版不做。

## 张量切分与 stitch

Attention side 按 `token_slice` 切：

```text
hidden_states
residual
topk_weights
topk_ids
router_logits
positions, 如果 stage context 需要
```

收到 connector 返回结果后按原 token order 拼回：

```python
full_output = hidden_states.clone()
for stage_idx, ubatch_slice in enumerate(ubatch_slices):
    full_output[ubatch_slice.token_slice] = stage_outputs[stage_idx]
hidden_states = full_output
```

request-boundary 切分保证 token order 是连续区间，因此 stitch 可以是简单
`index copy` 或 `torch.cat(stage_outputs, dim=0)`。为了后续扩展和安全，建议第一版
仍按 `token_slice` 写回。

## 与 PCP / DSA-CP 的风险边界

第一版建议按 backend 显式放行：

- SFA 无 CP：放行；
- SFA + PCP：在 stage-local `prefill_context_parallel_metadata` 重建完成后放行；
- SFA + DSA-CP：stage-local common metadata 重新 builder 后放行；
- SFA + PCP + DSA-CP：需要真实 NPU smoke；
- MLA CP / 通用 attention CP：先 fail fast，除非补齐对应 builder 的 stage-local
  long-seq metadata 验证；
- spec decode / hybrid attention / linear-attn enter-exit restore：先 fail fast 或单独
  建验证矩阵。

## 增量实施计划

### Phase 1：配置与 validation

- 增加 `async_moe_ubatching` 配置解析。
- 允许 `afdasyncconnector` 在该配置下进入 MoE-only 双 batch。
- 继续禁止 `parallel_config.use_ubatching`，防止安装 `UbatchWrapper`。
- 单测覆盖 unsupported matrix。

### Phase 2：request-boundary slice helper

- 新增 request-boundary split helper。
- 返回 `UBatchSlice` 兼容对象。
- 单测覆盖：
  - 偶数 request；
  - 奇数 request；
  - 单 request 关闭；
  - 大小不均匀 request；
  - 空 stage 拒绝。

### Phase 3：attention metadata sidecar

- Attention runner 继续返回 full metadata。
- 额外构造 async MoE sidecar。
- 无 PCP 场景先构造 stage common metadata 和 final stage attention metadata。
- 单测确认 full metadata 不变，sidecar stage metadata 坐标从 0 开始。

### Phase 4：MoE-only stage forward

- 新增 `forward_with_afd_v3()`：只在 MoE 层 stage 化。
- Dense 层保持 full batch。
- 增加 stage context manager。
- fake connector 单测验证：
  - dense 层调用次数不变；
  - MoE send/recv 按 stage 调用；
  - stitch 后 shape/order 正确；
  - context 退出后恢复 full metadata。

### Phase 5：PCP / SFA CP

- 抽取或新增 stage-local PCP metadata 生成 helper。
- SFA CP builder 使用 stage-local `prefill_context_parallel_metadata`。
- DSA-CP 通过重新 builder 自动生成 `DSACPContext`。
- 真实 NPU smoke 覆盖：
  - prefill；
  - PCP；
  - `afdasyncconnector`；
  - `--enforce-eager`；
  - 两 stage。

## 验证策略

CPU-safe 单测：

- 配置 validation；
- request-boundary split；
- sidecar metadata 容器；
- context manager restore；
- fake connector stage send/recv；
- stitch order。

Runtime/NPU 测试：

- no-CP baseline；
- SFA + PCP；
- SFA + DSA-CP；
- SFA + PCP + DSA-CP；
- 不均匀 request 长度；
- 单 request 回退单 stage。

性能验证：

- 不均匀 request 下 request-boundary split 的收益；
- 大单 request 场景是否应关闭双 batch。

## 开放问题

- PCPManager 的 long-seq metadata 生成逻辑如何最小改动地抽成 stage-local helper。
- request-boundary split 是否按 request 数对半，还是按 token 数在 request boundary
  上找最接近一半的位置。
- mixed batch 中 decode + prefill 是否直接禁用，还是只对 prefill request stage 化。
- 后续是否需要跨 dense 层延迟 combine，以获得更强流水，但这会显著扩大正确性风险。

## 当前推荐

第一版推荐实现为：

```text
afdasyncconnector
  + enforce eager
  + async_dp
  + compute_gate_on_attention
  + async_moe_ubatching
  + num_stages = 2
  + request-boundary split
  + full-batch dense/attention
  + MoE-only stage recv/attention/send/stitch
  + no-CP
```

这条路径改动面最小，也最符合 PD prefill 的目标：不改变全模型 forward 形态，只在
MoE attention 计算、payload 传输和 Attention 侧 stitch 边界引入双 stage 流水。
