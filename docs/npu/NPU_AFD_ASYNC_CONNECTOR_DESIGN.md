# NPU AFDAsyncConnector 设计

本文档设计 NPU 场景下新的 `AFDAsyncConnector`。该 connector 面向 vLLM
`async_dp` 分支，用 CAM 四个异步通信算子承载 Attention 与 FFN/MoE 之间的数据流，
并且不再使用当前 AFD 的 DP metadata send/recv control-plane。

本文档只描述设计，不要求当前阶段修改代码。后续实现时应继续遵守本仓库
`AGENTS.md` 中的 external plugin 边界：不修改 `../vllm` 或 `../vllm-ascend`，
所有行为由插件包、显式 class path、connector 注册、兼容 shim 或插件内测试提供。

## 背景

当前 NPU AFD 路径主要围绕 `camp2pconnector`：

```text
Attention runner
  -> 构造 AFD metadata
  -> 构造/发送 dp_metadata_list
  -> 模型层内 send_attn_output(hidden_states, metadata)
  -> recv_ffn_output(...)

FFN worker loop
  -> recv_dp_metadata_list()
  -> execute_ffn_step(dp_metadata_list, graph flags)
  -> per-layer/per-stage recv_attn_output()
  -> compute_ffn_output(...)
  -> send_ffn_output(...)
```

这里的 `dp_metadata_list` 同时承担两个职责：

- **控制面同步**：Attention 侧告诉 FFN 侧本 step 有哪些 stage、各 DP rank token
  数、是否 warmup / graph capture；
- **FFN step 触发**：FFN daemon loop 阻塞等待 `recv_dp_metadata_list()`，收到后才
  执行一次 FFN step。

`async_dp` 场景下，vLLM 会跳过跨 DP rank 的 batch/wave 协调。当前本地
`../vllm` 的 `async_dp` 分支已经做了两类修正：

- async DP 下不启用 MoE DP wave coordination；
- async DP 下 `set_forward_context()` / `coordinate_batch_across_dp()` 不做 DP
  all-reduce batch coordination。

因此，`AFDAsyncConnector` 不能继续假设 `forward_context.dp_metadata` 一定存在，
也不能继续把 FFN step 建立在单独的 DP metadata 消息之上。

## 目标

- 新增 Ascend NPU-only connector：`afdasyncconnector` / `AFDAsyncConnector`。
- `AFDAsyncConnector` 只能运行在 Ascend NPU 平台，不支持 CUDA、CPU 或其他后端。
- 使用该 connector 时必须启用 `--async-dp`。
- 使用该 connector 时 Attention 侧和 FFN 侧都必须是 eager 模式；第一版不支持
  Attention eager + FFN graph 或 Attention graph + FFN eager 的混合形态。
- 使用 CAM 四个算子接口：
  - `torch.ops.umdk_cam_op_lib.async_dispatch_send`
  - `torch.ops.umdk_cam_op_lib.async_dispatch_recv`
  - `torch.ops.umdk_cam_op_lib.async_combine_send`
  - `torch.ops.umdk_cam_op_lib.async_combine_recv`
- 初始化 connector 前必须按顺序导入 `torch`、`torch_npu`、
  `umdk_cam_op_lib`，加载真实 CAM 算子二进制；缺失时 fail fast。
- 不再发送和接收 `dp_metadata_list`。
- 走 connector 新增流程：新增 connector 名称、模块、factory 注册、connector
  capability、validation 和测试；不通过改造 `camp2pconnector` 或复用其 rank 语义来
  实现。
- 保留现有 `p2pconnector` / `camp2pconnector` 的行为，不因为 async connector
  改坏当前 GPU/NPU eager 路径。

## 非目标

第一版不覆盖以下能力：

- ACL graph capture / replay；
- 任一侧非 eager 的 AFD async 运行；
- NPU multistream；
- DBO / ubatching；
- 真实 CAM 通信算子性能优化；
- async DP 下多 Attention rank 到同一 FFN rank 的复杂乱序调度优化。

这些能力可以在 connector-driven 基础路径稳定后逐步补齐。

## 与当前 EngineCore Patch 的关系

当前 `afd_plugin.compat.patches.engine_core` 只在 AFD FFN role 下特殊处理
`EngineCore`：

- Attention role 仍走 vLLM 原生 `EngineCore` / `EngineCoreProc` 路径；
- FFN role 会跳过 scheduler / KV cache 初始化，并让 EngineCore busy loop 启动
  connector-driven FFN worker loop。

对 `AFDAsyncConnector` 来说，EngineCore patch 的总体方向仍然合理：FFN 侧仍然是
connector daemon，而不是普通 scheduler-driven engine。但 FFN worker loop 的内部
触发方式需要分支：

```text
camp2pconnector / p2pconnector:
  recv_dp_metadata_list()
  -> execute_ffn_step(dp_metadata_list, flags)

afdasyncconnector:
  execute_connector_driven_step()
  -> connector 内部通过 cam_dispatch_recv(...) 阻塞等待数据
```

也就是说，EngineCore patch 不需要因为 async connector 整体废弃，但 FFN worker /
model runner 不能再强制依赖 `recv_dp_metadata_list()`。

## Connector Contract 重设计

当前 `AFDConnectorBase` 把 DP metadata 收发作为普通方法暴露：

```python
send_dp_metadata_list(...)
recv_dp_metadata_list(...)
update_state_from_dp_metadata(...)
```

对于 `AFDAsyncConnector`，这些方法不应作为真实控制面使用。推荐给 connector 增加
显式 capability，而不是靠空方法隐式表达行为：

```python
class AFDConnectorBase:
    uses_dp_metadata_control_plane = True
    ffn_step_trigger = "dp_metadata"
```

各 connector 的语义：

```text
p2pconnector:
  uses_dp_metadata_control_plane = True
  ffn_step_trigger = "dp_metadata"

camp2pconnector:
  uses_dp_metadata_control_plane = True
  ffn_step_trigger = "dp_metadata"

afdasyncconnector:
  uses_dp_metadata_control_plane = False
  ffn_step_trigger = "connector"
```

`AFDAsyncConnector` 中 DP metadata 相关方法建议这样处理：

```python
def update_state_from_dp_metadata(...):
    return None

def send_dp_metadata_list(...):
    return None

def recv_dp_metadata_list(...):
    raise RuntimeError(
        "AFDAsyncConnector does not use DP metadata control-plane; "
        "FFN worker must use connector-driven loop"
    )
```

这样可以保证错误路径清晰：如果后续某个 runner 仍误调用
`recv_dp_metadata_list()`，会直接暴露 worker loop 没切换，而不是静默空转。

## 配置与校验

新增 connector 名称：

```text
afdasyncconnector
```

配置示例：

```json
{
  "afd": {
    "enabled": true,
    "role": "attention",
    "connector": "afdasyncconnector",
    "num_attention_servers": 8,
    "num_ffn_servers": 8,
	    "extra_config": {
	      "max_seq_len": 32768,
	      "expert_per_rank": 32,
	      "tp_size": 8,
	      "quant_mode": 1
	    }
	  }
	}
```

FFN role 同样使用 `connector: "afdasyncconnector"`，`role` 改为 `"ffn"`。

校验建议：

- `connector == "afdasyncconnector"` 时必须满足
  `vllm_config.parallel_config.async_dp == True`；
- `connector == "afdasyncconnector"` 时必须确认当前平台是 Ascend NPU：
  - worker class path 必须来自 `afd_plugin.v1.worker.ascend.*`；
  - runtime 初始化时必须能导入 `torch_npu` 和 `vllm_ascend`；
  - 如果能访问 `current_platform`，平台应为 Ascend/NPU；否则在 connector 初始化阶段
    fail fast；
- 只允许 Ascend NPU worker class path：
  - `afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker`
  - `afd_plugin.v1.worker.ascend.AFDNPUFFNWorker`
- 第一版要求 Attention 侧和 FFN 侧都传入 eager 配置：
  `model_config.enforce_eager == True`。这应作为启动约束分别在两个 role 的
  worker/model runner 初始化时校验，不能只在某一侧隐式假设；
- 第一版要求 `parallel_config.use_ubatching == False`；
- 第一版要求 multistream 相关配置关闭；
- `quant_mode` / `dynamicQuant` 按 CAM 接口允许 `0` 或 `1`，不要复用
  `camp2pconnector` 当前只允许 `quant_mode=0` 的限制；
- Attention 侧必须传入真实 top-k 路由信息；当前通过
  `compute_gate_on_attention=true` 在 Attention 侧计算 gate，缺失时必须 fail fast。

注意：NPU validation 应变成 connector-specific。`camp2pconnector` 保持现有限制，
`afdasyncconnector` 单独放开或新增限制，避免全局改动影响当前路径。

如果 Attention 和 FFN 分别由两个 `vllm serve` 进程启动，无法在单个进程中直接看到
对侧配置。第一版要求用户显式保证两侧都使用 `--enforce-eager`；插件侧在本进程内
验证本 role 的 eager 配置。后续如果需要跨进程启动一致性检查，可以通过 AFD
additional config hash 或 connector handshake 增加对侧 eager 标记，但第一版不引入
新的控制面 handshake。

## CAM Topology

CAM 示例中的 rank 语义是：

```text
rank < attentionRankSize:
  Attention rank

rank >= attentionRankSize:
  MoE / FFN rank
```

这和当前 P2P/CAMP2P 迁移代码中常见的 “FFN rank 在前，Attention rank 在后” 不同。
因此 `AFDAsyncConnector` 不应直接复用 `CAMP2PAFDConnector` 的 `world_rank` /
`p2p_rank` 语义。

推荐新增独立 topology helper：

```text
AFDAsyncTopology:
  role
  role_rank
  world_rank
  attention_rank_size
  expert_rank_size
  world_size
  expert_per_rank
  tp_size
```

rank 推导：

```text
Attention:
  world_rank = role_rank

FFN / MoE:
  world_rank = attention_rank_size + role_rank

world_size = attention_rank_size + expert_rank_size
```

其中：

- `attention_rank_size` 来自 `AFDConfig.num_attention_servers`；
- `expert_rank_size` 来自 `AFDConfig.num_ffn_servers`；
- `role_rank` 对 async DP 应优先使用 vLLM 的真实 DP rank / data parallel index；
- 如果 vLLM async DP 分支在某些路径保留 `data_parallel_index` 而改写
  `data_parallel_rank`，实现时必须明确使用哪个字段，并在测试里覆盖。

## CAM 四算子映射

### Attention 侧

Attention 侧每一层 FFN/MoE 之前调用 dispatch send：

```python
torch.ops.umdk_cam_op_lib.async_dispatch_send(
    hidden_states,
    expert_ids,
    comm_args,
    comm_id,
    max_seq_len,
    batch_size,
    hidden_size,
    topk,
    expert_rank_size,
    attention_rank_size,
    expert_per_rank,
    world_rank,
    world_size,
    layer_idx,
    tp_size,
    dynamic_quant,
)
```

随后等待 FFN/MoE 侧 combine send 的结果：

```python
output = torch.ops.umdk_cam_op_lib.async_combine_recv(
    placeholder_or_ref_tensor,
    expert_ids,
    expert_scales,
    comm_args,
    comm_id,
    batch_size,
    hidden_size,
    topk,
    expert_rank_size,
    attention_rank_size,
    expert_per_rank,
    world_rank,
    world_size,
)
```

`async_combine_recv` 的输出 shape / dtype 应与 dispatch send 的 `hidden_states`
一致。

### FFN / MoE 侧

FFN/MoE 侧通过 dispatch recv 阻塞接收：

```python
dispatch_out = torch.ops.umdk_cam_op_lib.async_dispatch_recv(
    placeholder_tensor,
    comm_args,
    comm_id,
    max_seq_len,
    hidden_size,
    topk,
    expert_rank_size,
    attention_rank_size,
    expert_per_rank,
    world_rank,
    world_size,
    tp_size,
    dynamic_quant,
)
```

返回值：

```text
(
  expandXOut,
  expandXOut_shared,
  dynamicScalesOut,
  dynamicScalesOut_shared,
  TokenNums_Rankid_Layeridx,
  Expert_tokens,
  Expert_tokens_shared,
)
```

`TokenNums_Rankid_Layeridx` 是 FFN 侧替代 DP metadata 的关键 payload，至少包含：

```text
index 0: 实际收到 token 总数
index 1: 本轮 token 来源 DP 组第一个 rank 的全局 id
index 2: layer index
index 3: 本轮接收起始专家 id
index 4: 本轮接收终止专家 id
后续: 每个 attention cp rank 的 token offset / expert token counts
```

FFN/MoE 计算完成后调用 combine send：

```python
torch.ops.umdk_cam_op_lib.async_combine_send(
    ffn_output,
    ffn_output_shared,
    comm_args,
    TokenNums_Rankid_Layeridx,
    comm_id,
    batch_size,
    hidden_size,
    topk,
    expert_rank_size,
    attention_rank_size,
    expert_per_rank,
    world_rank,
    world_size,
    tp_size,
)
```

### 真实 CAM 接口

`AFDAsyncConnector` 只调用真实 `torch.ops.umdk_cam_op_lib.*` 算子：

- 参数签名必须与 `cam_ops.py` 和当前 NPU 算子二进制一致；
- connector 初始化阶段先导入 `torch`、`torch_npu`、`umdk_cam_op_lib`，
  再检查四个 async CAM entry point 是否已注册；
- 缺失真实算子时直接报错，不在 Python 层注册替代实现；
- CPU-safe 单元测试可以 monkeypatch `torch.ops.umdk_cam_op_lib` 验证参数顺序；
- NPU opt-in 测试负责验证真实 `torch_npu` / CAM 环境。

## Attention 侧逻辑

### AFD metadata

Attention runner 仍需要向 forward context 注入 `AFDMetadata`，因为模型 wrapper 需要
拿到 connector：

```text
forward_context.additional_kwargs["afd_metadata"] = AFDMetadata(...)
forward_context.afd_metadata = AFDMetadata(...)
```

但当 connector `uses_dp_metadata_control_plane == False` 时：

- 不读取或强制要求 `forward_context.dp_metadata`；
- 不调用 `_ensure_dp_metadata()`；
- 不调用 `send_dp_metadata_list()`；
- `AFDMetadata.afd_tokens_lens` 使用本地 batch token 数；
- DP>1 + async_dp 下允许 `forward_context.dp_metadata is None`。

### top-k 路由信息

CAM dispatch send 必须拿到：

- `expert_ids`: shape `[tokenNum, topKNum]`, dtype `int32`;
- `expert_scales`: shape `[tokenNum, topKNum]`, dtype `float32`;
- `topk`: `topKNum`。

当前 AFD 路径的真实 top-k 仍在 FFN/MoE 侧计算；而 CAM `cam_dispatch_send()` 要求
Attention 侧传入 `expertIds` 和 `expertScales`。因此 `AFDAsyncConnector` 的完整
正确性路径最终必须把 top-k 计算前移到 Attention 侧，或者让 Attention 侧能够捕获
同一层的 gate 结果。

当前实现要求 Attention 侧传入真实 `topk_ids` / `topk_weights`：

```text
topk_ids:
  shape = [tokenNum, topKNum]
  dtype = int32

topk_weights:
  shape = [tokenNum, topKNum]
  dtype = float32
```

缺少真实 top-k 时必须 fail fast。真实 top-k 接入有两种选择：

1. **要求 gate-on-attention**：模型 wrapper 在 Attention 侧运行 gate，调用
   `send_attn_output(hidden_states, metadata, topk_ids=..., topk_weights=...)`；
2. **模型层 wrapper 捕获路由结果**：在 DeepSeek MoE 层中拆分 attention/ffn 时，
   让 connector 能拿到同一层的 top-k ids / weights。

如果拿不到真实 top-k 信息，不能继续调用 CAM dispatch send，应直接报错。

### Attention 层内调用顺序

推荐顺序：

```text
for layer in model.layers:
  hidden_states, residual = attention_part(...)

  topk_ids, topk_weights = compute_or_fetch_router_topk(...)

  connector.send_attn_output(
      hidden_states,
      metadata,
      topk_ids=topk_ids,
      topk_weights=topk_weights,
  )

  hidden_states = connector.recv_ffn_output(
      ref_tensor=hidden_states,
      topk_ids=topk_ids,
      topk_weights=topk_weights,
  )
```

`send_attn_output()` 内部映射到 `cam_dispatch_send()`；
`recv_ffn_output()` 内部映射到 `cam_combine_recv()`。

## FFN Worker / Runner 逻辑

### Worker loop 分支

当前 NPU FFN worker loop 是：

```text
while running:
  dp_metadata_list, is_graph_capturing, is_warmup = recv_dp_metadata_list()
  execute_ffn_step(dp_metadata_list, flags)
```

新增 connector-driven 分支：

```text
while running:
  if connector.ffn_step_trigger == "connector":
    model_runner.execute_connector_driven_step()
  else:
    dp_metadata_list, is_graph_capturing, is_warmup = recv_dp_metadata_list()
    model_runner.execute_ffn_step(dp_metadata_list, flags)
```

`AFDAsyncConnector` 的 `execute_connector_driven_step()` 不需要
`dp_metadata_list`。

### Runner step

第一版 eager-only runner 逻辑：

```text
execute_connector_driven_step()
  -> 创建 minimal Ascend forward context
  -> for layer_idx in range(num_layers):
       recv_output = connector.recv_attn_output(layer_idx=layer_idx)
       hidden_states = recv_output.hidden_states
       payload = recv_output
       metadata = recv_output.metadata

       forward_context.additional_kwargs["afd_metadata"] = metadata
       forward_context.dp_metadata = None
       set moe_layer_index

       ffn_output = model.compute_ffn_output(
           hidden_states=hidden_states,
           layer_idx=layer_idx,
           group_list=payload.group_list,
           dynamic_scales=payload.dynamic_scales,
           topk_weights=payload.topk_weights,
           topk_ids=payload.topk_ids,
           router_logits=payload.router_logits,
           row_idx=payload.row_idx,
           x_active_mask=payload.x_active_mask,
           cam_p2p_ep_name=payload.cam_p2p_ep_name or "",
       )

       connector.send_ffn_output(ffn_output, metadata)
```

对 `AFDAsyncConnector`，`recv_attn_output(layer_idx=...)` 内部调用
`cam_dispatch_recv()`，并从 `TokenNums_Rankid_Layeridx` 构造
`AFDRecvOutput`。

### AFDConnectorMetadata

`AFDConnectorMetadata` 仍可以作为 plugin 内部统一 payload 使用，但不再表示跨进程
DP metadata。建议：

```text
metadata.layer_idx:
  来自 runner 当前 layer_idx，或 TokenNums_Rankid_Layeridx[2]

metadata.stage_idx:
  async connector 第一版固定 0；如果后续支持 ubatch，再映射为 ubatch id

metadata.seq_lens:
  [actual_total_tokens]，来自 TokenNums_Rankid_Layeridx[0]

metadata.connector_data:
  保存 CAM dispatch recv 返回的 TokenNums_Rankid_Layeridx、
  Expert_tokens、Expert_tokens_shared、shared buffer、dynamic scales 等
```

`AFDRecvOutput` 可承载：

```text
hidden_states = expandXOut
dynamic_scales = dynamicScalesOut
group_list = Expert_tokens
connector_data = metadata.connector_data
```

如模型计算同时需要 shared expert buffer，应扩展 `AFDRecvOutput` 或放入
`metadata.connector_data`。不要用 `getattr`/`hasattr` 在 runtime 热路径隐藏字段不匹配；
实际 runtime 数据结构应直接访问成员，升级不兼容时让真实错误暴露。

## DP Metadata 重设计

### 旧语义

当前 `dp_metadata_list` 是 AFD FFN 的 step descriptor：

```text
stage id
token counts across DP ranks
max token count
warmup / graph capture flags
```

### async connector 新语义

`AFDAsyncConnector` 不再拥有单独 DP metadata 消息。step descriptor 分成两部分：

```text
Attention 本地 step:
  batch_size = hidden_states.shape[0]
  layer_idx
  topk ids / weights
  max_seq_len
  CAM topology parameters

FFN 接收 step:
  actual token count = TokenNums_Rankid_Layeridx[0]
  source rank/group = TokenNums_Rankid_Layeridx[1]
  layer idx = TokenNums_Rankid_Layeridx[2]
  expert range / expert token counts
```

因此：

- Attention 侧不再构造 vLLM `DPMetadata.make(...)`；
- Attention 侧不再发送 `dp_metadata_list`；
- FFN 侧不再接收 `dp_metadata_list`；
- FFN graph key 第一版不使用 DP metadata，直接禁用 graph；
- 真实 token shape 由 CAM op 返回值决定。

### 与 async_dp 的关系

async DP 的核心是各 DP rank 独立调度。`AFDAsyncConnector` 必须避免任何跨 DP rank
同步：

- 不调用 `coordinate_batch_across_dp()`；
- 不要求所有 Attention rank 同步进入同一个 step；
- 不要求每个 FFN step 覆盖所有 Attention rank；
- 不依赖 `current_wave` 或 DP coordinator wakeup。

如果一个 FFN rank 服务多个 Attention rank，CAM op 必须承担 rank/layer/token 的匹配
语义。插件层第一版不应重新实现一个 Python control-plane 去恢复同步 DP。

## CAM Comm 初始化

参考 `test_cam_ops.py`，CAM comm 初始化形态类似：

```python
comm_args = cam.create_comm_moe(
    comm_id,
    rank,
    world_size,
    max_seq_len,
    hidden_size,
    topk,
    expert_rank_size,
    init_endpoint,
    True,
).to("npu")
```

`AFDAsyncConnector` 初始化时需要：

- 按顺序导入 `torch`、`torch_npu`、`umdk_cam_op_lib`；
- 确认真实 `torch.ops.umdk_cam_op_lib.*` CAM ops 已注册；
- 创建 async CAM 专用 HCCL process group，rank 顺序为
  `[A0, A1, ..., F0, F1, ...]`，不同于 `camp2pconnector` 的 FFN-first
  通信域；
- 从该 process group 获取 HCCL comm name，并作为 CAM 算子入参
  `group_name`；
- 构造 `comm_args`；
- 保存 `comm_id`；
- 保存 topology 参数；
- runtime 模块层可以直接导入真实依赖，顶层插件入口仍保持 CPU-safe。

配置建议放在 `AFDConfig.extra_config`：

```text
cam_init_endpoint
comm_id
max_seq_len
topk
expert_per_rank
tp_size
dynamic_quant / quant_mode
```

不要把环境变量或 magic number 散落在 runtime 代码里；新增环境变量应集中在 helper
模块中定义和说明。

## 真实 CAM Op 策略

运行时只接受真实 CAM 算子：

```text
afd_plugin.compat.ascend.ops
  ensure_cam_ops_available()
```

运行时：

- `ensure_cam_ops_available()` 必须先导入 `torch`、`torch_npu`，
  再导入 `umdk_cam_op_lib` 触发 op 注册；
- 如果 `torch.ops.umdk_cam_op_lib.*` 真实存在，直接使用真实 op；
- 否则 fail fast，提示 CAM ops 未加载。

测试时：

- CPU-safe 单元测试只验证 connector 方法调用 op 的参数；
- NPU opt-in 测试验证真实 `torch_npu` / CAM 环境。

## Connector 新增流程

`AFDAsyncConnector` 必须按新增 connector 的流程落地，不能作为
`camp2pconnector` 的条件分支实现。推荐步骤：

1. 在配置层新增 connector 名称：

```text
SUPPORTED_AFD_CONNECTORS += ("afdasyncconnector",)
```

2. 新增独立 connector 模块：

```text
afd_plugin/connectors/npu/async_cam.py
  AFDAsyncConnector
  AFDAsyncConnectorData
  AFDAsyncTopology
```

该模块可以复用通用 helper，但不复用 `CAMP2PAFDConnector` 的 topology 和 DP metadata
control-plane。

3. 在 `AFDConnectorFactory` 注册：

```text
afdasyncconnector
  -> afd_plugin.connectors.npu.async_cam.AFDAsyncConnector
```

4. 在 connector base 或轻量 helper 中增加 capability：

```text
uses_dp_metadata_control_plane = False
ffn_step_trigger = "connector"
requires_eager = True
required_platform = "ascend"
```

旧 connector 默认保持：

```text
uses_dp_metadata_control_plane = True
ffn_step_trigger = "dp_metadata"
```

5. 在 validation 中增加 connector-specific 规则：

```text
afdasyncconnector:
  - 必须 async_dp
  - 必须 Ascend NPU worker
  - 必须 enforce_eager
  - 禁用 ubatching / ACL graph / multistream
  - Attention 侧必须有真实 top-k 来源

camp2pconnector:
  - 保持当前规则
```

6. 在 NPU Attention / FFN runner 中通过 capability 分支，而不是通过 connector 名称
散落判断：

```text
if connector.uses_dp_metadata_control_plane:
  走现有 dp_metadata send/recv 路径
else:
  走 async connector 路径
```

7. 测试按 connector 新增流程补齐：

```text
config/factory tests
validation tests
connector CAM op shape tests
Attention runner no-dp-metadata tests
FFN worker connector-driven loop tests
```

这个流程的目标是让 `AFDAsyncConnector` 成为一个清晰隔离的新 data-plane，而不是让
现有 CAMP2P 路径承担 async DP 的额外分支复杂度。

## 现有代码需要调整的区域

### config / factory

- `SUPPORTED_AFD_CONNECTORS` 增加 `"afdasyncconnector"`；
- `AFDConnectorFactory` 注册
  `afd_plugin.connectors.npu.async_cam.AFDAsyncConnector`；
- validation 增加 connector-specific 检查：
  - async DP 必选；
  - Ascend NPU 平台必选；
  - Attention / FFN 两侧各自进程内都必须 eager；
  - 禁用 ubatching / graph / multistream。

### connector base

- 增加 capability；
- 保持旧 connector 默认 `uses_dp_metadata_control_plane=True`；
- `AFDAsyncConnector` 覆盖为 false。
- `AFDAsyncConnector` 声明 `requires_eager=True` 和 `required_platform="ascend"`；
- `recv_dp_metadata_list()` 对 `AFDAsyncConnector` 是错误路径，不作为空轮询接口使用。

### NPU Attention runner

- `_install_afd_metadata_on_forward_context()` 中，如果 connector 不使用 DP metadata
  control-plane，跳过 `_send_dp_metadata()`；
- `_ensure_dp_metadata()` 不能在 async DP 下因为 `dp_metadata is None` 报错；
- 初始化时校验 `model_config.enforce_eager == True`；
- Attention 模型 wrapper 需要给 connector 传 top-k ids / weights；
- 通过 gate-on-attention 或模型层真实路由捕获得到 top-k。

### NPU FFN worker

- `_run_ffn_server_loop()` 根据 connector capability 选择：
  - DP metadata loop；
  - connector-driven loop。
- connector-driven loop 只用于 Ascend NPU eager；如果本进程不是 eager，应在启动阶段
  fail fast。

### NPU FFN model runner

- 新增 `execute_connector_driven_step()`；
- 新增无 `dp_metadata_list` 的 `_ffn_forward_connector_driven()`；
- 第一版禁用 ACL graph，避免 `_make_graph_key(dp_metadata_list)` 路径。

### model wrapper

- 当前 `send_attn_output(hidden_states, metadata)` 不够用；
- `AFDAsyncConnector` 需要 `topk_ids` / `topk_weights`；
- 需要在 DeepSeek MoE 拆分点明确真实路由信息来源。

## 风险与开放问题

### top-k 路由信息来源

这是最大功能风险。当前真实 top-k 在 FFN/MoE 侧计算，但 CAM dispatch send 必须在
Attention 侧拿到 `expertIds` / `expertScales`。

真实功能完成前必须明确以下方案之一：

- gate-on-attention：在 Attention 侧运行 gate；
- 模型层 wrapper 捕获：在 DeepSeek MoE 拆分点把同一层真实 top-k 传给 connector。

缺少真实 top-k 时必须 fail fast。

### FFN loop shutdown

没有 DP metadata poll 后，FFN 侧可能长时间阻塞在 `cam_dispatch_recv()`。如果 CAM op
没有 timeout 或 cancel 机制，`stop_ffn_server_loop()` 可能无法优雅退出。需要确认：

- CAM comm destroy 是否能唤醒阻塞 recv；
- 是否需要一个 sentinel send；
- 是否允许 daemon thread 在进程退出时硬退出。

### 多 Attention rank 异步乱序

async DP 下多个 Attention rank 不同步。若一个 FFN rank 同时服务多个 Attention
rank，CAM op 必须可靠区分 rank / layer / token payload。插件层不要用 Python
DP metadata control-plane 重新同步它们，否则会抵消 async DP 的目的。

### graph 与 shape key

当前 graph key 来自 DP metadata token shape。`AFDAsyncConnector` 第一版没有这个
metadata，因此应禁用 graph。后续如果要支持 ACL graph，需要用 CAM 返回的 token
metadata 或固定 `max_seq_len` 设计新的 graph key。

### rank 语义迁移

CAM topology 是 Attention-first，现有 P2P/CAMP2P 多处是 FFN-first。实现时应新增
独立 topology，避免用旧 `world_rank` 推导导致通信 rank 反转。

## 推荐落地阶段

### Phase 1: contract / validation / CAM connector

- 增加 connector capability；
- 注册 `afdasyncconnector`；
- 新增 `AFDAsyncConnector`；
- 增加 async_dp、Ascend NPU、两侧 eager、禁用 graph / ubatching / multistream 等
  connector-specific 校验；
- 禁用 DP metadata control-plane；
- 使用真实 CAM op 或测试替身验证四个 connector 方法的 payload 和调用顺序。

### Phase 2: Attention top-k 接入

- 在 NPU Attention 模型 wrapper 中接入真实 top-k；
- 没有真实 top-k 必须 fail fast；
- 跑通 Attention `dispatch_send -> combine_recv`；
- top-k 来源为 gate-on-attention 或模型层真实 top-k 捕获。

### Phase 3: FFN connector-driven loop

- NPU FFN worker 根据 capability 切换 loop；
- NPU FFN runner 新增无 DP metadata 的 connector-driven step；
- 跑通 `dispatch_recv -> compute_ffn_output -> combine_send`。

### Phase 4: 真实 NPU smoke

- 加载真实 CAM op；
- 先 1A1F eager；
- 再扩展多 A / 多 F；
- 记录 token metadata、rank、layer、shutdown 行为。

## 设计结论

`AFDAsyncConnector` 不应被实现成 “`send_dp_metadata_list()` 空实现的
`camp2pconnector` 变体”。它代表一种新的 AFD runtime 模式：

```text
旧模式:
  DP metadata control-plane 驱动 FFN step

AFDAsyncConnector:
  CAM data-plane / operator payload 驱动 FFN step
```

因此，后续实现重点是三件事：

- 用 connector capability 把 DP metadata loop 和 connector-driven loop 分开；
- 用 CAM op 的 `TokenNums_Rankid_Layeridx` 替代 FFN 侧 DP metadata；
- Attention 侧必须提供真实 top-k 路由信息。

这三点完成后，`AFDAsyncConnector` 才能真正符合 async DP 的语义，而不是把同步 DP 的
control-plane 换一个名字继续保留下来。
