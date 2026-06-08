# NPU AFDAsyncConnector CAM Stub Ops 设计

本文档说明 `AFDAsyncConnector` 第一版中 CAM 四个算子的 stub 方案。主设计见
`docs/npu/NPU_AFD_ASYNC_CONNECTOR_DESIGN.md`。

stub 的目标不是模拟真实 CAM 通信性能，也不是验证模型数值正确性。stub 只做两件事：

```text
1. 参数检验
2. 构造返回值
```

除此之外，stub 不建立通信语义，不保存输入，不维护跨 op 状态，也不模拟任何
send/recv。

## 目标

- 提供 `torch.ops.cam.*` 四个接口的打桩实现：
  - `cam_dispatch_send`
  - `cam_dispatch_recv`
  - `cam_combine_send`
  - `cam_combine_recv`
- 参数顺序与仓库根目录 `cam_ops.py` 保持一致；
- 返回结构与 `cam_ops.py` 描述保持一致；
- 支持 Ascend NPU runtime 中 opt-in 使用 stub，便于在没有真实 CAM op 时完成接口级
  参数检验和返回值构造；
- stub 必须显式启用，不能在真实 CAM op 缺失时静默替代。

## 非目标

- 不模拟真实跨进程、跨 rank、跨设备通信；
- 不验证 MoE 路由正确性；
- 不验证真实 expert token 排布；
- 不支持 ACL graph capture；
- 不作为性能测试依据；
- 不在 `use_stub_cam_ops=false` 时兜底注册。

## 开关策略

建议新增集中 helper：

```text
afd_plugin/compat/ascend/cam_stub_ops.py
  ensure_cam_ops_available(...)
  register_stub_cam_ops(...)
  is_cam_stub_ops_enabled(...)
```

运行时逻辑：

```text
if torch.ops.cam 四个真实 op 都存在:
  使用真实 op
elif afd_config.extra_config["use_stub_cam_ops"] == true:
  注册 stub op
else:
  fail fast: CAM ops are unavailable
```

配置示例：

```json
{
  "afd": {
    "enabled": true,
	    "role": "attention",
	    "connector": "afdasyncconnector",
	    "extra_config": {
	      "use_stub_cam_ops": true,
	      "stub_cam_max_tokens": 262144
	    }
	  }
	}
	```

	`use_stub_cam_ops` 只控制 CAM 四算子是否使用 stub。Attention 侧仍必须传入真实
	`topk_ids` / `topk_weights`，例如通过 `compute_gate_on_attention=true` 在
	Attention 侧计算 gate。

真实 NPU smoke、性能或精度验证必须关闭 `use_stub_cam_ops`。

## 注册方式

stub 使用 `torch.library.Library("cam", "DEF")` 注册到 `cam` namespace。原因：

- `AFDAsyncConnector` 的调用点必须是 `torch.ops.cam.*`，与真实 CAM 接口一致；
- vLLM 的 `direct_register_custom_op` 默认面向 `torch.ops.vllm.*` 注册路径，
  不适合作为 `cam` namespace 的主注册方式；
- stub 只用于参数检验和返回值构造，不承载 vLLM custom op 生命周期语义。

注册后的 op name 与真实接口一致：

```text
torch.ops.cam.cam_dispatch_send
torch.ops.cam.cam_dispatch_recv
torch.ops.cam.cam_combine_send
torch.ops.cam.cam_combine_recv
```

调用侧始终只访问 `torch.ops.cam.*`，避免 connector 代码未来从 stub 切换真实 CAM
时需要改调用点。

注册必须幂等：

```text
_CAM_STUB_OPS_REGISTERED = False

register_stub_cam_ops():
  if already registered:
    return
  register four ops
  mark registered
```

如果 op 已经由真实 `cam` 包注册，stub 不应覆盖真实实现。

## 行为边界

每个 stub op 的实现都应保持为无状态的两步函数：

```text
validate_args(...)
make_return(...)
```

四个 op 的具体行为：

```text
dispatch_send:
  参数检验
  构造完成标记 tensor

dispatch_recv:
  参数检验
  构造 7 元组返回值

combine_send:
  参数检验
  构造完成标记 tensor

combine_recv:
  参数检验
  构造 [batchSize, hiddenSize] 输出 tensor
```

stub 不提供任何跨 op 数据缓存、跨进程协调、共享内存或其它形式的模拟收发。
如果需要验证真实 send/recv 行为，必须切换到真实 CAM op。

## 接口设计

### `cam_dispatch_send`

签名：

```python
cam_dispatch_send(
    x,
    expertIds,
    commArgs,
    commId,
    maxSeqLen,
    batchSize,
    hiddenSize,
    topk,
    expertRankSize,
    attentionRankSize,
    expertPerRank,
    rank,
    worldSize,
    layerIndex,
    tpSize,
    dynamicQuant,
)
```

参数校验：

- `x.ndim == 2`
- `x.shape == (batchSize, hiddenSize)`
- `x.dtype in (float16, bfloat16)`；
- `expertIds.shape == (batchSize, topk)`
- `expertIds.dtype == int32`
- `rank < attentionRankSize`
- `worldSize == attentionRankSize + expertRankSize`

返回：

```text
done: Tensor
  shape = [1]
  dtype = int32 或 int64
  device = x.device
  value = 1
```

stub 不修改 `x`，也不保存 `x`。

### `cam_dispatch_recv`

签名：

```python
cam_dispatch_recv(
    x,
    commArgs,
    commId,
    batchSize,
    hiddenSize,
    topk,
    expertRankSize,
    attentionRankSize,
    expertPerRank,
    rank,
    worldSize,
    layerIndex,
    tpSize,
    dynamicQuant,
)
```

注意：`cam_ops.py` 注释中该参数名是 `batchSize`，语义更接近最大序列长度 /
接收 buffer 上限。stub 中建议称为 `maxSeqLenOrBatchSize`，但函数签名保持
`batchSize`，避免和真实 op 不一致。

参数校验：

- `x` 是占位 tensor；
- `rank >= attentionRankSize`
- `worldSize == attentionRankSize + expertRankSize`
- `expertPerRank > 0`
- `tpSize > 0`

返回 7 元组：

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

推荐 shape：

```text
effective_tokens = max(1, min(batchSize, stub_cam_max_tokens))
shared_tokens = max(1, effective_tokens // max(expertRankSize, 1))

expandXOut:
  shape = [effective_tokens, hiddenSize]
  dtype = x.dtype
  device = x.device

expandXOut_shared:
  shape = [shared_tokens, hiddenSize]
  dtype = x.dtype
  device = x.device

dynamicScalesOut:
  shape = [effective_tokens]
  dtype = float32
  device = x.device

dynamicScalesOut_shared:
  shape = [shared_tokens]
  dtype = float32
  device = x.device

TokenNums_Rankid_Layeridx:
  shape = [5 + tpSize * (2 + expertPerRank)]
  dtype = int64
  device = x.device

Expert_tokens:
  shape = [expertPerRank]
  dtype = int64
  device = x.device

Expert_tokens_shared:
  shape = [1]
  dtype = int64
  device = x.device
```

推荐填充值：

```text
TokenNums_Rankid_Layeridx[0] = effective_tokens
TokenNums_Rankid_Layeridx[1] = 0
TokenNums_Rankid_Layeridx[2] = layerIndex
TokenNums_Rankid_Layeridx[3] = 0
TokenNums_Rankid_Layeridx[4] = expertPerRank

Expert_tokens[:] = ceil(effective_tokens / expertPerRank) 的均匀近似分布
Expert_tokens_shared[0] = shared_tokens
dynamicScalesOut[:] = 1.0
dynamicScalesOut_shared[:] = 1.0
```

`TokenNums_Rankid_Layeridx[0]` 是 `AFDAsyncConnector.recv_attn_output()` 构造
`AFDConnectorMetadata.seq_lens` 的主要来源。

### `cam_combine_send`

签名：

```python
cam_combine_send(
    expandX,
    expandXShared,
    commArgs,
    expertTokenNums,
    commId,
    batchSize,
    hiddenSize,
    topk,
    expertRankSize,
    attentionRankSize,
    expertPerRank,
    rank,
    worldSize,
    tpSize,
)
```

参数校验：

- `rank >= attentionRankSize`
- `expandX.ndim == 2`
- `expandX.shape[1] == hiddenSize`
- `expertTokenNums.dtype == int64`
- `expertTokenNums.shape[0] >= 5`

返回：

```text
done: Tensor
  shape = [1]
  dtype = int32 或 int64
  device = expandX.device
  value = 1
```

stub 不保存 `expandX`。

### `cam_combine_recv`

签名：

```python
cam_combine_recv(
    expandX,
    expertIds,
    expertScales,
    commArgs,
    commId,
    batchSize,
    hiddenSize,
    topk,
    expertRankSize,
    attentionRankSize,
    expertPerRank,
    rank,
    worldSize,
)
```

参数校验：

- `rank < attentionRankSize`
- `expertIds.shape == (batchSize, topk)`
- `expertIds.dtype == int32`
- `expertScales.shape == (batchSize, topk)`
- `expertScales.dtype == float32`

返回：

```text
output:
  shape = [batchSize, hiddenSize]
  dtype = expandX.dtype
  device = expandX.device
```

如果 `expandX` 已经是 `[batchSize, hiddenSize]`，stub 可以返回 `expandX` 本身或
`expandX.clone()`。如果 `expandX` 是 `[1]` 占位 tensor，则创建 zero tensor。

推荐第一版返回 zero tensor。这样不会暗示 stub 有真实 combine 语义。

## 与 Top-k 的关系

`cam_dispatch_send` 和 `cam_combine_recv` 都需要 top-k 输入。`AFDAsyncConnector`
不再生成 stub top-k；Attention 侧必须传入真实 top-k，例如：

```text
topk_ids:
  int32, [batchSize, topk]
  每个 token 选中的 expert ids

topk_weights:
  float32, [batchSize, topk]
  每个 token 对应的 expert weights
```

CAM stub op 只检查这些 tensor 是否符合接口，不根据它们做真实路由。

## 与 AFDAsyncConnector 的交互

Attention 侧：

```text
AFDAsyncConnector.send_attn_output(...)
  -> torch.ops.cam.cam_dispatch_send(...)

AFDAsyncConnector.recv_ffn_output(...)
  -> torch.ops.cam.cam_combine_recv(...)
```

FFN 侧：

```text
AFDAsyncConnector.recv_attn_output(...)
  -> torch.ops.cam.cam_dispatch_recv(...)
  -> 从 TokenNums_Rankid_Layeridx 构造 AFDRecvOutput / AFDConnectorMetadata

AFDAsyncConnector.send_ffn_output(...)
  -> torch.ops.cam.cam_combine_send(...)
```

`AFDAsyncConnector` 不应该知道 stub 与真实 op 的差异。它只调用 `torch.ops.cam.*`。
是否注册 stub 由 compat helper 根据配置处理。

## 错误处理

stub 中的错误应该尽量早暴露：

- shape 不匹配：`ValueError`
- dtype 不匹配：`TypeError`
- rank 与 role 不匹配：`ValueError`
- `worldSize != attentionRankSize + expertRankSize`：`ValueError`
- `topk <= 0`、`hiddenSize <= 0`、`expertPerRank <= 0`：`ValueError`

不要使用宽泛 `try/except Exception` 吞掉真实错误。stub 是开发辅助层，不是容错层。

## 后续真实 CAM 接入

真实 CAM op 接入后：

- `use_stub_cam_ops=false`；
- `ensure_cam_ops_available()` 检查真实 `torch.ops.cam.*` 是否存在；
- 不注册 stub；
- `AFDAsyncConnector` 调用点不变；
- 若真实 op 返回结构和 `cam_ops.py` 不一致，应修改 connector 返回值解析，而不是
  调整 stub 去掩盖差异。

stub 的价值是让 connector 结构先稳定下来；真实功能是否正确，最终仍必须由 Ascend
NPU 真实 CAM op smoke 和精度验证确认。
