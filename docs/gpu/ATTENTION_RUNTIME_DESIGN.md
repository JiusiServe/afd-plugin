# Attention 侧 Runtime 详细设计

本文档描述 AFD plugin 中 Attention 侧的 runtime 设计。本文档基于
`docs/PHASE0_COMPATIBILITY_INVENTORY.md` 中已经确定的 Phase 0 决策。

## 设计目标

- Attention 侧继续使用原生 `vllm serve` 启动。
- 通过 `--worker-cls` 显式接入 AFD Attention worker。
- 不修改 vLLM `v0.23.0` 源码。
- 尽量复用 vLLM v1 原生 `GPUWorker` 和 `GPUModelRunner` 的生命周期、调度、
  KV cache 管理、scheduler output 处理和模型执行流程。
- AFD 相关行为尽量集中在 `AFDAttentionModelRunner`，worker 只负责注入
  model runner 和复用 worker 生命周期。

## 启动方式

Attention 侧通过 `vllm serve` 启动，并显式指定 `--worker-cls`：

```bash
vllm serve <model> \
  --worker-cls afd_plugin.v1.worker.AFDAttentionWorker \
  --additional-config '{
    "afd": {
      "enabled": true,
      "role": "attention",
      "connector": "p2pconnector",
      "host": "127.0.0.1",
      "port": 1239,
      "num_afd_stages": 3,
      "num_attention_servers": 2,
      "num_ffn_servers": 1
    }
  }'
```

配置从 `vllm_config.additional_config["afd"]` 读取，并解析成 plugin-owned
`AFDConfig`。不新增 `--afd-config`，不 patch `EngineArgs`，不 patch
`VllmConfig`。

## 核心类

### `AFDAttentionWorker`

`AFDAttentionWorker` 继承 vLLM v1 原生 `GPUWorker`。

职责：

- 作为 `--worker-cls` 的 dotted class path 入口。
- 复用 vLLM 原生 worker 的分布式初始化、设备初始化、模型加载、KV cache
  管理、sleep/wake、LoRA、profile 等生命周期。
- 在 model runner 构造位置创建 `AFDAttentionModelRunner`。
- 不承载 AFD Attention 业务逻辑；业务逻辑优先放到
  `AFDAttentionModelRunner`。

实现注意点：

- vLLM `GPUWorker.init_device()` 内部硬编码构造原生 `GPUModelRunner`，没有公开的
  model runner factory 参数。
- 因此 `AFDAttentionWorker` 不能只是空继承。它需要覆盖 model runner 构造路径。
- 第一版可以接受在 `AFDAttentionWorker.init_device()` 中复用原生逻辑并替换最后
  的 runner 构造段。后续如果 vLLM 提供 runner factory 扩展点，再收敛为更小的
  override。

### `AFDAttentionModelRunner`

`AFDAttentionModelRunner` 继承 vLLM v1 原生 `GPUModelRunner`。

职责：

- 解析 `additional_config["afd"]` 得到 Attention role 的 `AFDConfig`。
- 初始化 Attention 侧 AFD connector。
- 在模型 forward 前构造 AFD metadata。
- 在模型 forward 前向 FFN 侧发送 DP / ubatch metadata。
- 在模型 forward 中通过 plugin-owned model wrapper 触发 Attention 输出发送、
  FFN 输出接收。
- 支持 normal run 和 CUDA graph warmup/capture 路径中的 AFD metadata 发送。

## 执行路径

Attention 侧正常请求仍然来自 vLLM scheduler：

```text
OpenAI / vLLM request
  -> vLLM scheduler
  -> AFDAttentionWorker.execute_model(...)
  -> AFDAttentionModelRunner.execute_model(...)
  -> prepare scheduler output / input batch / ubatch slices
  -> build AFD metadata
  -> send DP metadata to FFN side through connector
  -> model forward
  -> plugin-owned model sends attention output to FFN side
  -> plugin-owned model receives FFN output
  -> normal vLLM postprocess / sampling / ModelRunnerOutput
```

Attention 侧仍然需要 KV cache，因为它负责 attention computation 和正常 request
调度。因此不要在 Attention 侧绕开 vLLM 原生 KV cache 管理。

## AFD Metadata 承载方式

原始 in-tree AFD 改动给 vLLM `ForwardContext` 增加了 `afd_metadata` 字段。
external plugin 第一版应避免 patch `vllm.forward_context`，优先使用已有
`ForwardContext.additional_kwargs`：

```python
set_forward_context(
    ...,
    additional_kwargs={
        "afd_metadata": afd_metadata,
    },
)
```

plugin-owned model wrapper 通过以下方式读取：

```python
forward_ctx = get_forward_context()
afd_metadata = forward_ctx.additional_kwargs.get("afd_metadata")
```

如果后续验证发现模型路径必须访问 `forward_ctx.afd_metadata` 属性，才重新讨论
受版本保护的 compat shim/patch。

## Connector 初始化

Attention connector 由 `AFDAttentionModelRunner` 持有。

建议生命周期：

```text
AFDAttentionModelRunner.__init__
  -> parse AFDConfig
  -> create connector
  -> init connector
```

后续可根据 P2P connector 的实际生命周期调整为 lazy init，但第一版保持简单。

## 与 Model 的关系

Attention 侧模型逻辑不 patch vLLM 原生 model module。通过 vLLM ModelRegistry
注册 plugin-owned model wrapper 或 implementation。

模型 wrapper 负责：

- 从 forward context 中读取 AFD metadata；
- 在 AFD enabled 时，把 attention output 发送到 FFN 侧；
- 接收 FFN output 并继续后续层或返回 hidden states；
- 在 AFD disabled 或 metadata 缺失时保持普通 vLLM 行为。

## 错误处理与校验

`AFDAttentionWorker` / `AFDAttentionModelRunner` 初始化时应校验：

- `additional_config["afd"]["enabled"] == true`
- `role == "attention"`
- `--worker-cls` 确实是 Attention worker
- connector 名称合法
- topology 字段存在且可解析
- vLLM 版本为已验证的 `v0.23.0`

如果 role 与 worker 不匹配，应 fail fast。

## 第一版最小实现范围

第一版 Attention 侧只实现：

- `AFDAttentionWorker` 继承 vLLM v1 `GPUWorker`
- `AFDAttentionModelRunner` 继承 vLLM v1 `GPUModelRunner`
- `additional_config["afd"]` 解析
- P2P connector 路径
- AFD metadata 构造
- forward context `additional_kwargs["afd_metadata"]`
- normal `execute_model()` 路径中的 metadata 发送

CUDA graph warmup/capture 路径、复杂 ubatching 和模型覆盖细节可以在 P2P 基线稳定后
逐步补齐。

## 待验证问题

- `AFDAttentionWorker.init_device()` 覆盖范围如何最小化。
- `AFDAttentionModelRunner.execute_model()` 是否必须复制较大段 vLLM 原逻辑，还是
  可以通过局部 helper 降低重复。
- `additional_kwargs["afd_metadata"]` 是否足以支撑 DeepSeek V2 和 Step3 model
  wrapper。
- CUDA graph capture 路径中 AFD metadata 发送是否需要单独 hook。
- ubatching 下 stage metadata 和 request/token slice 是否与原始 AFD 行为一致。
