# FFN 侧 Runtime 详细设计

本文档描述 AFD plugin 中 FFN 侧的 runtime 设计。FFN 侧比 Attention 侧更特殊：
它不接收普通 vLLM request，真实执行来源是 Attention 侧通过 AFD connector 发来的
hidden states 和 metadata。

本文档基于 `docs/PHASE0_COMPATIBILITY_INVENTORY.md` 中已经确定的 Phase 0 决策。

## 设计目标

- FFN 侧仍使用原生 `vllm serve` 启动。
- 不沿用原始 in-tree AFD commit 中新增的 `fserver` CLI。
- 不新增 `vllm fserver` 子命令。
- 不保留 plugin-owned executable entrypoint 目录；当前仓库不需要
  `afd_plugin.entrypoints`。
- 对 vLLM `EngineCore` 使用隔离的 compat patch，专门支持 FFN daemon 模式。
- 第一版不使用自定义 `scheduler-cls` 驱动执行。
- FFN 侧不接收普通 OpenAI/vLLM request。
- FFN 侧不做 KV cache 管理。
- FFN worker 在 vLLM worker 生命周期内启动自己的常驻 loop，从 AFD connector
  接收任务并执行 FFN。

## 启动方式

FFN 侧通过普通 `vllm serve` 启动。当前 EngineCore compat patch 已经让 FFN role
跳过正常 request/KV-cache 初始化路径：

```bash
vllm serve <model> \
  --worker-cls afd_plugin.v1.worker.AFDFFNWorker \
  --additional-config '{
    "afd": {
      "enabled": true,
      "role": "ffn",
      "connector": "p2pconnector",
      "host": "127.0.0.1",
      "port": 1239,
      "num_afd_stages": 3,
      "num_attention_servers": 2,
      "num_ffn_servers": 1
    }
  }'
```

第一版不传 `--scheduler-cls`。默认 scheduler 只作为 vLLM `EngineCore` 初始化后的
空转组件存在，不驱动 FFN 执行。

## 核心类

### `AFDFFNWorker`

`AFDFFNWorker` 继承 vLLM v1 原生 `GPUWorker`。

职责：

- 作为 `--worker-cls` 的 dotted class path 入口。
- 复用 vLLM 原生 worker 的进程、设备、分布式和模型加载生命周期。
- 创建 `GPUFFNModelRunner`。
- 向 vLLM `EngineCore` 表达“本 worker 不需要 KV cache”。
- 在 worker 初始化完成后启动 FFN 常驻 loop。
- 对意外的 scheduler-driven `execute_model()` 调用 fail fast。
- 在 shutdown 时停止 FFN loop 并关闭 connector。

建议覆盖点：

- `init_device()`：创建 `GPUFFNModelRunner`，而不是原生 `GPUModelRunner`。
- `get_kv_cache_spec()`：返回空 spec。
- `initialize_from_config(...)`：不分配 KV cache，初始化 AFD connector 并启动 FFN
  loop。
- `determine_available_memory()`：作为安全 no-op 或返回 0，理论上空 KV spec 下不应
  被调用。
- `execute_model(...)`：如果被调用，抛出明确错误。
- `shutdown()`：停止 loop，释放 connector 和线程资源。

### `GPUFFNModelRunner`

第一版可以直接迁移原始 AFD commit 中的 `GPUFFNModelRunner`，再逐步清理 import
和耦合。

职责：

- 加载完整模型权重或 FFN 所需模型部分。
- 初始化 AFD connector。
- 从 connector 接收 Attention 侧发送的 hidden states 和 metadata。
- 执行对应 layer/stage 的 FFN computation。
- 将 FFN output 通过 connector 发送回 Attention 侧。
- 支持 `FULL_DECODE_ONLY` warmup / CUDA graph capture / replay 路径；FFN graph
  cache 按 DP metadata shape 建 key。

## 执行路径

FFN 侧不是 request-driven，而是 connector-driven：

```text
vllm serve
  -> EngineCore 初始化
  -> 创建 AFDFFNWorker
  -> AFDFFNWorker 创建 GPUFFNModelRunner
  -> EngineCore 查询 KV cache spec
  -> AFDFFNWorker 返回空 KV cache spec
  -> EngineCore 调用 initialize_from_config(...)
  -> AFDFFNWorker 初始化 connector 并启动 FFN loop
  -> FFN loop 阻塞等待 Attention 侧 metadata / hidden states
  -> GPUFFNModelRunner 执行 FFN
  -> connector 返回 FFN output
```

普通 vLLM request 路径不应进入 FFN worker：

```text
OpenAI request
  -> scheduler
  -> AFDFFNWorker.execute_model(...)
  -> RuntimeError
```

这个 fail fast 行为用于尽早暴露错误部署或误发请求。

## Scheduler 方针

第一版 FFN 侧不使用自定义 scheduler：

- 不传 `--scheduler-cls`。
- 默认 scheduler 仅用于满足 vLLM `EngineCore` 的对象生命周期。
- FFN 执行不由 scheduler output 驱动。
- FFN 执行只由 AFD connector 输入驱动。

暂不设计 `AFDFFNScheduler`，原因：

- scheduler 在 vLLM 初始化流程中创建于 KV cache 初始化之后，不能解决“不需要
  KV cache”的核心问题。
- scheduler interface 不是稳定 public API，早引入会增加维护成本。
- FFN 侧没有普通 request，因此自定义 scheduler 的收益有限。

只有当默认 scheduler 在 no-request/empty-KV 场景下仍产生不可接受的
副作用时，才重新讨论 `AFDFFNScheduler`。

## KV Cache 方针

FFN 侧不需要 KV cache。第一版通过 worker 空 KV spec 避免 KV cache 管理：

```python
def get_kv_cache_spec(self) -> dict:
    return {}
```

预期 vLLM `EngineCore._initialize_kv_caches()` 看到所有 worker 的 KV spec 为空后：

- 不调用 memory profiling 路径；
- 构造空 KV cache config；
- 仍调用 worker `initialize_from_config(...)`。

`AFDFFNWorker.initialize_from_config(...)` 将这个 hook 用于启动 FFN runtime，而不是
分配 KV cache。

这条路径是第一版要重点验证的 P1 blocker。如果 vLLM 对空 KV cache config 仍有不
兼容假设，再考虑 compat shim 或 EngineCore patch。

## EngineCore 方针

当前实现 patch `EngineCore`，patch 位于 `afd_plugin.compat.patches.engine_core`。
这是 FFN daemon 模式的受控兼容层，不修改 vLLM 源码树。

原始 AFD in-tree 实现对 `EngineCore` 做了侵入式修改，用于 FFN role 下跳过普通
request loop 并显式触发 `start_ffn_server_loop`。external plugin 的当前方案是：

- 非 FFN config 完全走原生 `EngineCore` 路径；
- FFN config 下跳过正常 scheduler/KV cache 初始化，避免
  `HybridKVCacheCoordinator` 对 attention groups 的假设；
- 对插件延迟加载场景，patch `_initialize_kv_caches()`，返回空 KV cache config；
- `run_busy_loop()` 通过 executor RPC 启停 `start_ffn_server_loop` /
  `stop_ffn_server_loop`；
- shutdown 时停止 FFN loop 并释放 executor。

这个 patch 是幂等的，并通过 CPU-safe unit tests 覆盖。

## Connector Loop

FFN loop 建议由 `AFDFFNWorker` 管理线程生命周期，由 `GPUFFNModelRunner` 负责实际
计算：

```text
AFDFFNWorker.initialize_from_config(...)
  -> model_runner.initialize_afd_connector()
  -> start background loop

loop:
  while not shutdown:
    metadata = connector.recv_dp_metadata_list()
    if metadata indicates graph capture/warmup:
      model_runner.capture_model(...)
    else:
      model_runner.execute_ffn_step(...)
```

命名上不一定沿用原始 `execute_model(scheduler_output=None, dp_metadata_list=...)`。
后续类设计时可以考虑给 FFN runner 一个更语义化的方法，例如：

```python
execute_ffn_step(dp_metadata_list=...)
```

第一版如果直接迁移原始 `GPUFFNModelRunner`，可以先保持原方法签名，跑通后再重构。

## 错误处理与校验

`AFDFFNWorker` 初始化时应校验：

- `additional_config["afd"]["enabled"] == true`
- `role == "ffn"`
- `--worker-cls` 确实是 FFN worker
- connector 名称合法
- topology 字段存在且可解析
- vLLM 版本为已验证的 `v0.23.0`

`AFDFFNWorker.execute_model(...)` 被调用时应 fail fast，错误信息应说明：

- FFN worker 不接受 vLLM scheduled requests；
- FFN 执行由 AFD connector 驱动；
- 请确认普通请求没有被误发到 FFN 端口。

## 第一版最小实现范围

第一版 FFN 侧只实现：

- `AFDFFNWorker` 继承 vLLM v1 `GPUWorker`
- `GPUFFNModelRunner` 从原始 AFD commit 迁移
- `additional_config["afd"]` 解析
- P2P connector 路径
- 空 KV cache spec
- `initialize_from_config()` 启动常驻 loop
- `execute_model()` fail fast
- shutdown 停止 loop

`FULL_DECODE_ONLY` CUDA graph capture/replay 已接入 FFN loop；复杂拓扑、异构
A/F 比例、TP/PP collective 和 profiling 仍需要在 P2P 基线稳定后继续补齐。

## 待验证问题

- EngineCore patch 是否还需要覆盖更多目标 vLLM patch-level 版本。
- worker `initialize_from_config()` 是否仍应作为 FFN connector 初始化 fallback。
- FFN 端口暴露时是否需要部署层隔离，避免普通请求误发到 FFN endpoint。
- 默认 scheduler 空转是否有额外资源或日志副作用。
- `GPUFFNModelRunner` 从原始 AFD commit 迁移后，哪些 import 需要 compat helper。
- FFN loop 的异常如何传递回 vLLM worker/engine 进程，避免 silent failure。
- 多进程/多卡下 connector 初始化顺序是否稳定。
