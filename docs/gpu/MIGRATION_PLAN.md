# GPU AFD 迁移计划

本文档记录 GPU AFD external plugin 迁移的阶段规划、初始目录边界和当前设计决策。

## 迁移阶段

### Phase 0：基线与兼容性盘点

- 确认 vLLM `v0.23.0` 的 plugin hook 和 class-path 扩展点。
- 将 AFD commit 与目标 vLLM 版本做 diff，并按 config、connector、
  distributed state、engine、worker、model、CLI、example 分类。
- 判断哪些 in-tree 改动可以变成普通 plugin class，哪些需要兼容 shim。

### Phase 1：插件骨架、配置通道与校验

- 添加 Python packaging metadata，并提供 `vllm.general_plugins` entry point。
- 如果需要支持 macOS/本地无 CUDA wheel 的开发体验，参考 `dllm-plugin`，
  将 `vllm` 保持为可选 runtime extra。
- 添加幂等的 `register_afd()` 函数。
- 使用 vLLM 原生 `--additional-config` 的 `afd` namespace 作为 AFD 配置通道，
  不新增 `--afd-config`，不 patch `EngineArgs`，不 patch `VllmConfig` 增加
  `afd_config` 字段。
- 添加 plugin-owned `AFDConfig` 解析与基础 validation。
- 建立 runtime class 空壳，确保 `AFDAttentionWorker`、`AFDAttentionModelRunner`、
  `AFDFFNWorker`、`GPUFFNModelRunner` 的 dotted path 可 import/resolve。
- 添加 CPU-safe smoke tests，覆盖 import、注册、配置解析、validation 和
  class path resolve。
- Phase 1 不做真实 AFD 通信，不迁 P2P，不支持 ubatching，不支持 CUDA graph，
  不写 monkey patch。

### Phase 2：Attention Runtime MVP

- 实现 `AFDAttentionWorker`，继承 vLLM v1 原生 `GPUWorker`，通过
  `--worker-cls` 接入。
- 实现 `AFDAttentionModelRunner`，继承 vLLM v1 原生 `GPUModelRunner`。
- `AFDAttentionWorker` 负责注入 `AFDAttentionModelRunner`，其余 worker
  生命周期尽量复用 vLLM 原生逻辑。
- `AFDAttentionModelRunner` 负责解析 AFD config、初始化 dummy connector、
  构造 AFD metadata，并在 normal `execute_model()` 路径发送 DP/AFD metadata。
- 使用 `ForwardContext.additional_kwargs["afd_metadata"]` 承载 AFD metadata，
  第一版不 patch `vllm.forward_context`。
- 接入一个最小 plugin-owned model wrapper，验证 model forward 能读取
  `afd_metadata`。
- Phase 2 不做 FFN daemon loop，不做真实跨进程通信，不支持 ubatching，不支持
  CUDA graph。

### Phase 3：FFN Runtime MVP 与 Dummy Connector 闭环

- 实现 `AFDFFNWorker`，继承 vLLM v1 原生 `GPUWorker`，通过 `--worker-cls`
  接入；FFN serve 推荐使用 `--headless`。
- 第一版不传 `--scheduler-cls`，默认 scheduler 只作为 EngineCore 的空转组件，
  不驱动 FFN 执行。
- `AFDFFNWorker` 创建 `GPUFFNModelRunner`，`get_kv_cache_spec()` 返回空 spec，
  避免 FFN 侧 KV cache 管理。
- `AFDFFNWorker.initialize_from_config()` 初始化 connector 并启动 FFN 常驻 loop。
- `AFDFFNWorker.execute_model()` 对意外 scheduler 调用 fail fast。
- 第一版尽量不 patch `EngineCore`；只有默认 EngineCore 生命周期无法承载 FFN
  loop 时才讨论 `compat/patches`。
- 迁移第一版 `GPUFFNModelRunner`，可先基于原始 AFD 实现中的
  `GPUFFNModelRunner`。
- 完善 dummy connector，使 Attention 侧发送 hidden states/metadata，FFN 侧接收、
  执行最小 FFN step 或 passthrough，再返回给 Attention 侧。

### Phase 4：P2P Connector

- 在 dummy connector 闭环稳定后迁移真实 P2P connector。
- 迁移 connector metadata、rank mapping、process group 初始化和 send/recv
  hidden states 逻辑。
- 明确 Attention rank 到 FFN rank 的映射，以及 A/F 数量不等时的路由策略。
- 添加 GPU-gated 多进程测试。
- Phase 4 不同时引入 ubatching 和 CUDA graph，避免把通信问题和执行切片问题混在
  一起。

### Phase 5：Ubatching / DBO

- 第一版 MVP 明确不支持 AFD + ubatching；当 AFD enabled 且
  `parallel_config.use_ubatching` 为 true 时应 fail fast。
- Phase 5 专门支持 AFD + ubatching/DBO。
- 明确 `afd_stage_idx` 与 ubatch index 的关系、`num_of_stages` 的语义，以及
  padded/unpadded token lens 的使用规则。
- 解决每个 ubatch forward context 都需要独立 AFD metadata 的问题，优先避免
  patch；如必须 patch，只能放入 `afd_plugin.compat.patches`。
- 迁移或重设计原始 AFD 中的 `manual_dbo_yield` / `apply_dbo_yield` 行为。
- 增加 AFD + ubatching correctness tests。

### Phase 6：CUDA Graph

- 在 P2P connector 和 ubatching 语义稳定后支持 CUDA graph。
- normal run、warmup、capture、replay 路径都必须发送正确的 DP/AFD metadata。
- FFN 侧需要区分 warmup、graph capture 和 normal execution。
- 避免把 connector 通信等副作用错误 capture 进 CUDA graph。
- 明确哪些 AFD 路径需要 `enforce_eager`，哪些可以 graph capture。
- 添加 GPU-gated tests 和 profiling/debugging 文档。

### Phase 7：模型覆盖、拓扑与性能硬化

- 将 DeepSeek V2 和 Step3 模型中的 AFD 相关改动迁移为 plugin-owned model
  implementation 或 wrapper。
- 在 `register_afd()` 中通过 vLLM ModelRegistry 注册模型架构。
- 模型特定的 AFD 逻辑不要放进通用 connector 或 worker 模块。
- 文档化支持的 topology 组合、已知限制、必需环境变量和故障模式。
- 添加 FFN server 和 connector traffic 的 profiling/debugging 说明。
- 添加端到端 GPU integration tests 和 runbook。
- 在明确测试其他版本之前，兼容性说明都绑定到 vLLM `v0.23.0`。

## 初始目录结构

下面先只约定目录边界和目录名，暂不决定具体文件。类级别 API、文件拆分方式、
模块命名后续继续讨论；在讨论完成前，不要为了占位而提前创建细粒度文件。

```text
afd-plugin/
  AGENTS.md
  README.md
  pyproject.toml

  afd_plugin/
    compat/
      patches/
    connectors/
    distributed/
    models/
    runtime/

  docs/
  examples/
  tests/
```

### 目录职责

- `afd_plugin`：插件主包。顶层只放全局注册、配置、校验、轻量公共入口等跨目录
  模块；具体文件名后续再定。
- `afd_plugin.compat`：vLLM `v0.23.0` 的版本保护、延迟 import、兼容 helper
  和 shim。所有与目标 vLLM 版本强绑定的兼容逻辑优先集中在这里。
- `afd_plugin.compat.patches`：不得不 monkey patch vLLM 时使用的隔离区。这里的
  patch 必须幂等、受版本保护、有文档说明，并且只在没有可用 plugin/class-path
  扩展点时使用。
- `afd_plugin.connectors`：AFD 通信 contract 和 backend implementation。
- `afd_plugin.distributed`：从原始 `parallel_state.py` 改动中抽取出的
  AFD-specific distributed helper。这里应优先放插件自己调用的 helper，而不是
  patch。
- `afd_plugin.model_executor.models`：plugin-owned model implementation 或 wrapper。
- `afd_plugin.v1.worker`：vLLM 可通过显式 class path 加载的运行时 adapter/class，
  包括 worker、runner、ubatching、forward-context 相关能力。这里不作为 patch
  目录使用。
- `docs`：迁移说明、架构决策、operator runbook 和已知限制。
- `examples`：可运行示例和部署/online serving 样例。
- `tests`：测试目录。单元测试、集成测试、GPU-gated 测试的具体分层后续再定。

## 当前设计决策

- FFN 侧不要沿用原始 AFD commit 中新增的 `fserver` 入口；优先复用原生
  `vllm serve` 启动路径。
- FFNModelRunner 第一版可以直接使用或迁移原始 AFD 实现中的
  `GPUFFNModelRunner`。
- AttentionModelRunner 第一版继承 vLLM v1 原生 `GPUModelRunner`，并只加入
  AFD 必需的最小覆盖逻辑。
