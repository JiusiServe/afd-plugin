# Phase 0 兼容性盘点与迁移决策底稿

本文档是 AFD external plugin 迁移的 Phase 0 讨论底稿。目标是先把原始
in-tree AFD 改动逐项归类，判断在 vLLM `v0.19.1` 上哪些可以通过插件机制迁出，
哪些需要 adapter/shim，哪些可能不得不 patch。

本文档不定义最终类 API，也不要求立即创建对应文件。后续设计讨论应优先修改本
文档中的判断，再进入实现。

## 基线

- 目标 vLLM：同级目录 `../vllm`
- 已确认目标 tag：`v0.19.1`
- 原始 AFD 实现来源：vLLM 分支 `afd_gpu`
- 原始 AFD commit：`0ce8b91b937ec5d47b6902867c4275e0c5fb895e`
- 参考 external plugin：`../dllm-plugin`
- 核心约束：不修改 vLLM `v0.19.1` 源码

## vLLM v0.19.1 已确认扩展点

| 扩展点 | 状态 | 对 AFD 的意义 |
| --- | --- | --- |
| `vllm.general_plugins` | 可用 | 可注册 `register_afd()`，用于 model registration、轻量初始化和必要 shim 安装 |
| `VLLM_PLUGINS` | 可用 | 可控制是否加载 `afd` plugin |
| `--worker-cls` | 可用 | Attention/FFN 都可优先通过自定义 worker class 接入 |
| `--worker-extension-cls` | 可用 | 可作为补充扩展点，但不适合替换 model runner 构造流程 |
| `--scheduler-cls` | 可用 | FFN 第一版已确定不使用自定义 scheduler；仅作为后备方案 |
| ModelRegistry | 可用 | 可注册 AFD 版本的 DeepSeek V2 / Step3 model class |
| `vllm serve` | 可用 | FFN 侧不再走原始 `fserver`，优先复用 `vllm serve` 生命周期 |

关键观察：

- `WorkerWrapperBase.init_worker()` 会从 `parallel_config.worker_cls` 解析 dotted
  qualname，因此自定义 worker 是当前最重要的非侵入式入口。
- vLLM 原生 `GPUWorker.init_device()` 内部硬编码构造 `GPUModelRunner`。因此
  Attention 侧已确定使用自定义 `AFDAttentionWorker`，通过 `--worker-cls` 接入，
  并在 worker 内创建继承自 vLLM v1 `GPUModelRunner` 的
  `AFDAttentionModelRunner`。
- 原始 AFD commit 中新增的 `fserver` CLI 不符合当前设计方向。FFN 侧应走
  `vllm serve`，通过 role 配置和自定义 worker/model runner 进入 FFN runtime。
- 当前仓库不保留 `afd_plugin.entrypoints` 目录；除非后续出现原生
  `vllm serve` 与 class-path 入口无法覆盖的明确需求，否则不引入插件自有
  executable entrypoint。
- FFN 侧真实执行来源是 Attention 侧 connector 发送的 hidden states/metadata，
  不是 vLLM request scheduler。因此 FFN 第一版不使用自定义 scheduler 驱动执行。

## 从 v0.20.2 切换到 v0.19.1 的影响

当前仓库尚未落地 `afd_plugin/` 代码，因此切换目标版本主要影响设计基线和后续实现
参照，不涉及已有插件代码重写。

已确认 `v0.19.1` 仍具备 Phase 1 到 Phase 3 依赖的核心外部扩展点：

- `vllm.general_plugins` 和 `VLLM_PLUGINS` 均可用；
- `--worker-cls`、`--additional-config` 和 `--scheduler-cls` 均可用；
- `ForwardContext.additional_kwargs` 可用，仍可优先通过
  `additional_kwargs["afd_metadata"]` 承载 AFD metadata；
- `EngineCore._initialize_kv_caches()` 对空 KV spec 的处理路径与 `v0.20.2` 基本一致，
  FFN 侧“返回空 KV spec + 在 `initialize_from_config()` 启动 loop”的方案保留。

需要重新校准的版本差异：

- 原始 AFD commit 与 `v0.20.2` 更接近；切到 `v0.19.1` 后，`GPUModelRunner`、
  `GPUWorker`、`EngineCore`、`ForwardContext`、DeepSeek/Step3 模型相关 diff 更大。
- `v0.19.1` 的 `WorkerBase.compile_or_warm_up_model()` 返回 `float`，而 `v0.20.2`
  返回 `CompilationTimes`。本仓库当前只绑定 `v0.19.1`，第一版实现应按 `float`
  路径编写；若后续需要双版本支持，再放入 `afd_plugin.compat`。
- `v0.19.1` 的 `DPMetadata` 仍包含 `max_tokens_across_dp_cpu` 和 `chunked_sizes`
  相关逻辑，后续 ubatching/DBO 迁移必须按 `v0.19.1` 的结构重新验证。
- 原始 AFD 中对 DeepSeek V2、Step3 和 `GPUModelRunner` 的改动不能直接按
  `v0.20.2` 语义照搬，Phase 2 之后应以 `v0.19.1` 源码为实现基准做最小覆盖。

## 迁移方式标签

| 标签 | 含义 |
| --- | --- |
| Direct Port | 原代码可基本迁移到插件内，改 import 路径即可 |
| Subclass / Adapter | 继承或包装 vLLM 原类，通过 class path 或显式调用接入 |
| Registry | 通过 vLLM registry 注册，例如 ModelRegistry |
| Plugin Config | 通过插件自己的配置解析承载，不进入 vLLM 原生 config dataclass |
| Compat Shim | 小范围兼容层，适配 vLLM `v0.19.1` 的接口缺口 |
| Compat Patch | 运行时 monkey patch。只在没有扩展点时使用，必须幂等、版本保护、有测试 |
| Discard | 原始 in-tree 入口或改动不再沿用 |

## 原始 AFD 改动迁移矩阵

| 原始文件 | 原始作用 | 初步迁移方式 | 目标目录 | 风险 | 待确认问题 |
| --- | --- | --- | --- | --- | --- |
| `vllm/config/afd.py` | 定义 `AFDConfig` | Direct Port / Plugin Config | `afd_plugin/` 顶层或后续配置目录 | 中 | 不修改 `VllmConfig` 时，如何把 AFD config 传到 engine/worker/model runner |
| `vllm/config/__init__.py` | 暴露 `AFDConfig` | Discard | 无 | 低 | 插件内自行导出，不改 vLLM |
| `vllm/config/vllm.py` | 给 `VllmConfig` 增加 `afd_config`、hash、ubatching 校验例外 | Plugin Config；暂不 patch `VllmConfig` | `afd_plugin/compat` | 中 | 已确定使用 `additional_config["afd"]` 承载 AFD 配置；后续只需确认 ubatching 校验例外是否仍需要 shim |
| `vllm/engine/arg_utils.py` | 给 EngineArgs 添加 `--afd-config` 并传入 VllmConfig | Discard；使用原生 `--additional-config` | docs/runbook | 低 | 已确定不新增 `--afd-config`，不 patch CLI 参数 |
| `vllm/distributed/afd_transfer/*` | AFD connector package | Direct Port | `afd_plugin/connectors` | 中 | import 路径、torch distributed group、P2P 生命周期 |
| `vllm/distributed/parallel_state.py` | 增加 `init_afd_process_group` 等 AFD process group 能力 | Direct Helper + Patch 候选 | `afd_plugin/distributed` / `afd_plugin/compat/patches` | 高 | AFD group 是否只被插件调用；是否必须注册进 vLLM 全局 parallel state |
| `vllm/distributed/kv_transfer/.../mooncake_connector.py` | 删除一行导入/行为 | 暂不迁移 | 待定 | 中 | 是否与 AFD 无关或只是原分支本地修复 |
| `vllm/forward_context.py` | 增加 `AFDMetadata` 和 `ForwardContext.afd_metadata` | Compat Shim/Patch 候选 | `afd_plugin/runtime` / `afd_plugin/compat/patches` | 高 | 模型 forward 是否必须通过 vLLM 原生 `get_forward_context()` 读到 `afd_metadata` |
| `vllm/v1/engine/core.py` | FFN role 下跳过正常 engine init/busy loop，启动 FFN worker loop | Compat Patch | `afd_plugin/compat/patches` | 高 | 当前已用隔离 patch 支持 FFN daemon 模式，避免 HybridKVCacheCoordinator 初始化失败，并通过 RPC 启停 FFN loop |
| `vllm/v1/engine/utils.py` | engine 工具层改动 | 待盘点 | `afd_plugin/compat` | 中 | 需要读原始 diff 细节后决定 |
| `vllm/v1/executor/multiproc_executor.py` | 记录 AFD role/config | 可能 Discard / Shim | `afd_plugin/runtime` | 中 | 是否只是日志/状态缓存；自定义 worker 是否足够 |
| `vllm/v1/worker/gpu_worker.py` | FFN role 创建 `GPUFFNModelRunner`，增加 FFN loop RPC；Attention 默认仍使用原生 worker 路径 | Subclass / Adapter | `afd_plugin/runtime` | 高 | Attention 侧已确定使用 `AFDAttentionWorker`；FFN 侧已确定使用 `AFDFFNWorker`，worker 内启动常驻 loop，EngineCore RPC 触发仅作后备 |
| `vllm/v1/worker/gpu_model_runner.py` | Attention 侧 AFD runner 改动 | Subclass / Adapter | `afd_plugin/runtime` | 中 | 已确定 `AFDAttentionModelRunner` 继承 vLLM v1 `GPUModelRunner`；后续只需确认最小覆盖方法集合 |
| `vllm/v1/worker/gpu_ffn_model_runner.py` | FFN model runner 新实现 | Direct Port first | `afd_plugin/runtime` | 高 | 第一版可先使用原始 `GPUFFNModelRunner`，再逐步清理 import 和耦合 |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | microbatch wrapper AFD 改动 | Subclass/Shim 候选 | `afd_plugin/runtime` / `afd_plugin/compat` | 中 | 是否可由自定义 runner 内部处理，避免 patch 原 wrapper |
| `vllm/v1/worker/ubatching.py` | ubatching 元数据/流程补充 | Direct Helper / Shim 候选 | `afd_plugin/runtime` | 中 | 是否只被 AFD runner 调用 |
| `vllm/model_executor/models/deepseek_v2.py` | DeepSeek V2 AFD model 改动 | Registry + Model Wrapper | `afd_plugin/model_executor/models` | 高 | 继承原模型还是复制 AFD 版本；模型架构名如何注册和选择 |
| `vllm/model_executor/models/step3_text.py` | Step3 text AFD model 改动 | Registry + Model Wrapper | `afd_plugin/model_executor/models` | 高 | 同上 |
| `vllm/model_executor/models/step3_vl.py` | Step3 VL AFD model 改动 | Registry + Model Wrapper | `afd_plugin/model_executor/models` | 中 | 是否只需轻量 wrapper |
| `vllm/entrypoints/afd_ffn_server.py` | 新增 FFN server main | Discard | 无 | 低 | 当前设计明确不沿用 |
| `vllm/entrypoints/cli/fserver.py` | 新增 `vllm fserver` 子命令 | Discard | 无 | 低 | 当前设计明确不沿用 |
| `vllm/entrypoints/cli/main.py` | 注册 `fserver` 子命令 | Discard | 无 | 低 | 当前设计明确不修改 vLLM CLI |
| `vllm/entrypoints/cli/serve.py` | serve CLI 小改动 | 待盘点 / 尽量 Discard | `afd_plugin/compat` | 中 | 确认是否与 `vllm serve` FFN role 必需 |
| `examples/online_serving/afd_deepseek_v2/README.md` | 示例 runbook | Direct Port | `examples` / `docs` | 低 | 需改成 `vllm serve` 双 role 版本 |

## 当前建议的 Phase 0 结论

### 1. FFN 侧优先走自定义 worker，而不是自定义 CLI

原始实现通过 `fserver` + engine core 特殊 busy loop 启动 FFN server。当前设计改为
FFN 也走 `vllm serve`，因此 Phase 0 的重点变成：

- 如何用 `--worker-cls afd_plugin...AFDFFNWorker` 切换 FFN worker；
- 如何把 role/config 传入该 worker；
- 如何让 FFN worker 在 vLLM serve 生命周期中进入常驻接收循环；
- 是否仍需要 engine core patch 来触发 `start_ffn_server_loop`。

FFN 侧当前设计方针已确定为：

- FFN 侧不接收普通 vLLM/OpenAI request；
- FFN 侧真实执行由 Attention 侧通过 AFD connector 发送的 hidden
  states/metadata 驱动；
- FFN 第一版不传 `--scheduler-cls`，默认 scheduler 只作为 EngineCore 的空转组件；
- 普通 `vllm serve` 已验证可启动 FFN serve；
- `AFDFFNWorker` 继承 vLLM v1 `GPUWorker`；
- `AFDFFNWorker` 创建 `GPUFFNModelRunner`；
- `AFDFFNWorker.get_kv_cache_spec()` 返回空 spec，使 EngineCore 认为 FFN 不需要
  KV cache；
- `AFDFFNWorker.initialize_from_config()` 不分配 KV cache，而是初始化 connector
  并启动 FFN 常驻 loop；
- `AFDFFNWorker.execute_model()` 对意外 scheduler 调用 fail fast；
- EngineCore patch 作为当前 FFN daemon 模式的兼容实现，集中在
  `afd_plugin.compat.patches.engine_core`。

FFN 侧命令形态：

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

### 2. Model runner 切换应由自定义 worker 承担

vLLM `GPUWorker.init_device()` 硬编码创建原生 `GPUModelRunner`。因此：

- Attention 侧已确定通过 `--worker-cls` 显式传入 `AFDAttentionWorker`；
- `AFDAttentionWorker` 继承 vLLM v1 `GPUWorker`，复用原生 worker 生命周期；
- `AFDAttentionWorker` 负责创建 `AFDAttentionModelRunner`；
- `AFDAttentionModelRunner` 继承 vLLM v1 `GPUModelRunner`，只覆盖 AFD 必需逻辑；
- `AFDFFNWorker` 负责创建 `GPUFFNModelRunner`；
- 不假设 vLLM `v0.19.1` 有直接传入 model runner class 的公开参数。

Attention 侧命令形态：

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

### 3. AFDConfig 通过 `--additional-config` 承载（已确定）

原始实现把 `afd_config` 加入 `VllmConfig` 和 `EngineArgs`。作为 external plugin，
当前已确定不沿用这条路径：

- 不新增 `--afd-config`；
- 不 patch `EngineArgs`；
- 不 patch `VllmConfig` 来增加 `afd_config` 字段；
- 默认通过 vLLM 原生 `--additional-config` 的 `afd` namespace 承载 AFD 配置；
- 插件在 worker/model runner 等入口处从 `vllm_config.additional_config["afd"]`
  解析出 plugin-owned `AFDConfig`。

建议命令形态：

```bash
vllm serve <model> \
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

`additional_config` 会参与 vLLM `compute_hash()`。因此，后续需要单独决定
`afd_host`、`afd_port`、`afd_server_rank` 这类 runtime-only 字段是否进入
`additional_config["afd"]`，还是通过环境变量承载，以避免不必要地影响图/hash。

只有当 ubatching 校验或其它 vLLM 内部路径强依赖 `vllm_config.afd_config` 时，
才重新讨论受版本保护的 compat shim/patch。

### 4. Patch 隔离原则

Phase 0 暂定所有 patch 候选只能进入：

```text
afd_plugin/compat/patches/
```

不得把 monkey patch 混入 `connectors/`、`distributed/`、`models/` 或 `runtime/`。

## 高风险待确认问题

1. FFN role 暴露 OpenAI API server 时，部署层是否需要额外隔离来避免普通请求误发。
2. EngineCore patch 是否还需要覆盖更多目标 vLLM patch-level 版本。
3. AFD 的 `ForwardContext.afd_metadata` 是否必须存在于 vLLM 原生
   `ForwardContext` 对象上，还是可以由插件 model/runner 显式传递。
4. AFD process group 是否必须写入 vLLM `parallel_state` 的全局变量，还是可以由
   connector 持有独立 process group。
5. `GPUFFNModelRunner` 中依赖的 vLLM import 是否都存在于 `v0.19.1`，哪些需要
   迁移到 `compat`。
6. DeepSeek V2 / Step3 的 AFD model 改动是继承覆盖更稳，还是复制原始 AFD 版本更稳。
7. `AFDAttentionModelRunner` 的最小覆盖面是什么，是否需要改 scheduler output
   或 ubatching metadata。

## Phase 0 完成标准

Phase 0 完成时，应能明确回答：

- 第一版 package skeleton 需要哪些目录；
- 第一版需要哪些最小类；
- 哪些原始改动明确丢弃；
- 哪些原始改动明确通过 plugin class/registry 实现；
- 哪些点暂时列为 shim 候选；
- 哪些点确认为 patch 候选，并说明为什么没有可用扩展点；
- Phase 1 是否可以不写任何 monkey patch。

## 已确定决策

- AFD 配置默认通过 `--additional-config` 的 `afd` namespace 传入，例如
  `vllm_config.additional_config["afd"]`。
- 插件内部将 `additional_config["afd"]` 解析成 plugin-owned `AFDConfig`。
- 不新增 `--afd-config`。
- 不 patch `EngineArgs`。
- 不 patch `VllmConfig` 来增加 `afd_config` 字段。
- Attention 侧通过 `--worker-cls` 显式传入 `AFDAttentionWorker`。
- `AFDAttentionWorker` 继承 vLLM v1 `GPUWorker`。
- `AFDAttentionModelRunner` 继承 vLLM v1 `GPUModelRunner`。
- Attention 侧 AFD 行为优先放在 `AFDAttentionModelRunner` 中，worker 主要负责
  runner 注入和生命周期复用。
- FFN 侧通过 `--worker-cls` 显式传入 `AFDFFNWorker`。
- FFN 第一版不传 `--scheduler-cls`，默认 scheduler 只空转，不驱动执行。
- 普通 `vllm serve` 作为 FFN 启动路径。
- `AFDFFNWorker` 继承 vLLM v1 `GPUWorker`，创建 `GPUFFNModelRunner`。
- `AFDFFNWorker` 通过空 KV cache spec 避免 FFN 侧 KV cache 管理。
- `AFDFFNWorker.initialize_from_config()` 负责初始化 connector 并启动 FFN 常驻
  loop。
- `AFDFFNWorker.execute_model()` 对意外 scheduler 调用 fail fast。
- FFN 侧 EngineCore patch 是当前 FFN daemon 模式方案，必须保持幂等、版本保护和
  测试覆盖。

## 从 v0.19.1 迁移到 v0.23.0 的影响

插件运行时基线已从 `v0.19.1` 升级到 `v0.23.0`。上一节是原始 `v0.19.1` 迁移基线的
历史记录，保留不动；本节记录从 `v0.19.1` 升级到 `v0.23.0` 时实际遇到的接口变化与
对应处理。

仍然可用、与 `v0.19.1` 一致的扩展点：

- `vllm.general_plugins`、`VLLM_PLUGINS`、`--worker-cls`、`--additional-config`、
  `--scheduler-cls` 均可用；
- `ForwardContext.additional_kwargs` 仍可承载 `afd_metadata`；
- FFN 侧“返回空 KV spec + 在 `initialize_from_config()` 启动 loop”方案保留。

需要适配的 `v0.23.0` 接口变化：

- **`DPMetadata` 去掉了 `max_tokens_across_dp_cpu`**。`v0.19.1` 的 `DPMetadata` 含
  `max_tokens_across_dp_cpu` 和 `chunked_sizes`；`v0.23.0` 的原生 `DPMetadata` 只
  保留 `num_tokens_across_dp_cpu`（以及 `sp_local_sizes`，不再有 `chunked_sizes`
  方法）。AFD 在 `attention_model_runner._build_capture_dp_metadata` 和
  `ubatch_wrapper.build_ubatch_dp_metadata_list` 中，对 DP>1 走原生
  `DPMetadata.make()` 的路径重新挂回 `max_tokens_across_dp_cpu`，以维持下游
  （p2p 序列化、NPU forward context）对该字段的依赖。
- **`set_forward_context` / `DPMetadata.make` 对 `parallel_config` 字段的要求变多**。
  `v0.23.0` 会读取 `parallel_config.is_moe_model`（断言 `is not False`）和
  `tensor_parallel_size`，并在 `create_forward_context` 中读取
  `compilation_config.fast_moe_cold_start` / `static_forward_context`；相关单测的
  mock config 已补齐这些字段。
- **`compile_or_warm_up_model()` 返回类型变化**。`v0.19.1` 返回 `float`，`v0.23.0`
  返回 `CompilationTimes`（与 `v0.20.2` 一致）。
- 版本保护 `afd_plugin.compat.vllm.TARGET_VLLM_VERSION` 与 `pyproject.toml` 的
  vllm extra 均已指向 `0.23.0`。

