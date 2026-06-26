# NPU AFD async-DP 框架改动点迁移清单

本文档记录当前本地 `../vllm`、`../vllm-ascend` 中与 AFD async-DP 相关的
框架侧改动点，并逐项给出迁移到 `afd-plugin` 的最小化方案。

目标不是立即实现，而是先明确边界：哪些行为必须保留，哪些可以通过
plugin-owned class path、entry point 注册、AFD additional config、或受版本保护的
compat patch 承接，从而尽量不修改 `../vllm` 和 `../vllm-ascend` 源码树。

## 总体结论

可行方向是：用 AFD additional config 表达 async connector 模式，在插件内定义统一
判定 helper，例如 `is_afd_async_dp(vllm_config)`，再用少量 AFD-scoped compat patch
承接 vLLM engine/core-client 的调度差异。不要把方案建立在直接修改 vLLM
`ParallelConfig` dataclass 或新增 vLLM CLI 参数上。

推荐按核心 / 非核心拆分处理：

- 第一批最小可行链路：vLLM 配置判定、engine proc 选择、coordinator wave 关闭、
  client `FIRST_REQ` 跳过、forward context / DP batch coordination 跳过、NPU runner
  DP sync 跳过。
- 非核心相邻能力暂不迁移：native Ascend FusedMoE async stub、force load balance、
  layer sharding 放开、NPU profiler schedule。后续只在 NPU smoke 或性能验证证明
  必须时再单独评估。

## 核心来源 commit

本次迁移的 async-DP 核心功能主要来自以下 commit。

vLLM：

- `9da7e58d69cbcb66fa5da50e8661e6f9c760b26f`
  `feat: add async dp experiment switch`
  - 引入 async-DP 开关语义；
  - MoE DP rank 在 async-DP 下使用普通 `EngineCoreProc`；
  - 保留 DP/EP topology，以便 expert placement / expert weight loading 仍按原 DP rank
    工作。
- `e0396d82b23a4d8d987f5ee823396e800ebf1364`
  `fix: disable dp wave coordination for async dp`
  - coordinator 不再启用 MoE DP wave coordination；
  - DP client 不再发送 `FIRST_REQ` wave notification。
- `f3ce3f8e2067e05361a18b00e7e719e3cd796087`
  `fix: skip dp batch coordination for async dp`
  - `set_forward_context()` 不再构造 DP metadata；
  - `coordinate_batch_across_dp()` 不再做跨 DP rank batch coordination。

vLLM-Ascend：

- `bd144ed3cf27b217405028a1bcb931c8b111868f`
  `feat: add async dp moe stub path`
  - 核心迁移范围只包含 `vllm_ascend/worker/model_runner_v1.py` 中
    `_sync_metadata_across_dp()` 的改动，即 async-DP 下跳过 DP metadata sync；
  - 同 commit 中 `vllm_ascend/ops/fused_moe/fused_moe.py` 的 native MoE zero/stub
    改动暂不迁移.

其它相邻 commit，例如 native MoE stub 修补、force load balance、layer sharding、
profiler schedule，不属于当前 async-DP 核心迁移范围。

## vLLM 改动点

### 1. `ParallelConfig.async_dp` 配置字段

原代码位置：

- `../vllm/vllm/config/parallel.py`
- 当前 async 分支位置：`ParallelConfig.async_dp`

改动内容：

- 给 `ParallelConfig` 增加 `async_dp: bool = False`。
- 作为框架内所有 async-DP 分支的统一开关。

为什么 AFD async 需要：

- vLLM 默认 MoE DP 会认为所有 DP rank 需要同步 forward wave。
- AFD async connector 希望各 Attention DP rank 独立调度，并由 connector 与 FFN 侧
  建立数据流，不再用 vLLM 原生 DP wave 语义。

迁移到 afd-plugin 的最小方案：

- 不直接给 vLLM `v0.19.1` 的 pydantic dataclass 加字段。
- 在插件内新增统一 helper，例如：

```python
def is_afd_async_dp(vllm_config: object) -> bool:
    afd_config = parse_afd_config(vllm_config, validate=False)
    return afd_config.enabled and afd_config.connector == "afdasyncconnector"
```

- 对插件已有代码中直接读取 `parallel_config.async_dp` 的位置，改为 helper，或在
  plugin config patch 完成后动态设置 `parallel_config.async_dp = True` 作为兼容属性。
- 动态补属性只作为兼容桥，不作为用户配置入口。

优先级：第一批。

### 2. `--async-dp` CLI 参数和校验

原代码位置：

- `../vllm/vllm/engine/arg_utils.py`
- `EngineArgs.async_dp`
- `parallel_group.add_argument("--async-dp", ...)`
- `EngineArgs.create_engine_config()` 中 MoE / eager 校验
- 创建 `ParallelConfig(..., async_dp=self.async_dp, ...)`

改动内容：

- 新增 `--async-dp` CLI。
- 限制只能用于 MoE 模型。
- 限制必须 `--enforce-eager`。
- 把值传入 `ParallelConfig`。

为什么 AFD async 需要：

- AFD async connector 第一版要求 MoE + eager。
- async-DP 调度行为必须在 engine/core-client/model runner 路径上可见。

迁移到 afd-plugin 的最小方案：

- 不新增 vLLM CLI 参数；vLLM parser 在 plugin runtime 注册之前，强行扩展 CLI
  成本高且脆弱。
- 使用现有 `--additional-config` 表达：

```json
{
  "afd": {
    "enabled": true,
    "role": "attention",
    "connector": "afdasyncconnector"
  }
}
```

- MoE / eager / NPU worker class 等校验放在插件已有 validation 层。
- 当前插件已有 async connector 校验入口：
  `afd_plugin.compat.ascend.runtime.fail_if_unsupported_npu_afd_features()`。
- 需要把其中对 `parallel_config.async_dp` 的硬依赖改为插件 helper 或动态兼容属性。

优先级：第一批，但不按原 CLI 形式迁移。

### 3. MoE DP 下 async 时不用 `DPEngineCoreProc`

原代码位置：

- `../vllm/vllm/v1/engine/core.py`
- `EngineCoreProc.run_engine_core()`

改动内容：

- 原逻辑：`data_parallel and is_moe` 时一律创建 `DPEngineCoreProc`。
- async 分支：`parallel_config.async_dp` 为真时创建普通 `EngineCoreProc`，并传入
  `engine_index=dp_rank`。

为什么 AFD async 需要：

- `DPEngineCoreProc` 会按 DP wave 协调语义运行，包含 dummy batch 和全局 unfinished
  同步。
- AFD async Attention 侧需要每个 DP rank 独立按本地请求 step，不等待其他 DP rank。

迁移到 afd-plugin 的最小方案：

- 扩展现有 `afd_plugin.compat.patches.engine_core`。
- 当前 patch 已经处理 FFN daemon busy loop；新增 AFD async Attention 条件：
  当 `is_afd_async_dp(vllm_config)` 且 AFD role 为 `attention` 时，MoE DP engine
  创建普通 `EngineCoreProc`。
- patch 必须：
  - 只对 vLLM `0.19.1` 或允许的 dev 版本生效；
  - 幂等；
  - 非 AFD 配置完全委托原 vLLM 逻辑；
  - 有聚焦测试或 NPU smoke 覆盖选择分支。

优先级：第一批。

### 4. 关闭 MoE DP wave coordination

原代码位置：

- `../vllm/vllm/v1/engine/utils.py`
- `launch_core_engines()`
- `DPCoordinator(..., enable_wave_coordination=...)`

改动内容：

- 原逻辑：MoE 模型启动 DP coordinator 时启用 wave coordination。
- async 分支：`enable_wave_coordination = model_config.is_moe and not async_dp`。

为什么 AFD async 需要：

- 即便 engine proc 已换成普通 `EngineCoreProc`，coordinator 如果仍按 MoE wave 语义
  运行，client 侧和 coordinator 状态仍可能等待全局 wave。

迁移到 afd-plugin 的最小方案：

- patch `vllm.v1.engine.utils.launch_core_engines()` 中创建 coordinator 的参数。
- 或更小范围地 patch `DPCoordinator` 创建包装层，在 AFD async 下强制
  `enable_wave_coordination=False`。
- 保留 coordinator 的 stats / load-balance 功能，只关闭 wave coordination。

优先级：第一批。

### 5. client 不再发送 `FIRST_REQ` wave 唤醒

原代码位置：

- `../vllm/vllm/v1/engine/core_client.py`
- `DPAsyncMPClient.add_request_async()`

改动内容：

- 原逻辑：当 `engines_running` 为假时，向 coordinator 发送 `FIRST_REQ`。
- async 分支：`not engines_running and not async_dp` 时才发送。

为什么 AFD async 需要：

- `FIRST_REQ` 是 DP wave coordination 的一部分。
- AFD async 下不希望 coordinator 以全局 wave 模式驱动所有 MoE DP ranks。

迁移到 afd-plugin 的最小方案：

- patch `DPAsyncMPClient.add_request_async()`。
- AFD async 下跳过 `FIRST_REQ`。
- 与“关闭 MoE DP wave coordination”配套；两者最好同一个 compat patch 模块管理。

优先级：第一批。

### 6. `set_forward_context()` 不构造 `DPMetadata`

原代码位置：

- `../vllm/vllm/forward_context.py`
- `set_forward_context()`

改动内容：

- 原逻辑：DP>1 + MoE 下构造 `DPMetadata`，必要时调用
  `coordinate_batch_across_dp()`。
- async 分支：增加 `and not parallel_config.async_dp`，允许
  `forward_context.dp_metadata is None`。

为什么 AFD async 需要：

- AFDAsyncConnector 不再使用 DP metadata control-plane。
- Attention 侧不能再假设 `forward_context.dp_metadata` 必然存在。
- 原生 `set_forward_context()` 的 DP coordination 会引入跨 DP all-reduce，破坏 async
  调度。

迁移到 afd-plugin 的最小方案：

- 插件 runner 已有 capability 分支：当 connector
  `uses_dp_metadata_control_plane=False` 时跳过 `_send_dp_metadata()`。
- 仍建议 patch `vllm.forward_context.set_forward_context()`，在 AFD async 下跳过
  `DPMetadata` 创建，避免进入 vLLM 原生 all-reduce。
- patch 条件必须严格限定为 AFD async。

优先级：第一批。

### 7. `coordinate_batch_across_dp()` 直接返回

原代码位置：

- `../vllm/vllm/v1/worker/dp_utils.py`
- `coordinate_batch_across_dp()`

改动内容：

- 原逻辑：DP>1 时做 DP token 数同步、DP padding、native microbatch 协调。
- async 分支：直接返回 `(False, None, cudagraph_mode)`。

为什么 AFD async 需要：

- AFD async 不走 vLLM 原生 DP padding / microbatching。
- FFN step 由 connector 数据流触发，而不是 DP metadata。

迁移到 afd-plugin 的最小方案：

- patch `coordinate_batch_across_dp()`：
  - 如果 `is_afd_async_dp(parallel_config or vllm_config)` 为真，返回
    `(False, None, cudagraph_mode)`；
  - 否则调用原函数。
- 这个 patch 很小，适合单独加聚焦测试。

优先级：第一批。

### 8. async DP 调试日志

原代码位置：

- `../vllm/vllm/v1/engine/core.py`
- `_log_dp_schedule()`
- `EngineCoreProc.dp_step_counter`
- `_process_engine_step()` begin/end 日志

改动内容：

- 增加 schedule 和 step 级别日志，帮助观察各 DP rank 是否独立 step。

为什么 AFD async 需要：

- 主要用于诊断，不改变功能。

迁移到 afd-plugin 的最小方案：

- 不作为功能迁移的必需项。
- 如需要调试，可在插件 worker / runner 中增加 AFD 专属 debug 日志。
- 不建议为了日志 patch vLLM core。

优先级：可选。

## vLLM-Ascend 改动点

### 9. NPU runner 跳过 DP metadata sync

原代码位置：

- `../vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
- `NPUModelRunner._sync_metadata_across_dp()`

改动内容：

- DP size > 1 且 async-DP 时，不执行 CPU group all-reduce。
- 返回本 rank 的 `num_tokens_padded`，并返回 `num_tokens_across_dp=None`。

为什么 AFD async 需要：

- 原 DP sync 会强制所有 DP ranks 对齐 batch/wave。
- AFD async connector 不需要 native DP token coordination。

迁移到 afd-plugin 的最小方案：

- 优先不 patch vLLM-Ascend native `NPUModelRunner`。
- 走 plugin-owned `AFDNPUAttentionModelRunner`，其 `_sync_metadata_across_dp()` 已有
  connector-driven 分支。
- 把分支判断统一到 `is_afd_async_dp()` 或 connector capability，不直接依赖
  `parallel_config.async_dp`。

优先级：第一批。

### 10. native Ascend FusedMoE async stub（暂不迁移）

原代码位置：

- `../vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py`
- `AscendFusedMoE.forward()`
- `AscendFusedMoE.forward_impl()`
- `AscendSharedFusedMoE.forward()`
- `AscendSharedFusedMoE.forward_impl()`

改动内容：

- async-DP 时 native Ascend MoE 返回 zero/stub。
- 对 shared expert 场景返回 `(shared_out, routed_out)` 形状兼容的 zero/stub。

为什么 AFD async 需要：

- AFD async Attention 侧不应该真正执行 MoE FFN。
- 如果模型路径误入 native MoE，stub 可以避免重复计算或通信。

迁移到 afd-plugin 的最小方案：

- 暂不迁移，不 patch native fused MoE。
- AFD async DeepSeek 路径应强制使用 plugin-owned model wrapper：
  - Attention 侧只计算 attention / gate / topk；
  - 通过 connector 发送 hidden states 和 topk；
  - FFN 侧在 plugin model runner 中计算 MoE。
- 插件当前 `AFDDeepseekV2DecoderLayer.compute_attn_output()` 和
  `forward_with_afd_v2()` 已经承担这部分行为。
- 后续验证如果发现 `afdasyncconnector` + DeepSeek 仍进入 native
  `AscendFusedMoE.forward()`，优先视为 plugin model/runner 路径没有接管干净，
  先修正 class path、model registration、`compute_gate_on_attention` 校验或 FFN
  compute path。
- 只有在确认存在无法绕开的 native MoE 入口时，才重新讨论是否增加 AFD-scoped
  fallback patch；默认不做。

优先级：非核心，暂不迁移，仅保留为 smoke 验证关注点。

### 11. force load balance buffer

原代码位置：

- `../vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py`
- `../vllm-ascend/vllm_ascend/quantization/methods/w8a8_dynamic.py`

改动内容：

- 从 `additional_config` 读取：
  - `enable_force_load_balance`
  - `force_load_balance_topn_per_rank`
- 为 W8A8 MoE 初始化 fake topk buffer。
- W8A8 dynamic apply 时，如果 layer 开启 force load balance，用 fake topk ids
  替换原路由结果。

为什么可能与 AFD async 相关：

- 这更像 profile / 负载均衡辅助能力，不是 async-DP 调度本身。
- 可能用于避免 profile 或测试时 token 都集中到少数 EP ranks。

迁移到 afd-plugin 的最小方案：

- 先不搬 native patch。
- 插件已有 Attention-side topk 控制入口，可优先复用或增强：
  `afd_plugin.model_executor.models.deepseek_v2._force_balanced_topk_ids()`。
- 如果 FFN native W8A8 profile 仍需要 fake topk buffer，再把 buffer 逻辑迁到插件
  compute path，而不是 patch 整个 vLLM-Ascend quant method。

优先级：非核心，暂不迁移；仅在 profile / 性能验证证明需要时单独评估。

### 12. layer sharding 不再限制 PD producer

原代码位置：

- `../vllm-ascend/vllm_ascend/platform.py`
- `NPUPlatform._validate_layer_sharding_config()`

改动内容：

- 原逻辑：`additional_config.layer_sharding` 只能在 PD-disaggregated P node 使用。
- 新逻辑：该校验直接放开。

为什么可能与 AFD async 相关：

- 如果 AFD async 部署需要 layer sharding，但不是 vLLM-Ascend 所定义的 PD P node，
  原校验会阻塞启动。

迁移到 afd-plugin 的最小方案：

- 如果 AFD async 当前不依赖 layer sharding，则不搬。
- 如需要，在插件 Ascend compatibility patch 中只对 AFD config 放开：
  - AFD enabled 且 connector 为 `afdasyncconnector` 时跳过该校验；
  - 普通 vLLM-Ascend 行为不变。

优先级：非核心，暂不迁移；仅在 AFD async 明确依赖 layer sharding 时单独评估。

### 13. NPU profiler schedule 支持

原代码位置：

- `../vllm-ascend/vllm_ascend/worker/worker.py`
- `NPUWorker.__init__()`
- `NPUWorker.execute_model()`
- `NPUWorker.get_profiler()`

改动内容：

- 增加 `_profiler_uses_schedule`。
- torch profiler 支持 delay / wait / warmup / active / max iterations。
- execute_model 时调用 `profiler.step()`。

为什么可能与 AFD async 相关：

- 主要用于性能验证或 profile，不是 async-DP 功能路径。

迁移到 afd-plugin 的最小方案：

- 先不搬。
- 如果后续远程 NPU 性能验证需要相同行为，在 plugin-owned
  `AFDNPUAttentionWorker` / `AFDNPUFFNWorker` 中实现 profiler schedule。
- 不 patch native `NPUWorker`，除非必须复用 vLLM-Ascend profiler 入口。

优先级：非核心，暂不迁移；仅在远程 NPU 性能验证需要 profiler schedule 时单独评估。

## 插件内建议新增模块边界

建议新增或整理以下插件模块：

- `afd_plugin.compat.async_dp`
  - `is_afd_async_dp(vllm_config)`
  - `ensure_async_dp_compat_attr(vllm_config)`
  - AFD async 相关公共判定，不放在 hot path 里做复杂反射。

- `afd_plugin.compat.patches.async_dp_engine`
  - vLLM engine/core-client/utils 相关 patch。
  - 覆盖改动点 3、4、5。

- `afd_plugin.compat.patches.async_dp_forward_context`
  - `set_forward_context()` 和 `coordinate_batch_across_dp()` patch。
  - 覆盖改动点 6、7。

- 当前不新增 `afd_plugin.compat.patches.ascend_async_moe`。
  - 第 10 点默认不迁移；如果 smoke 发现 native FusedMoE 被误走，先修正插件接管路径。
  - 只有确认存在无法绕开的 native MoE 入口时，再重新设计 AFD-scoped fallback patch。

## 推荐实施顺序

1. 新增 `is_afd_async_dp()` 和 compat attr helper。
2. 调整插件内所有直接读 `parallel_config.async_dp` 的位置。
3. 扩展 engine core patch，让 AFD async Attention 使用普通 `EngineCoreProc`。
4. Patch coordinator wave coordination 和 client `FIRST_REQ`。
5. Patch forward context / DP coordination。
6. 跑聚焦测试或 NPU smoke，验证 patch 幂等和非 AFD 行为不变。
7. 再做 NPU smoke，确认 `afdasyncconnector` + DeepSeek 路径不会进入 native
   Ascend FusedMoE；如果会，优先修正插件接管路径，不迁移第 10 点 stub。
8. 根据 profile / 性能验证需要决定是否迁移 force load balance、layer sharding、
   profiler schedule。
