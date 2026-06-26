# NPU AFD async-DP 必要迁移点

本文档只记录迁移 AFD async-DP 核心功能必须处理的框架侧改动点。非核心相邻改动，
例如 native Ascend FusedMoE zero/stub、force load balance、layer sharding 放开、
NPU profiler schedule、额外调试日志等，不列入本文档。

## 核心来源 commit

vLLM：

- `9da7e58d69cbcb66fa5da50e8661e6f9c760b26f`
  `feat: add async dp experiment switch`
- `e0396d82b23a4d8d987f5ee823396e800ebf1364`
  `fix: disable dp wave coordination for async dp`
- `f3ce3f8e2067e05361a18b00e7e719e3cd796087`
  `fix: skip dp batch coordination for async dp`

vLLM-Ascend：

- `bd144ed3cf27b217405028a1bcb931c8b111868f`
  `feat: add async dp moe stub path`
  - 本文档只覆盖该 commit 中 `vllm_ascend/worker/model_runner_v1.py` 的改动。
  - 同 commit 中 `vllm_ascend/ops/fused_moe/fused_moe.py` 的 native MoE zero/stub
    不迁移。

## 迁移原则

- 不修改 `../vllm` 或 `../vllm-ascend` 源码树。
- 不按原样迁移 vLLM `--async-dp` CLI；插件侧用 AFD additional config 和
  `afdasyncconnector` 表达 async-DP 模式。
- 在插件内提供统一判定 helper，例如 `is_afd_async_dp(vllm_config)`。
- 如现有代码必须读取 `parallel_config.async_dp`，可以在插件 runtime 初始化后补一个
  兼容属性，但该属性只作为内部兼容桥，不作为用户配置入口。
- runtime 路径不要求 CPU-safe；验证以聚焦测试和真实 NPU smoke 为主。

## 必要改动点

### 1. async-DP 开关语义

原始来源：

- vLLM commit `9da7e58d69cbcb66fa5da50e8661e6f9c760b26f`
- `../vllm/vllm/config/parallel.py`
- `../vllm/vllm/engine/arg_utils.py`

原改动做了什么：

- 在 vLLM `ParallelConfig` 中新增 `async_dp: bool = False`。
- 在 `EngineArgs` 中新增 `--async-dp`。
- 创建 `ParallelConfig` 时传入 `async_dp`。

为什么需要迁移：

- 后续 engine/core-client/forward-context/runner 分支都需要一个统一的 async-DP
  判定条件。

迁移到 afd-plugin 的最小方案：

- 不修改 vLLM `ParallelConfig` dataclass，也不新增 vLLM CLI。
- 在插件中新增 helper：

```python
def is_afd_async_dp(vllm_config: object) -> bool:
    afd_config = parse_afd_config(vllm_config, validate=False)
    return afd_config.enabled and afd_config.connector == "afdasyncconnector"
```

- 对插件内直接读取 `parallel_config.async_dp` 的位置，改为使用该 helper，或先通过
  插件初始化补兼容属性：

```python
if is_afd_async_dp(vllm_config):
    vllm_config.parallel_config.async_dp = True
```

### 2. async-DP 配置校验

原始来源：

- vLLM commit `9da7e58d69cbcb66fa5da50e8661e6f9c760b26f`
- `../vllm/vllm/engine/arg_utils.py`

原改动做了什么：

- `--async-dp` 只能用于 MoE 模型。
- `--async-dp` 必须配合 eager 模式。

为什么需要迁移：

- AFD async connector 第一版依赖 MoE + eager 路径。
- 如果配置不满足，启动后容易在 runner、MoE、graph 或 DP sync 路径中失败。

迁移到 afd-plugin 的最小方案：

- 在插件 NPU validation 中校验 `afdasyncconnector`：
  - 模型必须是 MoE；
  - `model_config.enforce_eager == True`；
  - 使用 plugin-owned NPU worker / runner class path；
  - `compute_gate_on_attention == True`，确保 DeepSeek AFD async 路径不进入 native
    Ascend FusedMoE；
  - 不启用当前 async connector 不支持的 native ubatching / graph / multistream。

### 3. MoE DP rank 使用普通 `EngineCoreProc`

原始来源：

- vLLM commit `9da7e58d69cbcb66fa5da50e8661e6f9c760b26f`
- `../vllm/vllm/v1/engine/core.py`
- `EngineCoreProc.run_engine_core()`

原改动做了什么：

- 原逻辑：MoE + DP 时创建 `DPEngineCoreProc`。
- async-DP：MoE + DP 时创建普通 `EngineCoreProc(*args, engine_index=dp_rank, **kwargs)`。
- 仍设置 `parallel_config.data_parallel_rank = dp_rank`，保留 DP/EP topology。

为什么需要迁移：

- `DPEngineCoreProc` 会执行 DP wave coordination、dummy batch 和全局 unfinished sync。
- AFD async Attention 侧需要每个 DP rank 独立按本地请求 step。
- 但 expert placement / weight loading 仍需要知道真实 DP rank。

迁移到 afd-plugin 的最小方案：

- 扩展 `afd_plugin.compat.patches.engine_core`。
- 在 `is_afd_async_dp(vllm_config)` 且 AFD role 为 `attention` 时：
  - 保留 `parallel_config.data_parallel_index = dp_rank`；
  - 保留 `parallel_config.data_parallel_rank = dp_rank`；
  - 使用普通 `EngineCoreProc(..., engine_index=dp_rank, ...)`；
  - 不进入 `DPEngineCoreProc`。
- 非 AFD async 配置完全委托原 vLLM 逻辑。

### 4. 关闭 MoE DP wave coordination

原始来源：

- vLLM commit `e0396d82b23a4d8d987f5ee823396e800ebf1364`
- `../vllm/vllm/v1/engine/utils.py`
- `launch_core_engines()`

原改动做了什么：

- 原逻辑：MoE 模型启动 DP coordinator 时启用 wave coordination。
- async-DP：`enable_wave_coordination = model_config.is_moe and not async_dp`。

为什么需要迁移：

- 如果 coordinator 仍按 MoE wave 语义运行，即便 engine proc 已经换成普通
  `EngineCoreProc`，client 和 coordinator 状态仍可能等待全局 wave。

迁移到 afd-plugin 的最小方案：

- 在 AFD async 下 patch coordinator 创建逻辑，使 `enable_wave_coordination=False`。
- 保留 coordinator 的 stats / load-balance 功能，只关闭 wave coordination。

### 5. DP client 跳过 `FIRST_REQ` wave notification

原始来源：

- vLLM commit `e0396d82b23a4d8d987f5ee823396e800ebf1364`
- `../vllm/vllm/v1/engine/core_client.py`
- `DPAsyncMPClient.add_request_async()`

原改动做了什么：

- 原逻辑：当 `engines_running` 为假时，向 coordinator 发送 `FIRST_REQ`。
- async-DP：`not engines_running and not async_dp` 时才发送。

为什么需要迁移：

- `FIRST_REQ` 是 DP wave coordination 的一部分。
- AFD async 不希望 coordinator 以全局 wave 模式驱动 MoE DP ranks。

迁移到 afd-plugin 的最小方案：

- Patch `DPAsyncMPClient.add_request_async()`。
- AFD async 下跳过 `FIRST_REQ`。
- 与“关闭 MoE DP wave coordination”一起处理，避免 coordinator/client 状态不一致。

### 6. `set_forward_context()` 不构造 `DPMetadata`

原始来源：

- vLLM commit `f3ce3f8e2067e05361a18b00e7e719e3cd796087`
- `../vllm/vllm/forward_context.py`
- `set_forward_context()`

原改动做了什么：

- 原逻辑：DP>1 + MoE 时构造 `DPMetadata`，必要时调用
  `coordinate_batch_across_dp()`。
- async-DP：跳过该分支，允许 `forward_context.dp_metadata is None`。

为什么需要迁移：

- `AFDAsyncConnector` 不使用 DP metadata control-plane。
- async Attention 侧不能再假设 `forward_context.dp_metadata` 存在。
- 原生 DP metadata 创建会触发跨 DP all-reduce，破坏 async-DP 独立调度。

迁移到 afd-plugin 的最小方案：

- Patch `vllm.forward_context.set_forward_context()`。
- AFD async 下跳过 `DPMetadata` 创建。
- 插件 Attention runner 已应基于 connector capability 跳过 `_send_dp_metadata()`。

### 7. `coordinate_batch_across_dp()` 跳过跨 DP batch coordination

原始来源：

- vLLM commit `f3ce3f8e2067e05361a18b00e7e719e3cd796087`
- `../vllm/vllm/v1/worker/dp_utils.py`
- `coordinate_batch_across_dp()`

原改动做了什么：

- 原逻辑：DP>1 时执行 DP token 数同步、DP padding、native microbatch 协调。
- async-DP：直接返回 `(False, None, cudagraph_mode)`。

为什么需要迁移：

- AFD async 不走 vLLM 原生 DP padding / microbatching。
- FFN step 由 connector 数据流触发，不由 DP metadata 触发。

迁移到 afd-plugin 的最小方案：

- Patch `coordinate_batch_across_dp()`：
  - AFD async 下返回 `(False, None, cudagraph_mode)`；
  - 其它配置调用原函数。

### 8. NPU runner 跳过 DP metadata sync

原始来源：

- vLLM-Ascend commit `bd144ed3cf27b217405028a1bcb931c8b111868f`
- 只迁 `../vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
- `NPUModelRunner._sync_metadata_across_dp()`

原改动做了什么：

- DP size > 1 且 async-DP 时，不执行 CPU group all-reduce。
- 返回：

```python
return False, num_tokens_padded, None, cudagraph_mode
```

为什么需要迁移：

- vLLM-Ascend 的 NPU runner 也会在 DP>1 时同步 token metadata。
- AFD async Attention rank 需要本地独立调度，不能被 NPU runner 的 DP all-reduce 卡住。

迁移到 afd-plugin 的最小方案：

- 优先不 patch vLLM-Ascend native `NPUModelRunner`。
- 在 plugin-owned `AFDNPUAttentionModelRunner._sync_metadata_across_dp()` 中保留该行为。
- 判断条件使用 `is_afd_async_dp(vllm_config)` 或 connector capability，而不是直接依赖
  原生 `parallel_config.async_dp` 字段。

## 推荐实施顺序

1. 新增 `is_afd_async_dp()` 和可选 compat attr helper。
2. 调整插件内直接读取 `parallel_config.async_dp` 的位置。
3. 加强 AFD async validation。
4. 扩展 engine core patch，让 AFD async Attention 使用普通 `EngineCoreProc`。
5. Patch coordinator wave coordination 和 client `FIRST_REQ`。
6. Patch forward context 和 `coordinate_batch_across_dp()`。
7. 确认 `AFDNPUAttentionModelRunner._sync_metadata_across_dp()` 覆盖 async-DP skip sync。
8. 做 NPU smoke：确认各 Attention DP rank 可独立 step，且不会进入 native
   Ascend FusedMoE zero/stub 路径。
