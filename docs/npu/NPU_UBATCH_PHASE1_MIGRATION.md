# NPU uBatch Phase 1 迁移说明

## 结果

Phase 1 将 AFD NPU Attention ModelRunner 从“大方法复制”调整为“薄 runner +
稳定组件 + 版本 compat”。

## 迁移映射

| 原 runner 中的逻辑 | 新位置 |
| --- | --- |
| uBatch mode 与生命周期 | `v1/worker/npu/ubatch_plan.py` |
| native DBO 决策 | `NativeDBOPlanner` |
| async MoE request-boundary split | `AsyncMoePlanner` |
| async MoE metadata sidecar | `async_moe_ubatch.py` |
| AFD attention metadata | `attention_metadata_adapter.py` |
| ForwardContext 安装 | `forward_context.py` |
| graph capture 扩展 | `graph_capture.py` |
| DP token coordination | `compat/npu/v0191rc1/dp_coordination.py` |
| 上游 metadata fanout | `compat/npu/v0191rc1/attention_metadata_fanout.py` |
| 仍需复制的上游方法 | `compat/npu/v0191rc1/model_runner.py` |

## PCP 范围调整

升级不再要求 PCP，因此 Phase 1 之后做了以下收口：

- `AsyncMoePCPPlanner` 重命名为 `AsyncMoePlanner`；
- `AsyncMoePCPUbatch` 重命名为 `AsyncMoeUbatch`；
- 删除 PCP stage metadata adapter；
- 删除 `PCPManager` 状态事务；
- 删除 PCP debug 生产逻辑；
- 删除 PCP E2E 参数；
- NPU runtime 明确拒绝 PCP。

连接器的通用 rank 计算未删除，因为它不是 NPU ModelRunner PCP 执行适配，且可能被
其他后端或拓扑复用。

## 评审重点

1. `attention_model_runner.py` 是否只保留组合和稳定入口。
2. `compat/npu/v0191rc1/model_runner.py` 中每个 patch 是否有来源、原因和差异标记。
3. planner 是否不依赖 graph、stream 或可变全局状态。
4. scope 是否在异常路径清理。
5. async MoE 是否只使用 request-boundary slice。
6. `prefill_context_parallel_size > 1` 是否明确失败。

## 升级收益

升级新 vLLM-Ascend 时，主要重新检查版本 compat 的三个文件，而不是手工比较整个
AFD ModelRunner。稳定 planner、scope、metadata adapter 和 forward-context
协议只有在 AFD 自身需求变化时才需要修改。
