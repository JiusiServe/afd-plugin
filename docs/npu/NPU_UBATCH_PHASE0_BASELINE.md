# NPU uBatch Phase 0 基线

## 范围

Phase 0 固定以下版本的行为：

- vLLM `v0.19.1`
- vLLM-Ascend `v0.19.1rc1`
- AFD no-uBatch
- AFD native DBO uBatch
- AFD async MoE request-boundary uBatch

PCP 不属于当前升级范围。`prefill_context_parallel_size > 1` 必须失败。

## 已固定的契约

| 契约 | 验证方式 |
| --- | --- |
| NPU ModelRunner 基类和方法签名 | frozen JSON contract |
| attention metadata schema | frozen JSON contract |
| native DBO 决策 | planner 单测 |
| async MoE request-boundary split | planner 单测 |
| scope 成功/异常清理 | scope 单测 |
| metadata fanout | fake runner/builder 单测 |
| 不支持 PCP | feature validation 负向测试 |

冻结文件：

```text
tests/contracts/npu_model_runner_v0191rc1.json
```

生成命令：

```bash
python tools/compat/snapshot_npu_model_runner_contract.py \
  --vllm-root ../vllm \
  --vllm-ref v0.19.1 \
  --vllm-ascend-root ../vllm-ascend \
  --vllm-ascend-ref v0.19.1rc1
```

## CPU 验证

```bash
pytest -q tests/unit -m "not gpu and not vllm_runtime"
pytest -q tests/e2e/test_runner.py
ruff check .
ruff format --check .
```

## NPU 验证

至少运行两组真实硬件用例：

1. native DBO uBatch；
2. async MoE request-boundary uBatch。

记录启动命令、模型、设备拓扑、通过时间、关键 uBatch 日志和精度结果。PCP 用例
不再执行，也不作为升级门槛。

## Phase 0 退出条件

- frozen contract 可重复生成；
- no-uBatch/native DBO/async MoE CPU 测试通过；
- PCP 配置有明确失败测试；
- NPU E2E 参数中不再出现 PCP；
- 基线文档与代码支持矩阵一致。
