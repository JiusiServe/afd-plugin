# NPU uBatch ModelRunner 重构完成记录

## 最终支持矩阵

| 能力 | 状态 |
| --- | --- |
| no-uBatch | 支持 |
| native DBO uBatch | 支持 |
| async MoE request-boundary uBatch | 支持 |
| graph capture 扩展 | 支持 |
| NPU PCP | 不支持，配置时明确失败 |

## Phase 状态

| Phase | 内容 | 状态 |
| --- | --- | --- |
| 0 | 冻结当前 tag contract 与行为基线 | 完成 |
| 1 | 建立薄 runner 和版本 compat | 完成 |
| 2 | 隔离 control-plane/forward-context | 完成 |
| 3 | 引入 plan、scope 和两个 planner | 完成 |
| 4 | metadata fanout | 完成 |
| 5 | 删除 PCP 适配并增加负向验证 | 完成 |
| 6 | graph capture 收敛 | 完成 |
| 7 | contract、结构测试、CI 和 E2E 护栏 | 完成 |

## 保留的版本适配面

```text
afd_plugin/compat/npu/v0191rc1/
├── model_runner.py
├── attention_metadata_fanout.py
└── dp_coordination.py
```

它们分别承载上游 ModelRunner 方法 patch、attention metadata fanout 和 DP
coordination。稳定 AFD 目录不再承载 PCPManager 或 PCP stage metadata。

## 已删除的维护负担

- PCP stage metadata adapter；
- PCPManager snapshot/restore；
- PCP metadata cache 和 alias 隔离；
- PCP debug 中的生产状态管理；
- PCP contract method snapshot；
- PCP 单元测试和 E2E 参数；
- async MoE 名称中的 PCP 耦合。

## 升级门槛

升级到新的 vLLM/vLLM-Ascend 版本前后必须满足：

- frozen contract 差异已审阅；
- no-uBatch、native DBO、async MoE 单元测试通过；
- graph/eager 测试通过；
- PCP 负向验证通过；
- NPU native DBO 与 async MoE E2E 通过；
- vLLM 和 vLLM-Ascend 无本地修改。

详细升级步骤见
`docs/npu/NPU_UBATCH_MODEL_RUNNER_REFACTOR_PLAN.md`。
