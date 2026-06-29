# Phase 6 CUDA Graph Status

本文档记录 AFD plugin 当前 CUDA graph 实现状态。目标版本固定为
vLLM `v0.23.0`，不修改 `../vllm` 源码树；所有行为通过本插件 runtime、
connector、model wrapper、显式 class path 或受控 compat shim 提供。

## Current Status

Phase 6 的核心链路已经实现并进入 opt-in GPU 回归阶段：

- AFD 不再全局要求 `--enforce-eager`。
- Attention/FFN 侧只允许 vLLM `CUDAGraphMode.FULL_DECODE_ONLY`；`PIECEWISE`、
  `FULL`、`FULL_AND_PIECEWISE` 和未设置 graph mode 的非 eager 配置会 fail fast。
- FFN 侧实现 graph-keyed CUDA graph cache，按 DP metadata shape 做
  capture/replay。
- Attention 侧在 normal、warmup、capture、replay 路径发送 DP/AFD metadata，并把
  DP metadata control-plane send 移到正式 CUDA graph capture 外。
- P2P connector 已传递 `(dp_metadata_list, is_graph_capturing, is_warmup)`，并在
  FFN graph 模式下按 metadata shape 预分配接收 buffer。
- 两路 DBO/ubatching + `FULL_DECODE_ONLY` CUDA graph 已放行；其他 ubatch 数量仍
  fail fast。

当前仍未解决 role-based weight pruning；Attention 和 FFN 侧仍加载完整
DeepSeekV2 权重。

## Supported Matrix

| 场景 | 当前状态 | 备注 |
| --- | --- | --- |
| AFD eager | 支持 | 现有 `1A1F` / `2A2F` 基线 |
| Attention `FULL_DECODE_ONLY` | 支持 | 唯一支持的 vLLM CUDA graph mode |
| FFN graph-keyed capture/replay | 支持 | graph key 来自 DP metadata token shape |
| AFD + two-way DBO + eager | 支持 | Phase 5 结果 |
| AFD + two-way DBO + `FULL_DECODE_ONLY` | 支持 | 要求 `num_ubatches=2` |
| AFD + `PIECEWISE` | 不支持 | fail fast |
| AFD + `FULL` / `FULL_AND_PIECEWISE` | 不支持 | fail fast |
| AFD + DBO graph with `num_ubatches != 2` | 不支持 | fail fast |
| TP/PP 下 FFN graph cache | 未验证 | 当前 GPU E2E 固定 `TP=1` |

## Implementation Notes

### Policy

`afd_plugin.v1.worker.cuda_graph.validate_cuda_graph_mode()` 负责解析和校验 graph
策略。它保持 CPU-safe，不在 import 时加载 torch 或 vLLM CUDA runtime。

策略结果包含：

- `enabled`
- `mode_name`
- `allow_attention_full_decode_only`
- `enable_ffn_graph_cache`
- `allow_cuda_graph_with_ubatching`

保留了兼容入口 `fail_if_cuda_graph_enabled()`，但它现在委托给 mode validator，而
不是简单拒绝所有 graph 配置。

### Attention Metadata

`AFDAttentionModelRunner` 覆盖三类路径：

- normal run：在 `_model_forward()` 安装 AFD metadata 并发送普通 DP metadata。
- warmup：在 `_warmup_and_capture()` warmup loop 中设置 `_is_warmup=True`，FFN
  侧执行 warmup forward 但不登记正式 graph。
- capture：设置 `_afd_is_graph_capturing=True`。非 ubatch capture 会先发送单 stage
  DP metadata，然后 suppress capture 内重复发送；ubatch capture 由
  `AFDUBatchWrapper` 构造精确 padded ubatch slices，并在进入
  `torch.cuda.graph(...)` 前发送 per-stage DP metadata。

这保证各个 A 侧 DP 的 send metadata 和 F 侧 recv metadata 不落入正式 CUDA graph
capture 流程。

### FFN Graph Cache

`GPUFFNModelRunner` 维护：

```python
self.use_cuda_graph
self._cuda_graphs
self._graph_memory_pool
```

FFN run mode 由 `AFDGraphRunMode` 表示：

- `WARMUP`：执行 eager warmup，不存 graph。
- `CAPTURE`：按当前 graph key capture `_ffn_forward()`。
- `REPLAY`：命中 graph cache 后 replay。
- `EAGER`：graph miss 或 eager mode 下走 `_ffn_forward()` fallback。

graph key 第一版沿用原始 AFD 形状：

```text
tuple((stage_idx, tuple(meta.num_tokens_across_dp_cpu.tolist()))
      for stage_idx, meta in sorted(dp_metadata_list.items()))
```

### DBO / Ubatching

当前只支持 vLLM 两路 ubatch。DBO graph capture/replay 的 key 会从单 stage：

```text
[0:[64,64]]
```

变成双 stage：

```text
[0:[32,32],1:[32,32]]
```

在 2A2F、capture size 64 的场景下，总并发 64 往往不足以触发 live DBO 切分，因为
每个 DP rank 约 32 个 token，vLLM 会避免最后一个 ubatch 为空。把总并发提高到
128 后，每个 DP rank 约 64 个 token，可以触发 `32/32` 双 stage replay。

## Testing Progress

CPU-safe tests 已覆盖：

- CUDA graph policy：eager 允许、`FULL_DECODE_ONLY` 允许、非支持 mode 拒绝、
  两路 DBO graph 允许、其他 ubatch 数量拒绝。
- Attention metadata flag：normal、warmup、capture flag 会传到 connector。
- FFN graph key、replay、miss fallback、capture 时跳过 connector state update。
- AFD ubatch metadata cloning、per-ubatch DP metadata、两路 ubatch 校验。

GPU-gated pytest 已新增以下 opt-in cases：

- `test_deepseek_v2_eager_1a1f_end_to_end`
- `test_deepseek_v2_eager_2a2f_end_to_end`
- `test_deepseek_v2_full_decode_cudagraph_1a1f_end_to_end`
- `test_deepseek_v2_full_decode_cudagraph_2a2f_end_to_end`
- `test_deepseek_v2_full_decode_cudagraph_2a2f_dbo_replays_ubatch_graph`

CUDA graph cases 使用 `AFDDecodeBenchConnector`，对齐 `max-num-seqs`、
`max-num-batched-tokens` 和 CUDA graph capture size。DBO graph case 额外覆盖
live 请求期间的两 stage ubatch 路径。

最近一次 L20X 手工验证已经证明：

- `2A2F + FULL_DECODE_ONLY + AFDDecodeBenchConnector + DBO`
- capture size `64`
- 并发 `128`
- `128/128` completion 请求返回 `200`

## Running GPU E2E

```bash
AFD_GPU_E2E_MODEL=/home/jcz/models/DeepSeek-V2-Lite \
  uv run pytest -q -m gpu
```

常用覆盖变量：

- `AFD_GPU_E2E_GPUS=0,1,2,3`
- `AFD_GPU_E2E_GRAPH_CAPTURE_SIZE=64`
- `AFD_GPU_E2E_GRAPH_REQUESTS=64`
- `AFD_GPU_E2E_DBO_REQUESTS=128`
- `AFD_GPU_E2E_GRAPH_MAX_TOKENS=8`

## Remaining Gaps

- GPU graph tests 已加入 pytest，但仍是 opt-in，需要在 L20X/CI GPU 环境实际跑通后
  才能作为持续回归信号。
- graph 输出与 eager baseline 的 token 序列一致性还没有自动比较；当前 graph tests
  主要验证请求成功。
- 还没有 tiny torch module 级别的 CUDA graph capture/replay correctness unit test。
- strict graph miss fail-fast 还没实现；当前 graph miss 会 eager fallback。
- TP/PP、ratio > 1、EP collective、非 DeepSeekV2 模型的 graph 路径仍需独立验证。
- FFN graph cache 的显存预算尚未纳入 worker memory planning。

## Completion Criteria

Phase 6 可认为“功能闭环”完成的标准：

- L20X 上 `1A1F` / `2A2F` `FULL_DECODE_ONLY` GPU E2E 通过。
- L20X 上 `2A2F` `FULL_DECODE_ONLY + DBO` GPU E2E 通过，并能断言 live 两 stage replay。
- 非支持 graph mode 和非两路 ubatch graph 组合 fail fast 且错误清晰。
- CPU smoke tests 在无 CUDA、无 vLLM wheel 的本地开发环境干净通过或干净 skip。

后续 hardening 再补输出一致性、strict miss、TP/PP/ratio 拓扑和 graph memory runbook。
