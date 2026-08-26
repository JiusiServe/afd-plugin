# DSV4-Flash Prefill 性能仿真器

该工具在相同 16-die 预算下比较两种 Prefill 执行语义：

- CAMAsync AFD：Attention `DP2×TP4`（8 die）与 FFN `EP8`（8 die），两个 Attention DP 独立调度，FFN 使用共享 FCFS 队列；
- 合并部署：`DP4×TP4/SP4/global EP16`，各 DP 每层等待全局 wave；全局 dispatch/routed expert/combine collective 按四个 DP 的 query token 总量建模，combine 后的本地尾段按最重 DP 建模。

Python 后端是唯一仿真实现。CLI、HTTP API 和页面都调用同一套调度、离散事件和指标计算逻辑，前端不重复实现模型。

## 1. 快速开始

要求 Python 3.10 或更高版本。仿真器本身只使用标准库。

### 1.1 生成 msModeling profile

运行时只读取归一化 profile JSON，不会在每次页面操作时启动 msModeling：

```bash
python -m simulator profiles build \
  --msmodeling-root /path/to/msmodeling \
  --python /path/to/msmodeling/python \
  --output simulator/profiles/dsv4-flash-910c.json
```

默认网格：

- query anchors：`1,128,512,2K,4K,8K,16K,32K,64K,128K`；
- prefix anchors：`0,8K,32K,64K,96K,120K`；
- AFD Attention profile：8-device `DP2×TP4`，保留 `attention_router/afd_post`；
- AFD 单 FFN job profile：8-device `DP8×TP1×EP8`，关闭 SP，每 rank 输入 `ceil(stage_query_tokens/8)`，保留 `routed_experts/shared_expert`；
- 合并 profile：16-device `DP4×TP4×EP16`；
- `DeepSeek-V4-Flash`、msModeling `analytic`、sequence parallel、compile 路径。

最大的 query anchor 同时定义最大 context（默认 128K）。生成器会自动补 `prefix=0`、`query=1`、每个 prefix 的 `query=max_context-prefix` 边界点，以及 `prefix=max_context-1, query=1`，因此整个 `prefix+query<=max_context` 三角域都可插值。可通过 `--query-anchors`、`--prefix-anchors`、`--model-id`、`--device` 修改。非默认模型还应通过 `--hidden-size` 和 `--moe-top-k` 记录正确的模型 provenance；tooltip 的 Shape 直接来自 trace，TopK 来自 `--moe-top-k`。超出生成域的输入会报错，不做静默外推。`--keep-traces DIR` 可保留中间 Chrome trace。

### 1.2 启动页面

```bash
python -m simulator serve \
  --profiles simulator/profiles/dsv4-flash-910c.json \
  --host 127.0.0.1 \
  --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。

逐层关键路径时间线支持交互浏览：鼠标悬停或聚焦时间线后，按
`W` / `S` 缩放，按 `A` / `D` 左右平移，按 `R` 复位；也可以使用
鼠标滚轮缩放、按住拖拽平移、双击或点击“重置视图”复位。当前可见
时间范围和缩放倍数显示在时间线右上角。鼠标悬停在事件色块上会显示
阶段、资源、层号、起止时间、持续时间、token 数、批次和 uBatch 信息。
`W` / `S` / `A` / `D` 支持长按连续移动，动画速度与浏览器刷新率同步。
放大到事件色块有足够空间时，色块内会直接显示 Attention、Router、
Dispatch、FFN、Combine、Barrier 等阶段标签；空间不足时自动隐藏文字。
时间线按执行路径合并展示泳道：AFD 的 CAM dispatch/combine 事件并入对应
DP Attention 泳道，merged 的全局 EP16 阶段复制显示在四条 DP
Attention/FFN 泳道中。这里只改变前端布局，不改变后端事件和仿真结果。
merged 时间线还会把相邻的 combine collective 与本地
unpermute/TopK-weight 阶段合并成一个 `Combine` 色块，并将 SP 收尾阶段
显示为 `TP AllGather + HC Post`；profile 中的原始分项时延仍保持独立。
悬停 AFD 的 dispatch/combine 通信色块时，时间线会按相同的
layer、batch 和 uBatch 配对 Attention/CAM 与 FFN EP8 两端，并用带方向的
箭头显示数据流；横向跨度同时反映发送完成到接收开始之间的排队时间。
悬停 `FFN Compute`、`Routed Experts` 或 `Shared Expert` 时还会显示
msModeling trace 中 EP rank 的总输入 Shape、每个本地 Expert 的 GMM 实采
Shape、MoE TopK 和对应架构的 EP 数。Shape 直接从生成 profile 所用 trace
提取，不再根据请求 token 数推导。相同 Shape 的本地 Expert 会压缩显示为
`count × [tokens, hidden]`。运行点落在两个 query anchor 之间时，页面会同时
列出两个实采 Shape、各自的采样 Query 和线性插值权重。Shared Expert 不经过
TopK 路由，因此 TopK 显示“不适用”。Expert Shape 只随 query 变化，profile
仅在 `prefix=0` 的 query anchor 保存一份，避免随 prefix 重复数据。

### 1.3 CLI 仿真

```bash
python -m simulator simulate \
  --profiles simulator/profiles/dsv4-flash-910c.json \
  --config simulator/examples/continuous-prefix-cache.json \
  --output /tmp/dsv4-result.json

python -m simulator sweep \
  --profiles simulator/profiles/dsv4-flash-910c.json \
  --config simulator/examples/continuous-prefix-cache.json \
  --output /tmp/dsv4-sweep.json
```

## 2. 全部配置字段

配置文件顶层是 JSON object。没有提供的字段使用表中默认值。

### 2.1 Workload

| 字段 | 类型 / 默认值 | 说明 |
| --- | --- | --- |
| `mode` | `fixed` / `"fixed"` | `fixed`：所有请求在 `t=0` 可调度；`continuous`：按 arrival 配置持续生成请求。 |
| `fixed_lengths` | `int[]` / `[512,8192,2048,6144]` | 固定模式的完整 Prompt 长度。未使用 CSV 时生效。 |
| `length_mix` | `{tokens,weight}[]` | 持续模式的离散长度分布；`weight` 只需为正数，不要求预归一化。 |
| `csv_path` | `string|null` | CLI 读取的 CSV 路径；与 `csv_text` 互斥。页面上传会使用 `csv_text`。 |
| `csv_text` | `string|null` | CSV 原始内容；与 `csv_path` 互斥。 |
| `csv_sampling` | `cycle|sample` / `"cycle"` | 无时间戳 CSV 在持续模式下的选取方式：按行循环或有放回随机采样。 |

优先级：CSV > `fixed_lengths`/`length_mix`。固定模式下 CSV 每行回放一次；持续模式下无时间戳 CSV 提供经验长度分布。

### 2.2 `arrival`

| 字段 | 类型 / 默认值 | 说明 |
| --- | --- | --- |
| `arrival.kind` | `constant|poisson|trace` / `constant` | 固定间隔、泊松到达或回放 CSV 时间戳。 |
| `arrival.qps` | `float` / `1.0` | `constant`/`poisson` 的 offered QPS；trace 模式忽略。 |
| `arrival.duration_s` | `float` / `60` | warmup 后的统计窗口和请求生成时长。 |
| `arrival.warmup_s` | `float` / `10` | 持续负载预热时长；该区间到达的请求不进入最终指标。 |
| `arrival.seed` | `int` / `1024` | 泊松间隔随机种子；长度采样使用 `seed+1`。 |

`trace` 只支持持续模式，并要求 CSV 每一行都有 `arrival_time_ms`。第一条时间戳归零后按原间隔回放，只保留 `[0, warmup+duration)` 的请求；指标只统计 `[warmup, warmup+duration)` 内到达的请求，窗口内的空闲时间也进入吞吐分母。精确 trace 模式不支持自动 QPS 扫描。

### 2.3 `scheduler`

| 字段 | 类型 / 默认值 | 说明 |
| --- | --- | --- |
| `scheduler.policy` | `round_robin|vllm_queue_aware` / `round_robin` | `round_robin` 按到达顺序严格轮询；`vllm_queue_aware` 在请求到达时选择尚未完成请求数最少的 DP，同分时轮转。 |
| `scheduler.max_num_seqs` | `int` / `64` | 一个 DP scheduler batch 最多包含的请求/请求 chunk 数。 |
| `scheduler.max_num_batched_tokens` | `int` / `8192` | 一个 DP batch 的未缓存 query token 预算。 |
| `scheduler.chunked_prefill` | `bool` / `false` | 是否允许长 Prompt 分多个 Prefill batch。AFD chunked 路径是敏感性假设，不代表当前运行时已支持。 |
| `scheduler.chunk_size` | `int` / `max_num_batched_tokens` | 单请求每次最多调度的 query token；不得超过 batch token 预算。 |

每个 DP 使用 FIFO 装箱。non-chunked 请求的未缓存长度若超过 token 预算会直接报错。chunked 请求保留已计算 prefix，最终 chunk 完成才算请求完成。

`vllm_queue_aware` 对齐当前 vLLM 内置 DP 负载均衡的请求数口径和轮转平局规则。vLLM 还会在存在 waiting 请求且 KV 使用率超过 50% 时增加压力惩罚；本仿真器没有 Decode 与 KV 容量/驻留模型，因此不模拟该项，也不将 Prefix Cache 命中率误作 KV 容量压力。请求长度本身不参与该策略评分。

### 2.4 `prefix_cache`

| 字段 | 类型 / 默认值 | 说明 |
| --- | --- | --- |
| `prefix_cache.enabled` | `bool` / `false` | Prefix Cache 总开关。关闭时忽略 CSV cache 字段和全局命中参数。 |
| `prefix_cache.request_hit_rate` | `[0,1]` / `0` | 未提供请求级 cache 数据时，一个请求发生命中的概率。 |
| `prefix_cache.matched_prefix_ratio` | `[0,1]` / `0` | 命中请求中被缓存的前缀 token 比例。 |
| `prefix_cache.block_size` | `int` / `32` | 采样的缓存长度向下对齐到该 block；CSV 的真实 `cached_prefix_tokens` 不再对齐。 |
| `prefix_cache.lookup_fixed_ms` | `float` / `0` | 每个请求的固定缓存查找开销。 |
| `prefix_cache.lookup_per_block_ms` | `float` / `0` | 每个已缓存 block 的附加查找开销。 |
| `prefix_cache.seed` | `int` / `1024` | 命中与否的随机种子。两种架构复用同一采样结果。 |

计算语义：

```text
prefix_tokens = cached_prefix_tokens + 已完成的 chunk tokens
query_tokens  = 当前实际 Prefill tokens
```

Scheduler、FFN、Router 和 CAM 只处理 query tokens；Attention 用 `(prefix_tokens, query_tokens)` 查询 msModeling profile。输出同时报告逻辑输入 token 与实际计算 token。

merged 的 Attention 仍按每个 DP 的真实 query tokens 分别查表，barrier 等待最慢
DP。`merged_dispatch`、`routed_experts`、`merged_combine` 先求当前 wave 的
四 DP query tokens 总和，再用 `global_query_tokens/4` 查询对称 DP4 profile；
不足 1 token 时按最小 anchor 1 处理。combine 后的 `merged_combine_local`、
`shared_expert`、`merged_sp_post` 属于 per-DP 本地尾段，使用当前 wave 的
`max(dp_query_tokens)` 查询 profile。时间线的 `Token 数` 是该 phase 的 workload
口径：前三段显示全局总量，后三段显示最重 DP token；`Profile Query` 是实际
用于查表的 token 数。这里按需求将 dispatch 也视作全局 payload 的对称等效
近似；它不刻画发送端 DP token skew。若要研究 sender-side dispatch 尾延迟，
需要另行使用 `max(dp_query_tokens)` 或建立非对称通信 profile。

### 2.5 `afd`

| 字段 | 类型 / 默认值 | 说明 |
| --- | --- | --- |
| `afd.ubatch_split` | `request|token` / `request` | 两个 CAM MoE stage 的切分方式。请求模式按 token 总量选择最接近均衡的请求边界；token 模式可切开单个请求 chunk。无法形成两个非空 stage 时回退单 stage。 |

### 2.6 `cam`

每个 CAM leg 使用：

```text
latency_ms = fixed_ms + per_token_ms × stage_query_tokens
```

| 字段 | 默认值 `(fixed_ms, per_token_ms)` | 说明 |
| --- | --- | --- |
| `cam.calibrated` | `false` | 仅作结果可信度标记，不改变计算。 |
| `cam.dispatch_send` | `(0.11, 1/52000)` | Attention 侧 dispatch send。 |
| `cam.dispatch_recv` | `(0.10, 1/68000)` | FFN 侧 dispatch recv。 |
| `cam.combine_send` | `(0.10, 1/70000)` | FFN 侧 combine send。 |
| `cam.combine_recv` | `(0.12, 1/58000)` | Attention 侧 combine recv。 |
| `cam.<leg>.fixed_ms` | 见上 | 该 leg 固定启动时延。 |
| `cam.<leg>.per_token_ms` | 见上 | 每 query token 的线性时延。 |

默认值来自旧流水页面，只是未校准占位值。正式结论应使用 CAM microbenchmark 拟合值并设置 `calibrated=true`。

### 2.7 `slo`、`sweep` 和 TTFT

| 字段 | 类型 / 默认值 | 说明 |
| --- | --- | --- |
| `slo.ttft_limit_ms` | `float` / `1000` | Prefill TTFT 代理的 SLO 上限。 |
| `slo.target_ratio` | `(0,1]` / `0.99` | 自动容量判断要求的请求达标比例。 |
| `fixed_ttft_overhead_ms` | `float` / `0` | 加到每个请求 TTFT 上的 tokenizer/HTTP 等固定外部开销；不占用模拟计算资源。 |
| `sweep.min_qps` | `float` / `0.5` | 粗扫下界。 |
| `sweep.max_qps` | `float` / `64` | 粗扫上界。 |
| `sweep.coarse_points` | `int` / `10` | 对数间隔粗扫点数。 |
| `sweep.refinement_steps` | `int` / `7` | 最后一个 PASS 与第一个 FAIL 之间的二分轮数。 |
| `sweep.throughput_tolerance_ratio` | `(0,1]` / `0.99` | 除 SLO 外，要求 achieved throughput 至少达到 offered QPS 的比例。 |

QPS 扫描只支持 `mode="continuous"`，因为固定请求集没有 offered QPS；精确 CSV timestamp trace 也不支持改变 QPS。

TTFT 定义：

```text
TTFT_proxy = Prefill完成时间 - arrival_time + fixed_ttft_overhead_ms
```

它不包含首个 Decode step。持续模式停止到达后会 drain 完成，避免遗漏积压请求。

### 2.8 `output`

| 字段 | 类型 / 默认值 | 说明 |
| --- | --- | --- |
| `output.include_timeline` | `bool` / `true` | 是否返回逐层阶段事件。QPS 扫描内部会关闭。 |
| `output.timeline_max_events` | `int` / `20000` | 最大时间线事件数，超过后截断并设置 `timeline_truncated=true`。 |
| `output.include_requests` | `bool` / `true` | 是否返回每个请求的到达、DP、cache、完成时间和 TTFT。 |

## 3. CSV 格式

### 3.1 只有线上长度列表

```csv
input_length
512
8192
8192
32768
```

`input_length` 必填。重复行会保留，因此自然构成经验分布。

### 3.2 完整 trace

```csv
request_id,arrival_time_ms,input_length,cached_prefix_tokens
r001,0,8192,4096
r002,17,512,0
r003,21,32768,24576
```

| 列 | 必填 | 说明 |
| --- | --- | --- |
| `input_length` | 是 | 完整 Prompt token 数，必须为正整数。 |
| `request_id` | 否 | 原始请求标识；缺失时生成 `r1`、`r2`。 |
| `arrival_time_ms` | 否 | 线上到达时间。必须全部行都有或全部没有。 |
| `cached_prefix_tokens` | 否 | 真实缓存前缀长度，必须满足 `0 <= cached < input_length`。Prefix Cache 开启时覆盖全局采样。 |

未知列会被忽略。解析错误会报告 CSV 行号。

## 4. HTTP API

| 接口 | 说明 |
| --- | --- |
| `GET /api/defaults` | 返回全部默认配置与当前 profile 元数据。 |
| `POST /api/simulate` | Body 为完整配置 JSON；返回 AFD、合并结果和对比倍率。 |
| `POST /api/sweep` | Body 为完整配置 JSON；返回两种架构的 QPS 曲线和最大 SLO QPS。 |

页面与 API 同源，默认只监听 `127.0.0.1`。请求 body 上限为 10 MiB。

## 5. 输出指标

- `throughput_rps`：统计请求数除以包含 drain 的有效时长；
- `input_tokens_per_s`：逻辑完整 Prompt 吞吐；
- `compute_tokens_per_s`：扣除缓存前缀后的实际 Prefill token 吞吐；
- `ttft_mean/p50/p90/p99_ms`：Prefill TTFT 代理；
- `slo_attainment`：TTFT 不超过上限的请求比例；
- `slo_goodput_rps`：SLO 达标请求数除以有效时长；
- `utilization`：各 Attention DP 和 FFN/EP 关键资源的忙时比例；
- `barrier_wait_ms`：合并路径所有 DP 的同步等待总和；
- `attention_wait_ms`：AFD Attention 等待 FFN 返回的总和。

## 6. 当前模型边界

- 算子时延来自 msModeling analytic trace；CAM 时延来自独立参数模型。AFD profile 在归一化阶段按 phase 合成 Attention 与单 FFN job 两种 trace，并在 JSON metadata 中保存来源和命令。
- Attention batch 时延按各请求/chunk profile 求和；AFD FFN/MoE 使用 stage 总 query tokens 查表。FFN 侧假设 EP8 各 rank 均分单 job token，shared expert 也按每 rank `ceil(tokens/8)` 的 DP8×TP1 口径建模。
- merged 的 routed/combine 以及按需求近似的 dispatch 使用 `global_query_tokens/4`；combine-local/shared/SP 使用 `max(dp_query_tokens)`。
- 不模拟 Decode、KV transfer、prefix cache 容量/淘汰算法、MTP、graph、prefix cache lookup 并发、HBM OOM、EPLB 或真实专家负载偏斜。
- AFD DSV4 NPU 与 AFD chunked prefill 都是架构性能假设，不代表当前 afd-plugin 已完成对应 E2E 支持。

## 7. 测试

无额外测试依赖：

```bash
python -m unittest discover -s simulator/tests -v
```

测试覆盖 profile 插值/越界、CSV、Prefix Cache、固定/持续负载、chunked prefill、AFD 双 uBatch、合并屏障和 QPS 扫描复现性。
