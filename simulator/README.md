# DSV4-Flash Prefill 性能仿真器

该工具用可切换的等 die profile 比较两种 Prefill 执行语义：

- CAMAsync AFD：profile 指定 Attention `DP×TP` 与独立 FFN `EP8`，Attention DP 异步调度，FFN 使用共享 FCFS 队列；
- 合并部署：profile 指定 `DP×TP/global EP`，各 DP 每层等待全局 wave；全局 dispatch/routed expert/combine collective 按所有 DP 的 query token 总量建模，combine 后的本地尾段按最重 DP 建模。

Python 后端是唯一仿真实现。CLI、HTTP API 和页面都调用同一套调度、离散事件和指标计算逻辑，前端不重复实现模型。

所有比较强制使用相同的物理 die 数：`AFD Attention num_devices + AFD FFN num_devices = merged num_devices`。merged 的 TP、DP、EP 是同一组 die 上的并行维度，EP 不额外增加设备；例如 AFD `DP5×TP8 + 独立 FFN EP8` 使用 `40+8=48` die，对应 merged `DP6×TP8×EP48` 使用 48 die。不满足等 die 或 `DP×TP=num_devices` 的 profile 会在生成、组合、仿真或服务加载时直接报错。

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
- 合并 profile 默认使用 16-device `DP4×TP4×EP16`；可用 `--merged-dp-size 2 --merged-tp-size 8` 生成 `DP2×TP8×EP16`；
- `DeepSeek-V4-Flash`、msModeling `analytic`、sequence parallel、compile 路径。

最大的 query anchor 同时定义最大 context（默认 128K）。生成器会自动补 `prefix=0`、`query=1`、每个 prefix 的 `query=max_context-prefix` 边界点，以及 `prefix=max_context-1, query=1`，因此整个 `prefix+query<=max_context` 三角域都可插值。可通过 `--query-anchors`、`--prefix-anchors`、`--model-id`、`--device` 修改。非默认模型还应通过 `--hidden-size` 和 `--moe-top-k` 记录正确的模型 provenance；tooltip 的 Shape 直接来自 trace，TopK 来自 `--moe-top-k`。超出生成域的输入会报错，不做静默外推。`--keep-traces DIR` 可保留中间 Chrome trace。

默认设备名 `ATLAS_800_A3_752T_128G_DIE` 是 msModeling 的单 die profile：
每个 `--num-devices` 计数代表一个 64 GiB die；一张标称 128G 的 A3 卡包含
两个 die。因此页面中的 die 数除以二才是物理卡数。该 profile 的通信不是
全设备同速假设：rank 按 `[48, 8, 2]` 排列，并分别为跨 pod 的双层 CLOS、
pod 内单层 CLOS 和卡内 SIO 使用不同的带宽与延迟。

仓库内的 batch-64K profiles 覆盖以下等 die 拓扑。相同 AFD TP 和 FFN EP8
的行只复用一次 per-DP 实采点；增加 AFD DP replica 不会重新运行
msModeling。merged 的 TP 或全局 EP 变化时则重新生成 merged 点。运行时从
profile spec 读取 DP/TP/EP，不对不同拓扑做比例换算。

| die 数 | AFD Attention + 独立 FFN | merged 候选 |
| ---: | --- | --- |
| 16 | DP2×TP4 + EP8 | DP4×TP4×EP16；DP2×TP8×EP16 |
| 24 | DP4×TP4 + EP8 | DP3×TP8×EP24；DP6×TP4×EP24 |
| 32 | DP3×TP8 + EP8；DP6×TP4 + EP8 | DP4×TP8×EP32；DP8×TP4×EP32 |
| 40 | DP4×TP8 + EP8 | DP5×TP8×EP40 |
| 48 | DP5×TP8 + EP8 | DP6×TP8×EP48 |

仅生成新的 merged 拓扑时可跳过 AFD 调用：

```bash
python -m simulator profiles build \
  --msmodeling-root /path/to/msmodeling \
  --python /path/to/msmodeling/python \
  --topology merged \
  --merged-num-devices 48 \
  --merged-dp-size 6 \
  --merged-tp-size 8 \
  --merged-ep-size 48 \
  --output /tmp/merged-dp6tp8ep48.json
```

AFD 只增加 DP replica、TP 与独立 FFN EP 不变时，使用已有 per-DP 点组合，避免重新运行 msModeling：

```bash
python -m simulator profiles compose \
  --afd-profile simulator/profiles/existing-afd-tp8.json \
  --merged-profile /tmp/merged-dp6tp8ep48.json \
  --afd-dp-size 5 \
  --output simulator/profiles/afd-dp5tp8-vs-merged-dp6tp8.json
```

### 1.2 启动页面

```bash
python -m simulator serve \
  --profiles \
    simulator/profiles/dsv4-batch-64k.json \
    simulator/profiles/dsv4-batch-64k-dp2tp8.json \
    simulator/profiles/dsv4-batch-64k-afd-dp3tp8-merged-dp4tp8.json \
    simulator/profiles/dsv4-batch-64k-afd-dp5tp8-merged-dp6tp8.json \
    simulator/profiles/dsv4-batch-64k-afd-dp4tp4-merged-dp3tp8.json \
    simulator/profiles/dsv4-batch-64k-afd-dp6tp4-merged-dp8tp4.json \
    simulator/profiles/dsv4-batch-64k-afd-dp4tp8-merged-dp5tp8.json \
    simulator/profiles/dsv4-batch-64k-afd-dp4tp4-merged-dp6tp4.json \
  --host 127.0.0.1 \
  --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。服务会从全部输入 profile 中分别去重 AFD 和 merged 拓扑：AFD 下拉框列出所有 AFD 拓扑；选择后，merged 下拉框只列出总 die 数相同的选项，并默认选择该 AFD 原 profile 中的配对拓扑。同 die 下的其他 merged 拓扑仍可手动切换，例如 16-die AFD 可以比较 `DP4×TP4×EP16` 或 `DP2×TP8×EP16`。

逐层关键路径时间线支持交互浏览：鼠标悬停或聚焦时间线后，按
`W` / `S` 缩放，按 `A` / `D` 左右平移，按 `R` 复位；也可以使用
鼠标滚轮缩放、按住拖拽平移、双击或点击“重置视图”复位。当前可见
时间范围和缩放倍数显示在时间线右上角。鼠标悬停在事件色块上会显示
阶段、资源、层号、起止时间、持续时间、token 数、批次和 uBatch 信息。
`W` / `S` / `A` / `D` 支持长按连续移动，动画速度与浏览器刷新率同步。
放大到事件色块有足够空间时，色块内会直接显示 Attention、Router、
Dispatch、FFN、Combine、Barrier 等阶段标签；空间不足时自动隐藏文字。
时间线按执行路径合并展示泳道：AFD 的 CAM dispatch/combine 事件并入对应
DP Attention 泳道，merged 的全局 EP 阶段复制显示在所有 DP
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
| `csv_sampling` | `cycle|sample` / `"cycle"` | CSV 在合成到达过程下的长度选取方式：按行循环或有放回随机采样。 |

优先级：CSV > `fixed_lengths`/`length_mix`。固定模式下 CSV 每行回放一次；
持续模式下，`constant`/`poisson` 把 CSV 当作经验长度分布，即使文件带
`arrival_time_ms` 也会忽略该列；只有 `trace`/`scaled_trace` 会消费时间戳。

WebUI 还提供 MoonConv V4 Flash 的 `formal_0/1/2`（各 512 条）和
`screening`（128 条）内置 trace。它们只保留输入长度与相对到达时间；选择后
自动切换到持续模式和 `scaled_trace`，按配置 QPS 缩放原 trace 的时间轴。

### 2.2 `arrival`

| 字段 | 类型 / 默认值 | 说明 |
| --- | --- | --- |
| `arrival.kind` | `constant|poisson|scaled_trace|trace` / `constant` | 固定间隔、泊松到达、按 QPS 缩放 trace，或精确回放 CSV 时间戳。 |
| `arrival.qps` | `float` / `1.0` | `constant`/`poisson` 的 offered QPS；`scaled_trace` 的目标平均 QPS；精确 `trace` 忽略。 |
| `arrival.duration_s` | `float` / `60` | warmup 后的统计窗口和请求生成时长。 |
| `arrival.warmup_s` | `float` / `10` | 持续负载预热时长；该区间到达的请求不进入最终指标。 |
| `arrival.seed` | `int` / `1024` | 泊松间隔随机种子；长度采样使用 `seed+1`。 |

`trace` 只支持持续模式，并要求 CSV 每一行都有 `arrival_time_ms`。第一条时间戳归零后按原间隔回放，只保留 `[0, warmup+duration)` 的请求；指标只统计 `[warmup, warmup+duration)` 内到达的请求，窗口内的空闲时间也进入吞吐分母。精确 trace 模式不支持自动 QPS 扫描。

`scaled_trace` 同样要求完整时间戳。它保留所有相对间隔，包括同一时间戳形成的
零间隔 burst；然后统一缩放间隔，使其平均值为 `1000/arrival.qps` ms。
为了覆盖任意 warmup 和统计时长，trace 结束后按原请求及间隔顺序循环，并在
循环边界插入一个缩放后的平均间隔。因此每个完整循环的 offered QPS 精确等于
配置值。QPS sweep 可以使用 `scaled_trace`，每个扫描点重新缩放同一 arrival
形状。

WebUI 上传带 `arrival_time_ms` 的 CSV 时默认切换到持续模式和
`scaled_trace`。仍可手动切换为 `trace` 精确回放，或切换为
`constant`/`poisson` 仅使用其长度分布。

### 2.3 `scheduler`

| 字段 | 类型 / 默认值 | 说明 |
| --- | --- | --- |
| `scheduler.afd_policy` | `current_runtime|afd_wave_token_sum|afd_wave_token_square_sum|prefill_token_greedy|prefill_token_square_greedy|vllm_queue_aware|round_robin` / `current_runtime` | AFD Attention DP 的独立路由策略。 |
| `scheduler.merged_policy` | `current_runtime|merged_wave_token_sum|merged_wave_token_square_sum|prefill_token_greedy|prefill_token_square_greedy|vllm_queue_aware|round_robin` / `current_runtime` | merged DP 的独立路由策略。 |
| `scheduler.max_num_seqs` | `int` / `64` | 一个 DP scheduler batch 最多包含的请求/请求 chunk 数。 |
| `scheduler.chunked_prefill` | `bool` / `false` | 是否允许长 Prompt 分多个 Prefill batch。AFD chunked 路径是敏感性假设，不代表当前运行时已支持。 |
| `scheduler.afd.max_num_batched_tokens` | `int` / `8192` | AFD 单个 Attention DP batch 的未缓存 query token 预算。 |
| `scheduler.afd.chunk_size` | `int` / AFD `max_num_batched_tokens` | AFD 单请求每个 wave 最多调度的 query token。 |
| `scheduler.merged.max_num_batched_tokens` | `int` / `8192` | merged 单个 DP batch 的未缓存 query token 预算。 |
| `scheduler.merged.chunk_size` | `int` / merged `max_num_batched_tokens` | merged 单请求每个 wave 最多调度的 query token。 |

AFD 与 merged 分别使用自己的 batch token 预算和 chunk 上限；旧的共享 `scheduler.max_num_batched_tokens`、`scheduler.chunk_size` 字段已经移除。每个 DP 使用 FIFO 装箱。non-chunked 请求的未缓存长度若超过任一架构的 token 预算会直接报错。chunked 模式下，每个请求在一个 wave 中至多贡献一个 chunk；若该 chunk 没有用完对应架构的 batch token 预算，调度器会继续按 FIFO 扫描后续请求来填充当前 wave。未完成请求保留已计算 prefix 和原有相对顺序，并在下一 wave 优先调度；因此后到的短请求可能与长请求的首个 chunk 同 wave，并比长请求更早完成。最终 chunk 完成才算请求完成。

`current_runtime` 的架构差异如下：

- merged 使用 vLLM 0.26 的 async-DP DPLB。
- AFD async-DP 当前把 Attention engine core 替换为普通 `EngineCoreProc`，该路径不会发布 DPLB 的 waiting/running 统计；单 API frontend 下，client 只保留本地乐观计数，效果等价于轮询。因此 AFD 使用 `round_robin`。这是按当前代码路径得到的模型，多个相互独立的 API frontend 不在仿真范围内。

两侧策略可以任意组合。例如 AFD 使用 `afd_wave_token_square_sum`、merged 使用 `merged_wave_token_sum`。实验分支不兼容旧的单字段 `scheduler.policy`；出现该字段会直接报错，避免隐式地把同一策略套到两种不同执行语义上。

`vllm_queue_aware` 对齐 vLLM 0.26 的评分和更新机制：`score = 4 × waiting + running`，选择最低分 DP，同分时轮转扫描起点；engine 统计每 100 ms 更新一次，两次更新之间 client 会立即增加所选 DP 的本地 waiting 计数。请求 token 长度和 KV 使用率不参与这个版本的公式。仿真器显式跟踪 batch 外 waiting 与 batch 内 running 请求，chunked request 不会同时重复计数。

`prefill_token_greedy` 是纯 Prefill 实验策略。每个请求到达时使用最新状态选择 outstanding query tokens 最少的 DP；outstanding 包含排队请求尚未计算的 query tokens 和正在运行 batch 的 query tokens，Prefix Cache 已命中的 tokens 不计入。同分时轮转。正在运行的 batch 在完成前按完整 query tokens 计数，不估算 batch 内部进度。

`prefill_token_square_greedy` 使用每个未完成请求的 remaining query tokens 平方和作为 DP score，使长请求获得更高权重。chunk 完成后，其贡献从 `remaining_before²` 更新为 `remaining_after²`。该策略同样使用即时状态，不保留已完成工作的历史欠账。

`afd_wave_token_sum` 和 `afd_wave_token_square_sum` 是 AFD 的 FIFO wave 完成时间实验。候选请求只能追加到某个 DP 的本地队尾；调度器按 AFD 自己的 `max_num_batched_tokens`、`chunk_size` 与共享的 `max_num_seqs` 重建尚未运行的 FIFO wave，并加上运行中 wave 尚未执行的 Attention op，选择候选请求预计最先完成的 DP。候选在某个 wave 内完成后，不再把该 DP 后续排空时间计入候选分数。两种策略分别以每个 uBatch 内的 `Σquery_tokens` 和 `Σquery_tokens²` 估算 Attention 工作；运行中的 Attention op 按已经经过的时间扣除进度。FFN 不作为加法项，因为 token uBatch 会隐藏 Attention/FFN 中较短的一侧；当前策略刻意用于先观察 Attention 主导区，不建模瓶颈翻转与流水 bubble。merged 使用其独立选择的策略。

`merged_wave_token_sum` 和 `merged_wave_token_square_sum` 用相同的 FIFO 装箱约束预测同步 merged waves。对每个候选 DP，评分依次最小化：加入后的全局 wave 数、所有全局 wave 的预计 drain work、从当前状态到候选完成 wave 的预计 work。每个全局 wave 的 work 是该 wave 中最重 DP batch 的 `Σquery_tokens` 或 `Σquery_tokens²`。因此在某个 DP 新建本地 batch、但该序号的全局 wave 已由其他 DP 存在时，不会把它误判为新增全局 wave。已经运行或完成的 wave 不会重排。

`vllm_queue_aware` 可让任一侧使用正常工作的 DPLB；两个 Prefill greedy 策略用于旧的即时 outstanding-token 对照；`round_robin` 严格轮询。

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
所有 DP query tokens 总和，再用 `global_query_tokens/dp_size` 查询当前 profile；
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
| `output.timeline_max_events` | `int|null` / `20000` | 最大时间线事件数，超过后截断并设置 `timeline_truncated=true`；`null` 表示不截断，WebUI 使用该模式。 |
| `output.include_requests` | `bool` / `true` | 是否返回每个请求的到达、DP、cache、完成时间和 TTFT。 |

## 3. CSV 格式

### 3.1 内置 MoonConv V4 Flash 长度

内置数据来自
[`ShwStone/moonconv-wildchat-v4-flash-prefill`](https://huggingface.co/datasets/ShwStone/moonconv-wildchat-v4-flash-prefill)
revision `284a2326bbef3d5107995f52e38eeee9d0ccdb45`。这里只从 arrival
JSONL 提取 `actual_input_length` 和 `base_arrival_offset_ms`。formal 三个窗口
各 512 条，screening 为 128 条；总输入长度 min/max 为 891/63778。没有包含
prompt、token IDs、request ID 或 Mooncake trace index。

原始时间戳是 bursty 的，例如 formal_0 的 511 个相邻间隔中有 459 个为零。
内置数据默认使用 `scaled_trace`，所以这些并发 burst 不变，只有非零时间尺度
随目标 QPS 等比例变化。这里的 QPS 是完整循环的平均到达率，不会把 trace
改造成固定间隔或泊松过程。

### 3.2 只有线上长度列表

```csv
input_length
512
8192
8192
32768
```

`input_length` 必填。重复行会保留，因此自然构成经验分布。

### 3.3 完整 trace

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
| `GET /api/defaults` | 返回默认配置、`default_afd_topology_id`、去重后的 `afd_topologies` 与 `merged_topologies`。每个 AFD 条目提供同 die 的默认 merged ID。 |
| `GET /api/length-datasets/{id}` | 返回内置数据集的 `arrival_time_ms,input_length` CSV；可用 ID 由 `/api/defaults` 的 `length_datasets` 给出。 |
| `POST /api/simulate` | Body 为带 `afd_topology_id`、`merged_topology_id` 的完整配置 JSON；后端校验等 die 后返回对比结果。 |
| `POST /api/sweep` | Body 为带 `afd_topology_id`、`merged_topology_id` 的完整配置 JSON；返回两种架构的 QPS 曲线和最大 SLO QPS。 |

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
- merged 的 routed/combine 以及按需求近似的 dispatch 使用 `global_query_tokens/dp_size`；combine-local/shared/SP 使用 `max(dp_query_tokens)`。`dp_size`、`tp_size` 和 `ep_size` 均来自加载的 profile spec。
- 不模拟 Decode、KV transfer、prefix cache 容量/淘汰算法、MTP、graph、prefix cache lookup 并发、HBM OOM、EPLB 或真实专家负载偏斜。
- DeepSeek-V4-Flash 的 256 个 routed experts 若不能被 EP 整除，会按 rank
  分配 `floor(256/EP)` 或 `ceil(256/EP)` 个本地 experts；msModeling 当前
  rank 0 trace 会落在较重一侧。collective analytic 公式则对任意 EP 使用连续
  的 `log2(EP)`，不会额外刻画非 2 的幂规模在真实通信算法中的补尾轮次。
- AFD DSV4 NPU 与 AFD chunked prefill 都是架构性能假设，不代表当前 afd-plugin 已完成对应 E2E 支持。

## 7. 测试

无额外测试依赖：

```bash
python -m unittest discover -s simulator/tests -v
```

测试覆盖 profile 插值/越界、CSV、Prefix Cache、固定/持续负载、chunked prefill、AFD 双 uBatch、合并屏障和 QPS 扫描复现性。
