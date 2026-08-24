# DeepSeek-V4-Flash 推理 Shape 全流程

> 当前 A3 测试实例：模型实际输入 16 tokens，生成 32 tokens。覆盖 Prefill、Decode、混合注意力、MoE、LM Head、MTP 数据通路及 P/D 分离语义。

## 1. 证据规则与符号

为避免把推导误认为 profiler 实测，本文统一标记：

- **[实测]**：rank 12 Ascend profiler 的 operator shape。
- **[配置]**：模型配置或启动参数。
- **[代码]**：当前 vLLM / vLLM-Ascend / DSV4 实现确定的数据流。
- **[推导]**：由代码和 TP/DP/EP 规则得到，尚未从 Decode operator CSV 直接截取。

| 符号 | 含义 | 当前值 |
|---|---|---:|
| `B` | 有效请求数 | 1 |
| `S` | 序列长度 | Prefill 16；Decode 每步 1 |
| `T` | 展平 token 数 | `B×S` |
| `H` | hidden size | 4096 |
| `C` | mHC residual streams | 4 |
| `L` | Transformer 层数 | 43 |
| `V` | vocabulary size | 129280 |
| `N_h` | attention heads | 64；TP rank-local 16 |
| `D_h` | attention head dim | 512 |
| `TP/DP/EP` | 并行度 | 4 / 4 / 16 |

## 2. 当前测试实例

| 项目 | 当前值 |
|---|---|
| 模型 | `DeepSeek-V4-Flash-w8a8-mtp` |
| 硬件 | Ascend A3，16 卡 |
| block size | 128 |
| `max_model_len` | 8192 |
| `max_num_batched_tokens` | 4096 |
| `max_num_seqs` | 4 |
| 原始随机 prompt | 12 tokens |
| Chat template 后实际输入 | 16 tokens |
| 请求输出 | 32 tokens |
| profile | eager A，rank 12，带 stack/memory/shape |

```text
原始 prompt 12 tokens
  → chat template 后 16 tokens
  → Prefill forward 一次，采样 y1
  → Decode forward 31 次，依次采样 y2...y32
```

没有提前 EOS 时，总 forward 次数为 `1+31=32`。最后的 `y32` 采样后不再送回模型，因此最终已处理 KV 长度是 `16+31=47`，不是 48。

## 3. 三种 shape 视角

### 3.1 请求逻辑视角

```text
Prefill: [B=1,S=16] → T=16
Decode : [B=1,S=1]  → T=1（每步一个有效 token）
```

### 3.2 mHC residual 视角

```text
普通 hidden [T_local,4096]
→ 4-stream residual [T_local,4,4096]

Prefill residual [4,4,4096] [实测]
Decode residual  [1,4,4096] [实测]
```

Prefill 的逻辑 token 数是 16，但 sequence parallel 在 TP=4 内将 residual token 分为每 rank 4 个。`[4,4,4096]` 的第一个 4 是 local tokens，第二个 4 才是 mHC streams。

### 3.3 Attention 计算视角

```text
Prefill residual [4,4,4096]
  → hc_pre [4,4096]
  → TP all-gather [16,4096]
  → local Q [16,16,512]

Decode residual [1,4,4096]
  → hc_pre [1,4096]
  → graph/TP 对齐后的物理 token 维度 4
  → local Q [4,16,512] [推导]
```

Decode attention 第一维的 4 是图执行/通信对齐的物理槽位，不代表有 4 个有效请求；当前只有 1 个有效 token。

## 4. 从文本到 mHC residual

```text
文本 prompt
→ 原始 token_ids [12]
→ chat template 后 input_ids [1,16]
→ 展平 token_ids [16]
```

全局 embedding table 为 `[129280,4096]`；TP=4 后每 rank 持有 `[32320,4096]`。

```text
Prefill: ids [16] → logical hidden [16,4096]
                   → SP local [4,4096]
                   → mHC [4,4,4096]

Decode : id [1]   → hidden [1,4096]
                   → mHC [1,4,4096]
```

## 5. 43 层混合注意力排布

```text
layer 0 : SWA
layer 1 : SWA
layer 2 : CSA，compression stride=4
layer 3 : HCA，compression stride=128
layer 4 : CSA
layer 5 : HCA
...
layer 40: CSA
layer 41: HCA
layer 42: SWA
```

即 `[SWA, SWA, (CSA-c4, HCA-c128)×20, SWA]`。

- **DSA** 是整个稀疏/混合 attention backend 的总称，不是第四种 layer。
- **SWA** 是滑窗 attention。
- **CSA** 使用 c4 压缩 KV、Lightning Indexer 和 TopK 稀疏选择。
- **HCA** 使用 c128 层级压缩状态。

每层公共接口：

```text
x [T_local,4,4096]
 → hc_pre(attention) [T_local,4096]
 → Attention
 → hc_post [T_local,4,4096]
 → hc_pre(FFN) [T_local,4096]
 → FFN/MoE
 → hc_post [T_local,4,4096]
```

因此层与层之间始终是 Prefill `[4,4,4096]`、Decode `[1,4,4096]`。

## 6. mHC Pre / Post

`npu_hc_pre_v2`：

```text
Prefill input [4,4,4096] [实测] → output [4,4096] [代码]
Decode input  [1,4,4096] [实测] → output [1,4096] [代码]
```

它用学习到的 mixing 系数把 4 条 residual stream 合成一个 H=4096 hidden。子层完成后：

```text
sub-layer output [T_local,4096]
+ residual state [T_local,4,4096]
→ hc_post updated state [T_local,4,4096]
```

## 7. Attention Q、Cache 与输出投影

全局 64 heads，TP=4 后每 rank 计算 16 heads：

```text
Prefill Q [16,16,512] [实测]
           │  │  └─ head_dim
           │  └──── local heads=64/4
           └─────── attention token 数

Decode Q [4,16,512] [推导，物理图槽位为 4]
```

物理 cache capacity：

```text
SWA KV cache  [10123,128,1,512] [实测]
CSA c4 state  [10123,8,2048]    [实测]
HCA c128 state [10123,32,1024]  [实测]
```

`10123` 是整个运行预分配的物理 block 容量，不是当前请求长度；请求通过 block table 只引用其中少量 block。

输出投影：

```text
local attention output [T_attn,16,512]
→ reshape [T_attn,2,4096]        # 8 个全局 output groups / TP4
→ wo_a [T_attn,2,1024]
→ flatten [T_attn,2048]
→ wo_b + TP reduce-scatter
→ [T_local,4096]
```

所以 Prefill `[16,16,512] → [4,4096]`；Decode `[4,16,512] → [1,4096]` [推导]，再由 hc_post 写回 4 streams。

## 8. SWA / CSA / HCA 的关键 shape

### 8.1 SWA
SWA（Sliding Window Attention）仍然是 causal attention，但每个 query 最多只读取最近 128 个有效 token，而不是读取从序列开头到当前位置的全部 KV。

#### 8.1.1 当前 SWA 的输入和位置

当前 layer 排布中，纯 SWA 位于：

```text
layer 0、layer 1、layer 42
```

CSA/HCA 层内部也包含局部 SWA 分支，但还会额外使用压缩 KV 或 Indexer 结果。纯 SWA 的 Prefill profiler 输入：

```text
Q                 [16,16,512]       [实测]
shared-KV cache   [10123,128,1,512] [实测]
Attention output  [16,16,512]       [代码]
```

其中：

- `16`（Q 第一维）：本次 Prefill 的 token 数。
- `16`（Q 第二维）：TP=4 后每 rank 的 local query heads。
- `512`：每个 query head 的维度。
- `10123`：运行时预分配的物理 KV block 总容量。
- cache 第二维 `128`：当前 KV block size。
- cache 第三维 `1`：shared-KV，即多个 query heads 共享一组 KV 表示。

需要特别区分：当前 `block_size=128`，SWA window 也恰好为 128，但两者语义不同：

```text
block_size=128：KV cache 的物理分页粒度
window=128    ：每个 query 最多允许参与 attention 的逻辑历史长度
```

这里只是两个配置值碰巧相同，不能把一个 cache block 直接等同于 SWA 数学定义。

#### 8.1.2 SWA 的完整计算流程

进入 SWA 子层前，rank-local residual 是：

```text
Prefill X [4,4,4096]
Decode  X [1,4,4096]
```

第一步，mHC 生成 mapping 并执行 `hc_pre`：

```text
Prefill [4,4,4096] → [4,4096]
Decode  [1,4,4096] → [1,4096]
```

第二步，TP all-gather 恢复 attention token 维度：

```text
Prefill [4,4096] → [16,4096]
Decode  [1,4096] → physical graph slots [4,4096] [推导]
```

第三步，归一化、量化并生成 Q 与 shared KV。

##### Q LoRA：down projection 和 up projection

当前 Q 不是用一个 `Linear(4096,64×512)` 直接生成，而是经过两级低秩 projection：

```text
hidden size      H=4096
Q LoRA rank      Rq=1024
全局 Q heads     64
TP               4
rank-local heads 64/4=16
head_dim         512
```

第一级 `q_a_proj` 是 Q down projection：

```text
X                        [16,4096]
W_q_a                    [1024,4096]   # 按 [out,in] 表示
X · W_q_aᵀ               [16,1024]
```

这里的“down”指特征维从 4096 压到低秩空间 1024：

```text
4096 → 1024
```

随后执行 Q LoRA 中间归一化/动态量化，shape 不变：

```text
q_low_rank               [16,1024]
→ RMSNorm / DynamicQuant [16,1024]
```

第二级 `q_b_proj` 将低秩表示升回所有 Q heads。全局输出维度是：

```text
64 heads × 512 = 32768
```

TP=4 按 head/output 维切分后，每 rank 只计算：

```text
16 local heads × 512 = 8192
```

所以 rank-local up projection 是：

```text
q_low_rank               [16,1024]
W_q_b_local              [8192,1024]
q_flat_local             [16,8192]
reshape                   [16,16,512]
```

完整 Q 路径：

```text
[16,4096]
→ q_a_proj / down_proj
[16,1024]
→ RMSNorm + DynamicQuant
[16,1024]
→ q_b_proj / TP-local up_proj
[16,8192]
→ reshape，8192=16×512
[16,16,512]
```

第一维 16 始终是 Prefill token 数；reshape 新出现的第二维 16 是当前 TP rank 的 Q head 数，不是又生成了 16 份 token。

##### W8A8 在 projection 中做什么

`W8A8` 表示量化矩阵乘中：

```text
W8：weight 使用 8-bit 表示，通常为 INT8
A8：activation 使用 8-bit 表示，通常为动态 INT8
```

概念计算为：

```text
x_int8 = quantize(x_fp, x_scale)
w_int8 = quantize(w_fp, w_scale)
acc     = matmul(x_int8, w_int8ᵀ)    # 通常用更高精度累加
output  = dequantize(acc, x_scale, w_scale)
```

因此 Q down projection 可以理解为：

```text
activation [16,4096] --DynamicQuant--> INT8 [16,4096]
weight     [1024,4096]                INT8 [1024,4096]
INT8 matmul / higher-precision accumulate
→ dequantized q_low_rank [16,1024]
```

Q up projection同理：

```text
activation [16,1024] --DynamicQuant--> INT8 [16,1024]
weight     [8192,1024]                INT8 [8192,1024]
→ q_flat_local [16,8192]
```

当前 profile 中与这条路径对应的主要算子包括：

```text
RmsNorm / DynamicQuant / RmsNormDynamicQuant
QuantBatchMatmulV3
npu::npu_quant_matmul
view / unflatten / reshape
```

W8A8 改变的是存储 dtype、带宽、矩阵乘内核和缩放/反量化过程，不改变上述逻辑 shape。具体 scale 是 per-token、per-channel 还是其他粒度，应以对应量化 linear 的配置和 kernel 参数为准。

##### shared KV 支路

与 Q LoRA 并行，当前 hidden 还会生成 shared K/V representation：
KV down projection 的思路与 Q down projection 相同：都把 `[T,4096]` 压到更小的 latent space。区别是 Q 随后还通过 `q_b_proj` 升维并拆成 16 个 TP-local query heads，而当前 KV latent 直接保持为 1 个 shared head、每 token 512 维，经过归一化/Partial RoPE 后写入 shared-KV cache：

`[T,4096] → KV down_proj → [T,512] → [T,1,512] → cache`。

```text
shared KV current tokens [16,1,512]
```

所以进入后续 Partial RoPE 和 cache 写入前，两个关键结果为：

```text
Q projection result      [16,16,512]
shared KV current tokens [16,1,512]
```

这里 64 个全局 query heads 在 TP=4 后变成 16 个 local heads；shared KV 的 head 数为 1。

第四步，对 Q/K 的部分维度执行位置编码：

```text
K-like shared representation [16,1,1,512]
Q representation             [16,1,16,512]
→ InplacePartialRotaryMul
```

`partial` 表示 512 维 head 中只有配置指定的 RoPE 子空间参与旋转，其余维度保持原有内容。

第五步，把当前 token 的 shared KV 写入 paged KV cache：

```text
current shared KV [16,1,512]
slot_mapping      [16]
physical cache    [10123,128,1,512]
→ ScatterNdUpdateV2
```

`slot_mapping` 将每个逻辑 token 映射到 `(physical block, offset)`。写入 cache 和“这个 token 是否参与当前 query 的 SWA”是两件事：前者负责保存状态，后者由 attention metadata/window 决定。

第六步，生成 attention metadata：

```text
seq_lens / query positions / block table / slot mapping
+ causal constraint
+ sliding window=128
→ SparseAttnSharedkvMetadata
```

它为核心 kernel 准备每个 query 可以访问的逻辑 KV 范围及物理 cache 映射。

第七步，融合计算局部 attention：

```text
Q [T_attn,16,512]
+ shared-KV cache [10123,128,1,512]
+ SWA metadata
→ score = QKᵀ × scale
→ causal/window mask
→ softmax
→ probability × V
→ output [T_attn,16,512]
```

这些 score、mask、softmax 和 value aggregation 在 `SparseAttnSharedkv` 中融合完成，不会在 profile 中表现成一串独立的 `MatMul → MaskedFill → Softmax → MatMul` PyTorch 算子。

第八步，执行 grouped O projection。这里必须把 `wo_b` 本地矩阵乘和 TP reduce-scatter 分开理解。

第一阶段：将当前 rank 的 16 个 heads 分为 2 个 output groups。每组包含 8 个 heads：

```text
attention output            [16,16,512]
reshape                     [16,2,8,512]
每组拼接 8×512             [16,2,4096]
```

第二阶段：每个 group 分别执行 `wo_a` down projection：

```text
wo_a weight                 [2,1024,4096]  # [group,out,in]
input                       [16,2,4096]
wo_a output                 [16,2,1024]
flatten local groups        [16,2048]
```

其中当前 TP rank 有 2 个 groups，所以 `2048=2×1024`。全局 8 个 groups 对应的概念输入宽度是 `8×1024=8192`。

第三阶段：当前 rank 单独执行自己的 `wo_b` weight shard。按照 `[out,in]` 表示：

```text
local wo_b input            [16,2048]
local wo_b weight           [4096,2048]
matmul 使用的 weightᵀ       [2048,4096]
local partial output        [16,4096]
```

矩阵乘明确写成：

```text
[16,2048] @ [2048,4096]
→ [16,4096]
```

因此 `wo_b` 本身只完成：

```text
2048 → 4096
```

它不会把 token 数从 16 变成 4。由于每个 TP rank 只拥有全局 8 个 groups 中的 2 个，四个 rank 分别得到四份：

```text
rank0 partial [16,4096]
rank1 partial [16,4096]
rank2 partial [16,4096]
rank3 partial [16,4096]
```

每一份都只是完整 O projection 的部分和。

第四阶段才执行 TP reduce-scatter。逻辑上可以拆成 reduce 和 scatter 两步：

```text
Reduce：
rank0 partial + rank1 partial + rank2 partial + rank3 partial
→ complete O output [16,4096]

Scatter（沿 token/sequence-parallel 维）：
complete O output [16,4096]
→ TP4 分片
→ 每 rank [4,4096]
```

实际运行时 reduce-scatter 是一个 collective，完整的 `[16,4096]` 不一定在某个 rank 上单独物化；但逻辑 shape 应按上述两步理解。

Prefill 的完整 O projection：

```text
[16,16,512]
→ group reshape [16,2,4096]
→ wo_a [16,2,1024]
→ flatten [16,2048]
→ wo_b local matmul [16,4096]        # partial result
→ TP reduce-scatter [4,4096]         # rank-local final result
```

Decode 对应为：

```text
physical attention output [4,16,512]
→ group reshape [4,2,4096]
→ wo_a [4,2,1024]
→ flatten [4,2048]
→ wo_b local partial [4,4096]
→ TP reduce-scatter
→ valid local output [1,4096] [推导]
```

最后 `hc_post` 把 attention 输出注回四条 residual stream：

```text
Prefill [4,4096] + old [4,4,4096] → [4,4,4096]
Decode  [1,4096] + old [1,4,4096] → [1,4,4096]
```

#### 8.1.3 `seq_lens ≤ 128` 时

对于 query 位置 `p`（从 0 开始），causal attention 本来只能访问 `[0,p]`。当有效上下文不超过 128 时：

```text
可访问 KV = [0,p]
参与 token 数 = p+1
```

因此在当前测试中：

```text
Prefill 后 context=16
最终 Decode context=47
47 < 128
```

所有历史 token 都落在窗口内，所以这一组 profile 中 SWA 的数学结果等价于普通 causal full attention；只是执行的仍然是 SWA/DSA kernel。

#### 8.1.4 `seq_lens > 128` 时会发生什么

设当前 query 的绝对位置为 `p`，window size `W=128`。SWA 只允许访问：

```text
start = max(0, p-W+1)
end   = p
可访问位置 = [start,end]
```

因此最多参与 128 个 token。

例如：

```text
p=127：访问 [0,127]，共 128 tokens
p=128：访问 [1,128]，共 128 tokens；token 0 被排除
p=200：访问 [73,200]，共 128 tokens；[0,72] 不参与该 query
```

要区分三类“长度”：

1. **逻辑序列长度**仍会继续增长，例如 4096、8192；位置编号不会重置。
2. **本次 SWA 有效 attention 长度**在超过阈值后固定不超过 128。
3. **物理 KV cache 中是否仍保留旧 token**由 vLLM cache manager、block 复用和其他 attention 层的需要决定；不能仅从 cache capacity `[10123,...]` 判断旧 token 已经被物理删除。

对于 DSV4 尤其重要：旧 token 即使不再参与某个 SWA layer，也可能已经进入 CSA/HCA 的 compressed state，供其他 layer 的远程历史建模使用。因此：

```text
SWA：只看最近 128 tokens
CSA/HCA：通过压缩状态保留更长距离的信息
```

超过 128 后的计算量变化：

- Prefill：每个 query 最多与 128 个 KV 计算，稳定阶段复杂度约为 `O(S×128)`，而不是 full attention 的 `O(S²)`。
- Decode：每一步只有一个有效 query，SWA attention 的有效 KV 数固定为最多 128，因此这部分单步计算量不会随完整 context 线性增长。
- Cache/metadata：序列位置、block table 和压缩状态仍需更新；所以整模型 TPOT 不会因为 SWA 固定窗口而完全与 context length 无关。

#### 8.1.5 当前 profile 中的 SWA 算子汇总

| 阶段 | Profile 算子/内核 | 主要输入或作用 | 是否 SWA 专属 |
|---|---|---|---|
| mHC 合并 | `HcPre` / `_C_ascend::npu_hc_pre_v2` | `[T_local,4,4096] → [T_local,4096]` | 否，所有子层共用 |
| Norm/量化 | `RmsNorm`、`DynamicQuant`、`RmsNormDynamicQuant` | 为量化 projection 准备 hidden | 否 |
| Q/KV projection | `QuantBatchMatmulV3`、`npu::npu_quant_matmul` | 生成 Q 和 shared KV | 否，但 shape 由 SWA 配置决定 |
| 位置编码 | `InplacePartialRotaryMul` | Q/K 的部分 RoPE 维度 | DSA attention 共用 |
| KV cache 写入 | `ScatterNdUpdateV2` / `_C_ascend::npu_scatter_nd_update_v2` | 写入 `[10123,128,1,512]` cache | shared-KV 路径 |
| SWA metadata | `SparseAttnSharedkvMetadata` | 构造 causal/window/cache 映射 | 纯 SWA 调用中关键 |
| 核心 attention | `SparseAttnSharedkv` / `_C_ascend::npu_sparse_attn_sharedkv` | 融合 score、mask、softmax、V 聚合 | DSA 三类 attention 共用 |
| 输出投影 | `TransposeBatchMatMul`、`QuantBatchMatmulV3`、`MatMulV2` 等 | grouped `wo_a/wo_b` 和 TP 通信前后处理 | 否 |
| mHC 写回 | `HcPost` / `_C_ascend::npu_hc_post` | attention 输出注回 4 streams | 否 |

`SparseAttnSharedkv` 在当前 profile 中共出现 1376 次：

```text
43 layers × (1 Prefill + 31 Decode) = 43 × 32 = 1376
```

这证明它是整个 DSA backend 的共享核心，而不是只在 3 个纯 SWA layer 中出现。区分具体 layer 类型要看传入参数：

- 纯 SWA：Q、shared-KV cache 和 window metadata 有效，compressed/indexer 输入为空。
- CSA：额外传入 c4 compressed KV 与 Indexer/TopK 选择结果。
- HCA：额外使用 c128 层级压缩状态。

当前纯 SWA 的 profiler record 中，核心 kernel 的主要非空输入是：

```text
Q               [16,16,512]
shared-KV cache [10123,128,1,512]
其他 compressed/indexer tensor 为空
```

所以本次 16→47 token 的 profile 可以确认 SWA 的短上下文执行路径，但不能测出超过 128 后 window mask、cache block 复用和长 context metadata 的真实性能。要验证该边界，至少需要增加 `seq_lens=127/128/129/256` 四组 case。

### 8.2 CSA：C4A、Lightning Indexer 与稀疏主 Attention

CSA 不是简单的串行链路：

```text
KV cache → compressor → Indexer → window → RoPE
```

更准确的结构是：Q/raw shared-KV 先完成 projection 和 RoPE；随后局部 SWA、C4 压缩记忆和轻量 Indexer 构成并行/协作的长程信息路径；最后由 metadata 和 `SparseAttnSharedkv` 融合执行真正的高维 attention。

#### 8.2.1 CSA 的计算流程、算子与 shape

##### 步骤 1：mHC 与 Attention 输入

```text
rank-local residual              [4,4,4096]
→ HcPre / npu_hc_pre_v2
rank-local hidden                [4,4096]
→ TP all-gather
Attention hidden                 [16,4096]
```

##### 步骤 2：生成 Q 与 raw shared KV

Q LoRA：

```text
hidden                           [16,4096]
→ q_a_proj/down_proj
q low-rank                       [16,1024]
→ RMSNorm + DynamicQuant
                                 [16,1024]
→ q_b_proj，TP-local
q flat                           [16,8192]
→ reshape，8192=16×512
Q                                [16,16,512]
```

shared KV：

```text
hidden                           [16,4096]
→ KV down projection
shared KV                        [16,512]
→ reshape
                                 [16,1,512]
```

相关算子：

```text
RmsNorm / DynamicQuant / RmsNormDynamicQuant
QuantBatchMatmulV3
npu::npu_quant_matmul
view / reshape / unflatten
```

##### 步骤 3：RoPE 在 cache 写入与 TopK 选择之前执行

```text
Q                                [16,1,16,512]
K-like shared representation     [16,1,1,512]
→ InplacePartialRotaryMul
```

RoPE 作用于 Q/K 向量的配置子空间，不作用于 Indexer 输出、TopK indices 或 softmax 权重。

随后把当前 raw shared KV 写入 paged cache：

```text
current shared KV                [16,1,512]
slot_mapping                     [16]
physical raw KV cache            [10123,128,1,512]
→ ScatterNdUpdateV2
```

##### 步骤 4：C4A main compressor 构造 4:1 长期记忆

Profiler 直接观测：

```text
hidden input                     [16,4096]
wkv / wgate                      [1024,4096]
compressor state cache           [10123,8,2048]
absolute position embedding      [4,1024]
norm weight                      [512]
compressed cos/sin               [5,64]
```

概念 projection：

```text
candidate = X · Wkvᵀ             [16,1024]
gate      = X · Wgateᵀ           [16,1024]
```

压缩器按照连续 4 个原始 token 更新一个逻辑 compressed group：

```text
token 0...3                      → C4_0
token 4...7                      → C4_1
token 8...11                     → C4_2
token 12...15                    → C4_3
```

`APE [4,1024]` 用于区分一个 c4 group 内 offset 0、1、2、3。逻辑 compressed position 数近似为：

```text
N_c4 ≈ ceil(L/4)
```

`state cache [10123,8,2048]` 是融合 compressor 的物理累积/packing 布局，不能把第二维 8 直接解释成逻辑 c4 position 数。

相关算子：

```text
CompressorMetadata
Compressor
RmsNorm
InplacePartialRotaryMul           # compressed position 的 RoPE 子空间
```

##### 步骤 5：Indexer 使用独立的低维检索表示

CSA 还维护一条更便宜的 Indexer compressor 路径。Profiler 直接观测：

```text
hidden input                     [16,4096]
wkv / wgate                      [256,4096]
Indexer compressor state         [10123,8,512]
absolute position embedding      [4,256]
norm weight                      [128]
compressed cos/sin               [5,64]
```

Lightning Indexer 的输入：

```text
Indexer query                    [16,64,128]
Indexer key cache                [10123,128,1,128]
weights                          [16,64]
query scales                     [16,64]
key scales                       [10123,128,1]
block table                      [1,16]
TopK                             1024
```

它在 128 维检索空间中计算相关性：

```text
index_score[i,h,j]
  = index_Q[i,h,:] · index_K[j,0,:]
```

概念 score 和 TopK 输出：

```text
index scores                    [T_query,64,N_candidates]
selected indices                [T_query,64,K]
selected scores                 [T_query,64,K]
K                               ≤ 1024
```

实际融合 kernel 不一定显式物化这些大 tensor。Profile 能确认 query/key shape 和 TopK 标量，但 TopK indices 的准确物理布局要以 kernel 接口为准。

相关算子：

```text
VllmQuantLightningIndexerMetadata
VllmQuantLightningIndexer
```

main c4 compressor 与 Indexer compressor 的作用不同：

- main c4 compressor 生成供 C4A 使用的 512 维长期压缩 KV。
- Indexer compressor/Indexer 生成 128 维检索表示及 sparse indices。
- 两者不是“先把同一个 tensor 压缩，然后原封不动交给 Indexer”的单一串行操作。

##### 步骤 6：构造最终 Attention 候选集合

对于绝对 query 位置 `p`，SWA 最近窗口为：

```text
recent = [max(0,p-127),p]
|recent| ≤ 128
```

Indexer 返回远端 TopK positions。Metadata 需要处理：

```text
causal 越界
padding/无效 query
TopK 与 recent window 的重复位置
block table 到物理 cache slot 的映射
C4 compressed positions 的有效范围
```

可以把最终信息来源概念化为：

```text
recent raw KV（SWA，最多128）
∪ relevant remote KV（Indexer TopK，最多1024）
∪ C4 compressed KV（约L/4个逻辑 summaries）
```

但 `SparseAttnSharedkv` 会融合读取这些来源，不一定在内存中先构造一个真正的 concat tensor。

##### 步骤 7：加入 learnable attention sink

Profiler 中可看到 per-head sink 输入：

```text
sink                            [1,64]
```

对 head `h`，softmax 概念上变为：

```text
softmax([real_attention_logits, sink_h])
```

sink 对应的 value 视为零，因此它只吸收概率质量：

```text
O_h = Σⱼ p_j V_j + p_sink×0
```

这让某个 head 在当前 compressed/selected KV 都不相关时选择“什么也不读取”。它不是 mHC Sinkhorn，也不同于 Hamming/Indexer 配置里的整数 `sink=1`。

##### 步骤 8：执行完整 512 维 sparse attention

```text
Q                               [16,16,512]
raw shared-KV cache             [10123,128,1,512]
C4 compressed state/cache       物理布局由 compressor 管理
Indexer selected indices        logical [T,64,K]
→ SparseAttnSharedkvMetadata
→ SparseAttnSharedkv
Attention output                [16,16,512]
```

逻辑公式：

```text
score[i,h,j]
  = Q[i,h,:] · K_selected[j,0,:] / sqrt(512)

P[i,h,:]
  = softmax([score + causal/window mask, sink_h])

O[i,h,:]
  = Σⱼ P[i,h,j] · V_selected[j,0,:]
```

当前 profile 中 `SparseAttnSharedkv` 共出现：

```text
43 layers × 32 forwards = 1376 次
```

说明它是 SWA、CSA、HCA 共用的 DSA 核心；具体模式由 compressed/indexer 输入是否有效决定。

##### 步骤 9：grouped O projection 与 mHC 写回

```text
Attention output               [16,16,512]
→ group reshape                [16,2,4096]
→ wo_a                         [16,2,1024]
→ flatten                      [16,2048]
→ local wo_b                   [16,4096]   # TP partial
→ TP reduce-scatter            [4,4096]
→ HcPost + old residual
next residual                  [4,4,4096]
```

#### 8.2.2 CSA 的计算复杂度

定义：

```text
L      = seq_len
W      = SWA window = 128
K      = Indexer TopK = 1024
C      = c4 compression ratio = 4
D      = full attention dim = 512
d_idx  = Indexer dim = 128
N_cand = Indexer 实际扫描的 candidate 数
```

Compressor：

```text
Prefill：O(L)
Decode ：每步 O(1) 状态更新
```

昂贵的 512 维 selected attention：

```text
Prefill：O(L×(W+K)×D)
Decode ：O((W+K)×D)
```

当 W 和 K 固定时，这一部分 Prefill 对 L 近似线性，Decode 单步近似定长。

但 Indexer 为了找到 TopK，仍需对 candidate 计算低维 score：

```text
Prefill：O(L×N_cand×d_idx)
Decode ：O(N_cand×d_idx)
```

如果候选空间按 c4 压缩到约 `L/4`：

```text
Prefill Indexer：O(L²/4×d_idx)
Decode Indexer ：O(L/4×d_idx)
```

如果某一实现扫描 raw positions，则应把 `L/4` 换成 `L`。仅凭 operator CSV 无法确认 kernel 内部每个阶段的精确 candidate cardinality，因此不能把整个 Indexer 宣称为严格 O(1)。

C4A 如果对全部 c4 summaries 做 MQA：

```text
Prefill：O(L×(L/4)×D)
Decode ：O((L/4)×D)
```

实际融合实现可能对范围进一步约束，但 TopK 固定本身不能证明 C4A/Indexer 的检索成本定长。

因此 CSA 的准确复杂度结论是：

- full-dim、真正读取 V 的主 sparse attention 已从 `O(L²D)` 降为约 `O(L(W+K)D)`。
- Decode 的高成本主 attention 近似固定为最多 `128+1024` 个 positions。
- C4 compressor 是线性/增量的。
- Indexer 搜索和 C4A coarse attention 仍可能随 L 增长，并包含低维或压缩后的二次项。
- 所以“CSA 主 attention 近似定长”成立，但“整个 CSA 严格定长”不成立。

#### 8.2.3 CSA 的数学直觉

CSA 把长上下文问题拆成三个分辨率：

```text
最近历史：SWA 保留最多128个原始token，分辨率最高
中远历史：C4A 用4:1 summary保留连续语义
关键远端：Indexer 从长历史中挑最多1024个相关位置
```

可以类比为：

```text
SWA      = 眼前的高清画面
C4A      = 每4帧做一次摘要的时间轴
Indexer  = 根据当前问题检索最相关的旧帧
sink     = 没有相关内容时允许“不读取”
```

数学上，它避免对所有 `(query,key)` 组合都执行昂贵的 512 维 attention 和 V 聚合，而是先用压缩表示和 128 维 Indexer 做低成本筛选，再把高精度算力集中在局部窗口和少量 TopK 上。

---

### 8.3 HCA：C128A 层级压缩注意力

HCA 的基本结构是：

```text
HCA = SWA 最近原始 KV + C128A 超长程粗粒度 KV
```

HCA 不依赖 Lightning Indexer TopK。它直接以非常低的密度保存和读取长历史 summary。

#### 8.3.1 HCA 的计算流程、算子与 shape

##### 步骤 1：Q、raw shared KV 和 SWA

前半段与 CSA 相同：

```text
mHC residual                    [4,4,4096]
→ HcPre                         [4,4096]
→ TP all-gather                 [16,4096]

Q path:
[16,4096] → [16,1024] → [16,8192] → [16,16,512]

shared KV path:
[16,4096] → [16,512] → [16,1,512]

Q/K Partial RoPE
→ raw shared-KV cache           [10123,128,1,512]
```

最近最多 128 个 raw KV 由 SWA 提供高分辨率局部信息。

##### 步骤 2：C128 compressor 更新层级状态

Profiler 直接观测：

```text
hidden input                    [16,4096]
wkv / wgate                     [512,4096]
compressor state cache          [10123,32,1024]
absolute position embedding     [128,512]
norm weight                     [512]
compressed cos/sin              [1,64]
```

概念 projection：

```text
candidate = X · Wkvᵀ            [16,512]
gate      = X · Wgateᵀ          [16,512]
```

每连续 128 个 token 构成一个逻辑 summary：

```text
token 0...127                   → C128_0
token 128...255                 → C128_1
token 256...383                 → C128_2
...
```

逻辑 compressed position 数：

```text
N_c128 ≈ ceil(L/128)
```

`APE [128,512]` 区分 chunk 内 offset 0...127。`state cache [...,32,1024]` 是融合算子的物理累积/packing 布局，不能把 32 直接解释成固定的 logical summary 数。

相关算子：

```text
CompressorMetadata
Compressor
RmsNorm
InplacePartialRotaryMul          # compressed position encoding
```

##### 步骤 3：causal 地暴露已完成 summary

对当前 query，只能使用不包含未来 token 的 C128 state。当前尚未完成的 128-token chunk 继续更新 partial compressor state，局部细节由 SWA 覆盖。

例如位置 `p=200`：

```text
C128_0 总结 token 0...127，可以作为已完成长期记忆
当前 chunk token 128...200 尚未完成
最近原始细节由 SWA [73...200] 提供
```

当前测试 `L=16<128`：

```text
完整 C128 group 数 = 0
```

所以 profiler 能观察 compressor/state update，但不能评估多个 C128 summaries 参与 attention 时的真实性能。

##### 步骤 4：C128A 使用 MQA 读取压缩历史

逻辑输入：

```text
Q                               [T_query,16,512]
C128 K/V                        [N_c128,1,512]
```

所有 16 个 local Q heads 共享同一组 C128 KV：

```text
score128[i,h,j]
  = Q[i,h,:] · K128[j,0,:] / sqrt(512)

scores                          [T_query,16,N_c128]
output                          [T_query,16,512]
```

同样可以附加 per-head learnable sink：

```text
softmax([C128 logits, sink_h])
```

如果当前 query 与任何粗粒度 summary 都不相关，sink 可以吸收概率质量。

##### 步骤 5：融合 SWA 与 C128A 并输出

概念信息集合：

```text
recent raw KV                   ≤ 128 positions
C128 compressed KV              ≈ L/128 positions
```

核心算子：

```text
SparseAttnSharedkvMetadata
SparseAttnSharedkv
```

其融合执行 score、causal mask、sink softmax 和 V 聚合：

```text
Q                               [T_query,16,512]
→ SWA + C128A
Attention output                [T_query,16,512]
```

HCA 没有 Lightning Indexer/TopK 阶段。

随后与其他 attention 类型相同：

```text
Attention output                [16,16,512]
→ grouped wo_a                  [16,2,1024]
→ local wo_b partial            [16,4096]
→ TP reduce-scatter             [4,4096]
→ HcPost
next residual                   [4,4,4096]
```

#### 8.3.2 HCA 的计算复杂度

定义：

```text
L = seq_len
W = SWA window = 128
C = C128 compression ratio = 128
D = attention dim = 512
N_c128 ≈ L/128
```

Compressor：

```text
Prefill：O(L)
Decode ：每步 O(1) 增量更新
```

Decode 单步 Attention：

```text
O((W + L/128)×D)
```

它随 L 线性增长，但斜率只有 full attention 的约 `1/128`。当前 `max_model_len=8192` 时：

```text
最大 C128 summaries ≈ 8192/128 = 64
最大读取范围         ≈ 128 raw + 64 compressed = 192
```

最后一个 token 相比 full attention 的 8192 个 KV，attention 范围缩小约：

```text
8192/192 ≈ 42.7 倍
```

Prefill 总复杂度：

```text
SWA    ：O(L×W×D)
C128A  ：O(L×(L/128)×D)
合计   ：O(L×128×D + L²/128×D)
```

严格来说仍有缩小 128 倍的二次项。考虑 causal 平均历史，在 `L=8192` 时，粗略 pair 数为：

```text
SWA pairs     ≈ 8192×128 = 1,048,576
C128A pairs   ≈ 8192×32  =   262,144
```

在当前长度上限内，SWA 固定窗口项更大，所以整体表现会很接近以 L 为主的线性增长；但从渐近复杂度上不能把 Prefill 写成严格 O(L)。

HCA 的准确复杂度结论：

- Decode：`O((128+L/128)D)`，增长非常慢且当前配置下最多读取约 192 positions。
- Prefill：固定窗口线性项 + 缩小 128 倍的二次项。
- 相比 full attention `O(L²D)`，计算和访存都显著降低。

#### 8.3.3 HCA 的数学直觉

HCA 使用两种时间分辨率：

```text
最近 128 tokens：保留原始 KV，像高清短期记忆
更早的历史     ：每 128 tokens 压成一个 summary，像低帧率长期记忆
```

它假设：

- 近处 token 的词法、语法和局部依赖需要精确表示。
- 很远历史通常只需要保留主题、状态和宏观语义。
- 远端若确实需要逐 token 精确检索，则交给其他 CSA layer 的 Indexer/TopK 路径补充。

因此交替的 CSA/HCA layers 形成互补：

```text
CSA：4:1 中分辨率记忆 + 内容相关 TopK 精确检索
HCA：128:1 超低成本全局摘要
SWA：所有 layer 都保留最近窗口的高分辨率信息
```

可以类比为：

```text
SWA   = 最近几秒的高清视频
C4A   = 每4帧保存一次的中分辨率记录
C128A = 每128帧生成一张全局摘要图
Indexer = 按当前问题从旧记录中搜索关键帧
```

HCA 的价值不是精确恢复远端每个 token，而是用最多约 64 个 summary 为 8192-token 上下文提供廉价、始终可达的全局背景。

## 9. FFN 与 MoE：Hash Router、Learned Router、EP Dispatch 和 Expert MLP

每层 Attention 完成后，mHC 再执行一次 pre/post：

```text
residual                       [T_local,4,4096]
→ HcPre
MoE input x                    [T_local,4096]
→ Router + Routed/Shared Experts
MoE output y                   [T_local,4096]
→ HcPost
next residual                  [T_local,4,4096]
```

当前 Prefill rank 12：

```text
T_local=4
x                               [4,4096]
```

Decode 单请求：

```text
T_local=1
x                               [1,4096]
```

### 9.1 MoE 的统一数学形式

当前模型配置：

```text
全局 routed experts             E=256
每 token 激活 experts           K=6
Expert Parallel                 EP=16
每 EP rank local experts        256/16=16
expert hidden size              4096
expert intermediate size        2048
shared expert 数                1
```

无论 expert IDs 来自 Hash Router 还是 Learned Router，最终输出都可以写成：

```text
y_t = f_shared(x_t)
    + Σ(k=1...6) w_tk · f_{e_tk}(x_t)
```

其中：

- `e_tk`：token `t` 选择的第 `k` 个 routed expert ID。
- `w_tk`：对应 combine weight。
- `f_e`：第 `e` 个 expert 的 SwiGLU MLP。
- `f_shared`：每个 token 都执行的 shared expert，不经过 TopK 路由。

关键点是：虽然模型拥有 256 个 routed experts，一个 token 只执行其中 6 个，而不是执行全部 256 个。

### 9.2 前 3 层 Hash MoE

Hash MoE 的核心区别是：expert IDs 由 token ID 查表得到，不需要根据 hidden 做 `4096→256` 的 router projection。

#### 9.2.1 Hash 路由流程和 shape

当前 Prefill rank-local token：

```text
token IDs                       [4]
hash expert table               [129280,6]
```

查表：

```text
expert_ids[t,:] = hash_table[token_id[t],:]
```

Shape：

```text
token IDs                       [4]
hash table                      [129280,6]
→ selected expert IDs           [4,6]
```

例如：

```text
token_id=1234
hash_table[1234] = [7,31,48,106,192,233]
```

则该 token 固定发送给这 6 个 experts。相同 token ID 在相同 Hash layer 中会得到相同 expert 集合，因此路由具有确定性。

Profiler 中 Hash 路由核心输入：

```text
router affinity/workspace       [4,256]
token IDs                       [4]
hash table                      [129280,6]
```

对应算子：

```text
_C_ascend::moe_gating_top_k_hash
aclnnMoeGatingTopKHash
```

这里的 `[4,256]` 是融合 selector 使用的 affinity/workspace 形状；真正决定 Hash expert IDs 的直接证据是 `[4]` token IDs 和 `[129280,6]` table。Operator CSV 不记录算子输出值，因此 Hash combine weight 的精确归一化规则不能只凭 shape 推断；可以确定的是 expert IDs 不来自 hidden 的 learned top-k logits。

Decode 对应：

```text
token IDs                       [1]
selected expert IDs             [1,6]
```

#### 9.2.2 Hash MoE 的数学直觉

Hash Router 相当于一个预先学习/构造好的词表到专家映射：

```text
vocabulary token → 6 个固定专家
```

优点：

- 不需要每 token 执行大 router matmul。
- 相同词天然落到相同专家，行为稳定。
- 路由延迟低，容易形成稳定的 expert specialization。

代价：

- 路由只直接看到 token ID，看不到当前上下文 hidden。
- 同一个词在不同语境中仍先落到同一组 experts。
- 高频 token 可能天然造成热点，需要训练策略或运行时负载均衡处理。

因此前 3 层使用 Hash MoE，可以把较浅层、偏词法的特征以很低的 router 成本分发给专家；深层再改用上下文相关的 Learned Router。

### 9.3 后续 Learned MoE

Learned MoE 根据当前 token hidden 动态计算 256 个 expert affinity，因此同一个 token ID 在不同上下文、不同层可以选择不同 experts。

#### 9.3.1 Router projection

Profiler 直接观测到：

```text
MoE hidden                      [4,4096]
router weight                   [256,4096]
```

线性 projection：

```text
router_logits = x · W_routerᵀ

[4,4096] @ [4096,256]
→ [4,256]
```

profile 对应调用：

```text
aten::linear
aten::matmul
aclnnMatmul
```

融合 selector 的 learned-routing 输入：

```text
router affinity/logits          [4,256]
expert correction bias          [256]      [实测出现]
token ID/hash table             empty
```

而 Hash 路径是：

```text
router affinity/workspace       [4,256]
expert correction bias          empty
token IDs                       [4]
hash table                      [129280,6]
```

两条路径最终都进入当前 Ascend 融合 selector，因此 profiler 中都可能显示 `moe_gating_top_k_hash` 这个通用算子名；必须看非空输入才能区分 Hash 与 Learned 路由。

#### 9.3.2 Top6 选择和权重

概念流程：

```text
router logits                   [4,256]
→ scoring activation / correction bias
expert scores                   [4,256]
→ TopK(K=6)
expert IDs                      [4,6]
selected weights                [4,6]
```

DeepSeek 风格 learned routing 中，correction bias 可以参与“选择哪些 experts”，而最终 combine weight 通常由未加选择偏置的原始 affinity 归一化得到；具体 activation、group-limited TopK 和 scaling 顺序应以对应版本 selector 参数为准。当前 profile 能确认 `[256]` correction bias 和 Top6 路径，但不能从 CSV 恢复具体数值。

数学上：

```text
s_t = Router(x_t)               ∈ R^256
E_t = TopK(s_t,6)
w_t = Normalize(s_t[E_t])       ∈ R^6
```

Learned Router 的直觉是：

```text
当前 hidden 表示“这个 token 在此刻需要什么能力”
Router 根据语义把它送到最适合的 6 个专家
```

### 9.4 EP16 Dispatch：为什么 4 个 token 会变成 24，再变成 44

路由完成后，rank 12 最初有：

```text
local tokens                     4
experts per token                6
expert assignments              4×6=24
```

每个 token 会复制/引用 6 次，并附带：

```text
hidden                          [4096]
expert ID
combine weight
original token index
```

逻辑 dispatch buffer：

```text
[4,6,4096]
→ flatten assignments
[24,4096]
```

expert owner 由 EP16 决定。全局 256 experts 平均分布：

```text
rank 0  owns experts   0...15
rank 1  owns experts  16...31
...
rank 15 owns experts 240...255
```

每个 rank 按 destination EP rank 排序 token assignments，然后执行 All-to-All：

```text
local assignment buffer         [24,4096]
→ token permute / dispatch
→ HCCL All-to-All-V
→ received expert tokens        [N_recv,4096]
```

`N_recv` 由所有 EP ranks 路由到当前 rank 的 token 数决定，不等于本地原始 token 数。rank 12 profile 中出现：

```text
[44,4096]
[8,4096]
[10,4096]
[15,4096]
```

所以 `[44,4096]` 的含义是：在那个 layer/step，当前 rank 的 16 个 local experts 合计收到 44 个 expert-token assignments。它不是 batch size，也不是模型固定 shape。

相关 dispatch 算子/通信：

```text
MoeDistributeDispatchV2
npu_moe_token_permute
HcclAlltoAllV / all_to_all_single
```

### 9.5 Local Expert MLP：Grouped MatMul + SwiGLU

EP dispatch 后，当前 rank 只执行自己持有的 16 个 experts。

输入按 local expert 排序：

```text
expert 0 tokens
expert 1 tokens
...
expert 15 tokens
```

用 group list 描述每个 expert 对应的 token 区间，然后一次 grouped kernel 执行 16 个不同 MLP。

#### 9.5.1 Gate/Up projection

Profiler 示例：

```text
routed hidden                   [44,4096]
local expert gate/up weights    [16,4096,4096]
local expert scales             [16,4096]
group information               dynamic，覆盖44个assignments
local expert count              16
```

权重最后的 4096 实际包含：

```text
gate dim 2048 + up dim 2048 = 4096
```

对属于 expert `e` 的 token：

```text
z_e = x_e · W_gate_up[e]ᵀ

[N_e,4096] @ [4096,4096]
→ [N_e,4096]
→ split
  gate [N_e,2048]
  up   [N_e,2048]
```

#### 9.5.2 SwiGLU 与量化

```text
activated = SiLU(gate) ⊙ up

[N_e,2048] ⊙ [N_e,2048]
→ [N_e,2048]
```

当前融合算子将 grouped gate/up matmul、SwiGLU 和后续量化合并：

```text
_C_ascend::grouped_matmul_swiglu_quant_weight_nz
aclnnGroupedMatmulSwigluQuantWeightNZ
```

因此 profile 中不会分别看到每个 expert 的 16 次独立 matmul 和 SiLU。

#### 9.5.3 Down projection

每个 expert 再把 intermediate 2048 投回 hidden 4096：

```text
activated                       [N_e,2048]
W_down[e]                       [2048,4096]  # profiler物理权重布局
→ output                        [N_e,4096]
```

rank 12 profile 直接看到 local down weight 物理 shape：

```text
[16,2048,4096]
```

对应算子：

```text
npu::npu_grouped_matmul
aclnnGroupedMatmulWeightNz
```

合并 16 个 local experts 后，当前 rank 仍得到：

```text
local expert outputs            [N_recv,4096]
例如                           [44,4096]
```

### 9.6 Reverse All-to-All 与 Top6 Combine

Expert MLP 完成后，需要将每份 expert output 送回原 token 所在 rank：

```text
local expert outputs            [N_recv,4096]
→ expert-side unpermute
→ reverse HCCL All-to-All-V
→ original assignments          [T_local×6,4096]
```

当前 Prefill 示例恢复为：

```text
[24,4096]
→ reshape/group by original token
[4,6,4096]
```

然后按 router weight 合并：

```text
y_routed[t,:]
  = Σ(k=1...6) w[t,k] · y_expert[t,k,:]
```

Shape：

```text
expert outputs                  [4,6,4096]
combine weights                 [4,6]
→ weighted reduce on K dimension
routed result                   [4,4096]
```

相关算子：

```text
npu_moe_token_unpermute
MoeDistributeCombineV2
HcclAlltoAllV
```

最后加入 shared expert：

```text
shared result                   [4,4096]
routed result                   [4,4096]
→ add
MoE output                      [4,4096]
→ HcPost
next residual                   [4,4,4096]
```

### 9.7 Decode 的 shape

单请求 Decode rank-local token 数为 1：

```text
MoE input                       [1,4096]
Hash IDs 或 learned logits      [1,6] / [1,256]
Top6 assignments                [1,6]
flatten dispatch                [6,4096]
→ EP All-to-All
received tokens                 [N_recv_decode,4096]   # 动态
→ local grouped experts
→ reverse All-to-All
returned assignments            [6,4096]
→ weighted Top6 combine
MoE output                      [1,4096]
→ HcPost
residual                        [1,4,4096]
```

Decode 的 token 数很少，expert matmul 本身可能很小；All-to-All latency、同步和负载不均更容易成为 TPOT 瓶颈。

### 9.8 计算复杂度与性能直觉

#### Hash Router 与 Learned Router

Hash Router：

```text
查表成本约 O(T×K)
不需要 O(T×H×E) router matmul
```

Learned Router：

```text
router projection O(T×H×E)
= O(T×4096×256)
```

但相对 Expert MLP 和 EP 通信，router matmul通常不是最大成本。

#### Expert 计算量

每 token 只计算 K=6 个 routed experts：

```text
O(T×K×H×I)
```

其中：

```text
H=4096
I=2048
K=6
```

它与总 expert 数 E=256 不呈线性关系；增加总 experts 主要增加参数容量和路由选择空间，不会让每 token 执行全部 experts。

#### EP 通信量

Dispatch/Combine 数据量近似：

```text
O(T×K×H)
```

当前 Prefill 每 rank：

```text
4 tokens × 6 copies × 4096 elements
```

Decode 每 rank：

```text
1 token × 6 copies × 4096 elements
```

小 token 数下带宽未必饱和，通信启动、同步和跨 rank 尾部延迟更关键。

#### 负载不均

理想情况下，各 rank 接收接近相同数量的 expert assignments；实际由路由分布决定：

```text
step latency ≈ 最慢/最满 expert rank 的完成时间
```

所以 `N_recv=44/8/10/15` 的动态变化会直接影响 grouped matmul shape、通信量和尾部延迟。

当前 op statistic 中：

```text
MoeDistributeDispatchV2 约占 82.1% device op time
MoeDistributeCombineV2  约占  9.2%
```

这说明本次 profile 的主要设备时间集中在 MoE dispatch/combine，而不是 expert matmul本身。不过该 profile 带 stack/memory instrumentation，绝对比例不能直接当作无侵入性能结果，应该用 ops-only profile 复核。

### 9.9 当前 profile 的 AFD force-load-balance 影响

路由与 expert kernel 调用栈直接包含：

```text
afd_plugin/compat/patches/npu/force_load_balance.py
```

涉及位置包括：

```text
expert selector apply
expert MLP / communication apply
```

因此要区分两类事实：

- `[4,256]`、`[4]`、`[129280,6]`、`[16,4096,4096]`、`[16,2048,4096]` 等模型/算子 shape 是有效的。
- `[44,4096]`、`[8,4096]` 等具体 `N_recv` 是该次实际执行 shape，但可能已经受到 AFD force-load-balance patch 影响，不能直接代表原生 DSV4 learned/hash routing 的自然负载分布。

要研究原生路由质量、expert 热点和 load balance，需要单独做两组对照：

```text
A：关闭 force-load-balance，记录自然 expert histogram
B：开启 force-load-balance，记录修改后的 histogram
```

并同时比较：

```text
每层/每rank expert-token counts
max/mean load ratio
Dispatch/Combine 时间
Grouped expert kernel 时间
TTFT/TPOT
```

## 10. Final hidden、LM Head 与 MTP buffer

43 层后：

```text
Prefill residual [4,4,4096]
Decode residual  [1,4,4096]
```

最终 mHC head：

```text
Prefill [4,4,4096] → [4,4096]
Decode  [1,4,4096] → [1,4096]
```

Prefill 对每个请求选择最后一个有效 token hidden，LM head 输入为 `[1,4096]`。TP vocab logits 每 rank 是 `[1,32320]`，逻辑全局 logits 是 `[1,129280]`，采样得到 `next_token_id [1]`。

模型还可将合并前的 mHC hidden 写入 MTP buffer：

```text
[T,4,4096] → flatten [T,16384]
预分配 buffer [4096,16384]
```

buffer 第一维 4096 对应 `max_num_batched_tokens`，不是当前请求实际 token 数。

## 11. Prefill 端到端 shape 表

| 顺序 | 阶段 | 输入 | 输出 | 说明 |
|---:|---|---|---|---|
| 1 | Tokenizer/template | 文本 / `[12]` | `[1,16]` | 实际输入 16 tokens |
| 2 | TP Embedding | `[16]` | logical `[16,4096]` | rank vocab `[32320,4096]` |
| 3 | SP scatter | `[16,4096]` | `[4,4096]` | TP=4 |
| 4 | mHC 初始化 | `[4,4096]` | `[4,4,4096]` | 4 streams |
| 5 | Attention hc_pre | `[4,4,4096]` | `[4,4096]` | 输入实测 |
| 6 | TP all-gather | `[4,4096]` | `[16,4096]` | 完整 token 维 |
| 7 | Q projection | `[16,4096]` | `[16,16,512]` | local 16 heads |
| 8 | SWA/CSA/HCA | Q + caches | `[16,16,512]` | layer 类型不同 |
| 9 | O projection/RS | `[16,16,512]` | `[4,4096]` | 回到 local tokens |
| 10 | Attention hc_post | residual + output | `[4,4,4096]` | 写回 streams |
| 11 | FFN hc_pre | `[4,4,4096]` | `[4,4096]` | 每层第二次 hc_pre |
| 12 | Router | `[4,4096]` | `[4,256]`, top6 | Hash/learned |
| 13 | EP dispatch/experts | local 4×top6 | dynamic | routed tokens 动态 |
| 14 | EP combine | routed outputs | `[4,4096]` | 加权聚合 |
| 15 | FFN hc_post | residual + output | `[4,4,4096]` | 43 层重复 |
| 16 | Final mHC head | `[4,4,4096]` | `[4,4096]` | 合并 streams |
| 17 | Select last token | sequence hidden | `[1,4096]` | 每请求最后 token |
| 18 | LM head | `[1,4096]` | local `[1,32320]` | global `[1,129280]` |
| 19 | Sampler | logits | `[1]` | 产生 y1 |

## 12. Decode 单步 shape 表

| 顺序 | 阶段 | 输入 | 输出 | 证据 |
|---:|---|---|---|---|
| 1 | 输入 token | `[1]` | `[1]` | 逻辑 |
| 2 | TP Embedding | `[1]` | `[1,4096]` | 代码 |
| 3 | mHC residual | `[1,4096]` | `[1,4,4096]` | residual 实测 |
| 4 | Attention hc_pre | `[1,4,4096]` | `[1,4096]` | 输入实测 |
| 5 | 图槽位/TP 对齐 | 1 valid token | physical 4 slots | 推导 |
| 6 | Q projection | physical hidden | `[4,16,512]` | 推导 |
| 7 | Attention/KV append | Q + cache | `[4,16,512]` | 每步新增 1 个有效 KV |
| 8 | O projection/RS | `[4,16,512]` | `[1,4096]` | 推导 |
| 9 | Attention hc_post | residual + output | `[1,4,4096]` | 代码 |
| 10 | FFN hc_pre | `[1,4,4096]` | `[1,4096]` | 代码 |
| 11 | Router/experts | `[1,4096]` | `[1,4096]` | affinity `[1,256]` 推导 |
| 12 | FFN hc_post | residual + output | `[1,4,4096]` | 代码 |
| 13 | Final mHC head | `[1,4,4096]` | `[1,4096]` | 代码 |
| 14 | LM head | `[1,4096]` | local `[1,32320]` | global `[1,129280]` |
| 15 | Sampler | logits | `[1]` | 下一个 token |

上下文增长：

| Forward | 本次输入 | Forward 前 KV | Forward 后 KV | 输出 |
|---|---|---:|---:|---|
| Prefill | prompt 16 | 0 | 16 | `y1` |
| Decode 1 | `y1` | 16 | 17 | `y2` |
| Decode 2 | `y2` | 17 | 18 | `y3` |
| ... | ... | ... | ... | ... |
| Decode 31 | `y31` | 46 | 47 | `y32` |

## 13. P/D 分离语义

当前 profile 中 P 和 D 是同一服务实例内先后执行的 scheduler 阶段：

```text
P step：输入 16 tokens
  → 建立初始 SWA KV、CSA/HCA compressed state 和 metadata
  → 产生 y1

D steps ×31：每步输入 1 个有效 token
  → 更新 KV/compressed state
  → 产生下一个 token
```

如果部署为独立 P 实例和 D 实例，模型内部计算 shape 基本不变，变化的是状态所有权和传输：

```text
P 实例处理 [16] prompt
→ 将请求 cache/state/metadata 传给 D 实例
→ D 实例每步处理 [1] 有效 token
```

DSV4 的 P/D 传输不能只考虑普通 KV；还必须保证 compressed cache、Indexer 状态、block table、position/context metadata 等一致。线上具体 wire format 要以 AFD-plugin 当前 KV connector 代码和传输 trace 为准，不能仅从本次单实例 profiler 推断。

## 14. Compile / ACL Graph 对 shape 的影响

开启 `torch.compile + ACL Graph` 后逻辑 shape 不变，但执行图使用固定或分桶后的物理 shape：

```text
逻辑 Decode：1 valid token
物理图输入 ：graph size / padding / TP 对齐后的 token slots
```

当前 `max_num_seqs=4`，结合 FlashComm/TP 对齐，Decode 可使用 physical token size 4 的图。分析时必须区分：

- 真正的 `num_scheduled_tokens`。
- 捕获图的 batch/token size。
- 为静态图或通信对齐加入的 padding token。

这解释了为什么 Decode residual 直接观测为 `[1,4,4096]`，而 attention Q 推导为 `[4,16,512]`。

## 15. MTP 当前状态与启用后的 shape

模型目录包含 MTP 权重，但当前启动没有 speculative 配置，因此 MTP draft forward **未参与本次性能测试**；当前主模型只维护 `_mtp_hidden_buffer` 需要的数据。

如果以后启用 MTP，典型数据流为：

```text
draft token ids      [N_d]
token embedding      [N_d,4096]
previous mHC hidden  [N_d,16384]
reshape previous     [N_d,4,4096]
e_proj branch        [N_d,1,4096]
h_proj branch        [N_d,4,4096]
融合后                [N_d,4,4096]
MTP block            [N_d,4,4096]
MTP mHC head         [N_d,4096]
MTP logits           [N_d,129280]
```

`N_d` 是本次 speculative draft token 数，不固定为 1，也不等于 Prefill 的 16。

## 16. 本例的能力边界

本例可以确认 mHC residual、TP/SP、三种 attention、Hash/Learned MoE、Prefill/Decode 上下文增长和 graph padding 的主数据流，但不足以评价：

- 超过 128 tokens 后 SWA 窗口截断。
- 形成多个 c128 group 后的 HCA。
- 历史超过 TopK=1024 后 CSA 的稀疏收益。
- 多请求 continuous batching 的 graph/padding shape。
- 真正启用 MTP 后的额外成本和接受率。
- 独立 P/D 节点的实际传输 tensor 列表。

建议后续补测：

```text
Prefill length: 16, 128, 512, 1024, 4096
Decode context: 128, 512, 1024, 4096, 8192
Active seqs  : 1, 2, 4
Mode         : eager / compile+ACL Graph
```

## 17. 实测证据位置与一页总览

本地 profile：

```text
/mnt/d/cyj/afd/dsv4_rank12_profiles/
  A_eager_stack_memory_shapes/
    ASCEND_PROFILER_OUTPUT/operator_details.csv
```

关键实测摘要：

```text
npu_hc_pre_v2 Prefill input [4,4,4096]
npu_hc_pre_v2 Decode input  [1,4,4096]
sparse attention Prefill Q  [16,16,512]
Lightning Indexer query     [16,64,128]
Hash router affinity        [4,256]
Hash token ids              [4]
Hash table                  [129280,6]
```

```text
Text → tokens [1,16]
 → TP embedding/SP [4,4096]
 → mHC [4,4,4096]
 → 43×{hc_pre → TP gather → Q [16,16,512]
       → SWA/CSA-c4/HCA-c128
       → O projection/RS → MoE → hc_post}
 → final hidden [1,4096]
 → local logits [1,32320] / global [1,129280]
 → sample y1
 → 31×Decode：residual [1,4,4096]
              attention physical Q [4,16,512] [推导]
              → next token
 → 32 output tokens，最终已处理 KV 长度 47
```
## 18. mHC 机制详解：初始化、Projection、Sinkhorn 与 hc_post

这一节专门解释 hidden 如何扩展为 4 条 residual stream，以及 Attention/FFN 子层中的 `hc_pre` 和 `hc_post` 如何工作。

### 18.1 初始化：`unsqueeze` 只增加 residual-stream 轴

Embedding 或前置 hidden 的 rank-local shape 是：

```text
x: [T_local,H]，其中 H=4096
```

mHC 初始化在 hidden 维前插入一条 stream 轴，再扩展成 4 条：

```python
X = x.unsqueeze(1)       # [T_local,1,4096]
X = X.expand(-1,4,-1)    # [T_local,4,4096]
```

概念上等价于：

```text
X[t] = [x[t], x[t], x[t], x[t]]
```

当前测试：

```text
Prefill: [4,4096] → [4,1,4096] → [4,4,4096]
Decode : [1,4096] → [1,1,4096] → [1,4,4096]
```

这里不是 `Linear(4096,16384)`。初始化时四条 stream 内容相同，没有凭空产生新信息；它们在后续每层不同的 residual mixing 和子层输出注入中逐渐分化。`expand` 可以只是零拷贝 view，真正进入要求连续内存的融合算子时才可能物化。

### 18.2 一次 Projection 同时生成三组 mapping

设进入一个 Attention 或 FFN 子层前：

```text
X: [T,C,H]，C=4，H=4096
```

对每个 token 展平 stream 和 hidden：

```text
X_flat = reshape(X)
[T,4,4096] → [T,16384]
```

当前 rank 12 profiler 对 `npu_hc_pre_v2` 的直接观测是：

```text
X                  [4,4,4096]
projection weight  [24,16384]
附加参数            [3]
附加参数            [24]
```

因此 projection 主体可以写为：

```text
g = X_flat · Wᵀ + b

X_flat [T,16384]
W      [24,16384]
b      [24]
g      [T,24]
```

`24` 的来源不是经验值，而是三组 mapping 的元素数之和：

```text
H_pre : C   = 4
H_post: C   = 4
H_res : C²  = 16
----------------
总计          24
```

投影结果拆分为：

```python
g_pre, g_post, g_res = split(g, [4,4,16], dim=-1)

G_pre : [T,4]
G_post: [T,4]
G_res : [T,16] → [T,4,4]
```

融合算子还接收 `[3]` 和 `[24]` 两组参数。Operator CSV 可以确认其 shape，并能判断它们分别作用于三组 mapping 和 24 个输出分量；但仅凭 profile 不能严谨确定其源码变量名以及 bias、scale、激活的精确先后顺序。因此本文将确定的数据流写成：

```text
X_flat
→ learned projection
→ static/bias 与分组 scale
→ pre/post 激活或归一化
→ residual logits 的 Sinkhorn
→ H_pre、H_post、H_res
```

训练学习的是 `W`、静态项和缩放项；推理时每个 token 根据自己的当前 `X` 动态得到实际 mapping。因此不同 token、不同层、Attention 与 FFN 子层都可能得到不同系数。

### 18.3 `hc_pre`：4 条 stream 动态合成一条

激活/归一化后的：

```text
H_pre: [T,4]
X    : [T,4,4096]
```

逐 token 计算：

```text
z[t,h] = Σᵢ H_pre[t,i] · X[t,i,h]
```

Shape：

```text
[T,4] × [T,4,4096]
→ 按 stream 维加权求和
→ z [T,4096]
```

例如某个 token 的：

```text
H_pre[t] = [0.1,0.2,0.3,0.4]
```

则：

```text
z[t] = 0.1X₁[t] + 0.2X₂[t] + 0.3X₃[t] + 0.4X₄[t]
```

Attention 和 FFN 始终处理 H=4096 的 `z`，并不直接处理一个 16384 维 Transformer hidden。

### 18.4 Sinkhorn：把 residual logits 投影到近似双随机矩阵

Projection 产生的 residual logits：

```text
G_res: [T,4,4]
```

原始 logits 可以为负数，行列和也没有约束。Sinkhorn 先将其转成正数，再交替进行行、列归一化：

```python
M = exp(G_res / temperature)

for _ in range(num_iterations):
    M = M / M.sum(dim=-1, keepdim=True)  # 每行和归一到 1
    M = M / M.sum(dim=-2, keepdim=True)  # 每列和归一到 1

H_res = M
```

数值稳定实现通常在 log space 中完成：

```python
Z = G_res / temperature

for _ in range(num_iterations):
    Z = Z - logsumexp(Z, dim=-1, keepdim=True)
    Z = Z - logsumexp(Z, dim=-2, keepdim=True)

H_res = exp(Z)
```

最终：

```text
H_res[t,i,j] ≥ 0
Σⱼ H_res[t,i,j] ≈ 1   # 每行
Σᵢ H_res[t,i,j] ≈ 1   # 每列
```

例如：

```text
H_res =
[[0.7,0.1,0.1,0.1],
 [0.1,0.7,0.1,0.1],
 [0.1,0.1,0.7,0.1],
 [0.1,0.1,0.1,0.7]]
```

它保留每条 stream 的主要信息，同时允许信息在 stream 之间流动。双随机约束控制每一层 residual mixing 的总量，避免 43 层传播中持续放大、衰减或所有信息塌缩到单条 stream。

### 18.5 Attention 中完整的 Prefill shape

当前 Prefill rank-local 的 mHC/Attention 流程：

```text
输入 residual X                 [4,4,4096]
reshape                         [4,16384]
projection                      [4,24]
├─ H_pre                        [4,4]
├─ H_post                       [4,4]
└─ H_res                        [4,4,4]

hc_pre: H_pre 与 X 加权求和
                                [4,4096]
TP all-gather                   [16,4096]
Q projection                    [16,16,512]
SWA / CSA / HCA                 [16,16,512]
O projection + TP reduce-scatter
                                [4,4096]

hc_post:
branch output y                 [4,4096]
old residual X                  [4,4,4096]
H_post                          [4,4]
H_res                           [4,4,4]
→ next residual                 [4,4,4096]
```

Profiler 的 `npu_hc_post` 保留了 batch 维，因此直接显示为：

```text
y       [1,4,4096]
X       [1,4,4,4096]
H_post  [1,4,4]
H_res   [1,4,4,4]
```

去掉 `B=1` 后，正好对应上面的 `[T,...]` 表达。

### 18.6 Attention 中完整的 Decode shape

单请求 Decode 的有效 token 数为 1：

```text
输入 residual X                 [1,4,4096] [实测]
reshape                         [1,16384]
projection                      [1,24]
├─ H_pre                        [1,4]
├─ H_post                       [1,4]
└─ H_res                        [1,4,4]

hc_pre                          [1,4096]
graph/TP 对齐后物理 Q           [4,16,512] [推导]
Attention + O projection/RS     [1,4096]
hc_post                         [1,4,4096]
```

这里 mHC mapping 按 1 个有效 token 生成；Attention 侧的物理 token 维度 4 来自 ACL Graph/TP 对齐，不能误解成 `H_pre/H_post` 为 4 个请求生成。

### 18.7 `hc_post` 的计算公式

令子层输出为：

```text
y: [T,H]
```

则每个 token、每条输出 stream、每个 hidden 元素的更新为：

```text
X_next[t,j,h]
  = Σᵢ H_res[t,j,i] · X[t,i,h]
  + H_post[t,j] · y[t,h]
```

矩阵写法：

```text
X_mixed = H_res @ X
X_next  = X_mixed + H_post.unsqueeze(-1) * y.unsqueeze(1)

H_res                    [T,4,4]
X                        [T,4,4096]
X_mixed                   [T,4,4096]
H_post.unsqueeze(-1)      [T,4,1]
y.unsqueeze(1)            [T,1,4096]
broadcast product         [T,4,4096]
X_next                    [T,4,4096]
```

### 18.8 一个可手算的 `hc_post` 例子

为便于手算，将 `H=4096` 暂时缩成 `H=2`，令单个 token 的旧 residual 为：

```text
X₁=[1,0]
X₂=[0,1]
X₃=[1,1]
X₄=[2,0]

X shape: [4,2]
```

Attention 输出：

```text
y=[10,20]，shape [2]
```

Sinkhorn 后的：

```text
H_res=
[[0.7,0.1,0.1,0.1],
 [0.1,0.7,0.1,0.1],
 [0.1,0.1,0.7,0.1],
 [0.1,0.1,0.1,0.7]]
```

第一步，混合旧 residual：

```text
X₁_mixed = 0.7X₁+0.1X₂+0.1X₃+0.1X₄ = [1.0,0.2]
X₂_mixed = 0.1X₁+0.7X₂+0.1X₃+0.1X₄ = [0.4,0.8]
X₃_mixed = 0.1X₁+0.1X₂+0.7X₃+0.1X₄ = [1.0,0.8]
X₄_mixed = 0.1X₁+0.1X₂+0.1X₃+0.7X₄ = [1.6,0.2]
```

假设动态生成：

```text
H_post=[0.2,0.4,0.1,0.3]
```

第二步，将同一个 Attention 输出以不同权重注入每条 stream：

```text
X₁_next = [1.0,0.2] + 0.2[10,20] = [3.0,4.2]
X₂_next = [0.4,0.8] + 0.4[10,20] = [4.4,8.8]
X₃_next = [1.0,0.8] + 0.1[10,20] = [2.0,2.8]
X₄_next = [1.6,0.2] + 0.3[10,20] = [4.6,6.2]
```

Shape 全程保持：

```text
旧 residual [4,2]
→ H_res mixing [4,2]
+ H_post×y [4,2]
→ 新 residual [4,2]
```

真实模型只需把简化的 `H=2` 换回 `H=4096`：

```text
[T,4,4096] → hc_post → [T,4,4096]
```

因此 mHC 的核心不是把 Attention 输出变成四份完全独立的新特征，而是：旧 residual 先通过受约束的 `H_res` 重新路由，同一个子层输出再通过 token-dependent 的 `H_post` 以不同强度写入四条 stream。

### 18.9 最终合并与 MTP 展平

经过 43 层后，最终 mHC head 将 4 streams 合成为普通 hidden：

```text
Prefill [4,4,4096] → [4,4096]
Decode  [1,4,4096] → [1,4096]
```

只有为了保存 MTP 所需的完整 mHC 状态时，才会展平为：

```text
[T,4,4096] → [T,16384]
```

这个 `[T,16384]` 是 4 条 residual stream 的存储形式，不表示 Attention/FFN 的 hidden size 已从 4096 改成 16384。

## 19. 调度长度参数补充

- `max_model_len=8192`：单个请求允许模型处理的最大序列长度，近似满足 `prompt tokens + 已生成并送回模型的 tokens ≤ max_model_len`。
- `max_num_batched_tokens=4096`：一次 scheduler step 中所有请求合计最多调度的 token 数，不是单请求总长度。
- `max_num_seqs=4`：一次调度中最多容纳的 active sequence 数。

例如当前 16-token prompt 在 `max_model_len=8192` 下，理论上还能继续处理约 8176 个生成 token；但实际还会受到请求 `max_tokens`、EOS、显存/缓存和实现限制。


---

后续若拿到 Decode operator CSV 中精确的 Q/Indexer shape，以及独立 P/D connector 的实际传输 trace，应把相应 **[推导]** 项升级为 **[实测]**。
