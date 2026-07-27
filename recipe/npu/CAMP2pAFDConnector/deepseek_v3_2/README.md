# DeepSeek-V3.2 Synchronous Decode with CAMP2pAFDConnector on Ascend NPU

This recipe compares a conventional EP64 deployment with synchronous
Attention-FFN Disaggregation (AFD) deployments for DeepSeek-V3.2 decode
inference on Ascend NPUs. The AFD deployments use
`CAMP2pAFDConnector` to exchange activations between independent attention
and FFN workers.

## Image and model requirements

- Hardware: Ascend NPU, Atlas 900 A3 SuperPoD, 16 dies per node.
- Image: `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler`.
- Model: DeepSeek-V3.2 with W8A8 weights.
- AFD Plugin: install this repository in the container.

```bash
docker pull quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler
cd /path/to/afd-plugin
pip install -e . --no-build-isolation -v
```

## Topologies

| Topology | Nodes | Total dies | Layout |
|---|---:|---:|---|
| EP64 | 4 | 64 | DP64, EP64, TP1 |
| 48A16F | 4 | 64 | 48 attention ranks and 16 FFN ranks |
| 64A16F | 5 | 80 | 64 attention ranks and 16 FFN ranks |

Each node runs 16 local ranks. `DP_ADDRESS` is the address of the attention or
baseline node that owns DP rank 0. For AFD, `AFD_HOST` is the address of the
FFN node that owns FFN role rank 0. All attention and FFN processes must use
the same `AFD_HOST` and `AFD_PORT`.

### Limitations

- Forced expert load balancing replaces natural routed expert IDs with a
  deterministic synthetic cycle that uses four local experts per rank/NPU;
  it changes model outputs. Under this setup, 48A16F and 64A16F simulate the
  logical scales of 192A64F and 256A64F.
- Inputs are fixed at 16,384 or 32,768 tokens; outputs are synthetic and
  uniformly distributed from 512 to 1,536 tokens.
- `AFDDecodeBenchConnector` supplies the decode KV state.

## Experiment matrix

`BS` below is the per-rank vLLM batch size. It is used for
`--max-num-seqs`, `--max-num-batched-tokens`, and the `FULL_DECODE_ONLY` graph
capture size.

| Input | Topology | Per-rank BS | FFN BS | Max model length | AISBench requests |
|---|---|---:|---:|---:|---:|
| 16K | EP64 | 16 | N/A | 18,432 | 6144 |
| 16K | 48A16F | 28 | 84 | 18,432 | 8064 |
| 16K | 64A16F | 28 | 112 | 18,432 | 10752 |
| 32K | EP64 | 8 | N/A | 34,816 | 3072 |
| 32K | 48A16F | 14 | 42 | 34,816 | 4032 |
| 32K | 64A16F | 14 | 56 | 34,816 | 5376 |

The request count is:

```text
request count = DP size * per-rank BS * workload multiplier
workload multiplier = 6
```

The workload multiplier means that [AISBench](https://github.com/AISBench/benchmark) generates six complete waves of
the configured per-rank batch capacity. It provides enough requests to keep
the decode engines saturated after requests with shorter sampled output
lengths begin to finish. It increases the total amount of benchmark work; it
does not change the server-side per-rank BS.

For AFD, the FFN batch size is:

```text
FFN BS = attention ranks * attention BS / FFN ranks
```

This experiment uses 16 FFN ranks, so the four AFD cases evaluate to BS84,
BS112, BS42, and BS56.

## Important configuration

All six cases use forced expert balancing. The baseline EP64 case uses:

```json
{
  "enable_force_load_balance": true
}
```

EP64 already places four routed experts on each rank, so the baseline does not
set an explicit per-rank expert limit. The AFD cases use:

```json
{
  "enable_force_load_balance": true,
  "force_load_balance_topn_per_rank": 4
}
```

`enable_force_load_balance=true` enables benchmark-only forced expert load
balancing. After the model router computes its normal top-k routing result,
AFD replaces the routed expert IDs with a deterministic synthetic expert
cycle. The cycle distributes routed tokens evenly across EP ranks instead of
following the model's natural expert choices; the router-produced top-k
weights are retained. This produces a controlled, balanced MoE communication
load, but it changes model outputs and must not be used for accuracy or
production-serving evaluation.

For the AFD cases, `force_load_balance_topn_per_rank=4` includes four local
experts from each EP rank/NPU in that synthetic routing cycle. With this
mapping, the physical 48A16F and 64A16F deployments are used to simulate the
logical scales of 192A64F and 256A64F, respectively. These are simulated
logical scales; the physical deployments still use 48 or 64 attention dies
and 16 FFN dies.

The decode KV cache is populated through:

```json
{
  "kv_connector": "AFDDecodeBenchConnector",
  "kv_connector_module_path": "tools.benchmarks.decode_bench",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "fill_mean": 0.015,
    "fill_std": 0.0
  }
}
```

This recipe uses the synchronous `CAMP2pAFDConnector` path and does not enable
AFD async-DP. The `--async-scheduling` CLI option used by all cases is a vLLM
scheduler optimization and is unrelated to AFD async-DP or
`CAMAsyncAFDConnector`.

AFD attention and FFN workers also enable Dual Batch Overlap (DBO):

```text
--enable-dbo
--dbo-decode-token-threshold 2
--dbo-prefill-token-threshold 12
--ubatch-size 2
```

## Launcher variables

The three launchers in this directory run one node each:

- `baseline_ep64.sh`: one EP64 baseline node.
- `afd_attention.sh`: one AFD attention node.
- `afd_ffn.sh`: the AFD FFN node.

Common variables:

| Variable | Meaning |
|---|---|
| `MODEL_PATH` | DeepSeek-V3.2 W8A8 checkpoint path |
| `LOCAL_IP` | Current node communication IP |
| `NIC_NAME` | Current node communication interface; check it with `ifconfig` |
| `INPUT_LENGTH` | `16384` or `32768` |
| `BATCH_SIZE` | Per-rank baseline or attention BS |
| `ASCEND_RT_VISIBLE_DEVICES` | Defaults to all 16 local dies |

Baseline and attention launchers also use `DP_ADDRESS` and `DP_START_RANK`.
AFD launchers use `AFD_HOST`, `AFD_PORT` (default `29666`), and
`ATTENTION_RANKS` (`48` or `64`).

## Launch EP64

Set `DP_ADDRESS` to `<NODE0_IP>` on every node. Start node 0 first, followed by
nodes 1-3.

### 16K input, BS16

```bash
# node0
MODEL_PATH=<MODEL_PATH> LOCAL_IP=<NODE0_IP> DP_ADDRESS=<NODE0_IP> \
DP_START_RANK=0 INPUT_LENGTH=16384 BATCH_SIZE=16 \
bash baseline_ep64.sh

# node1
MODEL_PATH=<MODEL_PATH> LOCAL_IP=<NODE1_IP> DP_ADDRESS=<NODE0_IP> \
DP_START_RANK=16 INPUT_LENGTH=16384 BATCH_SIZE=16 \
bash baseline_ep64.sh

# node2
MODEL_PATH=<MODEL_PATH> LOCAL_IP=<NODE2_IP> DP_ADDRESS=<NODE0_IP> \
DP_START_RANK=32 INPUT_LENGTH=16384 BATCH_SIZE=16 \
bash baseline_ep64.sh

# node3
MODEL_PATH=<MODEL_PATH> LOCAL_IP=<NODE3_IP> DP_ADDRESS=<NODE0_IP> \
DP_START_RANK=48 INPUT_LENGTH=16384 BATCH_SIZE=16 \
bash baseline_ep64.sh
```

For the 32K case, use the same node mapping with:

```bash
INPUT_LENGTH=32768 BATCH_SIZE=8
```

## Launch 48A16F

Use `<ATTN_NODE0_IP>` as `DP_ADDRESS` and `<FFN_NODE_IP>` as `AFD_HOST` on all
four nodes. Start the FFN node and attention rank-0 node before the remaining
attention nodes.

### 16K input, attention BS28, FFN BS84

```bash
# FFN node
MODEL_PATH=<MODEL_PATH> LOCAL_IP=<FFN_NODE_IP> AFD_HOST=<FFN_NODE_IP> \
ATTENTION_RANKS=48 INPUT_LENGTH=16384 BATCH_SIZE=28 \
bash afd_ffn.sh

# attention node0, ranks 0-15
MODEL_PATH=<MODEL_PATH> LOCAL_IP=<ATTN_NODE0_IP> DP_ADDRESS=<ATTN_NODE0_IP> \
DP_START_RANK=0 AFD_HOST=<FFN_NODE_IP> ATTENTION_RANKS=48 \
INPUT_LENGTH=16384 BATCH_SIZE=28 bash afd_attention.sh

# attention node1, ranks 16-31
MODEL_PATH=<MODEL_PATH> LOCAL_IP=<ATTN_NODE1_IP> DP_ADDRESS=<ATTN_NODE0_IP> \
DP_START_RANK=16 AFD_HOST=<FFN_NODE_IP> ATTENTION_RANKS=48 \
INPUT_LENGTH=16384 BATCH_SIZE=28 bash afd_attention.sh

# attention node2, ranks 32-47
MODEL_PATH=<MODEL_PATH> LOCAL_IP=<ATTN_NODE2_IP> DP_ADDRESS=<ATTN_NODE0_IP> \
DP_START_RANK=32 AFD_HOST=<FFN_NODE_IP> ATTENTION_RANKS=48 \
INPUT_LENGTH=16384 BATCH_SIZE=28 bash afd_attention.sh
```

For the 32K case, use the same node mapping with:

```bash
INPUT_LENGTH=32768 BATCH_SIZE=14
```

The FFN launcher derives BS42 automatically.

## Launch 64A16F

Use `<ATTN_NODE0_IP>` as `DP_ADDRESS` and `<FFN_NODE_IP>` as `AFD_HOST` on all
five nodes. The FFN command is the same shape as 48A16F, with
`ATTENTION_RANKS=64`. Attention nodes use start ranks 0, 16, 32, and 48.

### 16K input, attention BS28, FFN BS112

```bash
# FFN node
MODEL_PATH=<MODEL_PATH> LOCAL_IP=<FFN_NODE_IP> AFD_HOST=<FFN_NODE_IP> \
ATTENTION_RANKS=64 INPUT_LENGTH=16384 BATCH_SIZE=28 \
bash afd_ffn.sh

# run once on each attention node with its own LOCAL_IP and DP_START_RANK
MODEL_PATH=<MODEL_PATH> LOCAL_IP=<ATTN_NODE_IP> DP_ADDRESS=<ATTN_NODE0_IP> \
DP_START_RANK=<0|16|32|48> AFD_HOST=<FFN_NODE_IP> ATTENTION_RANKS=64 \
INPUT_LENGTH=16384 BATCH_SIZE=28 bash afd_attention.sh
```

For the 32K case, use:

```bash
INPUT_LENGTH=32768 BATCH_SIZE=14
```

The FFN launcher derives BS56 automatically.

## AISBench configuration

Install [AISBench](https://github.com/AISBench/benchmark):

```bash
git clone https://github.com/AISBench/benchmark.git
cd benchmark
pip install -e . --use-pep517
pip install -r requirements/api.txt
```

### Synthetic dataset

Edit `ais_bench/datasets/synthetic/synthetic_config.py`. For 16K input:

```python
REQUEST_COUNT = 6144  # Select the value for the case from the matrix above.

synthetic_config = {
    "Type": "string",
    "RequestCount": REQUEST_COUNT,
    "TrustRemoteCode": False,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": 16384, "MaxValue": 16384},
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": 512, "MaxValue": 1536},
        },
    },
    "TokenIdConfig": {"RequestSize": REQUEST_COUNT, "PrefixLen": 0},
}
```

For 32K input, change only the input values and select the matching request
count:

```python
"Params": {"MinValue": 32768, "MaxValue": 32768}
```

### AISBench client

Edit `ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py`:

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

REQUEST_COUNT = 6144  # Select the value for the case from the matrix above.

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        path="<MODEL_PATH>",
        model="dsv3_2",
        request_rate=0,
        retry=2,
        host_ip="192.0.0.1",  # Replace with the actual proxy/API server IP.
        host_port=8006,  # Replace with the actual proxy/API server port.
        max_out_len=1536,
        batch_size=REQUEST_COUNT,
        trust_remote_code=True,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1024,
            ignore_eos=False,
        )
    )
]
```

Run the benchmark:

```bash
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf
```

## Results

```text
tokens/s/die = aggregate output token throughput / total deployed dies
```

The denominators are 64 for EP64, 64 for 48A16F, and 80 for 64A16F.

### 16K fixed input, 512-1,536 uniform output

![DeepSeek-V3.2 16K decode throughput per die](throughput_dsv3-2_16k.png)

In the 16K run, EP64 achieves 232.6 tokens/s/die, 48A16F achieves 220.3 tokens/s/die,
and 64A16F achieves 258.9 tokens/s/die. Using EP64 as the baseline,
the corresponding AFD changes are -5.3% for 48A16F and +11.3% for 64A16F.

### 32K fixed input, 512-1,536 uniform output

![DeepSeek-V3.2 32K decode throughput per die](throughput_dsv3-2_32k.png)

In the 32K run, EP64 achieves 168.2 tokens/s/die, 48A16F achieves 151.4 tokens/s/die,
and 64A16F achieves 183.3 tokens/s/die. Using EP64 as the baseline,
the corresponding AFD changes are -10.0% for 48A16F and +9.0% for 64A16F.
