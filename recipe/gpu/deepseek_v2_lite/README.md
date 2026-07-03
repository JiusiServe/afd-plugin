# DeepSeek-V2-Lite AFD Examples

End-to-end launch scripts for running DeepSeek-V2-Lite with the AFD
(Attention-FFN Disaggregation) plugin on vLLM `v0.19.1`.

## Prerequisites

- Install [LMCache](https://github.com/LMCache/LMCache). You can simply run `pip install lmcache`.
- Install [NIXL](https://github.com/ai-dynamo/nixl).
- At least 3 GPUs(A/H-class, tested against L20X).
- vLLM `v0.19.1` and the `afd-plugin` package installed in the same
  environment (see repository root `AGENTS.md`).
- DeepSeek-V2-Lite weights on disk. All scripts default to
  `/path/model_weights/DeepSeek-V2-Lite`; override with
  `MODEL_PATH=...` when launching.

## Directory layout

```
.
├── benchmark.sh                          # online serving benchmark client
├── prefill_decode_disaggregation/        # prefill_decode_disaggregation, 1P1A1F topology
│   ├── 1p1a1f_eager_dbo.sh
│   └── 1p1a1f_graph_dbo.sh
└── prefill_decode_colocated/             # prefill_decode_colocated, 2A2F topology
    ├── 2a2f_eager_dbo_dp1tp2.sh
    ├── 2a2f_eager_dbo_dp2tp1.sh
    ├── 2a2f_graph_dbo_dp1tp2.sh
    └── 2a2f_graph_dbo_dp2tp1.sh
```

### 2. Prefill/Decode Colocated — `2a2f`

2 processes, two GPUs each:

| GPUs   | Role              | Worker class           | Port  |
|--------|-------------------|------------------------|-------|
| 0, 1   | Attention    | `AFDAttentionWorker`   | 18305 |
| 2, 3   | FFN          | `AFDFFNWorker`         | 18305 |

The four variants cover the TP/DP cross product:

| File                            | DP | TP |
|---------------------------------|----|----|
| `2a2f_*_dp1tp2.sh`              | 1  | 2  |
| `2a2f_*_dp2tp1.sh`              | 2  | 1  |

## Running

Pick a script and execute it from the repository root. Each script
backgrounds its workers and writes per-worker logs (`afd_prefill.log`, `attn.log`, `ffn.log`) in the current directory.

### prefill_decode_colocated
```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
bash recipe/gpu/deepseek_v2_lite/prefill_decode_colocated/2a2f_graph_dbo_dp1tp2.sh
```

Wait for `attn.log` to print the `Application startup complete` line
before sending traffic.

### prefill_decode_disaggregation

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
bash recipe/gpu/deepseek_v2_lite/prefill_decode_disaggregation/1p1a1f_graph_dbo.sh
```

Wait for `attn.log` to print the `Application startup complete` line
before sending traffic.

```bash
uv run python3 ../vllm/examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_proxy_server.py --host 0.0.0.0 --port 18305     --prefiller-host 127.0.0.1 --prefiller-port 18301     --decoder-host 127.0.0.1   --decoder-port 18302
```

### Running the benchmark

Once the serving stack is up, run:

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
export RESULT_DIR=/path/results
export RESULT_FILENAME=2a2f_graph_dbo_dp1tp2.json
bash recipe/gpu/deepseek_v2_lite/benchmark.sh
```

It fires 1024 random requests (1024 input tokens / 128 output tokens) at
5 request rate with `--max-concurrency 32` against `127.0.0.1:18305`,
and dumps the JSON result to `$RESULT_DIR/$RESULT_FILENAME`.

## Common AFD configuration

Every AFD worker is wired through `--additional-config` with the same
shape; only `role` and `afd_size` differ between attention and FFN:

```jsonc
{
  "afd": {
    "enabled": true,
    "role": "attention",            // or "ffn"
    "connector": "p2pconnector",
    "host": "127.0.0.1",
    "port": 6269,
    "num_attention_ranks": 1,      // 2 in 2A2F
    "num_ffn_ranks": 1,            // 2 in 2A2F
    "extra_config": {
      "afd_size": "1A1F"             // "2A2F" in 2A2F
    }
  }
}
```

DBO (Dual Batch Overlap) is turned on for all examples with
`--dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12`.

### Switching eager → graph

Graph mode replaces `--enforce-eager` with:

```
--max-cudagraph-capture-size 64
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY",
                       "cudagraph_capture_sizes":[64]}'
```
