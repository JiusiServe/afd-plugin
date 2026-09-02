MODEL_PATH=${MODEL_PATH:-/home/models/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0}
export VLLM_USE_V2_MODEL_RUNNER=0

# Resolve this script's directory so the proxy is found regardless of the
# caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# NIXL transport tuning shared by every PD participant (prefill + decode).
export UCX_TLS=cuda_ipc,cuda_copy,tcp

# vLLM's native NixlConnector runs a per-engine ZMQ "side channel" for the
# metadata handshake. Every instance on this host must bind a DISTINCT side
# channel port; the prefiller returns its own host+port in kv_transfer_params
# so the decoder knows where to pull the KV from. The effective port is
# VLLM_NIXL_SIDE_CHANNEL_PORT + data_parallel_index, so with DP=2 the decode
# worker below actually claims ports 5603 and 5604.
export VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1

# Baseline (no AFD, no DBO): 2 prefill producers feed a single,
# non-disaggregated decode instance spanning 2 GPUs via --data-parallel-size
# 2. Compare 2p1a1f_graph_dbo.sh, where the decode side is itself split into
# a dedicated attention + FFN worker connected via P2pNcclAFDConnector.

# ----- Prefill producer #1 (sender, GPU0, HTTP 18301, side channel 5601) -----
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 VLLM_ENABLE_V1_MULTIPROCESSING=1 \
CUDA_VISIBLE_DEVICES=0 uv run vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 18301 \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager  \
  --max-num-seqs 64 \
  --max-num-batched-tokens 64 \
  --max-model-len 8192 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
  > afd_prefill0.log 2>&1 &

# ----- Prefill producer #2 (sender, GPU1, HTTP 18302, side channel 5602) -----
VLLM_NIXL_SIDE_CHANNEL_PORT=5602 VLLM_ENABLE_V1_MULTIPROCESSING=1 \
CUDA_VISIBLE_DEVICES=1 uv run vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 18302 \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager  \
  --max-num-seqs 64 \
  --max-num-batched-tokens 64 \
  --max-model-len 8192 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
  > afd_prefill1.log 2>&1 &

# ----- Decode worker (receiver, GPU2+GPU3, DP=2, HTTP 18303, side channel 5603) -----
# Plain, non-disaggregated decode: no AFD role split, no DBO. Holds the KV
# cache itself, so it is the sole NIXL consumer and pulls the prompt KV from
# whichever prefiller produced it.
VLLM_NIXL_SIDE_CHANNEL_PORT=5603 \
CUDA_VISIBLE_DEVICES=2,3 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --max-num-seqs 64 \
    --max-num-batched-tokens 64 \
    --max-cudagraph-capture-size 64 \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[64]
    }' \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
    --host 127.0.0.1 \
    --port 18303 \
    --trust-remote-code > attn.log 2>&1 &

# ----- Disaggregation proxy (client-facing endpoint, HTTP 18305) -----
# vLLM-native NIXL load-balance proxy. For each request it prefills on a
# prefiller (do_remote_decode), reads back the prefiller's NIXL handshake
# metadata (remote_engine_id / remote_block_ids / remote_host / remote_port) in
# kv_transfer_params, then hands it to the decoder with do_remote_prefill so the
# decoder pulls the KV. Send benchmark/client traffic to 18305, NOT to 18303.
# Wait for the prefill/decode servers to print "Application startup complete"
# before this proxy starts accepting traffic.
uv run python "$SCRIPT_DIR/../../../../../tools/proxy_server.py" \
    --host 127.0.0.1 \
    --port 18305 \
    --prefiller-hosts 127.0.0.1 127.0.0.1 \
    --prefiller-ports 18301 18302 \
    --decoder-hosts 127.0.0.1 \
    --decoder-ports 18303 \
    > proxy.log 2>&1

wait
