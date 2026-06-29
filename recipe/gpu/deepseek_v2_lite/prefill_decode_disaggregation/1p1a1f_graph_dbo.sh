MODEL_PATH=${MODEL_PATH:-/home/models/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0}

# Resolve this script's directory so the LMCache PD configs and the proxy are
# found regardless of the caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMCACHE_DIR="$SCRIPT_DIR/lmcache"

# NIXL transport tuning shared by every PD participant (sender + receiver).
export UCX_TLS=cuda_ipc,cuda_copy,tcp

# LMCache keys KV chunks by Python's string hash. Every PD participant (both
# prefill senders + the decode receiver) MUST share one hash seed, or the
# receiver computes different keys than the sender pushed and silently fails to
# match them (LMCache logs "PYTHONHASHSEED not set ... incorrect KV cache
# transfer", and the decoder shows hit/load = 0).
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

# ----- Prefill producer #1 (sender, GPU0, HTTP 18301) -----
LMCACHE_CONFIG_FILE="$LMCACHE_DIR/prefill.yaml" VLLM_ENABLE_V1_MULTIPROCESSING=1 \
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
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"producer1"}}' \
  > afd_prefill0.log 2>&1 &

# ----- Prefill producer #2 (sender, GPU1, HTTP 18302) -----
LMCACHE_CONFIG_FILE="$LMCACHE_DIR/prefill.yaml" VLLM_ENABLE_V1_MULTIPROCESSING=1 \
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
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"producer2"}}' \
  > afd_prefill1.log 2>&1 &

# ----- Decode attention worker (receiver, GPU2, HTTP 18305) -----
# Only the attention worker holds KV cache, so it is the sole PD receiver.
LMCACHE_CONFIG_FILE="$LMCACHE_DIR/decode.yaml" \
CUDA_VISIBLE_DEVICES=2 uv run vllm serve "$MODEL_PATH" \
    --worker-cls afd_plugin.v1.worker.AFDAttentionWorker \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "enabled": true,
            "role": "attention",
            "connector": "p2pconnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 1,
            "num_ffn_ranks": 1,
            "extra_config": {
                "afd_size": "1A1F"
            }
        }
    }' \
    --max-num-seqs 64 \
    --max-num-batched-tokens 64 \
    --enable-dbo \
    --dbo-decode-token-threshold 2 \
    --dbo-prefill-token-threshold 12 \
    --max-cudagraph-capture-size 64 \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[64]
    }' \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"lmcache_rpc_port":"consumer","skip_last_n_tokens":1}}' \
    --host 127.0.0.1 \
    --port 18303 \
    --trust-remote-code > attn.log 2>&1 &

# ----- Decode FFN worker (GPU3) -----
# The FFN worker carries no KV cache and is NOT a PD receiver: it gets no
# LMCACHE_CONFIG_FILE (so it stays on the inert local-cpu default and never
# binds the NIXL peer ports) and uses a distinct lmcache_rpc_port so its
# connector's IPC socket does not collide with the attention receiver's.
CUDA_VISIBLE_DEVICES=3 uv run vllm serve "$MODEL_PATH" \
    --worker-cls afd_plugin.v1.worker.AFDFFNWorker \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "enabled": true,
            "role": "ffn",
            "connector": "p2pconnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 1,
            "num_ffn_ranks": 1,
            "extra_config": {
                "afd_size": "1A1F"
            }
        }
    }' \
    --max-num-seqs 64 \
    --enable-dbo \
    --dbo-decode-token-threshold 2 \
    --dbo-prefill-token-threshold 12 \
    --max-num-batched-tokens 64 \
    --max-cudagraph-capture-size 64 \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[64]
    }' \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"lmcache_rpc_port":"consumer_ffn"}}' \
    --host 127.0.0.1 \
    --port 18304 \
    --trust-remote-code > ffn.log 2>&1 &

# ----- Disaggregation proxy (client-facing endpoint, HTTP 18300) -----
# Routes each request prefill -> decode and owns the ZMQ PULL socket (port 7500)
# that the prefill senders notify once KV has landed in the decoder PD buffer.
# Send benchmark/client traffic here, NOT to 18305.
# Wait for the prefill/attn/ffn servers to print "Application startup complete"
# before this proxy starts accepting traffic.
uv run --active python "$SCRIPT_DIR/disagg_proxy_server.py" \
    --host 127.0.0.1 \
    --port 18305 \
    --prefiller-host 127.0.0.1 \
    --prefiller-port 18301,18302 \
    --decoder-host 127.0.0.1 \
    --decoder-port 18303 \
    --decoder-init-port 7300 \
    --decoder-alloc-port 7400 \
    --proxy-host 127.0.0.1 \
    --proxy-port 7500 \
    > proxy.log 2>&1

wait