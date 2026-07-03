MODEL_PATH=${MODEL_PATH:-/home/models/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0}

CUDA_VISIBLE_DEVICES=0 uv run vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 18301 \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --max-num-seqs 64 \
  --max-num-batched-tokens 64 \
  --max-model-len 8192 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"producer1"}}' \
  > afd_prefill.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 uv run vllm serve "$MODEL_PATH" \
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
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"consumer"}}' \
    --host 127.0.0.1 \
    --port 18305 \
    --trust-remote-code > attn.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 uv run vllm serve "$MODEL_PATH" \
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
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"consumer"}}' \
    --host 127.0.0.1 \
    --port 18305 \
    --trust-remote-code > ffn.log 2>&1 &
