MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}

CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve "$MODEL_PATH" \
    --worker-cls afd_plugin.v1.worker.AFDAttentionWorker \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "enabled": true,
            "role": "attention",
            "connector": "p2pconnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_servers": 2,
            "num_ffn_servers": 2,
            "extra_config": {
                "afd_size": "2A2F"
            }
        }
    }' \
    --max-num-seqs 64 \
    --max-num-batched-tokens 64 \
    --enable-dbo \
    --dbo-decode-token-threshold 2 \
    --dbo-prefill-token-threshold 12 \
    --enforce-eager \
    --host 127.0.0.1 \
    --port 18305 \
    --trust-remote-code > attn.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 uv run vllm serve "$MODEL_PATH" \
    --worker-cls afd_plugin.v1.worker.AFDFFNWorker \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "enabled": true,
            "role": "ffn",
            "connector": "p2pconnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_servers": 2,
            "num_ffn_servers": 2,
            "extra_config": {
                "afd_size": "2A2F"
            }
        }
    }' \
    --max-num-seqs 64 \
    --enable-dbo \
    --dbo-decode-token-threshold 2 \
    --dbo-prefill-token-threshold 12 \
    --max-num-batched-tokens 64 \
    --enforce-eager \
    --host 127.0.0.1 \
    --port 18305 \
    --trust-remote-code > ffn.log 2>&1 &
