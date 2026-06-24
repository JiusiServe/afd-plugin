export PYTHONPATH="/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:$PYTHONPATH"
export VLLM_USE_V1=1

MODEL=/path/model_weights/dsv3_2

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8006 \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUFFNWorker \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": ['8']}'  \
  --max_num_seqs 8 \
  --seed 1024 \
  --served-model-name dsv3 \
  --max_num_batched_tokens 32 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 8192 \
  --quantization ascend \
  --additional-config '{
    "afd": {
      "enabled": true,
      "role": "ffn",
      "connector": "camp2pconnector",
      "host": "190.0.0.2",
      "port": 29666,
      "num_attention_servers": 16,
      "num_ffn_servers": 16,
      "afd_server_rank": 0
    }
  }' > ffn.log 2>&1 &
