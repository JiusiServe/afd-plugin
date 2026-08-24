
#!/bin/bash
# export AFD_CAMP2P_STUB_IO=1
export PYTHONPATH="/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:$PYTHONPATH"
MODEL=/home/admin/model-csi/model

VLLM_USE_V1=1 vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8006 \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": ['8']}'  \
  --max_num_seqs 8 \
  --quantization ascend \
  --max_num_batched_tokens 32 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 8192


# #!/bin/bash
# export PYTHONPATH="/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:$PYTHONPATH"
# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
# export MOONCAKE_CONFIG_PATH="/vllm-workspace/mooncake.json"
# export VLLM_USE_V1=1
# export ASCEND_RT_VISIBLE_DEVICES=$1
# export HCCL_IF_IP=$2
# export GLOO_SOCKET_IFNAME=eth0
# export TP_SOCKET_IFNAME=eth0
# export HCCL_SOCKET_IFNAME=eth0

# MODEL=/home/admin/model-csi/model

# vllm serve "$MODEL" \
# --served-model-name dsv3 \
# --host $2 \
# --port $3 \
# --tensor-parallel-size 8 \
# --enable-expert-parallel \
# --quantization ascend \
# --max-model-len 8192 \
# --max-num-batched-tokens 16384 \
# --max-num-seqs 16 \
# --gpu-memory-utilization 0.9 \
# --tokenizer-mode deepseek_v32 \
# --reasoning-parser deepseek_v3 \
# --trust-remote-code --no-enable-prefix-caching \
# --kv-transfer-config '{
#     "kv_connector": "MooncakeConnectorStoreV1",
#     "kv_role": "kv_both",
#     "kv_connector_extra_config": {
#         "use_layerwise": false,
#         "mooncake_rpc_port": "0",
#         "load_async": true,
#         "register_buffer": true
#     }
# }'