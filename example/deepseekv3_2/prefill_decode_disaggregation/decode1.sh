#!/bin/bash
# Decode node 1 template (2P2D, 16 NPU/node, A3).
#   $1 VISIBLE_DEVICES    $2 vllm port   $3 dp_size   $4 dp_rank
#   $5 dp_address  $6 rpc_port    $7 tp_size
export PYTHONPATH="/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:$PYTHONPATH"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export HCCL_BUFFSIZE=1024

nic_name="eth0"
local_ip="190.0.0.4"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

MODEL=/path/model_weights/dsv3_2

ADDITIONAL='{
  "enable_cpu_binding": false,
  "finegrained_tp_config": {"lmhead_tensor_parallel_size": 16},
  "afd": {
    "enabled": true,
    "role": "ffn",
    "connector": "camp2pconnector",
    "host": "190.0.0.4",
    "port": 29666,
    "num_attention_servers": 16,
    "num_ffn_servers": 16,
    "afd_server_rank": 0
  }
}'
COMPILATION='{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[3,15,24,33,42]}'
KV_TRANSFER='{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "36200",
  "engine_id": "2",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {"dp_size": 2, "tp_size": 8},
    "decode": {"dp_size": 16, "tp_size": 1}
  }
}'

vllm serve "$MODEL" \
  --port $2 \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUFFNWorker \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name dsv3 \
  --max-model-len 8192 \
  --max-num-batched-tokens 100 \
  --trust-remote-code \
  --max-num-seqs 14 \
  --gpu-memory-utilization 0.93 \
  --no-enable-prefix-caching \
  --quantization ascend \
  --tokenizer-mode deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --async-scheduling \
  --additional-config "$ADDITIONAL" \
  --kv-transfer-config "$KV_TRANSFER" \
  --compilation-config "$COMPILATION"
