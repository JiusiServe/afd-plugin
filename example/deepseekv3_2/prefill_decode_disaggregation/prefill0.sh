#!/bin/bash
# Prefill node 0 template (2P2D, 16 NPU/node, A3)
#   $1 VISIBLE_DEVICES    $2 vllm port   $3 dp_size   $4 dp_rank
#   $5 dp_address  $6 rpc_port    $7 tp_size
export PYTHONPATH="/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:$PYTHONPATH"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export HCCL_BUFFSIZE=1024

nic_name="eth0"
local_ip="190.0.0.1"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

MODEL=/path/model_weights/dsv3_2

ADDITIONAL='{"enable_cpu_binding" : false, "enable_sfa_cp":false}'
KV_TRANSFER='{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "36000",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {"dp_size": 2, "tp_size": 8},
    "decode": {"dp_size": 32, "tp_size": 1}
  }
}'

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name dsv3 \
  --max-model-len 8192 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.93 \
  --quantization ascend \
  --enforce-eager \
  --no-enable-prefix-caching \
  --tokenizer-mode deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --additional-config "$ADDITIONAL" \
  --kv-transfer-config "$KV_TRANSFER"
