#!/bin/bash
# Decode node 2 template (2P2D, 16 NPU/node, A3). ▒~V~R~T▒~V~R launch_dp.py 驱▒~V~R~J▒~V~R▒~V~R~@~B
# ▒~V~R~\▒~V~R▒~V~R~J~B▒~V~R~B▒~V~R▒~V~R~Q 16 个▒~V~R~^▒~V~R~K (rank 16-31)▒~V~R~Ldp_address ▒~V~R~L~G▒~V~R~P~Q d0 (decode ▒~V~R~D▒~V~R~Z~D▒~V~R~E▒~V~R享 broker)▒~V~R~@~B
# ▒~V~R~O~B▒~V~R~U▒~V~R▒~V~R~H▒~V~R~N run_dp_template.sh ▒~V~R~@▒~V~R~G▒~V~R▒~V~R~I:
#   $1 ▒~V~R~O▒~V~R▒~V~R~A设▒~V~R~G    $2 vllm 端▒~V~R~O▒~V~R   $3 dp_size   $4 dp_rank
#   $5 dp_address  $6 rpc_port    $7 tp_size
export PYTHONPATH="/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:$PYTHONPATH"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export HCCL_BUFFSIZE=1024
export AFD_NPU_FFN_PROFILER_ENABLE=true
export AFD_NPU_FFN_PROFILER_DIR=./profile/ffn
export AFD_NPU_FFN_PROFILER_SKIP_FIRST=50
export AFD_NPU_FFN_PROFILER_ACTIVE=20                  # ffn 默认 20 步

nic_name="eth0"
local_ip="33.182.141.132"        # ▒~V~R~V~R~\▒~V~R~V~R▒~V~R~V~R~\▒~V~R~V~R IP
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

MODEL=/home/admin/model-csi/model

ADDITIONAL='{
  "enable_cpu_binding": false,
  "finegrained_tp_config": {"lmhead_tensor_parallel_size": 16},
  "afd": {
    "enabled": true,
    "role": "ffn",
    "connector": "camp2pconnector",
    "host": "33.182.141.223",
    "port": 29666,
    "num_attention_servers": 16,
    "num_ffn_servers": 16,
    "afd_server_rank": 0
  }
}'
COMPILATION='{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":['64']}'
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

#  --async-scheduling \

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
  --async-scheduling \
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
  --additional-config "$ADDITIONAL" \
  --kv-transfer-config "$KV_TRANSFER" \
  --compilation-config "$COMPILATION"
  > d1.log 2>&1 &