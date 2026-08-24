#!/bin/bash
export ASCEND_RT_VISIBLE_DEVICES=$1
export PYTHONPATH="/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:$PYTHONPATH"
export HCCL_BUFFSIZE=2048

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AFD_NPU_FFN_PROFILER_ENABLE=true
export AFD_NPU_FFN_PROFILER_DIR="$SCRIPT_DIR/profile/ffn"
mkdir -p "$AFD_NPU_FFN_PROFILER_DIR"
export AFD_NPU_FFN_PROFILER_SKIP_FIRST=50
export AFD_NPU_FFN_PROFILER_ACTIVE=20
nic_name="eth0"
local_ip="33.215.118.135"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

MODEL=/a3_inference/itask/workdir/shared/jcz/model/dsv3.2
#MODEL=/home/admin/model-csi/model

VLLM_USE_V1=1 vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8006 \
  --load-format dummy \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUFFNWorker \
  --data-parallel-size 16 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": ['16']}'  \
  --max_num_seqs 16 \
  --seed 1024 \
  --max_num_batched_tokens 16 \
  --max-model-len 18432 \
  --gpu-memory-utilization 0.93 \
  --async-scheduling \
  --served-model-name dsv3 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --tokenizer-mode deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --kv-transfer-config '{
    "kv_connector": "AFDDecodeBenchConnector",
    "kv_connector_module_path": "afd_plugin.connectors.decode_bench",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "fill_mean": 0.015,
      "fill_std": 0.0
    }
  }' \
  --additional-config '{
    "enable_force_load_balance": true,
    "force_load_balance_topn_per_rank": 4,
    "afd": {
      "enabled": true,
      "role": "ffn",
      "connector": "camp2pconnector",
      "host": "33.215.118.135",
      "port": 29666,
      "num_attention_ranks": 48,
      "num_ffn_ranks": 16,
      "extra_config": {
        "afd_size": "48A16F"
      }
    }
  }' > ffn1.log 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > ffn1.pid
disown "$VLLM_PID"
