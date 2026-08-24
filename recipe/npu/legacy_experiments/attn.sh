#!/bin/bash
# export AFD_CAMP2P_STUB_IO=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH="/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:$PYTHONPATH"
export HCCL_BUFFSIZE=1024
# ---- NPU profiler (AFD_NPU_ATTENTION_PROFILER_*) ----
# 重要:trace 只在 vllm 正常 shutdown() 时 flush。停服务必须用 SIGTERM(kill -TERM / pkill -TERM),
#       绝不能 kill -9、进程也不能崩 —— 否则采不到/不完整。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AFD_NPU_ATTENTION_PROFILER_ENABLE=true
export AFD_NPU_ATTENTION_PROFILER_DIR="$SCRIPT_DIR/profile/attn"   # 产物落这(绝对路径)
export AFD_NPU_ATTENTION_PROFILER_SKIP_FIRST=50        # 跳过前 50 step(默认 1500);要稳态可调大
export AFD_NPU_ATTENTION_PROFILER_ACTIVE=10            # 采集 10 step

nic_name="eth0"
local_ip="33.182.142.7"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

MODEL=/home/admin/model-csi/model

VLLM_USE_V1=1 vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8006 \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --enable-expert-parallel \
  --max_num_batched_tokens 8 \
  --max_num_seqs 8 \
  --seed 1024 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.93 \
  --async-scheduling \
  --served-model-name dsv3 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --tokenizer-mode deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": ['8']}'  \
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
    "enable_cpu_binding": false,
    "finegrained_tp_config": {"lmhead_tensor_parallel_size": 16},
    "enable_force_load_balance": true,
    "afd": {
      "enabled": true,
      "role": "attention",
      "connector": "camp2pconnector",
      "host": "33.182.140.93",
      "port": 29666,
      "num_attention_servers": 16,
      "num_ffn_servers": 16,
      "afd_server_rank": 0
    }
  }' > attn.log 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > attn.pid
disown "$VLLM_PID"   # 脱离 shell,itask session 断了也活着;停的时候用 kill -TERM $(cat attn.pid)
