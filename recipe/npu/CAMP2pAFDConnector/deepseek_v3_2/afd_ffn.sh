#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V3.2 W8A8 model directory}"
: "${LOCAL_IP:?Set LOCAL_IP to the communication IP of this node}"
: "${AFD_HOST:?Set AFD_HOST to the IP of this FFN node}"
: "${ATTENTION_RANKS:?Set ATTENTION_RANKS to 48 or 64}"
: "${INPUT_LENGTH:?Set INPUT_LENGTH to 16384 or 32768}"
: "${BATCH_SIZE:?Set BATCH_SIZE to the attention-side per-rank batch size}"
: "${NIC_NAME:?Set NIC_NAME; find the NPU network interface with ifconfig}"

SERVER_PORT="${SERVER_PORT:-8006}"
AFD_PORT="${AFD_PORT:-29666}"
DEFAULT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$DEFAULT_VISIBLE_DEVICES}"
FFN_RANKS=16

case "$INPUT_LENGTH" in
  16384) MAX_MODEL_LEN=18432 ;;
  32768) MAX_MODEL_LEN=34816 ;;
  *)
    echo "INPUT_LENGTH must be 16384 or 32768, got $INPUT_LENGTH" >&2
    exit 2
    ;;
esac

case "$ATTENTION_RANKS" in
  48|64) ;;
  *)
    echo "ATTENTION_RANKS must be 48 or 64, got $ATTENTION_RANKS" >&2
    exit 2
    ;;
esac

if [[ "$INPUT_LENGTH" == 16384 && "$BATCH_SIZE" != 28 ]] ||
   [[ "$INPUT_LENGTH" == 32768 && "$BATCH_SIZE" != 14 ]]; then
  echo "AFD FFN requires BATCH_SIZE=28 for 16K or BATCH_SIZE=14 for 32K" >&2
  exit 2
fi

if (( ATTENTION_RANKS * BATCH_SIZE % FFN_RANKS != 0 )); then
  echo "ATTENTION_RANKS * BATCH_SIZE must be divisible by $FFN_RANKS" >&2
  exit 2
fi
FFN_BATCH_SIZE=$((ATTENTION_RANKS * BATCH_SIZE / FFN_RANKS))

export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,afd}"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"

ADDITIONAL_CONFIG="$(cat <<EOF
{
  "enable_force_load_balance": true,
  "force_load_balance_topn_per_rank": 4,
  "afd": {
    "role": "ffn",
    "connector": "CAMP2pAFDConnector",
    "host": "$AFD_HOST",
    "port": $AFD_PORT,
    "num_attention_ranks": $ATTENTION_RANKS,
    "num_ffn_ranks": 16
  }
}
EOF
)"

COMPILATION_CONFIG="$(cat <<EOF
{
  "cudagraph_mode": "FULL_DECODE_ONLY",
  "cudagraph_capture_sizes": [$FFN_BATCH_SIZE]
}
EOF
)"

KV_TRANSFER_CONFIG='{
  "kv_connector": "AFDDecodeBenchConnector",
  "kv_connector_module_path": "tools.benchmarks.decode_bench",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "fill_mean": 0.015,
    "fill_std": 0.0
  }
}'

exec env VLLM_USE_V1=1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$SERVER_PORT" \
  --worker-cls afd_plugin.v1.worker.npu.AFDNPUFFNWorker \
  --data-parallel-size "$FFN_RANKS" \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --max-num-batched-tokens "$FFN_BATCH_SIZE" \
  --max-num-seqs "$FFN_BATCH_SIZE" \
  --seed 1024 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.93 \
  --enable-dbo \
  --dbo-decode-token-threshold 2 \
  --dbo-prefill-token-threshold 12 \
  --ubatch-size 2 \
  --async-scheduling \
  --served-model-name deepseek_v3_2 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --tokenizer-mode deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --compilation-config "$COMPILATION_CONFIG" \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --additional-config "$ADDITIONAL_CONFIG"
