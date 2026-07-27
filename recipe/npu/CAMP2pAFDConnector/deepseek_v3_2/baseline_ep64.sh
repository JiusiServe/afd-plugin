#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V3.2 W8A8 model directory}"
: "${LOCAL_IP:?Set LOCAL_IP to the communication IP of this node}"
: "${DP_ADDRESS:?Set DP_ADDRESS to the node hosting DP rank 0}"
: "${DP_START_RANK:?Set DP_START_RANK to 0, 16, 32, or 48}"
: "${INPUT_LENGTH:?Set INPUT_LENGTH to 16384 or 32768}"
: "${BATCH_SIZE:?Set BATCH_SIZE to 16 for 16K or 8 for 32K}"
: "${NIC_NAME:?Set NIC_NAME; find the NPU network interface with ifconfig}"

SERVER_PORT="${SERVER_PORT:-8006}"
DP_RPC_PORT="${DP_RPC_PORT:-13345}"
DEFAULT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$DEFAULT_VISIBLE_DEVICES}"

case "$INPUT_LENGTH" in
  16384) MAX_MODEL_LEN=18432 ;;
  32768) MAX_MODEL_LEN=34816 ;;
  *)
    echo "INPUT_LENGTH must be 16384 or 32768, got $INPUT_LENGTH" >&2
    exit 2
    ;;
esac

case "$DP_START_RANK" in
  0|16|32|48) ;;
  *)
    echo "DP_START_RANK must be 0, 16, 32, or 48, got $DP_START_RANK" >&2
    exit 2
    ;;
esac

if [[ "$INPUT_LENGTH" == 16384 && "$BATCH_SIZE" != 16 ]] ||
   [[ "$INPUT_LENGTH" == 32768 && "$BATCH_SIZE" != 8 ]]; then
  echo "EP64 requires BATCH_SIZE=16 for 16K or BATCH_SIZE=8 for 32K" >&2
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
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

HEADLESS_ARGS=()
API_SERVER_ARGS=()
if [[ "$DP_START_RANK" == 0 ]]; then
  API_SERVER_ARGS=(--api-server-count 16)
else
  HEADLESS_ARGS=(--headless)
fi

COMPILATION_CONFIG="$(cat <<EOF
{
  "cudagraph_mode": "FULL_DECODE_ONLY",
  "cudagraph_capture_sizes": [$BATCH_SIZE]
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

ADDITIONAL_CONFIG='{
  "enable_force_load_balance": true
}'

exec env VLLM_USE_V1=1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$SERVER_PORT" \
  --data-parallel-size 64 \
  --data-parallel-size-local 16 \
  --data-parallel-start-rank "$DP_START_RANK" \
  --data-parallel-address "$DP_ADDRESS" \
  --data-parallel-rpc-port "$DP_RPC_PORT" \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --max-num-batched-tokens "$BATCH_SIZE" \
  --max-num-seqs "$BATCH_SIZE" \
  --seed 1024 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.93 \
  --async-scheduling \
  --served-model-name deepseek_v3_2 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --tokenizer-mode deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --compilation-config "$COMPILATION_CONFIG" \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --additional-config "$ADDITIONAL_CONFIG" \
  "${API_SERVER_ARGS[@]}" \
  "${HEADLESS_ARGS[@]}"
