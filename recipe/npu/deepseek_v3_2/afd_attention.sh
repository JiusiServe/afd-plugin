#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V3.2 W8A8 model directory}"
: "${LOCAL_IP:?Set LOCAL_IP to the communication IP of this node}"
: "${DP_ADDRESS:?Set DP_ADDRESS to the attention node hosting DP rank 0}"
: "${DP_START_RANK:?Set DP_START_RANK to the first attention rank on this node}"
: "${AFD_HOST:?Set AFD_HOST to the FFN node hosting FFN role rank 0}"
: "${ATTENTION_RANKS:?Set ATTENTION_RANKS to 48 or 64}"
: "${INPUT_LENGTH:?Set INPUT_LENGTH to 16384 or 32768}"
: "${BATCH_SIZE:?Set BATCH_SIZE to 28 for 16K or 14 for 32K}"
: "${NIC_NAME:?Set NIC_NAME to the NPU node communication interface; check it with ifconfig}"

SERVER_PORT="${SERVER_PORT:-8338}"
DP_RPC_PORT="${DP_RPC_PORT:-13377}"
AFD_PORT="${AFD_PORT:-29666}"
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

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

if (( DP_START_RANK < 0 || DP_START_RANK >= ATTENTION_RANKS || DP_START_RANK % 16 != 0 )); then
  echo "DP_START_RANK must be a 16-rank node boundary below ATTENTION_RANKS" >&2
  exit 2
fi

if [[ "$INPUT_LENGTH" == 16384 && "$BATCH_SIZE" != 28 ]] || \
   [[ "$INPUT_LENGTH" == 32768 && "$BATCH_SIZE" != 14 ]]; then
  echo "AFD attention requires BATCH_SIZE=28 for 16K or BATCH_SIZE=14 for 32K" >&2
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
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
if [[ "$DP_START_RANK" != 0 ]]; then
  HEADLESS_ARGS=(--headless)
fi

ADDITIONAL_CONFIG="$(printf \
  '{"enable_force_load_balance":true,"force_load_balance_topn_per_rank":4,"afd":{"enabled":true,"role":"attention","connector":"CAMP2pAFDConnector","host":"%s","port":%s,"num_attention_ranks":%s,"num_ffn_ranks":16,"extra_config":{"afd_size":"%sA16F"}}}' \
  "$AFD_HOST" "$AFD_PORT" "$ATTENTION_RANKS" "$ATTENTION_RANKS")"

exec env VLLM_USE_V1=1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$SERVER_PORT" \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker \
  --data-parallel-size "$ATTENTION_RANKS" \
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
  --compilation-config \
    "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$BATCH_SIZE]}" \
  --kv-transfer-config \
    '{"kv_connector":"AFDDecodeBenchConnector","kv_connector_module_path":"tools.benchmarks.decode_bench","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.015,"fill_std":0.0}}' \
  --additional-config "$ADDITIONAL_CONFIG" \
  "${HEADLESS_ARGS[@]}"
