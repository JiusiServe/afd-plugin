#!/usr/bin/env bash
# Startup isolation for DeepSeek-V4 Async CAM AFD.
# Usage: bash diagnose_worker_startup.sh native|attention-worker|ffn-worker
set -euo pipefail

MODE="${1:?usage: $0 native|attention-worker|ffn-worker}"
case "$MODE" in native|attention-worker|ffn-worker) ;; *) echo "unknown mode: $MODE" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/afd_dsv4_startup_diagnose}"
MODEL_PATH="${MODEL_PATH:-/mnt/sfs_turbo/models/DeepSeek-V4-Flash-w8a8-mtp}"
AFD_HOST="${AFD_HOST:-33.215.116.191}"
AFD_PORT="${AFD_PORT:-1239}"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/cann-9.0.1/set_env.sh}"

[[ -f "$CANN_SET_ENV" ]] || { echo "missing CANN environment: $CANN_SET_ENV" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CANN_SET_ENV"
export PYTHONPATH="$PLUGIN_ROOT:/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:${PYTHONPATH:-}"
export VLLM_PLUGINS="ascend,afd"
export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib/libopapi.so${LD_PRELOAD:+:$LD_PRELOAD}"
export ASCEND_CUSTOM_OPP_PATH="/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM:${ASCEND_CUSTOM_OPP_PATH:-}"
export HCCL_IF_IP="$AFD_HOST"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export TP_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
export HCCL_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${MODE}.log"
: >"$LOG_FILE"

if [[ "$MODE" == native ]]; then
  # Native control: use the container's proven DSV4 Flash INT8 shape.  TP8
  # does not fit this checkpoint on one 910A3 partition; DP4 x TP4 does.
  # Keep afd-plugin loaded for its DSV4 Ascend compatibility patch, but use
  # the stock NPU worker and native model implementation (no AFD worker).
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
  export VLLM_PLUGINS="ascend,afd"
  exec env VLLM_USE_V1=1 /usr/local/python3.12.13/bin/vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8910 --served-model-name dsv4-native-diagnose \
    --tensor-parallel-size 4 --data-parallel-size 4 --enable-expert-parallel \
    --enforce-eager --quantization ascend --tokenizer-mode deepseek_v4 \
    --max-model-len 8192 --max-num-batched-tokens 1024 --max-num-seqs 2 \
    --gpu-memory-utilization 0.90 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
    --trust-remote-code >>"$LOG_FILE" 2>&1
fi

if [[ "$MODE" == attention-worker ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  worker="afd_plugin.v1.worker.npu.AFDNPUAttentionWorker"
  role="attention"
  port=8911
  gpu_memory_utilization=0.85
else
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
  worker="afd_plugin.v1.worker.npu.AFDNPUFFNWorker"
  role="ffn"
  port=8912
  gpu_memory_utilization=0.90
fi

# Keep the real role-local 8-card shape: a single rank cannot hold the FFN
# experts.  The absent peer role makes CAM rendezvous block only after model
# construction, which is the intended boundary for this diagnostic.
ADDITIONAL_CONFIG="{\"afd\":{\"role\":\"$role\",\"connector\":\"CAMAsyncAFDConnector\",\"async\":true,\"host\":\"$AFD_HOST\",\"port\":$AFD_PORT,\"num_attention_ranks\":8,\"num_ffn_ranks\":8,\"compute_gate_on_attention\":true,\"connector_extra_config\":{\"dynamicQuant\":1,\"attn_ranks_per_dp\":1,\"async_moe_ubatching\":false}}}"
exec env VLLM_USE_V1=1 /usr/local/python3.12.13/bin/vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$port" --served-model-name "dsv4-$role-diagnose" \
  --worker-cls "$worker" --tensor-parallel-size 1 --data-parallel-size 8 \
  --enable-expert-parallel --enforce-eager --quantization ascend \
  --tokenizer-mode deepseek_v4 --max-model-len 8192 \
  --max-num-batched-tokens 1024 --max-num-seqs 2 --gpu-memory-utilization "$gpu_memory_utilization" \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
  --trust-remote-code --additional-config "$ADDITIONAL_CONFIG" >>"$LOG_FILE" 2>&1
