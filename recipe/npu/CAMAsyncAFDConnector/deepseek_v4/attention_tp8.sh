#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Keep Attention and FFN on the same DSV4 Flash MTP checkpoint. See ffn_ep8.sh.
MODEL_PATH="/mnt/sfs_turbo/models/DeepSeek-V4-Flash-w8a8-mtp"
export MODEL_PATH
# Keep the rendezvous/HCCL address local to the current task Pod.  itask may
# migrate a stopped task, leaving a stale inherited IP in its environment.
AFD_HOST="${AFD_HOST_OVERRIDE:-$(awk '!/^#/ && index($1, "127.") != 1 && $1 != "::1" {print $1; exit}' /etc/hosts)}"
: "${AFD_PORT:=1239}"
: "${API_PORT:=8900}"
NIC_NAME="${NIC_NAME_OVERRIDE:-eth0}"
: "${MAX_NUM_BATCHED_TOKENS:=1024}"
: "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${AFD_DP_SIZE:=1}"
: "${AFD_TP_SIZE:=8}"
: "${AFD_SHARED_FFN_POOL:=false}"
export ASCEND_RT_VISIBLE_DEVICES

# Keep the runtime environment identical to the proven DSV4 AFD
# recipes.  In particular, worker subprocesses need CANN's driver and ATB
# libraries in addition to the CAM operator library below.
: "${CANN_SET_ENV:=/usr/local/Ascend/cann-9.0.1/set_env.sh}"
if [[ ! -f "$CANN_SET_ENV" ]]; then
  echo "CANN environment script not found: $CANN_SET_ENV" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CANN_SET_ENV"

# The nightly A3 image's CANN setup script does not export the driver or
# toolkit runtime directories. torch_npu needs both before vLLM imports it.
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/cann-9.0.1/aarch64-linux/lib64:/usr/local/Ascend/cann-9.0.1/runtime/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PLUGIN_ROOT:/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:${PYTHONPATH:-}"
export VLLM_PLUGINS="ascend,afd"
export AFD_FORCE_BALANCED_TOPK_IDS=0
# ``set_env.sh`` owns the CANN runtime library order.  Prepending CAM's
# devlib/op_api directories here makes vllm_ascend_C initialize against a
# mixed runtime before NPUWorker applies the role-local device mapping.
# CAM discovery uses ASCEND_CUSTOM_OPP_PATH below and must not override it.
# Required by the CAM Async Connector user guide: custom-op discovery and
# both op_api paths must precede the inherited CANN loader path.
export ASCEND_CUSTOM_OPP_PATH="/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api:${LD_LIBRARY_PATH}"
# CAM's vendor library must be named libopapi.so: the runtime resolves the
# aclnn symbols through that SONAME inside spawned worker processes.
CAM_CUST_OPAPI="/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib/libopapi.so"
export LD_PRELOAD="$CAM_CUST_OPAPI${LD_PRELOAD:+:$LD_PRELOAD}"
export HCCL_IF_IP="$AFD_HOST"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$AFD_HOST"
export HCCL_BUFFSIZE=4096
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
# Attention constructs the external CAM communicator while its model runner is
# being created.  Match the FFN-side connection window for the full 16-rank
# world so it can wait for post-load FFN initialization.
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}"
# Keep worker creation consistent with the FFN endpoint; see ffn_ep8.sh.
# The task environment may export ``forkserver``.  Python 3.12's forkserver
# fails while restoring this runtime's signal handlers, so this recipe must
# override (rather than default) the inherited setting.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export AFD_FORCE_SPAWN_MULTIPROCESSING=1

ADDITIONAL_CONFIG="$(printf '%s' "{
  \"enable_force_load_balance\": false,
  \"afd\": {
    \"role\": \"attention\",
    \"connector\": \"CAMAsyncAFDConnector\",
    \"async\": true,
    \"host\": \"$AFD_HOST\",
    \"port\": $AFD_PORT,
    \"num_attention_ranks\": 8,
    \"num_ffn_ranks\": 8,
    \"compute_gate_on_attention\": true,
    \"connector_extra_config\": {
      \"dynamicQuant\": 1,
      \"attn_ranks_per_dp\": $AFD_TP_SIZE,
      \"shared_ffn_pool\": $AFD_SHARED_FFN_POOL,
      \"async_moe_ubatching\": true
    }
  }
}")"

# Load from the same shared checkpoint with bounded per-rank concurrency.
exec env VLLM_USE_V1=1 /usr/local/python3.12.13/bin/vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$API_PORT" \
  --api-server-count 1 \
  --served-model-name dsv4-async \
  --worker-cls afd_plugin.v1.worker.npu.AFDNPUAttentionWorker \
  --data-parallel-size "$AFD_DP_SIZE" \
  --tensor-parallel-size "$AFD_TP_SIZE" \
  --enable-expert-parallel \
  --enforce-eager \
  --quantization ascend \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --block-size 128 \
  --max-model-len 8192 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.70 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --trust-remote-code \
  --additional-config "$ADDITIONAL_CONFIG"
