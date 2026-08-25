#!/usr/bin/env bash
set -euo pipefail

# Two A3-task DSV4 async CAM deployment.
#
# Attention is one global DP3TP8 vLLM process group, started once on each
# node: node1 owns DP0-1 (NPUs 0-15) and hosts the API; node2 owns DP2
# (NPUs 0-7) and joins headlessly.  Both Attention invocations use node1 as
# the vLLM DP coordinator.  CAM rendezvous remains local to each CAM DP group.
# node2 also hosts the shared FFN EP8 pool on NPUs 8-15.
: "${ROLE:?ROLE must be attention or ffn}"
: "${NODE_IP:?NODE_IP is required}"
: "${ATTENTION_NODE_ID:=1}"  # 1 = node1 (DP0-1), 2 = node2 (DP2)
: "${API_PORT:=8900}"
: "${DP_RPC_PORT:=29550}"
: "${NIC_NAME:=eth0}"
: "${VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${PROFILE_VARIANT:=none}"
: "${PROFILE_ROOT:=/tmp/dsv4_afd_profiles}"
: "${GPU_MEMORY_UTILIZATION:=0.70}"
: "${HCCL_BUFFSIZE:=512}"

case "$ROLE" in
  attention|ffn) ;;
  *) echo "ROLE must be attention or ffn" >&2; exit 2 ;;
esac

if [[ "$ROLE" == attention ]]; then
  case "$ATTENTION_NODE_ID" in
    1)
      ATTN_DP_SIZE_LOCAL=2
      ATTN_DP_START_RANK=0
      ATTN_HEADLESS_ARGS=()
      API_SERVER_ARGS=(--api-server-count 1)
      ;;
    2)
      ATTN_DP_SIZE_LOCAL=1
      ATTN_DP_START_RANK=2
      ATTN_HEADLESS_ARGS=(--headless)
      API_SERVER_ARGS=()
      ;;
    *)
      echo "ATTENTION_NODE_ID must be 1 or 2 for ROLE=attention" >&2
      exit 2
      ;;
  esac
fi

case "$PROFILE_VARIANT" in
  none)
    PROFILER_ARGS=()
    ;;
  full)
    PROFILE_DIR="$PROFILE_ROOT/afd_${ROLE}_node${ATTENTION_NODE_ID}_full"
    PROFILER_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_memory\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":10,\"max_iterations\":9,\"warmup_iterations\":0,\"active_iterations\":10,\"wait_iterations\":0}")
    ;;
  ops)
    PROFILE_DIR="$PROFILE_ROOT/afd_${ROLE}_node${ATTENTION_NODE_ID}_ops"
    PROFILER_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":10,\"max_iterations\":9,\"warmup_iterations\":0,\"active_iterations\":10,\"wait_iterations\":0}")
    ;;
  *)
    echo "PROFILE_VARIANT must be none, full, or ops" >&2
    exit 2
    ;;
esac

MODEL_PATH=/mnt/sfs_turbo/models/DeepSeek-V4-Flash-w8a8-mtp
PLUGIN_ROOT=/a3_inference/itask/workdir/wb02363348/bjf_afd/code/afd-plugin
NODE1_IP=33.182.143.180
NODE2_IP=33.182.143.182

source /usr/local/Ascend/cann-9.0.1/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export PYTHONPATH="$PLUGIN_ROOT:/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:${PYTHONPATH:-}"
export VLLM_PLUGINS=ascend,afd
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export AFD_FORCE_SPAWN_MULTIPROCESSING=1
CAM_VENDOR=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM
CAM_OPAPI_DIR="$CAM_VENDOR/op_api/lib"
CAM_OPAPI="$CAM_OPAPI_DIR/libopapi.so"
export ASCEND_CUSTOM_OPP_PATH="$CAM_VENDOR:${ASCEND_CUSTOM_OPP_PATH:-}"
# umdk_cam_op_lib resolves libopapi.so by name on first request.  Put CAM's
# vendor implementation before CANN's stock libopapi.so, which lacks CAM ops.
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:"$CAM_OPAPI_DIR":"$CAM_VENDOR/op_api":/usr/local/Ascend/cann-9.0.1/aarch64-linux/lib64:/usr/local/Ascend/cann-9.0.1/runtime/lib64:${LD_LIBRARY_PATH:-}
export CAM_CUST_OPAPI_LIB_PATH="$CAM_OPAPI"
export LD_PRELOAD="$CAM_OPAPI${LD_PRELOAD:+:$LD_PRELOAD}"
export HCCL_IF_IP="$NODE_IP"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_BUFFSIZE HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800
export OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

if [[ "$ROLE" == attention ]]; then
  # Keep all Attention ranks in one vLLM DP3TP8 process group.  The connector
  # derives AFD role rank from vLLM's global DP rank and TP rank, so no manual
  # per-node role-rank offset is used here.
  PARALLEL_ARGS=(
    --data-parallel-size 3
    --data-parallel-size-local "$ATTN_DP_SIZE_LOCAL"
    --data-parallel-start-rank "$ATTN_DP_START_RANK"
    --data-parallel-address "$NODE1_IP"
    --data-parallel-rpc-port "$DP_RPC_PORT"
    --tensor-parallel-size 8
    "${ATTN_HEADLESS_ARGS[@]}"
  )
  WORKER=afd_plugin.v1.worker.npu.AFDNPUAttentionWorker
  MODEL_NAME=dsv4-afd-attention
else
  PARALLEL_ARGS=(--data-parallel-size 8 --tensor-parallel-size 1)
  API_SERVER_ARGS=(--api-server-count 1)
  WORKER=afd_plugin.v1.worker.npu.AFDNPUFFNWorker
  MODEL_NAME=dsv4-afd-ffn
fi

ADDITIONAL_CONFIG="{\"enable_force_load_balance\":false,\"afd\":{\"role\":\"$ROLE\",\"connector\":\"CAMAsyncAFDConnector\",\"async\":true,\"host\":\"$NODE1_IP\",\"port\":1239,\"num_attention_ranks\":24,\"num_ffn_ranks\":8,\"compute_gate_on_attention\":true,\"connector_extra_config\":{\"dynamicQuant\":1,\"attn_ranks_per_dp\":8,\"shared_ffn_pool\":true,\"async_moe_ubatching\":true,\"cam_rendezvous_hosts\":[\"$NODE1_IP\",\"$NODE1_IP\",\"$NODE2_IP\"],\"scheduler_host\":\"$NODE2_IP\"}}}"

exec env VLLM_USE_V1=1 /usr/local/python3.12.13/bin/vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$API_PORT" "${API_SERVER_ARGS[@]}" --served-model-name "$MODEL_NAME" \
  --worker-cls "$WORKER" "${PARALLEL_ARGS[@]}" --enable-expert-parallel \
  --enforce-eager --quantization ascend --tokenizer-mode deepseek_v4 \
  --block-size 128 --max-model-len 8192 --max-num-batched-tokens 1024 --max-num-seqs 2 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --trust-remote-code --no-enable-prefix-caching --enable-chunked-prefill \
  --additional-config "$ADDITIONAL_CONFIG" "${PROFILER_ARGS[@]}"
