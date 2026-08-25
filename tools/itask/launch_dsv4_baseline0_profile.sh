#!/usr/bin/env bash
set -euo pipefail

# DP4 x TP4 x EP16 native DSV4 baseline on one 16-NPU A3 task.
# PROFILE_VARIANT=full records stack+shapes; PROFILE_VARIANT=ops records ops only.
: "${PROFILE_VARIANT:=full}"
: "${API_PORT:=9000}"
: "${PROFILE_ROOT:=/tmp/dsv4_dualnode_profiles}"

case "$PROFILE_VARIANT" in
  full) with_stack=true; record_shapes=true ;;
  ops) with_stack=false; record_shapes=false ;;
  *) echo "PROFILE_VARIANT must be full or ops, got $PROFILE_VARIANT" >&2; exit 2 ;;
esac

MODEL_PATH=/mnt/sfs_turbo/models/DeepSeek-V4-Flash-w8a8-mtp
PROFILE_DIR="$PROFILE_ROOT/baseline0_$PROFILE_VARIANT"
mkdir -p "$PROFILE_DIR"

source /usr/local/Ascend/cann-9.0.1/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH=/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:${PYTHONPATH:-}
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=4096
export HCCL_OP_EXPANSION_MODE=AIV

# v0.26 stops only when ``profiling_for_iters > max_iterations``.  Nine is
# therefore the value that records exactly ten active engine iterations.
PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":$with_stack,\"torch_profiler_record_shapes\":$record_shapes,\"torch_profiler_with_memory\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":10,\"max_iterations\":9,\"warmup_iterations\":0,\"active_iterations\":10,\"wait_iterations\":0}"

exec env VLLM_USE_V1=1 /usr/local/python3.12.13/bin/vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$API_PORT" --served-model-name dsv4-baseline0 \
  --data-parallel-size 4 --tensor-parallel-size 4 --enable-expert-parallel \
  --enforce-eager --quantization ascend --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 --block-size 128 --max-model-len 8192 \
  --max-num-batched-tokens 1024 --max-num-seqs 2 --gpu-memory-utilization 0.70 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --trust-remote-code --no-enable-prefix-caching --enable-chunked-prefill \
  --profiler-config "$PROFILER_CONFIG"
