#!/bin/bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH="/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:${PYTHONPATH:-}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV

JEMALLOC=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
if [[ -f "$JEMALLOC" ]]; then
  export LD_PRELOAD="$JEMALLOC${LD_PRELOAD:+:$LD_PRELOAD}"
fi

MODEL=/mnt/sfs_turbo/models/DeepSeek-V4-Flash-w8a8-mtp
LOG_FILE=${LOG_FILE:-dsv4_baseline.log}
PROFILER_DIR=${PROFILER_DIR:-/a3_inference/itask/workdir/wb02363348/bjf_afd/code/afd/profiles/dsv4_stack_memory_shapes}
mkdir -p "$PROFILER_DIR"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8900 \
  --served-model-name dsv4 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --quantization ascend \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --block-size 128 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILER_DIR\",\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_memory\":true,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true}" \
  --enforce-eager \
  >"$LOG_FILE" 2>&1