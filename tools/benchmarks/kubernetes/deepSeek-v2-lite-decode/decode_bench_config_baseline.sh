#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Baseline decode-bench recipe: a single, non-disaggregated vLLM instance
# (no AFD, no DBO) for comparison against the 2A2F recipe.

RESULT_PREFIX="baseline_noafd_dp4tp1"

setup_recipe() {
  : # AFD plugin intentionally not installed for the baseline run
}

launch_servers() {
  CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL_PATH" \
      --data-parallel-size 4 \
      --tensor-parallel-size 1 \
      --enable-expert-parallel \
      --kv-transfer-config "$DECODE_BENCH_KV_CONFIG" \
      --max-num-seqs 64 \
      --max-num-batched-tokens 64 \
      --max-cudagraph-capture-size 64 \
      --compilation-config '{
          "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[64]
      }' \
      --host 127.0.0.1 \
      --port 18305 \
      --trust-remote-code > attn.log 2>&1 &
}
