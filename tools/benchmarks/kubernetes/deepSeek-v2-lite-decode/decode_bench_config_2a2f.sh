#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# AF-disaggregated (2A2F) decode-bench recipe: two vLLM instances -- an
# attention instance (with the decode-bench connector attached) and an ffn
# instance -- connected via P2pNcclAFDConnector, with DBO enabled. Compare
# recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_disaggregation/
# 2p1a1f_graph_dbo.sh, where the decode side pulls real KV from a separate 1P
# prefill instance via a proxy instead of fabricating it.

RESULT_PREFIX="2a2f_graph_dbo_dp2tp1"

setup_recipe() {
  uv pip install nixl
  # .git is not copied into the pod, so setuptools_scm cannot infer a
  # version; pin one explicitly for the editable build.
  SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv pip install --no-deps -e "$REPO"
}

launch_servers() {
  # --- attention instance (decode-bench connector attached here) --------------
  CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
      --data-parallel-size 2 \
      --tensor-parallel-size 1 \
      --enable-expert-parallel \
      --additional-config '{
          "afd": {
              "role": "attention",
              "connector": "P2pNcclAFDConnector",
              "host": "127.0.0.1",
              "port": 6269,
              "num_attention_ranks": 2,
              "num_ffn_ranks": 2
          }
      }' \
      --kv-transfer-config "$DECODE_BENCH_KV_CONFIG" \
      --max-num-seqs 64 \
      --max-num-batched-tokens 64 \
      --enable-dbo \
      --dbo-decode-token-threshold 2 \
      --dbo-prefill-token-threshold 12 \
      --max-cudagraph-capture-size 64 \
      --compilation-config '{
          "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[64]
      }' \
      --host 127.0.0.1 \
      --port 18305 \
      --trust-remote-code > attn.log 2>&1 &

  # --- ffn instance (no decode-bench connector) --------------------------------
  CUDA_VISIBLE_DEVICES=2,3 vllm serve "$MODEL_PATH" \
      --data-parallel-size 2 \
      --tensor-parallel-size 1 \
      --enable-expert-parallel \
      --additional-config '{
          "afd": {
              "role": "ffn",
              "connector": "P2pNcclAFDConnector",
              "host": "127.0.0.1",
              "port": 6269,
              "num_attention_ranks": 2,
              "num_ffn_ranks": 2
          }
      }' \
      --max-num-seqs 64 \
      --enable-dbo \
      --dbo-decode-token-threshold 2 \
      --dbo-prefill-token-threshold 12 \
      --max-num-batched-tokens 64 \
      --max-cudagraph-capture-size 64 \
      --compilation-config '{
          "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[64]
      }' \
      --host 127.0.0.1 \
      --port 18305 \
      --trust-remote-code > ffn.log 2>&1 &
}
