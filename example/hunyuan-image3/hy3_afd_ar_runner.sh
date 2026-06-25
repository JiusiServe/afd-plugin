#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${AFD_MODEL:-${AFD_GPU_E2E_MODEL:-}}"

if [[ -z "${MODEL}" ]]; then
  echo "Set AFD_MODEL to a local HunyuanImage3-Instruct model path." >&2
  exit 2
fi

# Default FFN deploy config restricts the process to AR Stage 0 only,
# preventing --omni from also loading the DiT Stage 1 (which hardcodes
# devices="2,3" and causes OOM on unintended GPUs).
FFN_DEPLOY_CONFIG="${AFD_FFN_DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy/hunyuan_image3_ar_afd_ffn.yaml}"
ATTN_DEPLOY_CONFIG="${AFD_ATTN_DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy/hunyuan_image3_ar_afd_attn.yaml}"
exec python "${SCRIPT_DIR}/run.py" \
  --model "${MODEL}" \
  --vllm-bin "${AFD_VLLM_BIN:-vllm}" \
  --num-attention-servers 2 \
  --num-ffn-servers 2 \
  --attention-gpus "${AFD_ATTENTION_GPUS:-0,1}" \
  --ffn-gpus "${AFD_FFN_GPUS:-2,3}" \
  --api-port-base "${AFD_API_PORT_BASE:-18000}" \
  --afd-port "${AFD_PORT:-6239}" \
  --max-tokens "${AFD_MAX_TOKENS:-8}" \
  --startup-timeout "${AFD_STARTUP_TIMEOUT:-900}" \
  --ffn-start-delay "${AFD_FFN_START_DELAY:-25}" \
  --common-vllm-arg=--trust-remote-code \
  --ffn-vllm-arg=--deploy-config \
  --ffn-vllm-arg="${FFN_DEPLOY_CONFIG}" \
  --attention-vllm-arg=--deploy-config \
  --attention-vllm-arg="${ATTN_DEPLOY_CONFIG}" \

