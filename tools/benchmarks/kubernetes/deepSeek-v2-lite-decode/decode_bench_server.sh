#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Decode-only benchmark server launcher. Recipe-specific detail (how many
# vLLM instances to start and with which flags) lives in a config file passed
# as the first argument; see decode_bench_config_2a2f.sh and
# decode_bench_config_baseline.sh for the two recipes currently defined.
#
# This is the DECODE side of prefill-decode (PD) disaggregation with the
# prefill instance faked out: the AFDDecodeBenchConnector reports every
# prompt token except the last as externally computed and fills the KV cache
# with dummy values, so requests skip prefill and go straight into decode --
# letting you stress the decode path with arbitrary ISL, with no prefill
# instance, no LMCache producer, and no proxy. Throughput/latency are
# meaningful; generated text is garbage.
#
# The connector lives in tools/benchmarks/ and is NOT shipped in the wheel, so
# it is loaded purely via kv_connector_module_path and needs the repo root on
# PYTHONPATH (inherited by the vLLM scheduler/worker subprocesses).
#
# Usage:
#   MODEL_PATH=/path/to/DeepSeek-V2-Lite \
#     ./decode_bench_server.sh decode_bench_config_2a2f.sh
# then, once the instance(s) are ready, drive load against the decode port:
#   MODEL_PATH=/path/to/DeepSeek-V2-Lite tools/benchmarks/vllm_bench.sh
#
# A config file must define a `launch_servers` function that backgrounds one
# or more `vllm serve` processes (using the MODEL_PATH and
# DECODE_BENCH_KV_CONFIG variables set below) and returns without waiting.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <config-file>" >&2
  exit 1
fi
CONFIG_FILE="$1"

# --- make tools.benchmarks.decode_bench importable in vLLM subprocesses -------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}

# dummy-KV fill params for the decode-bench connector
FILL_MEAN=${FILL_MEAN:-0.015}
FILL_STD=${FILL_STD:-0.0}
DECODE_BENCH_KV_CONFIG=$(cat <<JSON
{"kv_connector":"AFDDecodeBenchConnector","kv_connector_module_path":"tools.benchmarks.decode_bench","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":${FILL_MEAN},"fill_std":${FILL_STD}}}
JSON
)

# --- per-GPU DCGM profiling (tensor core / SM / DRAM activity) ----------------
# 1002 SM_ACTIVE, 1003 SM_OCCUPANCY, 1004 PIPE_TENSOR_ACTIVE, 1005 DRAM_ACTIVE,
# 1013/1014/1015 tensor core INT8/FP16/FP64 active. The cluster's DCGM
# host-engine pods aren't on hostNetwork, so they're only reachable at their
# own pod IP (not the node's hostIP); run.sh resolves the host-engine pod on
# our node and drops its IP in /work/DCGM_HOST_IP before REPO_READY.
DCGM_HOST_ARGS=()
[ -f /work/DCGM_HOST_IP ] && DCGM_HOST_ARGS=(--host "$(cat /work/DCGM_HOST_IP)")
if command -v dcgmi >/dev/null 2>&1 && [ "${#DCGM_HOST_ARGS[@]}" -gt 0 ]; then
  dcgmi dmon "${DCGM_HOST_ARGS[@]}" -e 1002,1003,1004,1005,1013,1014,1015 -d 1000 > dcgm.log 2>&1 &
  DCGM_PID=$!
  trap '[ -n "${DCGM_PID:-}" ] && kill "$DCGM_PID" 2>/dev/null || true' EXIT
else
  echo "dcgmi not found or DCGM host-engine unresolved; skipping DCGM profiling" > dcgm.log
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

launch_servers

wait
