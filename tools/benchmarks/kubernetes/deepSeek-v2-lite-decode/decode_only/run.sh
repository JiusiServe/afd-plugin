#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Orchestrate a DeepSeek-V2-Lite AFD recipe run end-to-end:
#   1. download the model to deepseek-v2-lite-pvc (Job)
#   2. launch the serve+bench pod for the requested recipe, run
#      `vllm serve`, benchmark, copy results out, delete the pod
#
# Usage: AFD_PLUGIN_IMAGE=<image> ./run.sh <baseline|2a2f>
#   baseline -- plain, non-disaggregated `vllm serve` (no AFD, no DBO)
#   2a2f     -- 2a2f_graph_dbo_dp1tp2 AF-disaggregated recipe
#
# AFD_PLUGIN_IMAGE must point at an image built from
# docker/Dockerfile.k8s-bench and pushed somewhere the cluster can pull it.
#
# Requires an authenticated `kubectl`/`oc` session in the target namespace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RECIPE="${1:-}"
case "$RECIPE" in
  baseline | 2a2f) ;;
  *) echo "Usage: $0 <baseline|2a2f>" >&2; exit 1 ;;
esac

IMAGE="${AFD_PLUGIN_IMAGE}"
LOCAL_RESULTS="${LOCAL_RESULTS:-${SCRIPT_DIR}/results}"
JOB=deepseek-v2-lite-downloader
PVC=deepseek-v2-lite-pvc
POD=afd-deepseek-v2-lite-serve-bench

command -v envsubst >/dev/null || { echo "envsubst (gettext) is required" >&2; exit 1; }

run_stage() {
  local label="$1" pod="${POD}"

  echo "=== [${label}] apply serve+bench pod ==="
  # shellcheck disable=SC2016
  TEMPLATE_RECIPE="${label}" TEMPLATE_RESULT_PREFIX="${label}" TEMPLATE_IMAGE="${IMAGE}" \
    envsubst '${TEMPLATE_RECIPE} ${TEMPLATE_RESULT_PREFIX} ${TEMPLATE_IMAGE}' \
    < "${SCRIPT_DIR}/serve-bench-pod.yaml" | kubectl apply -f -

  echo "=== [${label}] waiting for pod to reach Running ==="
  until [ "$(kubectl get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; do
    phase="$(kubectl get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    [ "$phase" = "Failed" ] && { echo "pod Failed"; kubectl logs "${pod}" --tail=50; exit 1; }
    sleep 10
  done

  kubectl exec "${pod}" -- mkdir -p /work

  echo "--- [${label}] streaming pod logs until benchmark completes ---"
  kubectl logs -f "pod/${pod}" &
  local log_pid=$!

  echo "=== [${label}] waiting for benchmark completion sentinel (/work/BENCH_DONE) ==="
  until kubectl exec "pod/${pod}" -- test -f /work/BENCH_DONE 2>/dev/null; do
    sleep 15
  done
  kill "${log_pid}" 2>/dev/null || true

  echo "=== [${label}] copy results to ${LOCAL_RESULTS} ==="
  mkdir -p "${LOCAL_RESULTS}"
  kubectl cp "${pod}:/models/results" "${LOCAL_RESULTS}"
  echo "=== [${label}] results copied ==="

  echo "=== [${label}] deleting pod ${pod} ==="
  kubectl delete pod "${pod}" --ignore-not-found
}

echo "=== [1/2] apply model PVC + downloader job ==="
if kubectl get pvc "${PVC}" >/dev/null 2>&1; then
  echo "PVC ${PVC} already exists; skipping download job"
else
  kubectl apply -f "${BENCHMARK_DIR}/pvc.yaml"
  kubectl delete job "${JOB}" --ignore-not-found
  kubectl apply -f "${BENCHMARK_DIR}/download-job.yaml"

  echo "=== waiting for download job to complete (timeout 30m) ==="
  kubectl wait --for=condition=complete "job/${JOB}" --timeout=30m
fi

echo "=== [2/2] ${RECIPE} run ==="
run_stage "${RECIPE}"

echo "=== results copied ==="
ls -la "${LOCAL_RESULTS}"
echo
echo "Run complete; pod has been deleted."
echo "Delete the downloader job when done:"
echo "  kubectl delete job ${JOB}"
