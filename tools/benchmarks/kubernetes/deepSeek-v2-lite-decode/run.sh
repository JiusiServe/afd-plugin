#!/usr/bin/env bash
# Orchestrate a DeepSeek-V2-Lite AFD recipe run end-to-end:
#   1. download the model to deepseek-v2-lite-pvc (Job)
#   2. launch the serve+bench pod for the requested recipe, copy the local
#      afd-plugin repo in, build the venv, run `vllm serve`, benchmark, copy
#      results out, delete the pod
#
# Usage: ./run.sh <baseline|2a2f>
#   baseline -- plain, non-disaggregated `vllm serve` (no AFD, no DBO)
#   2a2f     -- 2a2f_graph_dbo_dp1tp2 AF-disaggregated recipe
#
# Requires an authenticated `kubectl`/`oc` session in the target namespace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -ne 1 ] || [ ! -f "${SCRIPT_DIR}/decode_bench_config_${1}.sh" ]; then
  echo "Usage: $0 <recipe>, where <recipe> has a decode_bench_config_<recipe>.sh in ${SCRIPT_DIR}" >&2
  exit 1
fi
RECIPE="$1"
RECIPE_CONFIG="decode_bench_config_${RECIPE}.sh"

LOCAL_RESULTS="${LOCAL_RESULTS:-${SCRIPT_DIR}/results}"
JOB=deepseek-v2-lite-downloader
PVC=deepseek-v2-lite-pvc
POD=afd-deepseek-v2-lite-serve-bench

command -v envsubst >/dev/null || { echo "envsubst (gettext) is required" >&2; exit 1; }

run_stage() {
  local label="$1" recipe_config="$2" pod="${POD}"

  echo "=== [${label}] apply serve+bench pod ==="
  TEMPLATE_RECIPE_CONFIG="${recipe_config}" \
    envsubst '${TEMPLATE_RECIPE_CONFIG}' \
    < "${SCRIPT_DIR}/serve-bench-pod.yaml" | kubectl apply -f -

  echo "=== [${label}] waiting for pod to reach Running ==="
  until [ "$(kubectl get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; do
    phase="$(kubectl get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    [ "$phase" = "Failed" ] && { echo "pod Failed"; kubectl logs "${pod}" --tail=50; exit 1; }
    sleep 10
  done

  echo "=== [${label}] resolving on-node DCGM host-engine pod IP ==="
  local node dcgm_ip
  node="$(kubectl get pod "${pod}" -o jsonpath='{.spec.nodeName}')"
  dcgm_ip="$(kubectl get pod -n nvidia-gpu-operator -l app=nvidia-dcgm \
    --field-selector "spec.nodeName=${node}" \
    -o jsonpath='{.items[0].status.podIP}' 2>/dev/null || true)"
  kubectl exec "${pod}" -- mkdir -p /work
  if [ -n "${dcgm_ip}" ]; then
    echo "DCGM host-engine for node ${node}: ${dcgm_ip}"
    kubectl exec "${pod}" -- bash -c "echo '${dcgm_ip}' > /work/DCGM_HOST_IP"
  else
    echo "WARNING: could not resolve DCGM host-engine pod IP for node ${node}; DCGM profiling will be skipped"
  fi

  echo "=== [${label}] copying local afd-plugin repo into pod ==="
  kubectl exec "${pod}" -- mkdir -p /work/afd-plugin
  # copy repo contents (excluding heavy .git) then signal readiness
  tar --exclude=.git --exclude=runs/results -C "${SCRIPT_DIR}/../../../.." -cf - . \
    | kubectl exec -i "${pod}" -- tar -xf - -C /work/afd-plugin
  kubectl exec "${pod}" -- touch /work/afd-plugin/REPO_READY

  echo "--- [${label}] streaming pod logs until benchmark completes ---"
  kubectl logs -f "pod/${pod}" &
  local log_pid=$!

  echo "=== [${label}] waiting for benchmark completion sentinel (/work/BENCH_DONE) ==="
  until kubectl exec "pod/${pod}" -- test -f /work/BENCH_DONE 2>/dev/null; do
    sleep 15
  done
  kill "${log_pid}" 2>/dev/null || true

  echo "=== [${label}] copy results to ${LOCAL_RESULTS}/${label} ==="
  mkdir -p "${LOCAL_RESULTS}/${label}"
  kubectl cp "${pod}:/models/results" "${LOCAL_RESULTS}/${label}"
  echo "=== [${label}] results copied ==="

  echo "=== [${label}] deleting pod ${pod} ==="
  kubectl delete pod "${pod}" --ignore-not-found
}

echo "=== [1/2] apply model PVC + downloader job ==="
if kubectl get pvc "${PVC}" >/dev/null 2>&1; then
  echo "PVC ${PVC} already exists; skipping download job"
else
  kubectl apply -f "${SCRIPT_DIR}/pvc.yaml"
  kubectl delete job "${JOB}" --ignore-not-found
  kubectl apply -f "${SCRIPT_DIR}/download-job.yaml"

  echo "=== waiting for download job to complete (timeout 30m) ==="
  kubectl wait --for=condition=complete "job/${JOB}" --timeout=30m
fi

echo "=== [2/2] ${RECIPE} run ==="
run_stage "${RECIPE}" "${RECIPE_CONFIG}"

echo "=== results copied ==="
ls -la "${LOCAL_RESULTS}"
echo
echo "Run complete; pod has been deleted."
echo "Delete the downloader job when done:"
echo "  kubectl delete job ${JOB}"
