#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Deploy a serve pod so an external load generator
# (e.g. inference-perf) can benchmark it:
#   1. resolve MODEL_ID (the HF repo id, defaults to deepseek-ai/DeepSeek-V2-Lite,
#      overridable via the MODEL_ID env var); create the model PVC (name
#      defaults to deepseek-v2-lite-pvc, overridable via the PVC_NAME env
#      var) if it doesn't exist yet. The serve pod always serves MODEL_ID --
#      vLLM downloads the weights into the PVC's HF_HOME cache on a cold
#      volume and reuses them warm on every later run.
#   2. launch the serve pod, which runs the selected AFD recipe script
#      with the client-facing proxy rebound to 0.0.0.0 so it's reachable
#      off-pod
#   3. apply a Service in front of the proxy
#   4. wait for the serve pod to report ready
#   5. apply inference-perf-config.yaml + inference-perf-pod.yaml (the load
#      generator) and stream its logs until the run completes
#   6. copy inference-perf's reports out via copy-reports.sh
#
# The serve pod is left running afterwards (this script does NOT delete it).
#
# Usage: AFD_PLUGIN_IMAGE=<image> ./run.sh [recipe-script-path]
#
# AFD_PLUGIN_IMAGE must point at an image built from
# docker/Dockerfile.k8s-bench and pushed somewhere the cluster can pull it.
#
# recipe-script-path is the AFD recipe script to run inside the serve pod,
# given as a path relative to the afd-plugin repo root baked into the image
# (/opt/afd-plugin). Defaults to the 2P1A1F graph+DBO recipe; can also be set
# via the RECIPE_SCRIPT_PATH env var. Example:
#   ./run.sh recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_disaggregation/baseline.sh
#
# GPU_COUNT sets the pod's nvidia.com/gpu request/limit (default 4, matching
# the 2 prefill + 1 attention + 1 FFN workers the default recipe launches).
#
# MODEL_ID sets the HF repo id served by the pod (default
# deepseek-ai/DeepSeek-V2-Lite).
#
# PVC_NAME sets the name of the PersistentVolumeClaim used to cache the
# model weights (default deepseek-v2-lite-pvc).
#
# Requires an authenticated `kubectl`/`oc` session in the target namespace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_K8S_DIR="$(cd "${SCRIPT_DIR}/../../../../recipe/gpu/P2pNcclAFDConnector/kubernetes" && pwd)"

IMAGE="${AFD_PLUGIN_IMAGE}"
RECIPE_SCRIPT_PATH="${1:-${RECIPE_SCRIPT_PATH:-recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_graph_dbo_dp2tp1.sh}}"
GPU_COUNT="${GPU_COUNT:-4}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V2-Lite}"
PVC="${PVC_NAME:-deepseek-v2-lite-pvc}"
POD=vllm-pod
SVC=vllm-service
INF_POD=inference-perf

command -v envsubst >/dev/null || { echo "envsubst (gettext) is required" >&2; exit 1; }

echo "=== [1/6] apply PVC ==="
if kubectl get pvc "${PVC}" >/dev/null 2>&1; then
  echo "PVC ${PVC} already exists; serving ${MODEL_ID} from its warm HF_HOME cache"
else
  echo "PVC ${PVC} does not exist; creating it empty -- vLLM will download ${MODEL_ID} into it on first use"
  # shellcheck disable=SC2016
  PVC_NAME="${PVC}" envsubst '${PVC_NAME}' < "${RECIPE_K8S_DIR}/pvc.yaml" | kubectl apply -f -
fi

echo "=== [2/6] apply serve pod (recipe: ${RECIPE_SCRIPT_PATH}) ==="
kubectl delete pod "${POD}" --ignore-not-found
# shellcheck disable=SC2016
TEMPLATE_IMAGE="${IMAGE}" TEMPLATE_MODEL="${MODEL_ID}" RECIPE_SCRIPT_PATH="${RECIPE_SCRIPT_PATH}" GPU_COUNT="${GPU_COUNT}" PVC_NAME="${PVC}" \
  envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${RECIPE_SCRIPT_PATH} ${GPU_COUNT} ${PVC_NAME}' \
  < "${RECIPE_K8S_DIR}/serve-bench-pod.yaml" | kubectl apply -f -

echo "=== [3/6] apply Service in front of the proxy ==="
kubectl apply -f "${RECIPE_K8S_DIR}/service.yaml"

echo "=== waiting for pod to reach Running ==="
until [ "$(kubectl get pod "${POD}" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; do
  phase="$(kubectl get pod "${POD}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  [ "$phase" = "Failed" ] && { echo "pod Failed"; kubectl logs "${POD}" --tail=50; exit 1; }
  sleep 10
done

echo "=== [4/6] waiting for the disaggregation stack to report ready ==="
kubectl logs -f "pod/${POD}" &
log_pid=$!
while true; do
  pod_logs="$(kubectl logs "pod/${POD}" 2>/dev/null || true)"
  echo "$pod_logs" | grep -q "stack READY" && break
  if echo "$pod_logs" | grep -q "ERROR:"; then
    echo "serve pod hit an error during startup"
    kubectl logs "${POD}" --tail=200
    exit 1
  fi
  if [ "$(kubectl get pod "${POD}" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Failed" ]; then
    echo "pod Failed"
    kubectl logs "${POD}" --tail=200
    exit 1
  fi
  sleep 10
done
kill "${log_pid}" 2>/dev/null || true

echo
echo "=== ${POD} is up; proxy is exposed via Service ${SVC} ==="
echo "From another pod in this namespace (e.g. inference-perf), reach it at:"
echo "  http://${SVC}:18305"

echo
echo "=== [5/6] apply inference-perf load-generator pod (service is ready) ==="
kubectl delete pod "${INF_POD}" --ignore-not-found
# shellcheck disable=SC2016
MODEL_ID="${MODEL_ID}" \
  envsubst '${MODEL_ID}' \
  < "${SCRIPT_DIR}/inference-perf-config.yaml" | kubectl apply -f -
kubectl apply -f "${SCRIPT_DIR}/inference-perf-pod.yaml"

echo "=== waiting for ${INF_POD} pod to start ==="
until [ "$(kubectl get pod "${INF_POD}" -o jsonpath='{.status.phase}' 2>/dev/null)" != "Pending" ]; do
  sleep 5
done

echo "--- streaming ${INF_POD} logs until the load test completes ---"
kubectl logs -f "pod/${INF_POD}" || true

# `kubectl logs -f` can return early on a dropped/reset connection well
# before the pod itself finishes -- don't trust it as a completion signal.
# Explicitly poll for a terminal phase so reports aren't copied out from a
# load test that's still running.
echo "=== waiting for ${INF_POD} pod to reach a terminal phase ==="
until INF_PHASE="$(kubectl get pod "${INF_POD}" -o jsonpath='{.status.phase}' 2>/dev/null)"; \
    [ "${INF_PHASE}" = "Succeeded" ] || [ "${INF_PHASE}" = "Failed" ]; do
  sleep 10
done
echo
echo "=== ${INF_POD} pod phase: ${INF_PHASE} ==="

echo "=== [6/6] copying inference-perf reports out ==="
"${SCRIPT_DIR}/copy-reports.sh"

echo
echo "Tail the serve pod's logs any time with:"
echo "  kubectl logs -f pod/${POD}"
echo "Delete the serve pod when done (frees the GPUs) and the Service:"
echo "  kubectl delete pod ${POD}"
echo "  kubectl delete -f ${RECIPE_K8S_DIR}/service.yaml"
