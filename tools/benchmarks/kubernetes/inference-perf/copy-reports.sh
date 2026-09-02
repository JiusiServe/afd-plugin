#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Copy the inference-perf-reports PVC (written to by the inference-perf pod,
# per storage.local_storage.path: /reports in inference-perf-config.yaml's
# ConfigMap) out of the cluster to a local directory.
#
# The inference-perf pod (restartPolicy: Never) normally exits once its load
# test finishes, and `kubectl cp`/`kubectl exec` cannot reach a
# terminated/completed container -- so this spins up a short-lived helper
# pod that just mounts the same PVC read-only, copies out of THAT, then
# deletes the helper pod. Works whether the inference-perf pod is still
# around (Completed) or has already been deleted; only the PVC needs to
# still exist.
#
# Usage: ./copy-reports.sh [local-dir]
#   local-dir defaults to ./reports
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${1:-${SCRIPT_DIR}/reports}"
PVC=inference-perf-reports
HELPER=inference-perf-reports-copy

kubectl get pvc "${PVC}" >/dev/null 2>&1 || { echo "PVC ${PVC} not found" >&2; exit 1; }

echo "=== [1/3] starting helper pod to mount ${PVC} read-only ==="
kubectl delete pod "${HELPER}" --ignore-not-found
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${HELPER}
  labels:
    app: inference-perf
    role: reports-copy
spec:
  restartPolicy: Never
  containers:
    - name: copy
      image: busybox:1.36
      command: ["sleep", "3600"]
      volumeMounts:
        - name: reports
          mountPath: /reports
          readOnly: true
  volumes:
    - name: reports
      persistentVolumeClaim:
        claimName: ${PVC}
        readOnly: true
EOF

echo "=== waiting for helper pod to reach Running ==="
until [ "$(kubectl get pod "${HELPER}" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; do
  phase="$(kubectl get pod "${HELPER}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  if [ "$phase" = "Failed" ]; then
    echo "helper pod Failed"
    kubectl describe pod "${HELPER}" | tail -30
    kubectl delete pod "${HELPER}" --ignore-not-found
    exit 1
  fi
  sleep 5
done

mkdir -p "${LOCAL_DIR}"
echo "=== [2/3] copying ${HELPER}:/reports -> ${LOCAL_DIR} ==="
kubectl cp "${HELPER}:/reports/." "${LOCAL_DIR}"

echo "=== [3/3] cleaning up helper pod ==="
kubectl delete pod "${HELPER}" --ignore-not-found

echo "=== reports copied to ${LOCAL_DIR} ==="
ls -la "${LOCAL_DIR}"
