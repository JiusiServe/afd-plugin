#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Fetch tools/datasets/cp8sp50k_custom_dataset_text_matched_token_ids.jsonl
# (Git LFS, excluded from the default LFS fetch), reshape it into the
# ShareGPT format inference-perf-config.yaml's `data.type: shareGPT` loader
# expects (see convert_dataset_to_sharegpt.py), and upload the result onto
# the inference-perf-dataset PVC so the inference-perf pod can read it from
# /datasets/sharegpt.json.
#
# The inference-perf pod runs quay.io/inference-perf/inference-perf:latest,
# which has no access to this repo or its Git LFS storage, so this all runs
# locally (same authenticated git/kubectl session run.sh already assumes)
# and pushes the result in via a short-lived helper pod + `kubectl cp` --
# the same pattern copy-reports.sh uses, in reverse (upload, not download).
#
# Usage: ./prepare-dataset.sh [target-output-tokens]
#   target-output-tokens defaults to 256, passed through to
#   convert_dataset_to_sharegpt.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATASET_REL_PATH="tools/datasets/cp8sp50k_custom_dataset_text_matched_token_ids.jsonl"
TARGET_OUTPUT_TOKENS="${1:-256}"
PVC=inference-perf-dataset
HELPER=inference-perf-dataset-upload

if kubectl get pvc "${PVC}" >/dev/null 2>&1; then
  echo "PVC ${PVC} already exists; skipping dataset prep"
  exit 0
fi

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v git-lfs >/dev/null || { echo "git-lfs is required" >&2; exit 1; }

echo "=== [1/4] fetching ${DATASET_REL_PATH} via git lfs pull ==="
git -C "${REPO_ROOT}" lfs pull --include="${DATASET_REL_PATH}" --exclude=""

TMP_FILE="$(mktemp --suffix=.json)"
trap 'rm -f "${TMP_FILE}"' EXIT

echo "=== [2/4] converting dataset to ShareGPT format (target output tokens: ${TARGET_OUTPUT_TOKENS}) ==="
python3 "${SCRIPT_DIR}/convert_dataset_to_sharegpt.py" \
  "${REPO_ROOT}/${DATASET_REL_PATH}" "${TMP_FILE}" \
  --target-output-tokens "${TARGET_OUTPUT_TOKENS}"

echo "=== [3/4] creating ${PVC} and uploading via helper pod ==="
kubectl apply -f "${SCRIPT_DIR}/dataset-pvc.yaml"
kubectl delete pod "${HELPER}" --ignore-not-found
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${HELPER}
  labels:
    app: inference-perf
    role: dataset-upload
spec:
  restartPolicy: Never
  containers:
    - name: upload
      image: busybox:1.36
      command: ["sleep", "3600"]
      volumeMounts:
        - name: dataset
          mountPath: /datasets
  volumes:
    - name: dataset
      persistentVolumeClaim:
        claimName: ${PVC}
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

echo "=== [4/4] copying ${TMP_FILE} -> ${HELPER}:/datasets/sharegpt.json ==="
kubectl cp "${TMP_FILE}" "${HELPER}:/datasets/sharegpt.json"

kubectl delete pod "${HELPER}" --ignore-not-found

echo "=== dataset ready on PVC ${PVC} (/datasets/sharegpt.json) ==="
