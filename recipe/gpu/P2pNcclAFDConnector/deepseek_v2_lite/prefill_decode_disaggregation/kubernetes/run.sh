#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Drive the DeepSeek-V2-Lite 2P1A1F prefill/decode-disaggregation recipe on a
# Kubernetes/OpenShift cluster.
#
# Bring-up and workload are separate commands on purpose: `up` costs 10-20
# minutes (four vLLM instances + cudagraph capture), so one stack can serve
# several `bench` runs before you tear it down.
#
# The pod runs the recipe script from this directory's parent unmodified; this
# script only supplies the cluster plumbing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POD=afd-dsv2lite-pd-1a1f
PVC=deepseek-v2-lite-pvc
PROXY_PORT=18305

usage() {
  cat <<'USAGE'
Usage: run.sh <command> [options]

Commands:
  up        create the PVC and serving pod, wait until the stack is ready
  bench     run a request-rate ladder against a ready pod, copy results out
  down      delete the pod, freeing the 4 GPUs (the PVC is kept)
  all       up, then bench, then down
  status    print pod phase and readiness
  logs      follow the pod's launcher output

up options:
  --image REF        benchmark image (default: $AFD_PLUGIN_IMAGE), required
  --model ID|PATH    Hub id, or a /models path for pre-staged weights
                     (default: deepseek-ai/DeepSeek-V2-Lite)
  --recipe FILE      recipe script in the parent directory
                     (default: 2p1a1f_graph_dbo.sh; 2p1a1f_eager_dbo.sh starts
                     much faster and is the one to use for a smoke test)
  --pull-secret NAME dockerconfigjson Secret that can pull the image
                     (default: $AFD_PLUGIN_PULL_SECRET, else ghcr-push)
  --timeout MIN      how long to wait for readiness (default: 90); the pod
                     keeps waiting past this, so a timeout here is not a
                     failure -- check `run.sh status`

bench options:
  --isl N            random-dataset input length      (default: 1024)
  --osl N            random-dataset output length     (default: 128)
  --rates "R..."     space-separated request rates    (default: "5 10 20 40 inf")
  --num-prompts N    prompts per rate point           (default: 1024)
  --concurrency N    max in-flight requests           (default: 32)
  --tag NAME         result set name, also the in-pod and local subdirectory
                     (default: isl<ISL>-osl<OSL>)
  --results DIR      local destination (default: ./results)

down options:
  --delete-pvc       also delete the model cache PVC (default: keep it)

Examples:
  # smoke test: fast startup, one rate point, then tear down
  ./run.sh all --image $AFD_PLUGIN_IMAGE --recipe 2p1a1f_eager_dbo.sh \
      --rates 5 --num-prompts 64

  # bring up once, sweep twice, tear down
  ./run.sh up --image $AFD_PLUGIN_IMAGE
  ./run.sh bench --isl 1024 --tag isl1024
  ./run.sh bench --isl 4096 --tag isl4096
  ./run.sh down
USAGE
}

die() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------- up ---------
IMAGE="${AFD_PLUGIN_IMAGE:-}"
MODEL="deepseek-ai/DeepSeek-V2-Lite"
RECIPE_SCRIPT="2p1a1f_graph_dbo.sh"
# dockerconfigjson Secret used to pull the (private) benchmark image.
PULL_SECRET="${AFD_PLUGIN_PULL_SECRET:-ghcr-push}"
# The pod's own bring-up budget is longer than this (40 min per server log,
# four logs, plus the proxy poll), so a timeout here means "stop waiting", not
# "the pod died" -- the stack may still come up. `run.sh status` says which.
UP_TIMEOUT_MIN=90

# ------------------------------------------------------------- bench ---------
ISL=1024
OSL=128
RATES="5 10 20 40 inf"
NUM_PROMPTS=1024
CONCURRENCY=32
TAG=""
LOCAL_RESULTS="${SCRIPT_DIR}/results"

# -------------------------------------------------------------- down ---------
DELETE_PVC=0

parse_options() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --image)       IMAGE="$2"; shift 2 ;;
      --model)       MODEL="$2"; shift 2 ;;
      --recipe)      RECIPE_SCRIPT="$2"; shift 2 ;;
      --pull-secret) PULL_SECRET="$2"; shift 2 ;;
      --timeout)     UP_TIMEOUT_MIN="$2"; shift 2 ;;
      --isl)         ISL="$2"; shift 2 ;;
      --osl)         OSL="$2"; shift 2 ;;
      --rates)       RATES="$2"; shift 2 ;;
      --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
      --concurrency) CONCURRENCY="$2"; shift 2 ;;
      --tag)         TAG="$2"; shift 2 ;;
      --results)     LOCAL_RESULTS="$2"; shift 2 ;;
      --delete-pvc)  DELETE_PVC=1; shift ;;
      -h|--help)     usage; exit 0 ;;
      *)             die "unknown option: $1 (see --help)" ;;
    esac
  done
  [ -n "$TAG" ] || TAG="isl${ISL}-osl${OSL}"
}

pod_phase() {
  kubectl get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || true
}

pod_exec() {
  kubectl exec "$POD" -- "$@"
}

require_ready() {
  [ "$(pod_phase)" = "Running" ] || die "pod $POD is not Running (run: $0 up)"
  pod_exec test -f /work/SERVER_READY 2>/dev/null \
    || die "pod $POD is up but the stack is not ready (run: $0 status)"
}

cmd_up() {
  [ -n "$IMAGE" ] || die "no image: pass --image or set AFD_PLUGIN_IMAGE"
  [ -f "${SCRIPT_DIR}/../${RECIPE_SCRIPT}" ] \
    || die "no such recipe script: ${SCRIPT_DIR}/../${RECIPE_SCRIPT}"
  command -v envsubst >/dev/null || die "envsubst (gettext) is required"
  # Catch a missing pull secret here: without it the pod reaches Pending and
  # sits in ImagePullBackOff holding its 4-GPU reservation until torn down.
  kubectl get secret "$PULL_SECRET" >/dev/null 2>&1 \
    || die "pull secret '$PULL_SECRET' not found in this namespace (see --pull-secret)"

  echo "=== model cache volume ==="
  # The PVC is the pod's HF cache (HF_HOME). On a cold volume the first run
  # downloads the weights inline, which keeps the GPUs allocated while it
  # happens; every later run starts warm.
  #
  # Create it only when absent. A PVC spec is immutable once bound, so
  # re-applying pvc.yaml over an existing claim is rejected outright (the
  # cluster records the defaulted storageClassName that the manifest omits) --
  # which would abort bring-up on every run after the first.
  if kubectl get pvc "$PVC" >/dev/null 2>&1; then
    echo "PVC ${PVC} already exists; keeping it (cache stays warm)"
  else
    kubectl apply -f "${SCRIPT_DIR}/pvc.yaml"
  fi

  echo "=== apply serving pod (recipe: ${RECIPE_SCRIPT}, model: ${MODEL}) ==="
  kubectl delete pod "$POD" --ignore-not-found --wait=true
  TEMPLATE_IMAGE="$IMAGE" \
  TEMPLATE_MODEL="$MODEL" \
  TEMPLATE_RECIPE_SCRIPT="$RECIPE_SCRIPT" \
  TEMPLATE_PULL_SECRET="$PULL_SECRET" \
    envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${TEMPLATE_RECIPE_SCRIPT} ${TEMPLATE_PULL_SECRET}' \
    < "${SCRIPT_DIR}/serve-pod.yaml" | kubectl apply -f -

  echo "=== waiting for pod to be scheduled ==="
  # Bounded: a pod stuck in ImagePullBackOff stays Pending forever while still
  # holding its 4-GPU reservation, so report the kubelet's reason and bail out
  # rather than spinning.
  local sched_deadline=$(( SECONDS + 600 ))
  until [ "$(pod_phase)" = "Running" ]; do
    [ "$(pod_phase)" = "Failed" ] && {
      kubectl logs "$POD" --tail=50 || true
      die "pod Failed before starting"
    }
    local waiting
    waiting="$(kubectl get pod "$POD" \
      -o jsonpath='{.status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || true)"
    case "$waiting" in
      ImagePullBackOff|ErrImagePull|CreateContainerConfigError|CreateContainerError|InvalidImageName)
        kubectl describe pod "$POD" 2>/dev/null | sed -n '/Events:/,$p' | tail -15 || true
        die "pod cannot start: $waiting (see events above); GPUs freed by: $0 down"
        ;;
    esac
    if [ "$SECONDS" -ge "$sched_deadline" ]; then
      kubectl describe pod "$POD" 2>/dev/null | sed -n '/Events:/,$p' | tail -15 || true
      die "pod still not Running after 10m (phase: $(pod_phase), waiting: ${waiting:-none})"
    fi
    sleep 10
  done

  echo "--- streaming pod logs ---"
  echo "    startup is ~10-20m (four instances + cudagraph capture),"
  echo "    plus a one-time model download if ${PVC} is cold"
  kubectl logs -f "pod/${POD}" &
  local log_pid=$!

  local deadline=$(( SECONDS + UP_TIMEOUT_MIN * 60 ))
  while :; do
    if pod_exec test -f /work/SERVER_READY 2>/dev/null; then
      kill "$log_pid" 2>/dev/null || true
      echo "=== stack READY ==="
      return 0
    fi
    if pod_exec test -f /work/SERVER_FAILED 2>/dev/null; then
      kill "$log_pid" 2>/dev/null || true
      die "bring-up failed; per-worker logs were dumped above (also: $0 logs)"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      kill "$log_pid" 2>/dev/null || true
      die "timed out after ${UP_TIMEOUT_MIN}m waiting for readiness"
    fi
    sleep 15
  done
}

cmd_bench() {
  require_ready

  echo "=== rate ladder: isl=${ISL} osl=${OSL} prompts=${NUM_PROMPTS}"
  echo "    concurrency=${CONCURRENCY} rates='${RATES}' tag=${TAG} ==="

  local rc=0
  for rate in $RATES; do
    echo "----- rate=${rate} -----"
    # Each rate is an independent `vllm bench serve` run against the proxy, NOT
    # the attention server: traffic must go through the proxy so requests are
    # prefilled remotely and the decoder pulls KV over NIXL. AFD_MODEL comes
    # from the pod's env, so the benchmark always names the model the servers
    # were actually started with.
    kubectl exec "$POD" -- env \
      RATE="$rate" ISL="$ISL" OSL="$OSL" NUM_PROMPTS="$NUM_PROMPTS" \
      CONCURRENCY="$CONCURRENCY" TAG="$TAG" PROXY_PORT="$PROXY_PORT" \
      bash -c '
        set -euo pipefail
        OUT="/work/results/$TAG"
        mkdir -p "$OUT"
        vllm bench serve \
          --host 127.0.0.1 \
          --port "$PROXY_PORT" \
          --endpoint /v1/completions \
          --model "$AFD_MODEL" \
          --trust-remote-code \
          --dataset-name random \
          --random-input-len "$ISL" \
          --random-output-len "$OSL" \
          --num-prompts "$NUM_PROMPTS" \
          --max-concurrency "$CONCURRENCY" \
          --request-rate "$RATE" \
          --ignore-eos \
          --percentile-metrics ttft,tpot,itl,e2el \
          --save-result \
          --result-dir "$OUT" \
          --result-filename "rate_${RATE}.json" \
          2>&1 | tee "$OUT/rate_${RATE}.log"
        exit "${PIPESTATUS[0]}"
      ' || { rc=$?; echo "rate=${rate} FAILED (rc=${rc})"; }
  done

  echo "=== collecting per-worker logs ==="
  pod_exec bash -c "cp -f /work/*.log /work/results/${TAG}/ 2>/dev/null || true"

  echo "=== copying results to ${LOCAL_RESULTS}/${TAG} ==="
  mkdir -p "$LOCAL_RESULTS"
  kubectl cp "${POD}:/work/results/${TAG}" "${LOCAL_RESULTS}/${TAG}"
  ls -la "${LOCAL_RESULTS}/${TAG}"

  if [ "$rc" -ne 0 ]; then
    # Return rather than die: `all` must still tear the pod down and free the
    # GPUs when a rate point fails.
    echo "warning: at least one rate point failed (rc=${rc})" >&2
    return "$rc"
  fi
  echo "=== bench complete ==="
}

cmd_down() {
  echo "=== deleting pod ${POD} ==="
  kubectl delete pod "$POD" --ignore-not-found
  if [ "$DELETE_PVC" = "1" ]; then
    echo "=== deleting PVC ${PVC} (model cache will need re-downloading) ==="
    kubectl delete pvc "$PVC" --ignore-not-found
  fi
}

cmd_status() {
  local phase; phase="$(pod_phase)"
  echo "pod:   ${POD}"
  echo "phase: ${phase:-<absent>}"
  [ "$phase" = "Running" ] || return 0
  if pod_exec test -f /work/SERVER_READY 2>/dev/null; then
    echo "state: READY (proxy on :${PROXY_PORT})"
  elif pod_exec test -f /work/SERVER_FAILED 2>/dev/null; then
    echo "state: FAILED during bring-up (see: $0 logs)"
  else
    echo "state: starting"
  fi
  echo "model: $(kubectl get pod "$POD" -o jsonpath='{.spec.containers[0].env[?(@.name=="AFD_MODEL")].value}')"
  echo "recipe: $(kubectl get pod "$POD" -o jsonpath='{.spec.containers[0].env[?(@.name=="AFD_RECIPE_SCRIPT")].value}')"
}

cmd_logs() {
  kubectl logs -f "pod/${POD}"
}

COMMAND="${1:-}"
[ -n "$COMMAND" ] || { usage; exit 1; }
shift
parse_options "$@"

case "$COMMAND" in
  up)     cmd_up ;;
  bench)  cmd_bench ;;
  down)   cmd_down ;;
  all)
    cmd_up
    bench_rc=0
    cmd_bench || bench_rc=$?
    cmd_down
    [ "$bench_rc" -eq 0 ] || exit "$bench_rc"
    ;;
  status) cmd_status ;;
  logs)   cmd_logs ;;
  -h|--help|help) usage ;;
  *)      die "unknown command: $COMMAND (see --help)" ;;
esac
