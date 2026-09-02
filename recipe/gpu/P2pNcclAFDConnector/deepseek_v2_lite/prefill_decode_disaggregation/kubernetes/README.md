# 2P1A1F prefill/decode disaggregation on Kubernetes

Runs [`../2p1a1f_eager_dbo.sh`](../2p1a1f_eager_dbo.sh) (or the `graph`
variant) unmodified on a Kubernetes/OpenShift cluster. Five processes, one pod,
one 4-GPU node, all on `127.0.0.1`:

| GPU | Role | Port | KV wiring |
|-----|------|------|-----------|
| 0 | Prefill #0 | 18301 | `NixlConnector` `kv_producer` |
| 1 | Prefill #1 | 18302 | `NixlConnector` `kv_producer` |
| 2 | Decode — attention | 18303 | `NixlConnector` `kv_consumer` |
| 3 | Decode — FFN | *(none)* | no KV cache; AFD p2p on 6269 |
| — | Proxy | **18305** | send all traffic here |

Two manifests, applied with `kubectl`: [`pvc.yaml`](pvc.yaml) (the model cache)
and [`serve-pod.yaml`](serve-pod.yaml) (the stack).

## 0. Prerequisites

- `kubectl`/`oc` authenticated to the namespace; `envsubst` (gettext) locally.
- A node with 4 free GPUs.
- Two Secrets:

```bash
kubectl create secret generic hf-token-secret --from-literal=token=<hf_token>

kubectl create secret docker-registry ghcr-push \
  --docker-server=ghcr.io \
  --docker-username=<github_user> \
  --docker-password=<pat_with_write:packages>
```

`ghcr-push` is only needed to **push** the image the cluster builds. The image
is published public, so the serving pod pulls it with no credentials and
carries no `imagePullSecrets`.

## 1. Build the image on the cluster

Bakes `nixl` + an editable `afd-plugin` install + the repo sources (including
the recipe script) into `vllm/vllm-openai:v0.26.0`. Rebuild after editing a
recipe script. Build on the cluster, not locally — `uv` segfaults under QEMU on
Apple Silicon. (`docker/Dockerfile.ci` will not work: no `nixl`, so KV transfer
fails.)

Create the BuildConfig once:

```bash
export IMAGE=ghcr.io/<user>/afd-plugin-k8s-bench:<tag>

cat <<EOF | kubectl apply -f -
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: afd-plugin-k8s-bench
spec:
  completionDeadlineSeconds: 5400
  source: {type: Binary, binary: {}}
  strategy:
    type: Docker
    dockerStrategy: {dockerfilePath: docker/Dockerfile.k8s-bench}
  output:
    to: {kind: DockerImage, name: ${IMAGE}}
    pushSecret: {name: ghcr-push}
  resources:
    requests: {cpu: "4", memory: 8Gi}
    limits: {cpu: "8", memory: 16Gi}
EOF
```

Build from the **repo root** and require `PHASE=Complete` (a "Push successful"
log line alone is not proof):

```bash
cd <repo-root>
oc start-build afd-plugin-k8s-bench --from-dir=. --follow
kubectl get build -l buildconfig=afd-plugin-k8s-bench \
  -o custom-columns='NAME:.metadata.name,PHASE:.status.phase'
```

Retarget the tag before a later build:

```bash
kubectl patch bc afd-plugin-k8s-bench --type=merge \
  -p '{"spec":{"output":{"to":{"kind":"DockerImage","name":"'"$IMAGE"'"}}}}'
```

**Make the package public after the first push.** GHCR creates new packages
private, and the serving pod supplies no pull credentials: on
github.com/users/<user>/packages/container/afd-plugin-k8s-bench/settings set
visibility to Public. Verify from a machine with no registry login:

```bash
docker manifest inspect $IMAGE >/dev/null && echo public
```

A private image surfaces later as `ImagePullBackOff` on the serving pod, not as
a build failure.

## 2. Deploy

### 2a. Model cache PVC

`HF_HOME` points at this volume, so a cold claim downloads ~30 GB inline on
first use (holding the GPUs while it does) and every later run starts warm.
Create it only when absent — a bound PVC's spec is immutable, so re-applying
`pvc.yaml` over an existing claim is rejected:

```bash
kubectl get pvc deepseek-v2-lite-pvc || kubectl apply -f pvc.yaml
```

### 2b. Serving pod

`serve-pod.yaml` carries three placeholders. Render it with `envsubst`, naming
the variables **explicitly** — a bare `envsubst` would also eat the `$VAR`
references in the pod's inline shell script and break bring-up:

```bash
export TEMPLATE_IMAGE=$IMAGE
export TEMPLATE_MODEL=deepseek-ai/DeepSeek-V2-Lite   # or /models/<dir> if pre-staged
export TEMPLATE_RECIPE_SCRIPT=2p1a1f_eager_dbo.sh    # or 2p1a1f_graph_dbo.sh

envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${TEMPLATE_RECIPE_SCRIPT}' \
  < serve-pod.yaml | kubectl apply -f -
```

Replacing a previous run: `kubectl delete pod afd-dsv2lite-pd-1a1f --wait=true`
first — the pod is `restartPolicy: Never` and is not managed by a controller.

Weights already staged on the PVC skip the download:

```bash
export TEMPLATE_MODEL=/models/DeepSeek-V2-Lite
```

### 2c. Wait for readiness

The pod launches the recipe, waits for all four instances plus the proxy, then
writes a marker into `/work`. Watch it come up:

```bash
kubectl wait --for=jsonpath='{.status.phase}'=Running pod/afd-dsv2lite-pd-1a1f --timeout=10m
kubectl logs -f pod/afd-dsv2lite-pd-1a1f
```

Bring-up is a few minutes for `2p1a1f_eager_dbo.sh` and ~20 for
`2p1a1f_graph_dbo.sh` (cudagraph capture), plus the one-time download on a cold
PVC. Poll for the outcome:

```bash
# READY when this exits 0
kubectl exec afd-dsv2lite-pd-1a1f -- test -f /work/SERVER_READY

# FAILED when this exits 0; per-worker log tails are already in `kubectl logs`
kubectl exec afd-dsv2lite-pd-1a1f -- test -f /work/SERVER_FAILED
```

A pod stuck `Pending`/`ContainerCreating` is holding its 4-GPU reservation —
check `kubectl describe pod afd-dsv2lite-pd-1a1f` and tear it down (step 4)
rather than waiting it out.

## 3. Verify

```bash
MODEL=deepseek-ai/DeepSeek-V2-Lite   # must match TEMPLATE_MODEL exactly
kubectl exec afd-dsv2lite-pd-1a1f -- curl -s \
  http://127.0.0.1:18305/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","prompt":"The capital of France is","max_tokens":24,"temperature":0}'
```

Traffic must go to **18305** — only the proxy performs the remote-prefill
handshake that makes disaggregation happen. To drive it locally instead:
`kubectl port-forward pod/afd-dsv2lite-pd-1a1f 18305:18305`.

## 4. Tear down

The pod holds 4 GPUs until deleted:

```bash
kubectl delete pod afd-dsv2lite-pd-1a1f          # keeps the model cache PVC
kubectl delete pvc deepseek-v2-lite-pvc          # also drops the cache
```

## Notes

- **Nothing binds 18304.** The FFN worker runs the p2p connector loop and never
  serves HTTP; readiness uses `AFD FFN EngineCore started` for it, not
  `Application startup complete`. Do not probe 18304.
- **Prefill is chunked at 64 tokens**, so TTFT is dominated by chunking, not by
  the NIXL transfer. `max-model-len` is 8192, so `ISL + OSL` must stay under it.
- **`uv run` is shimmed in-pod** so the recipe script stays byte-identical to
  what a local user runs.
- **CSI-restricted volumes need a `nodeSelector`.** A GPU request alone does not
  express which nodes can mount the model PVC. `serve-pod.yaml` ships
  `scale: "true"` for IBM Spectrum Scale; adjust for your cluster.
- **Single-node by construction** — a multi-pod topology would need the AFD p2p
  endpoint and NIXL side channels reachable across pod IPs.
