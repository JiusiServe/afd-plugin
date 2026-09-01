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

## 0. Prerequisites

- `kubectl`/`oc` authenticated to the namespace; `envsubst` locally.
- A node with 4 free GPUs.
- Two Secrets:

```bash
kubectl create secret generic hf-token-secret --from-literal=token=<hf_token>

kubectl create secret docker-registry ghcr-push \
  --docker-server=ghcr.io \
  --docker-username=<github_user> \
  --docker-password=<pat_with_write:packages>
```

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

## 2. Deploy

```bash
export AFD_PLUGIN_IMAGE=$IMAGE
./run.sh up --recipe 2p1a1f_eager_dbo.sh
```

Returns when all four instances are up and the proxy answers `/healthcheck`.
Eager takes a few minutes; `2p1a1f_graph_dbo.sh` ~20 (cudagraph capture).

Weights come from the PVC (`HF_HOME=/models/.hf_home`); a cold volume downloads
~30 GB inline while holding the GPUs. If already staged, skip it:

```bash
./run.sh up --recipe 2p1a1f_eager_dbo.sh --model /models/DeepSeek-V2-Lite
```

## 3. Verify

```bash
./run.sh status     # expect: state: READY (proxy on :18305)

MODEL=/models/DeepSeek-V2-Lite   # must match ./run.sh status
kubectl exec afd-dsv2lite-pd-1a1f -- curl -s \
  http://127.0.0.1:18305/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","prompt":"The capital of France is","max_tokens":24,"temperature":0}'
```

Traffic must go to **18305** — only the proxy performs the remote-prefill
handshake that makes disaggregation happen. To drive it locally instead:
`kubectl port-forward pod/afd-dsv2lite-pd-1a1f 18305:18305`.

## 4. Benchmark (optional)

One bring-up serves several runs. Results land in `./results/<tag>/` as
`rate_<R>.json` (throughput, TTFT/TPOT/ITL/E2EL) plus the per-worker logs.

```bash
./run.sh bench --isl 1024 --tag isl1024
./run.sh all --recipe 2p1a1f_eager_dbo.sh --rates 5 --num-prompts 64  # smoke
```

## 5. Tear down

The pod holds 4 GPUs until freed:

```bash
./run.sh down                 # keeps the model cache PVC
./run.sh down --delete-pvc    # also drops the cache
```

## Reference

Commands: `up`, `bench`, `down`, `all`, `status`, `logs`. Run `./run.sh --help`
for all options. Most-used: `--image`, `--model`, `--recipe`, `--pull-secret`
(default `ghcr-push`), `--timeout` (default 90 min), `--isl`/`--osl`,
`--rates`, `--num-prompts`, `--tag`.

## Notes

- **Nothing binds 18304.** The FFN worker runs the p2p connector loop and never
  serves HTTP; readiness uses `AFD FFN EngineCore started` for it, not
  `Application startup complete`. Do not probe 18304.
- **Prefill is chunked at 64 tokens**, so TTFT is dominated by chunking, not by
  the NIXL transfer. `max-model-len` is 8192, so `ISL + OSL` must stay under it.
- **Benchmarks use `--ignore-eos`** — output text is meaningless; it measures
  serving performance, not quality.
- **`uv run` is shimmed in-pod** so the recipe script stays byte-identical to
  what a local user runs.
- **CSI-restricted volumes need a `nodeSelector`.** A GPU request alone does not
  express which nodes can mount the model PVC. `serve-pod.yaml` ships
  `scale: "true"` for IBM Spectrum Scale; adjust for your cluster.
- **Single-node by construction** — a multi-pod topology would need the AFD p2p
  endpoint and NIXL side channels reachable across pod IPs.
