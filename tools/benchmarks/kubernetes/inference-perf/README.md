# Benchmark an AFD recipe with inference-perf (Kubernetes)

Deploys an AFD serving endpoint on a Kubernetes/OpenShift cluster --
prefill/decode disaggregation or colocation, whichever recipe script you
point it at -- and drives load against it with
[inference-perf](https://github.com/kubernetes-sigs/inference-perf) to
produce throughput/latency reports.

The serve pod runs the chosen recipe script unmodified (aside from rebinding
the client-facing port off loopback) and stays up afterwards so you can
point your own client at it too.

## Prerequisites

- An authenticated `kubectl` (or `oc`) session pointed at the target
  namespace, with permission to create Pods, Services, and
  PersistentVolumeClaims.
- `envsubst` (part of `gettext`) installed locally -- used to render
  [pvc.yaml](../../../../recipe/gpu/P2pNcclAFDConnector/kubernetes/pvc.yaml)
  and
  [serve-bench-pod.yaml](../../../../recipe/gpu/P2pNcclAFDConnector/kubernetes/serve-bench-pod.yaml)
  before `kubectl apply`.
- A benchmark image built from
  [docker/Dockerfile.k8s-cuda](../../../../docker/Dockerfile.k8s-cuda)
  (base `vllm/vllm-openai:v0.26.0` with an editable `afd-plugin`
  install, repo sources baked in) and pushed somewhere the cluster can pull
  it from. Set `AFD_PLUGIN_IMAGE` to that image ref when running `run.sh`.
  Build and push it from the afd-plugin repo root:

  ```bash
  IMAGE=<registry>/<repo>:<tag>
  docker build -f docker/Dockerfile.k8s-cuda -t "$IMAGE" .
  docker push "$IMAGE"
  ```

  Use a registry your cluster's nodes can pull from (and `docker login` to it
  first if it requires auth).
- A `hf-token-secret` Secret in the namespace with a `token` key holding a
  Hugging Face access token (used by the serve pod, and optionally by
  inference-perf, to download from the Hub):

  ```bash
  kubectl create secret generic hf-token-secret --from-literal=token=<hf_token>
  ```

- Cluster nodes with enough GPUs for the recipe you pick (see `GPU_COUNT`
  below -- it must match the number of workers the recipe launches).

## Usage

```bash
AFD_PLUGIN_IMAGE=<image> ./run.sh [recipe-script-path]
```

`recipe-script-path` selects which AFD recipe script the serve pod runs,
given as a path relative to the afd-plugin repo root baked into the image
(`/opt/afd-plugin`), e.g. any script under
[recipe/gpu/P2pNcclAFDConnector](../../../../recipe/gpu/P2pNcclAFDConnector).
Can also be set via the `RECIPE_SCRIPT_PATH` env var. Defaults to a
DeepSeek-V2-Lite prefill/decode-colocation recipe; for example, to run a
disaggregation recipe instead:

```bash
AFD_PLUGIN_IMAGE=<image> ./run.sh recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_disaggregation/2p1a1f_graph_dbo.sh
```

Other env vars `run.sh` reads:

- `MODEL_ID` -- the HF repo id the pod serves (default
  `deepseek-ai/DeepSeek-V2-Lite`). Must match whatever the chosen recipe
  script expects.
- `PVC_NAME` -- name of the PersistentVolumeClaim used to cache the model
  weights (default `deepseek-v2-lite-pvc`). Reused across runs so the model
  only needs to download once.
- `GPU_COUNT` -- the pod's `nvidia.com/gpu` request/limit (default `4`).
  Must match the number of workers the chosen recipe launches (e.g. 2
  prefill + 1 attention + 1 FFN for `2p1a1f_graph_dbo.sh`, or 2 attention +
  2 FFN for `2a2f_graph_dbo_dp2tp1.sh`).

This:

1. Resolves `MODEL_ID` and `PVC_NAME`, then creates the PVC (via
   [pvc.yaml](../../../../recipe/gpu/P2pNcclAFDConnector/kubernetes/pvc.yaml))
   if it doesn't already exist. The serve pod always serves `MODEL_ID`;
   vLLM downloads the weights into the PVC's `HF_HOME` cache on a cold
   volume and reuses them warm on every later run.
2. Renders
   [serve-bench-pod.yaml](../../../../recipe/gpu/P2pNcclAFDConnector/kubernetes/serve-bench-pod.yaml)
   with `AFD_PLUGIN_IMAGE`, `MODEL_ID`, `PVC_NAME`, `recipe-script-path`, and
   `GPU_COUNT`, and applies it.
3. Applies
   [service.yaml](../../../../recipe/gpu/P2pNcclAFDConnector/kubernetes/service.yaml)
   -- a Service in front of the proxy, named `vllm-service`.
4. Waits for the pod to reach `Running`, then streams pod logs until the
   serve pod prints `stack READY`, and prints the Service endpoint.
5. Renders [inference-perf-config.yaml](inference-perf-config.yaml) -- the
   load generator's config, preconfigured to hit `vllm-service:18305` --
   with `MODEL_ID`, applies it, then applies
   [inference-perf-pod.yaml](inference-perf-pod.yaml) (the load-generator
   Pod and its reports PVC) and streams its logs until the run completes.
6. Runs [copy-reports.sh](copy-reports.sh) to pull inference-perf's reports
   out to `./reports` (see
   [Copying inference-perf reports out](#copying-inference-perf-reports-out)
   below for how that works even after the `inference-perf` pod exits).

The serve pod (and its Service) are left running afterwards; `run.sh`
does **not** delete them, so you can re-run step 5 (re-apply
`inference-perf-config.yaml`/`inference-perf-pod.yaml`, or just re-run
`run.sh` with the same `RECIPE_SCRIPT_PATH`) without redeploying the model.

Note that the recipe scripts themselves bind the client-facing port to
`127.0.0.1` (loopback-only, since they're normally run and benchmarked from
within the same pod/host); `serve-bench-pod.yaml` patches a copy of the
selected script at startup to rebind that port to `0.0.0.0` so the Service
can actually reach it. Every internal worker port stays on `127.0.0.1`,
since only the client-facing endpoint (a proxy in disaggregation recipes,
the attention server itself in colocation recipes) is meant to be
client-facing.

Once you're done, delete the serve pod (frees the GPUs) and the Service:

```bash
kubectl delete pod vllm-pod
kubectl delete -f ../../../../recipe/gpu/P2pNcclAFDConnector/kubernetes/service.yaml
```

## Load profile

[inference-perf-config.yaml](inference-perf-config.yaml) drives a synthetic
`shared_prefix` workload (250 prompt groups x 5 prompts/group, a 7000-token
shared system prompt, 256-token questions, 256-token outputs) at a Poisson
arrival process that ramps through rate stages 5 -> 40 requests/sec (60s
each). Edit the `load` and `data` sections there to change the request rate
schedule or prompt shape.

## Copying inference-perf reports out

`inference-perf-pod.yaml` writes its reports to `/reports` (backed by the
`inference-perf-reports` PVC), and the `inference-perf` pod exits
(`restartPolicy: Never`) once its load test finishes. `kubectl cp`/`kubectl
exec` can't reach a terminated container, so use
[copy-reports.sh](copy-reports.sh) instead of copying directly from the
`inference-perf` pod -- it mounts the `inference-perf-reports` PVC read-only
from a short-lived helper pod and copies from there, so it works whether the
`inference-perf` pod is still around (`Completed`) or already deleted:

```bash
./copy-reports.sh [local-dir]   # defaults to ./reports
```

## What's running

`serve-bench-pod.yaml` runs whichever recipe script
`recipe-script-path`/`RECIPE_SCRIPT_PATH` points at, unmodified from
`/opt/afd-plugin` (baked into the image), with `uv run` stripped at
execution time via a `uv` shim on `PATH` -- the pod's base image already
provides the right Python environment, so `uv run` is unnecessary and,
unlike in a plain `uv`-managed dev checkout, isn't guaranteed to resolve
correctly. The shim rewrites `uv run <cmd>` to just `<cmd>` and otherwise
passes through to the real `uv` binary.

Per-instance logs land in `/work` inside the pod, one per worker the recipe
launches (e.g. `afd_prefill0.log`, `afd_prefill1.log`, `attn.log`,
`ffn.log`, `proxy.log` for a disaggregation recipe).
