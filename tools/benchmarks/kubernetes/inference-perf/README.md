# DeepSeek-V2-Lite 2P1A1F serve pod (Kubernetes)

Deploys a real prefill/decode-disaggregated DeepSeek-V2-Lite serving
endpoint on a Kubernetes/OpenShift cluster, for benchmarking with an
external load generator such as
[inference-perf](https://github.com/kubernetes-sigs/inference-perf).

Unlike [decode_only](../decode_only), this pod does **not** fabricate KV or
run a benchmark internally -- it runs the real
[2p1a1f_graph_dbo.sh](../../../../recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_disaggregation/2p1a1f_graph_dbo.sh)
recipe (2 prefill producers, 1 attention decode worker, 1 FFN decode worker,
all connected via `P2pNcclAFDConnector`/NIXL, with DBO enabled) and stays up
so you can point your own client at it. Generated text and throughput
numbers are real, not fabricated.

## Prerequisites

- An authenticated `kubectl` (or `oc`) session pointed at the target
  namespace, with permission to create Pods, Jobs, and PersistentVolumeClaims.
- `envsubst` (part of `gettext`) installed locally -- used to render
  [serve-bench-pod.yaml](serve-bench-pod.yaml) before `kubectl apply`.
- `git` with the `git-lfs` extension installed locally -- used by
  [prepare-dataset.sh](prepare-dataset.sh) to fetch
  [tools/datasets/cp8sp50k_custom_dataset_text_matched_token_ids.jsonl](../../../datasets/cp8sp50k_custom_dataset_text_matched_token_ids.jsonl),
  which is excluded from the default LFS fetch.
- A benchmark image built from
  [docker/Dockerfile.k8s-bench](../../../../docker/Dockerfile.k8s-bench)
  (base `vllm/vllm-openai:v0.26.0` with an editable `afd-plugin`
  install, repo sources baked in) and pushed somewhere the cluster can pull
  it from. Set `AFD_PLUGIN_IMAGE` to that image ref when running `run.sh`.
  Build and push it from the afd-plugin repo root:

  ```bash
  IMAGE=<registry>/<repo>:<tag>
  docker build -f docker/Dockerfile.k8s-bench -t "$IMAGE" .
  docker push "$IMAGE"
  ```

  Use a registry your cluster's nodes can pull from (and `docker login` to it
  first if it requires auth).
- A `hf-token-secret` Secret in the namespace with a `token` key holding a
  Hugging Face access token (used both by the model-download Job and the
  serve pod):

  ```bash
  kubectl create secret generic hf-token-secret --from-literal=token=<hf_token>
  ```

- Cluster nodes with 4 GPUs available (the recipe launches 2 prefill
  producers + 1 attention worker + 1 FFN worker, one per GPU).

## Usage

```bash
AFD_PLUGIN_IMAGE=<image> ./run.sh [recipe-script-path]
```

`recipe-script-path` selects which AFD recipe script the serve pod runs,
given as a path relative to the afd-plugin repo root baked into the image
(`/opt/afd-plugin`). Defaults to
[2p1a1f_graph_dbo.sh](../../../../recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_disaggregation/2p1a1f_graph_dbo.sh);
can also be set via the `RECIPE_SCRIPT_PATH` env var. For example, to run the
non-AFD baseline instead:

```bash
AFD_PLUGIN_IMAGE=<image> ./run.sh recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_disaggregation/baseline.sh
```

This:

1. Applies [model-config.yaml](../model-config.yaml) -- a ConfigMap holding
   `MODEL_ID` (the HF hub id) and `MODEL_PATH` (the on-disk path, which also
   doubles as the string vLLM serves the model as), read by every step
   below -- then applies [pvc.yaml](../pvc.yaml) and, if the PVC
   doesn't already exist, runs [download-job.yaml](../download-job.yaml) to
   fetch `MODEL_ID` onto it (skipped on subsequent runs once the PVC is
   populated). Shared with [decode_only](../decode_only) -- if you've already
   run that, this step is a no-op.
2. Renders [serve-bench-pod.yaml](serve-bench-pod.yaml) with
   `AFD_PLUGIN_IMAGE`, `recipe-script-path`, and `GPU_COUNT` and applies it.
3. Applies [service-route.yaml](service-route.yaml) -- a Service in front of
   the proxy, named `vllm-service`.
4. Waits for the pod to reach `Running`, then streams pod logs until the
   disaggregation proxy reports `Application startup complete` in
   `/work/proxy.log`, and prints the Service endpoint.
5. Runs [prepare-dataset.sh](prepare-dataset.sh) to fetch
   `tools/datasets/cp8sp50k_custom_dataset_text_matched_token_ids.jsonl`
   (a real-workload prompt-length distribution) via `git lfs pull`, reshape
   it into the ShareGPT format inference-perf's dataset loader expects
   (adding a synthetic ~256-token filler reply per prompt so the load stays
   decode-focused -- see the comment in
   [inference-perf-config.yaml](inference-perf-config.yaml) for why), and
   upload it onto the `inference-perf-dataset` PVC. Skipped on subsequent
   runs once that PVC is populated.
6. Renders [inference-perf-config.yaml](inference-perf-config.yaml) -- the
   load generator's config, preconfigured to hit the Service from step 3
   and read prompts from the PVC step 5 populated -- with
   `MODEL_ID`/`MODEL_PATH` read back from `model-config.yaml`, applies it,
   then applies [inference-perf-pod.yaml](inference-perf-pod.yaml) (the
   load-generator Pod and its reports/dataset PVCs) and streams its logs
   until the run completes.
7. Runs [copy-reports.sh](copy-reports.sh) to pull inference-perf's reports
   out to `./reports` (see
   [Copying inference-perf reports out](#copying-inference-perf-reports-out)
   below for how that works even after the `inference-perf` pod exits).

The serve pod (and its Service) are left running afterwards; `run.sh`
does **not** delete them, so you can re-run steps 5-7
(`prepare-dataset.sh`, `inference-perf-config.yaml`,
`inference-perf-pod.yaml`, and `copy-reports.sh`) again without
redeploying the model.

Note that the recipe scripts themselves bind the proxy to `127.0.0.1`
(loopback-only, since they're normally run and benchmarked from within the
same pod/host); `serve-bench-pod.yaml` patches a copy of the selected script
at startup to rebind just the proxy to `0.0.0.0` so the Service can actually
reach it (do NOT send traffic directly to the attention worker's port). The
internal prefill/attention/FFN `vllm serve` instances are left on
`127.0.0.1`, since only the proxy talks to them and only the proxy is
client-facing.

Once you're done, delete the serve pod (frees the GPUs) and the Service:

```bash
kubectl delete pod vllm-pod
kubectl delete -f service-route.yaml
```

Delete the downloader Job when you no longer need it:

```bash
kubectl delete job afd-model-downloader
```

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

[serve-bench-pod.yaml](serve-bench-pod.yaml) runs whichever recipe script
`recipe-script-path`/`RECIPE_SCRIPT_PATH` points at (default
[2p1a1f_graph_dbo.sh](../../../../recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_disaggregation/2p1a1f_graph_dbo.sh))
unmodified from `/opt/afd-plugin` (baked into the image), with `uv run`
stripped at execution time via a `uv` shim on `PATH` -- the pod's base
image already provides the right Python environment, so `uv run` is
unnecessary and, unlike in a plain `uv`-managed dev checkout, isn't
guaranteed to resolve correctly. The shim rewrites `uv run <cmd>` to just
`<cmd>` and otherwise passes through to the real `uv` binary.

Per-instance logs land in `/work` inside the pod: `afd_prefill0.log`,
`afd_prefill1.log`, `attn.log`, `ffn.log`, and `proxy.log`.
