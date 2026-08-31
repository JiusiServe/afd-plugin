# DeepSeek-V2-Lite decode-bench (Kubernetes)

Runs a decode-only throughput/latency benchmark for DeepSeek-V2-Lite on a
Kubernetes/OpenShift cluster, in two variants:

- **baseline** -- a single, non-disaggregated `vllm serve` instance
  (`--data-parallel-size 4`), no AFD, no DBO.
- **2a2f** -- the AF-disaggregated recipe: an attention instance and an ffn
  instance connected via `P2pNcclAFDConnector`, with DBO enabled.

Both variants use the `AFDDecodeBenchConnector` to fabricate KV for every
prompt token except the last, so requests skip prefill and go straight into
decode. Throughput/latency numbers are meaningful; generated text is
garbage. See the header comments in
[decode_bench_server.sh](../../decode_bench_server.sh) for details.

## Prerequisites

- An authenticated `kubectl` (or `oc`) session pointed at the target
  namespace, with permission to create Pods, Jobs, and PersistentVolumeClaims.
- `envsubst` (part of `gettext`) installed locally -- used to render
  [serve-bench-pod.yaml](serve-bench-pod.yaml) for the requested recipe
  before `kubectl apply`.
- A benchmark image built from
  [docker/Dockerfile.ci](../../../../docker/Dockerfile.ci)
  (base `vllm/vllm-openai:v0.26.0` with an editable `afd-plugin`
  install, repo sources baked in) and pushed somewhere the cluster can pull
  it from. Set `AFD_PLUGIN_IMAGE` to that image ref when running `run.sh`.
  Build and push it from the afd-plugin repo root:

  ```bash
  IMAGE=<registry>/<repo>:<tag>
  docker build -f docker/Dockerfile.ci -t "$IMAGE" .
  docker push "$IMAGE"
  ```

  Use a registry your cluster's nodes can pull from (and `docker login` to it
  first if it requires auth).
- A `hf-token-secret` Secret in the namespace with a `token` key holding a
  Hugging Face access token (used both by the model-download Job and the
  serve+bench Pod):

  ```bash
  kubectl create secret generic hf-token-secret --from-literal=token=<hf_token>
  ```

- Cluster nodes with 4 GPUs available (both recipes request
  `nvidia.com/gpu: "4"`).

## Usage

```bash
AFD_PLUGIN_IMAGE=<image> ./run.sh <baseline|2a2f>
```

This:

1. Applies [pvc.yaml](pvc.yaml) and, if the PVC doesn't already exist, runs
   [download-job.yaml](download-job.yaml) to fetch `deepseek-ai/DeepSeek-V2-Lite`
   onto it (skipped on subsequent runs once the PVC is populated).
2. Renders [serve-bench-pod.yaml](serve-bench-pod.yaml) for the requested
   recipe and applies it.
3. Waits for the pod to reach `Running`.
4. Streams pod logs, waits for the vLLM server(s) to start, then runs the
   request-rate ladder (`10 20 40 80 inf` req/s) via `request_generator.sh`.
5. Copies `/models/results` from the pod to a local `results` folder,
   then deletes the pod (freeing the GPUs).

The downloader Job and PVC are left in place for reuse across runs. Delete
the Job when you no longer need it:

```bash
kubectl delete job deepseek-v2-lite-downloader
```

To run both variants back-to-back for comparison, just invoke `run.sh`
twice -- they share the same pod name and GPU request, so the second run's
pod won't come up until the first one's is deleted at the end of its stage:

```bash
./run.sh baseline && ./run.sh 2a2f
```

Set `LOCAL_RESULTS` to change where results land locally (default:
`./results`).

## Recipe selection

Both recipes share [decode_bench_server.sh](../../decode_bench_server.sh) (the
in-pod server launcher) and [serve-bench-pod.yaml](serve-bench-pod.yaml) (the
pod template) -- there are no more per-recipe config files. `run.sh` maps the
`<baseline|2a2f>` argument to two values, rendered into the pod template via
`envsubst`.

An editable `afd-plugin` install is not conditional on the
recipe: it's baked into the `AFD_PLUGIN_IMAGE` image at build time (see
[docker/Dockerfile.ci](../../../../docker/Dockerfile.ci)), so
every recipe run uses the same image.
