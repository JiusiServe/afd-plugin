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
[decode_bench_server.sh](decode_bench_server.sh) for details.

## Prerequisites

- An authenticated `kubectl` (or `oc`) session pointed at the target
  namespace, with permission to create Pods, Jobs, and PersistentVolumeClaims.
- `envsubst` (part of `gettext`) installed locally -- used to render
  [serve-bench-pod.yaml](serve-bench-pod.yaml) for the requested recipe
  before `kubectl apply`.
- A `hf-token-secret` Secret in the namespace with a `token` key holding a
  Hugging Face access token (used both by the model-download Job and the
  serve+bench Pod):

  ```bash
  kubectl create secret generic hf-token-secret --from-literal=token=<hf_token>
  ```

- Cluster nodes with 4 GPUs available (both recipes request
  `nvidia.com/gpu: "4"`) and a storage class matching
  [pvc.yaml](pvc.yaml)'s `storageClassName` (`ibm-spectrum-scale-fileset` by
  default -- adjust if your cluster uses a different one).
- Optional: an NVIDIA DCGM host-engine DaemonSet in the `nvidia-gpu-operator`
  namespace, for per-GPU profiling during the run. If it isn't found,
  `run.sh` prints a warning and profiling is skipped -- the benchmark still
  runs.
- The `afd-plugin` repo checked out locally; `run.sh` tars up the repo root
  (excluding `.git` and `runs/results`) and streams it into the pod, so run
  it from a clean/relevant checkout.

## Usage

```bash
./run.sh <baseline|2a2f>
```

This:

1. Applies [pvc.yaml](pvc.yaml) and, if the PVC doesn't already exist, runs
   [download-job.yaml](download-job.yaml) to fetch `deepseek-ai/DeepSeek-V2-Lite`
   onto it (skipped on subsequent runs once the PVC is populated).
2. Renders [serve-bench-pod.yaml](serve-bench-pod.yaml) for the requested
   recipe (via `decode_bench_config_<recipe>.sh`) and applies it.
3. Waits for the pod to reach `Running`, copies the local repo in, and
   signals readiness.
4. Streams pod logs, waits for the vLLM server(s) to start, then runs the
   request-rate ladder (`10 20 40 80 inf` req/s) via `request_generator.sh`.
5. Copies `/models/results` from the pod to `./results/<recipe>/` locally,
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

## Recipe-specific files

Both recipes share [decode_bench_server.sh](decode_bench_server.sh) (the
in-pod server launcher) and [serve-bench-pod.yaml](serve-bench-pod.yaml) (the
pod template). What differs per recipe lives in
`decode_bench_config_<recipe>.sh`:

- `RESULT_PREFIX` -- filename prefix for benchmark result JSON files.
- `setup_recipe()` -- recipe-specific venv setup (e.g. installing `nixl` and
  an editable `afd-plugin` for 2a2f; a no-op for baseline).
- `launch_servers()` -- the `vllm serve` invocation(s) for the recipe.
