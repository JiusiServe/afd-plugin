# A3 vLLM 0.26 CAM Async AFD deployment

This is the reproducible single-node, 16-NPU deployment for the DeepSeek-V4
Flash INT8 CAM async recipe. It starts Attention as `DP2 x TP4` on NPUs `0-7`
and the shared FFN pool as `DP8 x TP1 x EP8` on NPUs `8-15`.

## 1. Create the task

From the local AFD workspace, use the checked-in task wrapper:

```bash
bash run_itask.sh afd_bjf_26
```

The wrapper selects the validated nightly A3 image, 16 cards, host networking,
the DeepSeek-V4 model reference, and this task workdir:

```text
/a3_inference/itask/workdir/wb02363348/bjf_afd/code
```

Wait for `itask list` to report `Running` before executing commands in the
task.

## 2. Pin the runtime sources

Inside the task, pin the two editable source trees that come with the image:

```bash
git -C /vllm-workspace/vllm checkout --detach v0.26.0
git -C /vllm-workspace/vllm-ascend checkout -B releases/v0.26.0rc \
  --track origin/releases/v0.26.0rc
```

Expected revisions are `vllm` `568afb3a` and `vllm-ascend` `80d8c194f`.

## 3. Install CAM and AFD

Synchronize the local `afd-plugin` directory to the task first:

```bash
cd /path/to/afd-plugin
itask sync afd_bjf_26 --verbose
```

Then, inside the task:

```bash
source /usr/local/Ascend/cann-9.0.1/set_env.sh
cd /a3_inference/itask/workdir/wb02363348/bjf_afd/code/afd-plugin

bash afd_plugin/connectors/npu/bin/CAM_ascend910_93_openEuler_aarch64.run
# CAM installs its vendor op-api as libcust_opapi.so, but the spawned runtime
# resolves CAM aclnn symbols through the libopapi.so SONAME.
mv /usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib/libcust_opapi.so \
  /usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib/libopapi.so
/usr/local/python3.12.13/bin/python -m pip install \
  afd_plugin/connectors/npu/bin/umdk_cam_op_lib-209.0.0b1-cp312-cp312-linux_aarch64.whl

# CAM async uses the four CAM vendor operators. It does not need the legacy
# AFD A2E/E2A build, whose rebuild can fail on a shared NFS worktree that
# already contains build artifacts.
AFD_BUILD_ASCEND_OPS=0 /usr/local/python3.12.13/bin/python -m pip install \
  -v --no-build-isolation --no-deps -e .
```

Do not use `AFD_BUILD_ASCEND_OPS=0` for `CAMP2pAFDConnector`; that connector
needs AFD's A2E/E2A extension and should be built in a clean worktree.

## 4. Runtime-loader requirement for this nightly image

The image's `set_env.sh` does not export all dynamic-library directories needed
by `torch_npu`. Both `ffn_ep8.sh` and `attention_tp8.sh` therefore prepend:

```text
/usr/local/Ascend/driver/lib64/driver
/usr/local/Ascend/driver/lib64
/usr/local/Ascend/cann-9.0.1/aarch64-linux/lib64
/usr/local/Ascend/cann-9.0.1/runtime/lib64
```

Keep this before importing vLLM or `torch_npu`. The scripts also configure the
CAM `op_api` path and preload CAM's renamed `libopapi.so`.

Validate the installed CAM operators before a full model launch:

```bash
source /usr/local/Ascend/cann-9.0.1/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/cann-9.0.1/aarch64-linux/lib64:/usr/local/Ascend/cann-9.0.1/runtime/lib64:/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib:/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api:$LD_LIBRARY_PATH
export CAM_CUST_OPAPI_LIB_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib/libopapi.so
python -c 'from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available; ensure_cam_async_ops_available(); print("AFD_CAM_OPS_OK")'
```

## 5. Start the DP2 deployment

Run the launcher from the AFD repository root:

```bash
START_DELAY_SECONDS=120 \
  bash recipe/npu/CAMAsyncAFDConnector/deepseek_v4/launch_dp2tp4.sh
```

It starts FFN first, waits for its model workers and CAM rendezvous to be
ready, then starts Attention. Logs and process IDs are written to:

```text
/tmp/afd_dsv4_async_dp2tp4/ffn.log
/tmp/afd_dsv4_async_dp2tp4/attention.log
/tmp/afd_dsv4_async_dp2tp4/ffn.pid
/tmp/afd_dsv4_async_dp2tp4/attention.pid
```

The Attention OpenAI endpoint is `http://127.0.0.1:8900`; FFN listens on
`8901` only as the connector endpoint.

## 6. Check startup

```bash
tail -f /tmp/afd_dsv4_async_dp2tp4/ffn.log
tail -f /tmp/afd_dsv4_async_dp2tp4/attention.log
```

Start sending requests only after both logs report that their API server is
running. A failure in either role should be diagnosed from its own log before
restarting the pair.
