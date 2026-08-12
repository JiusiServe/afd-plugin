# DeepSeek-V3.2 Async CAM v0.26 Accuracy Deployment

This recipe launches the full, unmodified DeepSeek-V3.2 model on two
16-NPU Ascend 910C nodes:

- Attention node: DP2, TP8, FlashComm1/SP enabled;
- FFN node: DP16, TP1, EP16, FlashComm1 disabled;
- AFD-managed two-stage token-balanced MoE ubatching;
- forced expert load balancing disabled in both the environment and
  `additional_config`.

Both launch scripts set `--gpu-memory-utilization 0.8`. The remaining device
memory is intentional headroom for CAM communication buffers and runtime
allocations on this multi-node topology.

The launch scripts intentionally do not use SSH or manage the other node's
processes. Start each role through the cluster job system so teardown remains
owned by that system.

## Prerequisites

Use the v0.26 runtime and CAM packages documented in the
[CAM Async Connector User Guide](../../../../../docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md).
The complete W8A8 checkpoint must be available at the same path on both nodes.
Its `config.json` must report `model_type=deepseek_v32` (or architecture
`DeepseekV32ForCausalLM`) and `num_hidden_layers=61`.

Set the CAM operator paths on both nodes before launching:

```bash
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api:${LD_LIBRARY_PATH}
```

## Launch

Use the Attention node's communication IP as `AFD_HOST` on both nodes. The
default CAM port, maximum model length, and batch limits may be overridden, but
`MAX_NUM_BATCHED_TOKENS` must remain identical on both roles.

On the FFN node:

```bash
MODEL_PATH=/path/to/DeepSeek-V3.2-W8A8 \
LOCAL_IP=<ffn-node-ip> \
AFD_HOST=<attention-node-ip> \
NIC_NAME=<npu-nic> \
bash recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/v0_26_accuracy/ffn_ep16.sh
```

On the Attention node:

```bash
MODEL_PATH=/path/to/DeepSeek-V3.2-W8A8 \
LOCAL_IP=<attention-node-ip> \
AFD_HOST=<attention-node-ip> \
NIC_NAME=<npu-nic> \
bash recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/v0_26_accuracy/attention_dp2tp8.sh
```

Wait for `http://127.0.0.1:8000/v1/models` on the Attention node before sending
inference or evaluation requests. A post-metadata-isolation run using the
complete 61-layer model, token split, and the fixed first 200 GSM8K samples
reported `0.975` exact match (`195/200`). This is a fixed-prefix regression
result, not a full-dataset accuracy claim. E2E automation is maintained
separately from this deployment recipe.
