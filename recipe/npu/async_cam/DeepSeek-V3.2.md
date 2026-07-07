# DeepSeek-V3.2 async_cam Recipe

This recipe describes how to run DeepSeek-V3.2 with the AFD async CAM
connector on Ascend NPU.

## Introduction

`async_cam.py` provides an Ascend CAM backed asynchronous AFD connector for
the Attention-FFN disaggregation path. It uses CAM async dispatch/combine
operators to move MoE activations between attention-side and FFN-side ranks,
allowing the prefill computation to be split across independent attention and
expert workers.

Current limitation: async CAM currently supports only the prefill stage in a
prefill/decode disaggregated deployment.

## Image and Hardware Requirements

- Hardware: Ascend 910C only.
- Image: `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler`.

```bash
docker pull quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler
```

## Installing Operator Packages

Run the following commands from the repository root inside the container:

```bash
bash afd_plugin/connectors/npu/bin/CAM_ascend910_93_openEuler_aarch64.run
pip install afd_plugin/connectors/npu/bin/umdk_cam_op_lib-208.1.0b1-cp311-cp311-linux_aarch64.whl
```

## Experiment Configuration

### Model

- Model: DeepSeekV3.2 W8A8.
- Hardware: 2 Node Ascend 910C.
- Because of device-count limits, the experiment uses a reduced model with the
  first 10 layers only: 3 dense layers and 7 MoE layers.

### Benchmark

Dataset:

- File: `cp8sp50k_custom_dataset_token_ids.jsonl`
- Download URL: `[TBD](https://example.com/to-be-filled)`.

```bash
vllm bench serve \
  --backend vllm \
  --model deepseek_v3_2 \
  --tokenizer /path/to/DeepSeek-V3.2 \
  --tokenizer-mode auto \
  --trust-remote-code \
  --endpoint /v1/completions \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name custom \
  --dataset-path /path/to/cp8sp50k_custom_dataset_token_ids.jsonl \
  --skip-chat-template \
  --custom-output-len -1 \
  --request-rate 10 \
  --no-oversample \
  --disable-shuffle \
  --num-warmups 0 \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 25,50,90,95,99 \
  --save-result \
  --save-detailed \
  --result-dir ./bench_results \
  --result-filename ttft_by_prompt_len.json
```

### Baseline

- Topology: `DP4PCP8`.

<details>
<summary>Node0 Deployment Command (DP4PCP8)</summary>

```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=1

vllm serve /path/to/DeepSeek-V3.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 0 \
  --data-parallel-address 33.215.117.43 \
  --data-parallel-rpc-port 29550 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --block-size 128 \
  --quantization ascend \
  --seed 1024 \
  --served-model-name deepseek_v3_2 \
  --enable-expert-parallel \
  --max-num-seqs 8 \
  --max-model-len 70000 \
  --max-num-batched-tokens 140000 \
  --no-enable-chunked-prefill \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enable-expert-parallel
```

</details>

<details>
<summary>Node1 Deployment Command (DP4PCP8)</summary>

```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=1

vllm serve /path/to/DeepSeek-V3.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --headless \
  --data-parallel-address 33.215.117.43 \
  --data-parallel-rpc-port 29550 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --block-size 128 \
  --quantization ascend \
  --seed 1024 \
  --served-model-name deepseek_v3_2 \
  --enable-expert-parallel \
  --max-num-seqs 8 \
  --max-model-len 70000 \
  --max-num-batched-tokens 140000 \
  --no-enable-chunked-prefill \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enable-expert-parallel
```
</details>

### AFD async CAM

- Attention side: `DP3PCP8`.
- FFN side: `EP8`.
- Connector: `afdasyncconnector`.
- Current scope: PD-disaggregated prefill stage only.

<details>
<summary>Node0 Attention Deployment Command (DP0-DP1)</summary>

```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV

export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export AFD_FORCE_BALANCED_TOPK_IDS=1

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
vllm serve /path/to/DeepSeek-V3.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker \
  --max-num-seqs 32 \
  --enforce-eager \
  --served-model-name deepseek_v3_2 \
  --quantization ascend \
  --max-model-len 70000 \
  --max-num-batched-tokens 140000 \
  --tensor-parallel-size 1 \
  --data-parallel-size 3 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 0 \
  --data-parallel-address 33.215.117.43 \
  --data-parallel-rpc-port 29550 \
  --prefill-context-parallel-size 8 \
  --no-enable-chunked-prefill \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.8 \
  --additional-config '{
    "afd": {
      "enabled": true,
      "connector": "afdasyncconnector",
      "async": true,
      "role": "attention",
      "host": "33.215.117.43",
      "port": 1239,
      "num_attention_ranks": 24,
      "num_ffn_ranks": 8,
      "afd_role_rank": 0,
      "compute_gate_on_attention": true,
      "extra_config": {
        "quant_mode": 0,
        "dynamicQuant": 1,
        "async_moe_ubatching": true,
        "async_moe_num_ubatches": 2,
        "async_moe_split": "request",
        "attn_ranks_per_dp": 8
      }
    }
  }'
```

</details>

<details>
<summary>Node1 Attention Deployment Command (DP2)</summary>

```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV

export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export AFD_FORCE_BALANCED_TOPK_IDS=1

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve /path/to/DeepSeek-V3.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker \
  --max-num-seqs 32 \
  --served-model-name deepseek_v3_2 \
  --enforce-eager \
  --quantization ascend \
  --max-model-len 70000 \
  --max-num-batched-tokens 140000 \
  --tensor-parallel-size 1 \
  --data-parallel-size 3 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 2 \
  --data-parallel-address 33.215.117.43 \
  --data-parallel-rpc-port 29550 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.8 \
  --headless \
  --prefill-context-parallel-size 8 \
  --no-enable-chunked-prefill \
  --enable-expert-parallel \
  --additional-config '{
    "afd": {
      "enabled": true,
      "connector": "afdasyncconnector",
      "async": true,
      "role": "attention",
      "host": "33.215.117.43",
      "port": 1239,
      "num_attention_ranks": 24,
      "num_ffn_ranks": 8,
      "afd_role_rank": 0,
      "compute_gate_on_attention": true,
      "extra_config": {
        "quant_mode": 0,
        "dynamicQuant": 1,
        "async_moe_ubatching": true,
        "async_moe_num_ubatches": 2,
        "async_moe_split": "request",
        "attn_ranks_per_dp": 8
      }
    }
  }'
```

</details>

<details>
<summary>Node1 FFN Deployment Command (EP8)</summary>

```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV

export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export AFD_FORCE_BALANCED_TOPK_IDS=1

ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
vllm serve /path/to/DeepSeek-V3.2 \
  --port 8001 \
  --worker-cls afd_plugin.v1.worker.ascend.AFDNPUFFNWorker \
  --max-num-seqs 2 \
  --enforce-eager \
  --quantization ascend \
  --max-num-batched-tokens 140000 \
  --served-model-name deepseek_v3_2 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --additional-config '{
    "afd": {
      "enabled": true,
      "connector": "afdasyncconnector",
      "async": true,
      "role": "ffn",
      "host": "33.215.117.43",
      "port": 1239,
      "num_attention_ranks": 24,
      "num_ffn_ranks": 8,
      "afd_role_rank": 0,
      "compute_gate_on_attention": true,
      "extra_config": {
        "quant_mode": 0,
        "dynamicQuant": 1,
        "async_moe_ubatching": true,
        "async_moe_num_ubatches": 2,
        "async_moe_split": "request",
        "attn_ranks_per_dp": 8
      }
    }
  }'
```

</details>

## Experiment Results

TBD.
