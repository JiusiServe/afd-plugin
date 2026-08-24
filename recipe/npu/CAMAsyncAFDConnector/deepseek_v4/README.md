# DeepSeek-V4 Flash-INT8 Async CAM target baseline

This recipe is the M0--M4 target-only baseline. It deliberately excludes
AFD-managed async MoE ubatching.

Both roles must use the same model checkpoint and identical AFD topology. The
first smoke configuration is one Attention rank and one FFN rank:

```json
{
  "afd": {
    "role": "attention",
    "connector": "CAMAsyncAFDConnector",
    "async": true,
    "host": "<attention-rank-0-ip>",
    "port": 1239,
    "num_attention_ranks": 1,
    "num_ffn_ranks": 1,
    "compute_gate_on_attention": true,
    "connector_extra_config": {
      "dynamicQuant": 1,
      "attn_ranks_per_dp": 1,
      "async_moe_ubatching": false
    }
  }
}
```

On the FFN process, change only `role` to `"ffn"`. Start FFN before Attention.
Use `--enforce-eager`, `--quantization ascend`, and the native DSV4 tokenizer
configuration (`--tokenizer-mode deepseek_v4`). Do not enable vLLM DBO,
`--num-ubatches` or async MoE ubatching.

For the single-node 16-card baseline, use the checked-in scripts:

```bash
bash recipe/npu/CAMAsyncAFDConnector/deepseek_v4/launch_8a8f.sh
tail -f /tmp/afd_dsv4_async/ffn.log
tail -f /tmp/afd_dsv4_async/attention.log
```

The launch shape is eight Attention DP ranks on devices `0-7` and eight FFN
EP ranks on devices `8-15`; it follows the existing DSV4 AFD launch
shape and is the practical minimum for the full model.

For M4, expand the same topology to two Attention and two FFN ranks:

```json
{
  "num_attention_ranks": 2,
  "num_ffn_ranks": 2,
  "connector_extra_config": {
    "dynamicQuant": 1,
    "attn_ranks_per_dp": 1,
    "async_moe_ubatching": false
  }
}
```

The HCCL rank order is `[A0, A1, F0, F1]`. For each scale-out run, verify both
roles share `host`, `port`, rank counts, `attn_ranks_per_dp`, model revision,
and CAM operator installation. Compare greedy output with native DSV4 before
measuring throughput.
