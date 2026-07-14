---
layout: post
title: "Announcing vLLM AFD Plugin: Disaggregating Attention and FFN for Flexible MoE Serving"
author: "AFD Plugin Contributors"
summary: "What vLLM AFD Plugin adds to the vLLM ecosystem: Attention–FFN disaggregation for MoE serving, GPU and Ascend NPU backends, connector-based execution, and graph and ubatching support."
image: /assets/figures/2026-07-14-vllm-afd-plugin/vllm-afd-plugin-architecture.svg
tags:
  - inference
  - moe
  - ecosystem
---

We are excited to introduce [**vLLM AFD Plugin**](https://github.com/vllm-project/afd-plugin), an experimental external plugin that brings **Attention–FFN Disaggregation (AFD)** to vLLM.

Mixture-of-Experts (MoE) inference combines two very different kinds of work inside every transformer layer. Attention is stateful and closely coupled to request scheduling and the KV cache, while the FFN or expert path is dominated by routed expert computation and all-to-all communication. Serving both paths in one worker topology forces them to share the same scaling, execution, and communication choices.

vLLM AFD Plugin separates these paths into independently deployed Attention and FFN services while preserving vLLM's request lifecycle and OpenAI-compatible API on the Attention side. The project currently supports NVIDIA GPUs and Ascend NPUs, synchronous and asynchronous connectors, DeepSeek V2/V3-family model wrappers, and eager, graph, and dual-batch execution paths within clearly validated limits.

> vLLM AFD Plugin is currently experimental. The initial implementation targets vLLM `0.19.1`, and more large-scale testing is still needed across models, topologies, and hardware.

## Why Attention–FFN Disaggregation?

MoE serving systems must balance several competing demands:

1. **Different scaling dimensions.** Attention capacity follows request state, sequence length, and KV-cache pressure. Expert capacity follows token routing and expert load. AFD gives each path its own rank topology instead of requiring one shared layout.
2. **Different runtime responsibilities.** Attention needs scheduling, KV-cache coordination, and sampling. FFN execution only needs activations, routing metadata, and a way to return expert outputs. Splitting the services lets the FFN side run as a lightweight connector-driven daemon.
3. **Backend-specific communication.** CUDA and Ascend expose different collective libraries, graph runtimes, and optimized MoE operators. A common connector contract keeps the model-facing flow stable while allowing each backend to own its data path.
4. **Room for communication/computation overlap.** Asynchronous dispatch and MoE ubatching can overlap independent stages instead of serializing all expert work behind the Attention path.

AFD does not yet imply role-pruned model loading: in the current implementation, both services load the full model weights and split the forward execution path. Role-specific weight loading is an important next step.

## Inside the Architecture

![vLLM AFD Plugin runtime architecture](../assets/vllm-afd-plugin-architecture.svg)

The plugin integrates through vLLM's `vllm.general_plugins` entry point, explicit `--worker-cls` paths, and the standard `--additional-config` channel. It does not require edits to the vLLM source tree.

The runtime has three main parts:

* **Request-driven Attention service.** The Attention worker retains vLLM's scheduler, KV cache, batching, model lifecycle, and sampling path. A plugin-owned model runner installs AFD metadata into the forward context and publishes data-parallel, ubatch, layer, and graph state to the FFN side.
* **Connector data and control plane.** At each split layer, the model wrapper sends Attention hidden states to the FFN service and receives the computed FFN output. A backend-neutral connector interface carries both tensors and the metadata required to interpret them.
* **Connector-driven FFN service.** The FFN worker has no request traffic, scheduler, or KV cache. A background loop receives metadata and activations, invokes `compute_ffn_output()` on the plugin-owned model wrapper, and sends the result back to Attention. Requests are always sent to the Attention API server.

This boundary is deliberately narrow. vLLM continues to own the serving control plane where its existing abstractions fit, while the plugin owns the AFD workers, model runners, connectors, metadata, model split points, and a small set of version-scoped compatibility patches.

### Connector and backend support

| Connector | Backend | Execution | Recommended stage | Graph support |
| --- | --- | --- | --- | --- |
| `P2pNcclAFDConnector` | CUDA | Synchronous P2P | Decode | `FULL_DECODE_ONLY` CUDA graph |
| `CAMP2pAFDConnector` | Ascend NPU | Synchronous CAMP2P/HCCL | Decode | `FULL_DECODE_ONLY` ACL graph |
| `CAMAsyncAFDConnector` | Ascend NPU | Asynchronous CAM | Prefill | Not currently supported |

The same high-level exchange—Attention output to FFN, FFN output back to Attention—is shared across connectors. Backend packages remain separate so CUDA graph behavior, ACL graph behavior, NCCL communication, and Ascend custom operators do not leak into one another.

### Key features

* **Native vLLM serving surface.** Existing vLLM users still launch with `vllm serve`, send requests to an OpenAI-compatible endpoint, and configure the runtime through `--additional-config`.
* **GPU and Ascend implementations.** GPU workers extend vLLM v1 classes, while NPU workers extend vLLM-Ascend classes directly. Shared behavior lives in configuration, topology, metadata, and connector contracts rather than cross-device inheritance.
* **Synchronous AFD for decode throughput.** `P2pNcclAFDConnector` and `CAMP2pAFDConnector` synchronously exchange Attention activations and FFN outputs, allowing the two roles to scale independently in throughput-oriented decode deployments. Their current graph paths use `FULL_DECODE_ONLY` semantics on CUDA and ACL, respectively.
* **Asynchronous AFD for prefill.** `CAMAsyncAFDConnector` uses CAM asynchronous dispatch and combine operators to decouple prefill Attention ranks from expert workers. Together with AFD-managed MoE ubatching, it overlaps independent Attention and FFN stages to reduce pipeline stalls. This path currently targets the prefill stage in a prefill/decode-disaggregated deployment and does not yet support graph execution.
* **MoE model integration.** The plugin registers wrappers for DeepSeek V2/V3-family architectures, including DeepSeek V3.2, and GLM MoE DSA. The wrapper exposes separate Attention and FFN computations while reusing upstream layer implementations.
* **Graph and ubatching paths.** The synchronous GPU and NPU connectors support decode-only graph capture. Dual Batch Overlap is supported with exactly two ubatches, and CAM async provides AFD-managed MoE ubatching for its prefill path.

## A Performance Snapshot

### Synchronous AFD Decode Throughput with `CAMP2pAFDConnector`

<!-- TODO: Add the synchronous AFD decode benchmark setup, results, figure, and analysis. -->

### Asynchronous AFD Prefill Performance with `CAMAsyncAFDConnector`

The repository includes an early CAM async experiment on two Ascend 910C nodes using a DeepSeek V3.2 W8A8 model reduced to 10 layers. The comparison uses forced expert balancing and contrasts a `DP4PCP8 TP1` baseline with an AFD layout consisting of Attention `DP3PCP8 TP1` plus FFN `EP8`.

![Median TTFT comparison for the CAM async experiment](../../recipe/npu/cam_async/text_matched_dp_afd_median_ttft.png)

Across the measured request rates, the AFD configuration lowers median/P50 time to first token. At 12 requests per second, median TTFT decreases from **15.1 seconds to 8.0 seconds**, a reduction of approximately **47%**. At both 10 and 12 requests per second, the measured gap is about 7.2 seconds.

**Note**: These numbers are a focused validation of the CAM async execution path, not a general performance claim for full DeepSeek V3.2 or every AFD topology.

## Getting Started

The current implementation requires Python 3.10–3.13 and targets vLLM `0.19.1`.

Deployment commands depend on the backend, connector, model, and rank topology. Instead of duplicating configurations here, use the maintained [AFD Plugin recipes](https://github.com/vllm-project/afd-plugin/tree/main/recipe):

* **GPU synchronous AFD:** the [DeepSeek V2 Lite P2P NCCL recipes](https://github.com/vllm-project/afd-plugin/tree/main/recipe/gpu/p2p_nccl/deepseek_v2_lite) cover decode-oriented colocated and prefill/decode-disaggregated deployments, eager and CUDA graph execution, and multiple DP/TP layouts.
* **Ascend NPU asynchronous prefill AFD:** the [DeepSeek V3.2 CAM async recipe](https://github.com/vllm-project/afd-plugin/blob/main/recipe/npu/cam_async/DeepSeek-V3.2.md) documents the required environment, topology, AFD configuration, benchmark setup, and current limitations.

Refer to the repository [README](https://github.com/vllm-project/afd-plugin#readme) and recipe directory for the latest installation steps, supported connector matrix, configuration fields, and complete launch commands.

## Current Scope and Roadmap

The project intentionally exposes its current boundaries: exact vLLM version pinning, model runner v1 only, full weights on both roles, decode-only graph modes, exactly two ubatches for DBO, and hardware-gated end-to-end testing.

The next phase of development will focus on:

* **Broader vLLM compatibility and upstream alignment:** track newer vLLM and vLLM-Ascend releases, evaluate model runner v2, keep compatibility patches minimal, and contribute generally useful abstractions upstream as they mature.
* **More flexible execution:** extend graph modes, ubatch counts, asynchronous stages, and validated rank topologies.
* **Production-scale validation:** publish repeatable accuracy, latency, throughput, stability, and multi-node results on full models and realistic workloads.
* **Expanded model and connector coverage:** add MoE architectures and backend transports through the existing model-wrapper and connector interfaces.
* **Multimodal and vLLM-Omni integration:** explore how AFD can integrate with [vLLM-Omni](https://github.com/vllm-project/vllm-omni) and heterogeneous multimodal pipelines, including its application within autoregressive (AR), Diffusion Transformer (DiT), and other stages that can benefit from independently scaled Attention and FFN execution.
* **Heterogeneous hardware and low-latency serving:** explore deploying Attention and FFN roles across different accelerator types and interconnects, together with connector, scheduling, placement, and computation-communication overlap optimizations that reduce time to first token and inter-token latency.


## Join the Community

vLLM AFD Plugin is at an early stage, and feedback from model, serving, and hardware communities will shape its direction.

* **Code and documentation:** [github.com/vllm-project/afd-plugin](https://github.com/vllm-project/afd-plugin)
* **Runtime design docs:** [GPU Attention/FFN and Ascend Attention/FFN designs](https://github.com/vllm-project/afd-plugin/tree/main/docs)
* **Issues and feature requests:** [GitHub Issues](https://github.com/vllm-project/afd-plugin/issues)

Let's build a more composable and hardware-aware future for MoE serving together.
