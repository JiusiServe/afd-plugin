"""End-to-end pass over the async GPU connector's public API, two processes.

Rank 0 runs the Attention side (``send_attn_output`` / ``recv_ffn_output``),
rank 1 runs the FFN side (``recv_ffn_work_item`` / ``send_ffn_work_item_output``)
with the real grouped-GEMM helper. Everything between the gate and the combined
result is exercised: routing, one-sided dispatch, local expert compute, the
write-back, and the weighted reduction.

Run with two GPUs::

    python tests/e2e/async_gpu_connector_e2e.py

Deliberately *not* launched with torchrun. ``init_afd_process_group`` builds its
own TCPStore on the AFD port, and under torchelastic every rank is forced to
``is_master=False`` (``torch/distributed/rendezvous.py:188``), so no rank hosts
the store and the group never forms. Production launches the two roles as
separate ``vllm serve`` processes, which this mirrors.
"""

import multiprocessing as mp
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.gpu.async_gpu import GpuAsyncAFDConnector
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.model_executor.models.gpu.deepseek_v2_attention_gate import (
    compute_attention_gate_moe_ffn,
)

NUM_TOKENS = 48
HIDDEN = 128
INTERMEDIATE = 256
TOPK = 4
NUM_EXPERTS = 8
NUM_LAYERS = 3
PORT = 29655
WORLD_PORT = 29656
SCALING = 1.7


def build_connector(role: str, local_rank: int) -> GpuAsyncAFDConnector:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=HIDDEN,
                num_experts_per_tok=TOPK,
                n_routed_experts=NUM_EXPERTS,
            ),
            dtype=torch.bfloat16,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=NUM_TOKENS),
        additional_config={
            "afd": {
                "role": role,
                "connector": "GpuAsyncAFDConnector",
                "async": True,
                "compute_gate_on_attention": True,
                "num_attention_ranks": 1,
                "num_ffn_ranks": 1,
                "port": PORT,
                "connector_extra_config": {"ring_depth": 1},
            },
        },
    )
    afd_config = AFDConfig(
        role=role,
        connector="GpuAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        num_attention_ranks=1,
        num_ffn_ranks=1,
        host="127.0.0.1",
        port=PORT,
    )
    connector = GpuAsyncAFDConnector(
        rank=local_rank,
        local_rank=local_rank,
        vllm_config=vllm_config,
        afd_config=afd_config,
        role_rank=0,
    )
    connector.init_afd_connector()
    return connector


def make_weights(device):
    generator = torch.Generator(device="cpu").manual_seed(11)
    w13 = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator)
        / HIDDEN**0.5
    ).to(device, torch.bfloat16)
    w2 = (
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator=generator)
        / INTERMEDIATE**0.5
    ).to(device, torch.bfloat16)
    return w13, w2


def make_layer_inputs(layer_idx, device):
    gen = torch.Generator(device="cpu").manual_seed(100 + layer_idx)
    x = torch.randn(NUM_TOKENS, HIDDEN, generator=gen).to(device, torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(NUM_EXPERTS, generator=gen)[:TOPK] for _ in range(NUM_TOKENS)],
    ).to(device, torch.int32)
    topk_weights = torch.rand(NUM_TOKENS, TOPK, generator=gen).to(device)
    return x, topk_ids, topk_weights


def reference_moe(x, w13, w2, topk_ids, topk_weights):
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            hidden = x[token].to(torch.float32) @ w13[expert].to(torch.float32).T
            gate, up = hidden.chunk(2, dim=-1)
            y = (F.silu(gate) * up) @ w2[expert].to(torch.float32).T
            out[token] += float(topk_weights[token, slot]) * y * SCALING
    return out


def init_world(rank: int) -> None:
    """Mimic a single `vllm serve`: a private default group of size 1.

    The connector bootstraps NVSHMEM on the AFD group itself, so the default
    group deliberately does *not* span both roles -- that is the topology the
    real deployment has.
    """
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{WORLD_PORT + rank}",
        world_size=1,
        rank=0,
    )


def run_attention(rank: int) -> None:
    init_world(0)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    connector = build_connector("attention", rank)
    w13, w2 = make_weights(device)

    for layer_idx in range(NUM_LAYERS):
        x, topk_ids, topk_weights = make_layer_inputs(layer_idx, device)
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=layer_idx,
                stage_idx=0,
                seq_len=NUM_TOKENS,
            ),
        )
        connector.send_attn_output(
            x,
            context,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        got = connector.recv_ffn_output(ref_tensor=x, ubatch_idx=0)
        expected = reference_moe(x, w13, w2, topk_ids, topk_weights)
        torch.testing.assert_close(
            got.to(torch.float32),
            expected,
            rtol=8e-2,
            atol=8e-2,
        )
        print(
            f"[A] layer {layer_idx}: combined output matches reference MoE", flush=True
        )

    print("PASS: async GPU connector end-to-end", flush=True)
    connector.close()


def run_ffn(rank: int) -> None:
    init_world(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    connector = build_connector("ffn", rank)
    w13, w2 = make_weights(device)
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            experts=SimpleNamespace(
                routed_experts=SimpleNamespace(w13_weight=w13, w2_weight=w2),
                _shared_experts=None,
                routed_scaling_factor=SCALING,
            ),
        ),
    )

    for _ in range(NUM_LAYERS):
        while True:
            try:
                work_item = connector.recv_ffn_work_item(
                    stage_idx=0,
                    max_num_tokens=NUM_TOKENS,
                )
                break
            except TimeoutError:
                continue
        states = work_item.context.states
        payload = compute_attention_gate_moe_ffn(
            layer,
            hidden_states=work_item.hidden_states,
            group_list=states.group_list,
            expand_x_shared=None,
        )
        connector.send_ffn_work_item_output(work_item, payload)
        print(
            f"[F] layer {work_item.layer_idx}: served "
            f"{work_item.num_tokens} routed tokens",
            flush=True,
        )

    connector.close()


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise SystemExit("this test needs two visible GPUs")
    mp.set_start_method("spawn", force=True)
    procs = [
        mp.Process(target=run_ffn, args=(1,)),
        mp.Process(target=run_attention, args=(0,)),
    ]
    for proc in procs:
        proc.start()
    failed = False
    for proc in procs:
        proc.join(timeout=300)
        if proc.exitcode != 0:
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
