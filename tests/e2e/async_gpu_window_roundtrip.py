"""Two-rank A->F->A round trip over SymmWindow with real routing data.

Run with two GPUs::

    torchrun --nproc_per_node=2 tests/e2e/async_gpu_window_roundtrip.py


Rank 0 plays one Attention rank, rank 1 one FFN rank holding every expert.
The FFN side runs an identity "expert", so the recombined result must equal the
weighted sum of the inputs -- the same invariant the CPU self-check asserts,
but now carried across the wire.
"""

import os

import torch
import torch.distributed as dist

from afd_plugin.connectors.gpu.async_gpu import plan_dispatch
from afd_plugin.connectors.gpu.symm_window import (
    SlotLayout,
    SymmWindow,
    encode_header,
)

NUM_TOKENS = 64
HIDDEN = 128
TOPK = 6
FFN_SIZE = 1
EXPERT_PER_RANK = 16
RING_DEPTH = 2
NUM_LAYERS = 3


def main() -> None:
    rank = int(os.environ["RANK"])
    # Device index must come from LOCAL_RANK: under a GPU reservation the
    # visible devices are remapped, so the global rank can be out of range.
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")

    layout = SlotLayout.build(
        expert_per_rank=EXPERT_PER_RANK,
        partial_cap=NUM_TOKENS * TOPK,
        token_cap=NUM_TOKENS,
        # This smoke test ships every token as shared, and FFN_SIZE is 1, so the
        # per-rank split is the whole batch.
        shared_cap=-(-NUM_TOKENS // FFN_SIZE),
        hidden_size=HIDDEN,
        payload_itemsize=2,
    )
    window = SymmWindow(
        num_regions=1,
        ring_depth=RING_DEPTH,
        layout=layout,
        payload_dtype=torch.bfloat16,
        device=device,
        group=dist.group.WORLD,
        rank=rank,
        world_size=2,
    )
    if rank == 0:
        print(
            f"slot={layout.slot_bytes / 2**10:.0f}KiB "
            f"window={window.total_bytes / 2**10:.0f}KiB",
            flush=True,
        )

    for layer_idx in range(NUM_LAYERS):
        ring = layer_idx % RING_DEPTH
        gen = torch.Generator(device="cpu").manual_seed(layer_idx)
        hidden = torch.randn(NUM_TOKENS, HIDDEN, generator=gen).to(
            device, torch.bfloat16
        )
        topk_ids = torch.stack(
            [
                torch.randperm(EXPERT_PER_RANK, generator=gen)[:TOPK]
                for _ in range(NUM_TOKENS)
            ],
        ).to(device, torch.int32)
        topk_weights = torch.rand(NUM_TOKENS, TOPK, generator=gen).to(device)

        # Both ranks build the plan from the same inputs. Rank 0 sends from it;
        # rank 1 only uses it as an oracle for what should have arrived.
        plan = plan_dispatch(
            topk_ids,
            topk_weights,
            ffn_size=FFN_SIZE,
            expert_per_rank=EXPERT_PER_RANK,
        )

        if rank == 0:
            counts_host = plan.counts.cpu().tolist()
            # Shared rows are a contiguous range; this rank owns all of them.
            shared = slice(0, NUM_TOKENS)
            header = encode_header(
                layout,
                seq=layer_idx + 1,
                src_role_rank=0,
                layer_idx=layer_idx,
                stage_idx=0,
                num_tokens=NUM_TOKENS,
                routed_tokens=NUM_TOKENS * TOPK,
                shared_tokens=shared.stop - shared.start,
                topk=TOPK,
                flags=0,
                expert_counts=counts_host,
                segment_start=0,
            )
            # Capacity form: the whole batch and every partial go out, and the
            # rank-1 side locates its own run from the header.
            window.write_slot(
                peer=1,
                region=0,
                ring=ring,
                header=header,
                expand_idx=plan.expand_idx,
                weights=plan.weights,
                routed_x=hidden,
                shared_x=hidden[shared],
            )

            # Wait for the expert output and reduce it.
            while True:
                arrived = window.poll()
                if arrived is not None:
                    break
            got = arrived.header
            assert got.layer_idx == layer_idx, (got.layer_idx, layer_idx)
            assert got.src_role_rank == 0, got.src_role_rank
            assert got.routed_tokens == NUM_TOKENS * TOPK, got.routed_tokens

            # A reply is a whole batch, so combine adds it without an index.
            acc = (
                window.local_routed(
                    arrived.region,
                    arrived.ring,
                    got.num_tokens,
                )
                .to(torch.float32)
                .clone()
            )
            acc[shared] += window.local_shared(
                arrived.region, arrived.ring, got.shared_tokens
            ).to(torch.float32)
            expected = (
                hidden.to(torch.float32) * topk_weights.sum(dim=1, keepdim=True)
                + hidden.to(torch.float32)  # identity shared expert
            )
            torch.testing.assert_close(acc, expected, rtol=2e-2, atol=2e-2)
            print(f"layer {layer_idx}: combine matches reference", flush=True)
        else:
            while True:
                arrived = window.poll()
                if arrived is not None:
                    break
            got = arrived.header
            assert got.layer_idx == layer_idx, (got.layer_idx, layer_idx)
            assert sum(got.expert_counts) == got.routed_tokens
            shipped = window.local_routed(arrived.region, arrived.ring, got.num_tokens)
            shared_rows = window.local_shared(
                arrived.region, arrived.ring, got.shared_tokens
            )
            expand = window.local_expand_idx(
                arrived.region, arrived.ring, got.routed_tokens, got.segment_start
            ).to(torch.int64)
            weights = window.local_weights(
                arrived.region, arrived.ring, got.routed_tokens, got.segment_start
            )
            # Gathering by the partial indices must reproduce the sender's rows.
            expanded = shipped.index_select(0, expand)
            torch.testing.assert_close(expanded, hidden.index_select(0, expand))

            # Identity experts, then the weighted reduce back to token rows.
            reduced = torch.zeros(
                got.num_tokens, HIDDEN, dtype=torch.float32, device=device
            )
            reduced.index_add_(
                0, expand, expanded.to(torch.float32) * weights.unsqueeze(1)
            )

            echo = encode_header(
                layout,
                seq=layer_idx + 1,
                src_role_rank=0,
                layer_idx=got.layer_idx,
                stage_idx=got.stage_idx,
                num_tokens=got.num_tokens,
                routed_tokens=got.routed_tokens,
                shared_tokens=got.shared_tokens,
                topk=TOPK,
                flags=0,
                expert_counts=got.expert_counts,
                segment_start=got.segment_start,
            )
            window.write_slot(
                peer=0,
                region=0,
                ring=arrived.ring,
                header=echo,
                expand_idx=None,
                weights=None,
                routed_x=reduced,
                shared_x=shared_rows.clone(),
            )
            print(
                f"layer {layer_idx}: dispatch payload verified, echoed back", flush=True
            )

    dist.barrier()
    if rank == 0:
        print("PASS: A->F->A round trip over SymmWindow", flush=True)
    window.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
