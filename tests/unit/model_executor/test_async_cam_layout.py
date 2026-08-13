# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.model_executor.models.npu import async_cam_layout  # noqa: E402
from afd_plugin.model_executor.models.npu.async_cam_layout import (  # noqa: E402
    AsyncMoeUbatchMetadata,
)
from afd_plugin.model_executor.npu.async_cam_ubatching import (  # noqa: E402
    AsyncMoeStage,
)


def test_sp_layout_transposes_full_shards_into_stage_shards(monkeypatch):
    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 10), input_tokens=10),
            AsyncMoeStage(slice(0, 1), slice(10, 15), input_tokens=6),
        ],
        parent_input_tokens=16,
        use_sequence_parallel=True,
    )
    global_hidden = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    global_residual = global_hidden + 100
    positions = torch.arange(16)
    scaling = torch.ones(2, 16)
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(async_cam_layout, "get_tp_group", lambda: tp_group)

    for tp_rank, expected_positions in (
        (0, [[0, 1, 2, 3, 4], [10, 11, 12]]),
        (1, [[5, 6, 7, 8, 9], [13, 14, 0]]),
    ):
        tp_group.rank_in_group = tp_rank
        local_slice = slice(tp_rank * 8, (tp_rank + 1) * 8)

        def all_gather(tensor, token_dim):
            assert token_dim == 0
            assert tensor.shape[1] == 4
            return torch.cat((global_hidden, global_residual), dim=-1)

        monkeypatch.setattr(
            async_cam_layout,
            "tensor_model_parallel_all_gather",
            all_gather,
        )
        stage_inputs = async_cam_layout.build_async_moe_stage_inputs(
            global_hidden[local_slice],
            global_residual[local_slice],
            positions,
            scaling,
            metadata,
        )

        assert [stage.tolist() for stage in stage_inputs.positions] == (
            expected_positions
        )
        assert [int(stage.shape[0]) for stage in stage_inputs.hidden_states] == [
            5,
            3,
        ]
        assert [tuple(stage.shape) for stage in stage_inputs.llama_4_scaling] == [
            (2, 5),
            (2, 3),
        ]
        monkeypatch.setattr(
            async_cam_layout,
            "tensor_model_parallel_all_gather",
            lambda tensor, token_dim: (
                global_hidden[:10]
                if int(tensor.shape[token_dim]) == 5
                else torch.cat(
                    (
                        global_hidden[10:15],
                        global_hidden.new_zeros((1, 2)),
                    ),
                    dim=0,
                )
            ),
        )
        restored = async_cam_layout.restore_async_moe_stage_outputs(
            stage_inputs.hidden_states,
            metadata,
        )
        expected_restored = global_hidden.clone()
        expected_restored[15].zero_()
        assert torch.equal(restored, expected_restored[local_slice])


def test_replicated_layout_removes_and_restores_parent_padding():
    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 3), input_tokens=3),
            AsyncMoeStage(slice(0, 1), slice(3, 5), input_tokens=2),
        ],
        parent_input_tokens=8,
        use_sequence_parallel=False,
    )
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    positions = torch.arange(8)

    stage_inputs = async_cam_layout.build_async_moe_stage_inputs(
        hidden_states,
        None,
        positions,
        None,
        metadata,
    )

    assert [stage[:, 0].tolist() for stage in stage_inputs.hidden_states] == [
        [0.0, 2.0, 4.0],
        [6.0, 8.0],
    ]
    assert [stage.tolist() for stage in stage_inputs.positions] == [
        [0, 1, 2],
        [3, 4],
    ]
    restored = async_cam_layout.restore_async_moe_stage_outputs(
        stage_inputs.hidden_states,
        metadata,
    )
    assert torch.equal(restored[:5], hidden_states[:5])
    assert torch.count_nonzero(restored[5:]) == 0


def test_plain_tp_cam_boundary_shards_and_restores_replicated_tokens(monkeypatch):
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(async_cam_layout, "get_tp_group", lambda: tp_group)
    hidden_states = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    topk_weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    topk_ids = torch.arange(10, dtype=torch.int32).reshape(5, 2)
    router_logits = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    padded_output = torch.arange(12, dtype=torch.float32).reshape(6, 2) + 100

    monkeypatch.setattr(
        async_cam_layout,
        "tensor_model_parallel_all_gather",
        lambda tensor, token_dim: padded_output,
    )

    expected_hidden_rows = (
        hidden_states[:3],
        torch.cat((hidden_states[3:], hidden_states.new_zeros((1, 2)))),
    )
    for tp_rank in range(2):
        tp_group.rank_in_group = tp_rank
        payload = async_cam_layout.prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            router_logits,
            use_sequence_parallel=False,
        )

        assert torch.equal(payload.hidden_states, expected_hidden_rows[tp_rank])
        assert payload.hidden_states.shape[0] == 3
        assert payload.topk_weights.shape[0] == 3
        assert payload.topk_ids.shape[0] == 3
        assert payload.router_logits is not None
        assert payload.router_logits.shape[0] == 3
        assert payload.layout.parent_tokens == 5
        assert payload.layout.padded_tokens == 6
        assert payload.layout.requires_tp_all_gather is True

        local_output = padded_output[tp_rank * 3 : (tp_rank + 1) * 3]
        restored = async_cam_layout.restore_cam_dispatch_output(
            local_output,
            payload.layout,
        )
        assert torch.equal(restored, padded_output[:5])
