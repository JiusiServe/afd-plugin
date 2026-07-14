from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import (
    AFDA2FTransferPayload,
    AFDConnectorFactory,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.connectors.npu import camp2p as camp2p_module
from afd_plugin.connectors.npu.camp2p import (
    CAMP2pAFDConnector,
    CAMP2PTransferState,
    build_camp2p_topology,
)


class _FakeDPMetadata:
    def __init__(self, values):
        self.num_tokens_across_dp_cpu = values


def _vllm_config(*, num_ubatches: int = 1, n_shared_experts: int = 0):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            num_ubatches=num_ubatches,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=2,
                n_routed_experts=4,
                n_shared_experts=n_shared_experts,
            ),
        ),
    )


def _afd_config(*, role: str, rank: int = 0, extra_config: dict | None = None):
    return AFDConfig(
        enabled=True,
        connector="CAMP2pAFDConnector",
        role=role,
        afd_role_rank=rank,
        num_attention_ranks=4,
        num_ffn_ranks=2,
        extra_config=extra_config or {},
    )


def test_camp2p_factory_creates_connector():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
    )

    assert isinstance(connector, CAMP2pAFDConnector)
    assert not connector.is_initialized
    assert connector.max_num_reqs == 8


def test_camp2p_topology_matches_original_rank_layout():
    attn0 = build_camp2p_topology(_afd_config(role="attention", rank=0), 0)
    attn1 = build_camp2p_topology(_afd_config(role="attention", rank=1), 1)
    attn2 = build_camp2p_topology(_afd_config(role="attention", rank=2), 2)
    ffn1 = build_camp2p_topology(_afd_config(role="ffn", rank=1), 1)

    assert (attn0.world_rank, attn0.p2p_rank, attn0.dp_metadata_destinations) == (
        2,
        2,
        (0,),
    )
    assert (attn1.world_rank, attn1.p2p_rank, attn1.dp_metadata_destinations) == (
        3,
        3,
        (1,),
    )
    assert not attn2.participates_in_p2p_group
    assert (ffn1.world_rank, ffn1.p2p_rank) == (1, 1)


def test_camp2p_create_recv_metadata_uses_original_contiguous_af_grouping():
    rank0 = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn", rank=0),
    )
    rank1 = CAMP2pAFDConnector(
        1,
        1,
        _vllm_config(),
        _afd_config(role="ffn", rank=1),
    )
    dp_metadata_list = {0: _FakeDPMetadata([2, 3, 5, 7])}

    metadata0 = rank0.create_recv_metadata(
        dp_metadata_list=dp_metadata_list,
        ubatch_idx=0,
        layer_idx=3,
    )
    metadata1 = rank1.create_recv_metadata(
        dp_metadata_list=dp_metadata_list,
        ubatch_idx=0,
        layer_idx=3,
    )

    assert metadata0.seq_lens == [5]
    assert metadata1.seq_lens == [12]
    assert isinstance(metadata0.transfer_state, CAMP2PTransferState)
    assert isinstance(metadata0.transfer_state, AFDTransferState)
    assert metadata0.transfer_state.batch_size == 5
    assert metadata0.transfer_state.h == 16
    assert metadata0.transfer_state.k == 2


def test_camp2p_ignores_mix_placement_for_connector_metadata():
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(n_shared_experts=3),
        _afd_config(
            role="ffn",
            rank=0,
            extra_config={"mix_placement": True},
        ),
    )

    metadata = connector.create_recv_metadata(
        dp_metadata_list={0: _FakeDPMetadata([2, 3, 5, 7])},
        ubatch_idx=0,
        layer_idx=3,
    )

    assert metadata.transfer_state.k == 2
    assert metadata.transfer_state.moe_expert_num == 4
    assert metadata.transfer_state.shared_expert_num == 0


def test_camp2p_update_metadata_keeps_original_handle_shape():
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn", rank=0),
    )
    metadata = connector.create_recv_metadata(
        dp_metadata_list={0: _FakeDPMetadata([2, 3, 5, 7])},
        ubatch_idx=0,
        layer_idx=0,
    )
    recv_output = AFDA2FTransferPayload(
        hidden_states="hidden",
        metadata=metadata,
        topk_ids="ids",
        topk_weights="weights",
        expand_idx="expand",
        ep_recv_counts="counts",
        atten_batch_size="atten",
    )

    connector.update_metadata(metadata, recv_output)

    assert metadata.transfer_state.handle == [
        "ids",
        "weights",
        "expand",
        "counts",
        "atten",
    ]


def test_camp2p_init_creates_one_hccl_group_per_ubatch(monkeypatch):
    calls = []

    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(camp2p_module, "ensure_cam_p2p_ops_available", lambda: None)
    monkeypatch.setattr(camp2p_module, "_register_camp2p_custom_ops", lambda: None)

    def fake_init_afd_process_group(**kwargs):
        calls.append(kwargs)
        backend = SimpleNamespace(
            get_hccl_comm_name=lambda rank: f"hccl:{kwargs['group_name']}:{rank}",
        )
        return SimpleNamespace(
            group_name=kwargs["group_name"],
            _get_backend=lambda device: backend,
        )

    monkeypatch.setattr(
        camp2p_module,
        "init_afd_process_group",
        fake_init_afd_process_group,
    )
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(num_ubatches=2),
        _afd_config(role="attention", rank=0),
    )

    connector.init_afd_connector()

    assert [call["group_name"] for call in calls[:2]] == ["afd", "afd1"]
    assert connector.hccl_comm_name_list == ["hccl:afd:2", "hccl:afd1:2"]
    assert connector.hccl_comm_name == "hccl:afd:2"
    assert connector.hccl_comm_name2 == "hccl:afd1:2"
    assert (
        camp2p_module._get_group_ep(
            0,
            connector.hccl_comm_name,
            connector.hccl_comm_name2,
            "",
        )
        == "hccl:afd:2"
    )
    assert (
        camp2p_module._get_group_ep(
            1,
            connector.hccl_comm_name,
            connector.hccl_comm_name2,
            "",
        )
        == "hccl:afd1:2"
    )


def test_camp2p_send_attn_custom_op_receives_all_hccl_names(monkeypatch):
    torch = pytest.importorskip("torch")
    captured = {}
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(num_ubatches=2),
        _afd_config(role="attention", rank=0),
    )
    connector._initialized = True
    connector.hccl_comm_name = "hccl0"
    connector.hccl_comm_name2 = "hccl1"
    connector.hccl_comm_name3 = ""
    hidden_states = torch.empty((3, 16))
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=1,
        seq_len=3,
    )

    def fake_set_forward_context_transfer_state(data, *, ubatch_idx=None):
        captured["transfer_state"] = data
        captured["ubatch_idx"] = ubatch_idx

    def fake_send_attn_output(*args):
        captured["args"] = args
        return args[0]

    monkeypatch.setattr(
        camp2p_module,
        "_set_forward_context_transfer_state",
        fake_set_forward_context_transfer_state,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "afd_camp2p_send_attn_output",
        fake_send_attn_output,
        raising=False,
    )

    output = connector.send_attn_output(hidden_states, metadata)

    assert output is None
    assert captured["ubatch_idx"] == 1
    assert captured["args"][1:4] == ("hccl0", "hccl1", "")
    assert captured["args"][4] == 3
    assert captured["transfer_state"].batch_size == 3


def test_camp2p_init_fails_cleanly_without_ascend_runtime():
    connector = CAMP2pAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention", rank=0),
    )

    with pytest.raises(RuntimeError, match="AFD Ascend custom ops|torch-npu"):
        connector.init_afd_connector()
