from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig, afd_config_from_mapping
from afd_plugin.connectors import AFDConnectorFactory, AFDConnectorMetadata

pytest.importorskip("torch")
from afd_plugin.connectors.npu import async_cam as async_cam_module  # noqa: E402
from afd_plugin.connectors.npu.async_cam import (  # noqa: E402
    AFDAsyncConnector,
    AFDAsyncConnectorData,
    build_async_topology,
)


class _FakeTensor:
    def __init__(self, shape, *, dtype="bf16", device="npu:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def new_zeros(self, shape):
        return _FakeTensor(shape, dtype=self.dtype, device=self.device)

    def remainder(self, value):
        del value
        return self

    def unsqueeze(self, dim):
        shape = list(self.shape)
        shape.insert(dim, 1)
        return _FakeTensor(tuple(shape), dtype=self.dtype, device=self.device)

    def expand(self, *shape):
        return _FakeTensor(shape, dtype=self.dtype, device=self.device)

    def contiguous(self):
        return self


class _FakeCamOps:
    def __init__(self):
        self.calls = []

    def cam_dispatch_send(self, *args):
        self.calls.append(("dispatch_send", args))
        return args[0]

    def cam_dispatch_recv(self, *args):
        self.calls.append(("dispatch_recv", args))
        batch_size = args[4]
        hidden_size = args[5]
        topk = args[6]
        expert_rank_size = args[7]
        attention_rank_size = args[8]
        return (
            _FakeTensor((batch_size, hidden_size)),
            _FakeTensor((batch_size, topk), dtype="int32"),
            _FakeTensor((batch_size, topk), dtype="fp32"),
            _FakeTensor((batch_size * topk,), dtype="int32"),
            _FakeTensor((expert_rank_size,), dtype="int32"),
            _FakeTensor((attention_rank_size,), dtype="int32"),
            _FakeTensor((batch_size,), dtype="int32"),
        )

    def cam_combine_send(self, *args):
        self.calls.append(("combine_send", args))
        return args[0]

    def cam_combine_recv(self, *args):
        self.calls.append(("combine_recv", args))
        batch_size = args[6]
        hidden_size = args[7]
        return _FakeTensor((batch_size, hidden_size))


class _FakeTorch:
    def __init__(self):
        self.float32 = "fp32"
        self.int32 = "int32"
        self.ops = SimpleNamespace(cam=_FakeCamOps())

    def arange(self, stop, *, dtype, device):
        return _FakeTensor((stop,), dtype=dtype, device=device)

    def full(self, shape, value, *, dtype, device):
        del value
        return _FakeTensor(shape, dtype=dtype, device=device)

    def zeros(self, shape, *, dtype, device):
        return _FakeTensor(shape, dtype=dtype, device=device)

    def ones(self, shape, *, dtype, device):
        return _FakeTensor(shape, dtype=dtype, device=device)


def _vllm_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            tensor_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=2,
                n_routed_experts=8,
            ),
        ),
    )


def _afd_config(*, role: str, rank: int = 0, use_stub: bool = True):
    return AFDConfig(
        enabled=True,
        connector="afdasyncconnector",
        role=role,
        afd_server_rank=rank,
        num_attention_servers=4,
        num_ffn_servers=2,
        extra_config={"use_stub_cam_ops": use_stub},
    )


def _topk_payload(batch_size: int, topk: int = 2):
    return {
        "topk_ids": _FakeTensor((batch_size, topk), dtype="int32"),
        "topk_weights": _FakeTensor((batch_size, topk), dtype="fp32"),
    }


def test_config_accepts_async_connector_name():
    config = afd_config_from_mapping(
        {
            "enabled": True,
            "role": "attention",
            "connector": "afdasyncconnector",
        },
    )

    assert config.connector == "afdasyncconnector"


def test_async_connector_factory_creates_import_safe_connector():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
    )

    assert isinstance(connector, AFDAsyncConnector)
    assert not connector.is_initialized
    assert connector.uses_dp_metadata_control_plane is False
    assert connector.ffn_step_trigger == "connector"
    assert connector.requires_eager is True
    assert connector.required_platform == "ascend"


def test_async_topology_uses_cam_attention_first_rank_layout():
    attn = build_async_topology(_afd_config(role="attention", rank=3), 3)
    ffn = build_async_topology(
        _afd_config(role="ffn", rank=1),
        1,
        num_routed_experts=8,
    )

    assert attn.world_rank == 3
    assert ffn.world_rank == 5
    assert ffn.world_size == 6
    assert ffn.expert_per_rank == 4


def test_async_connector_disables_dp_metadata_control_plane():
    connector = AFDAsyncConnector(0, 0, _vllm_config(), _afd_config(role="ffn"))

    connector.update_state_from_dp_metadata({0: object()})
    connector.send_dp_metadata_list({0: object()})

    with pytest.raises(RuntimeError, match="does not use the DP metadata"):
        connector.recv_dp_metadata_list()


def test_async_connector_calls_cam_stub_shaped_ops(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = AFDAsyncConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention", use_stub=False),
    )
    connector._initialized = True
    connector.comm_args = _FakeTensor((1,), dtype="int64")
    connector._placeholder = _FakeTensor((8, 16))
    hidden_states = _FakeTensor((3, 16))
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=2,
        stage_idx=0,
        seq_len=3,
    )

    connector.configure_metadata(metadata, batch_size=3)
    output = connector.send_attn_output(
        hidden_states,
        metadata,
        **_topk_payload(3),
    )
    combined = connector.recv_ffn_output(
        ref_tensor=hidden_states,
        ubatch_idx=0,
    )

    assert output is hidden_states
    assert combined.shape == (3, 16)
    assert fake_torch.ops.cam.calls[0][0] == "dispatch_send"
    assert fake_torch.ops.cam.calls[1][0] == "combine_recv"
    assert fake_torch.ops.cam.calls[0][1][5:11] == (3, 16, 2, 2, 4, 4)
    assert isinstance(metadata.connector_data, AFDAsyncConnectorData)


def test_async_ffn_side_dispatch_recv_and_combine_send(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = AFDAsyncConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn", use_stub=False),
    )
    connector._initialized = True
    connector.comm_args = _FakeTensor((1,), dtype="int64")
    connector._placeholder = _FakeTensor((8, 16))

    recv_output = connector.recv_attn_output(batch_size=4, layer_idx=1)
    connector.send_ffn_output(recv_output.hidden_states, recv_output.metadata)

    assert recv_output.hidden_states.shape == (4, 16)
    assert recv_output.topk_ids.shape == (4, 2)
    assert recv_output.group_list is recv_output.ep_recv_counts
    assert fake_torch.ops.cam.calls[0][0] == "dispatch_recv"
    assert fake_torch.ops.cam.calls[1][0] == "combine_send"


def test_async_select_experts_maps_legacy_global_num_experts(monkeypatch):
    calls = []
    fake_package = ModuleType("vllm_ascend")
    fake_ops = ModuleType("vllm_ascend.ops")
    fake_fused_moe = ModuleType("vllm_ascend.ops.fused_moe")
    fake_selector = ModuleType("vllm_ascend.ops.fused_moe.experts_selector")

    def select_experts(*, num_experts=-1, **kwargs):
        calls.append((num_experts, kwargs))
        return "weights", "ids"

    fake_selector.select_experts = select_experts
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe", fake_fused_moe)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.experts_selector",
        fake_selector,
    )
    connector = AFDAsyncConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
    )

    result = connector.select_experts(router_logits="logits", global_num_experts=8)

    assert result == ("weights", "ids")
    assert calls == [(8, {"router_logits": "logits"})]


def test_async_connector_stub_mode_bypasses_cam_ops(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    attn_connector = AFDAsyncConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
    )
    attn_connector._initialized = True
    attn_connector.comm_args = _FakeTensor((1,), dtype="int64")
    attn_connector._placeholder = _FakeTensor((8, 16))
    hidden_states = _FakeTensor((3, 16))
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=2,
        stage_idx=0,
        seq_len=3,
    )
    attn_connector.configure_metadata(metadata, batch_size=3)

    output = attn_connector.send_attn_output(
        hidden_states,
        metadata,
        **_topk_payload(3),
    )
    combined = attn_connector.recv_ffn_output(
        ref_tensor=hidden_states,
        ubatch_idx=0,
    )

    ffn_connector = AFDAsyncConnector(0, 0, _vllm_config(), _afd_config(role="ffn"))
    ffn_connector._initialized = True
    ffn_connector.comm_args = _FakeTensor((1,), dtype="int64")
    ffn_connector._placeholder = _FakeTensor((8, 16))
    recv_output = ffn_connector.recv_attn_output(batch_size=4, layer_idx=1)
    ffn_connector.send_ffn_output(recv_output.hidden_states, recv_output.metadata)

    assert output is hidden_states
    assert combined is hidden_states
    assert recv_output.hidden_states.shape == (4, 16)
    assert recv_output.topk_ids.shape == (4, 2)
    assert recv_output.group_list is recv_output.ep_recv_counts
    assert fake_torch.ops.cam.calls == []
