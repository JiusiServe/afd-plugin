from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig, afd_config_from_mapping

pytest.importorskip("torch")

from afd_plugin.connectors import (  # noqa: E402
    AFDA2FTransferPayload,
    AFDConnectorFactory,
    AFDTransferMetadata,
    AFDDPMetadata,
    AFDControlPayload,
    AFDF2ATransferPayload,
    AFDTransferState,
)
from afd_plugin.connectors.npu import async_cam as async_cam_module  # noqa: E402
from afd_plugin.connectors.npu.async_cam import (  # noqa: E402
    AFD_ASYNC_CAM_GROUP_NAME,
    CAM_COMM_ID,
    AFDAsyncTransferState,
    CAMAsyncAFDConnector,
    build_async_topology,
)


class _FakeScalar:
    def __init__(self, value):
        self._value = int(value)

    def item(self):
        return self._value


class _FakeTensorLike:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, item):
        start = "" if item.start is None else item.start
        stop = "" if item.stop is None else item.stop
        return f"{self.name}[{start}:{stop}]"


class _FakeTensor:
    def __init__(self, shape, *, dtype="bf16", device="npu:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def new_zeros(self, shape):
        return _FakeTensor(shape, dtype=self.dtype, device=self.device)

    def new_empty(self, shape):
        return _FakeTensor(shape, dtype=self.dtype, device=self.device)


class _FakeCamOps:
    def __init__(self):
        self.calls = []

    def async_dispatch_send(self, *args):
        self.calls.append(("dispatch_send", args))
        return args[0]

    def async_dispatch_recv(self, *args):
        self.calls.append(("dispatch_recv", args))
        batch_size = args[3]
        hidden_size = args[4]
        expert_per_rank = args[8]
        tp_size = args[11]
        return (
            _FakeTensor((batch_size, hidden_size)),
            _FakeTensor((max(1, batch_size // 2), hidden_size)),
            _FakeTensor((batch_size,), dtype="fp32"),
            _FakeTensor((max(1, batch_size // 2),), dtype="fp32"),
            _FakeTensor((5 + tp_size * (2 + expert_per_rank),), dtype="int64"),
            _FakeTensor((expert_per_rank,), dtype="int64"),
            _FakeTensor((1,), dtype="int64"),
        )

    def async_combine_send(self, *args):
        self.calls.append(("combine_send", args))
        return args[0]

    def async_combine_recv(self, *args):
        self.calls.append(("combine_recv", args))
        batch_size = args[5]
        hidden_size = args[6]
        return _FakeTensor((batch_size, hidden_size))


class _FakeTorch:
    def __init__(self):
        self.bfloat16 = "bf16"
        self.float16 = "fp16"
        self.float32 = "fp32"
        self.int32 = "int32"
        self.int64 = "int64"
        self.ops = SimpleNamespace(umdk_cam_op_lib=_FakeCamOps())

    def empty(self, shape, *, dtype, device):
        return _FakeTensor(shape, dtype=dtype, device=device)


def _vllm_config(*, tp_size: int = 1, pcp_size: int = 1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=pcp_size,
            tensor_parallel_size=tp_size,
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


def _afd_config(*, role: str, rank: int = 0, extra_config=None):
    return AFDConfig(
        enabled=True,
        connector="CAMAsyncAFDConnector",
        role=role,
        afd_role_rank=rank,
        num_attention_ranks=4,
        num_ffn_ranks=2,
        extra_config={} if extra_config is None else dict(extra_config),
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
            "connector": "CAMAsyncAFDConnector",
        },
    )

    assert config.connector == "CAMAsyncAFDConnector"


def test_async_connector_factory_creates_import_safe_connector():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
    )

    assert isinstance(connector, CAMAsyncAFDConnector)
    assert not connector.is_initialized
    assert connector.uses_dp_metadata_control_plane is False
    assert connector.ffn_step_trigger == "connector"
    assert connector.tp_size == 1


def test_async_connector_uses_attn_ranks_per_dp_for_cam_tp_size():
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(tp_size=4, pcp_size=2),
        _afd_config(role="attention", extra_config={"attn_ranks_per_dp": "3"}),
    )

    assert connector.tp_size == 3


@pytest.mark.parametrize("value", [True, "bad"])
def test_async_connector_rejects_invalid_attn_ranks_per_dp(value):
    with pytest.raises(TypeError, match="extra_config.attn_ranks_per_dp"):
        CAMAsyncAFDConnector(
            0,
            0,
            _vllm_config(),
            _afd_config(role="attention", extra_config={"attn_ranks_per_dp": value}),
        )


def test_async_connector_rejects_nonpositive_attn_ranks_per_dp():
    with pytest.raises(ValueError, match="extra_config.attn_ranks_per_dp"):
        CAMAsyncAFDConnector(
            0,
            0,
            _vllm_config(),
            _afd_config(role="attention", extra_config={"attn_ranks_per_dp": 0}),
        )


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


def test_async_connector_init_creates_attention_first_hccl_group(monkeypatch):
    calls = []
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    monkeypatch.setattr(
        async_cam_module,
        "ensure_cam_async_ops_available",
        lambda: None,
    )

    def fake_init_afd_process_group(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(group_name=kwargs["group_name"])

    monkeypatch.setattr(
        async_cam_module,
        "init_afd_process_group",
        fake_init_afd_process_group,
    )
    monkeypatch.setattr(
        async_cam_module,
        "_hccl_comm_name",
        lambda group, rank: f"hccl:{group.group_name}:{rank}",
    )
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn", rank=1),
    )

    connector.init_afd_connector()

    assert calls == [
        {
            "backend": "hccl",
            "init_method": "tcp://127.0.0.1:29500",
            "world_size": 6,
            "rank": 5,
            "group_name": AFD_ASYNC_CAM_GROUP_NAME,
            "timeout": calls[0]["timeout"],
        },
    ]
    assert connector.cam_pg is not None
    assert connector.group_name == f"hccl:{AFD_ASYNC_CAM_GROUP_NAME}:5"
    assert connector.comm_args.shape == (1,)
    assert connector.comm_args.dtype == fake_torch.float16
    assert connector._placeholder.shape == (1,)


def test_async_connector_disables_dp_metadata_control_plane():
    connector = CAMAsyncAFDConnector(0, 0, _vllm_config(), _afd_config(role="ffn"))
    payload = AFDControlPayload(
        dp_metadata_list={0: AFDDPMetadata([1])},
        is_graph_capturing=False,
        is_warmup=False,
    )

    connector.update_state_from_dp_metadata(payload)
    connector.send_dp_metadata_list(payload)

    with pytest.raises(RuntimeError, match="does not use the DP metadata"):
        connector.recv_dp_metadata_list()


def test_async_connector_calls_cam_shaped_ops(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(pcp_size=3),
        _afd_config(
            role="attention",
            extra_config={"attn_ranks_per_dp": 3},
        ),
    )
    connector._initialized = True
    connector.comm_args = _FakeTensor((1,), dtype="fp16")
    connector._placeholder = _FakeTensor((8, 16))
    hidden_states = _FakeTensor((3, 16))
    metadata = AFDTransferMetadata.create_attention_metadata(
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

    assert output is None
    assert combined.shape == (3, 16)
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][0] == "dispatch_send"
    assert fake_torch.ops.umdk_cam_op_lib.calls[1][0] == "combine_recv"
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][1][3] == CAM_COMM_ID
    assert fake_torch.ops.umdk_cam_op_lib.calls[1][1][4] == CAM_COMM_ID
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][1][5:11] == (3, 16, 2, 2, 4, 4)
    assert fake_torch.ops.umdk_cam_op_lib.calls[1][1][5:11] == (3, 16, 2, 2, 4, 4)
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][1][14] == 3
    assert isinstance(metadata.connector_data, AFDAsyncTransferState)
    assert isinstance(metadata.connector_data, AFDTransferState)


def test_async_ffn_side_dispatch_recv_and_combine_send(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(pcp_size=2),
        _afd_config(role="ffn", extra_config={"attn_ranks_per_dp": 2}),
    )
    connector._initialized = True
    connector.comm_args = _FakeTensor((1,), dtype="fp16")
    connector._placeholder = _FakeTensor((8, 16))

    recv_output = connector.recv_attn_output(batch_size=4, layer_idx=1)
    connector.send_ffn_output(recv_output.hidden_states, recv_output.metadata)

    assert recv_output.hidden_states.shape == (4, 16)
    assert recv_output.topk_ids is None
    assert recv_output.dynamic_scales.shape == (4,)
    assert recv_output.expand_x_shared.shape == (2, 16)
    assert recv_output.dynamic_scales_shared.shape == (2,)
    assert recv_output.group_list is recv_output.ep_recv_counts
    assert recv_output.ep_recv_counts_shared.shape == (1,)
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][0] == "dispatch_recv"
    assert fake_torch.ops.umdk_cam_op_lib.calls[1][0] == "combine_send"
    assert fake_torch.ops.umdk_cam_op_lib.calls[0][1][11] == 2
    assert fake_torch.ops.umdk_cam_op_lib.calls[1][1][13] == 2
    assert fake_torch.ops.umdk_cam_op_lib.calls[1][1][3] is recv_output.atten_batch_size


def test_async_combine_send_requires_dispatch_recv_token_metadata(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(async_cam_module, "torch", fake_torch)
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn"),
    )
    connector._initialized = True
    metadata = AFDTransferMetadata.create_ffn_metadata(
        layer_idx=1,
        stage_idx=0,
        seq_lens=[4],
    )
    metadata.connector_data = AFDAsyncTransferState(
        batch_size=4,
        hidden_size=16,
        topk=2,
        layer_idx=1,
        expert_token_nums=_FakeTensor((4,), dtype="int64"),
    )

    with pytest.raises(RuntimeError, match="TokenNums_Rankid_Layeridx"):
        connector.send_ffn_output(_FakeTensor((4, 16)), metadata)


def test_async_ffn_work_item_uses_cam_layer_and_token_metadata(monkeypatch):
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn"),
    )
    connector.ffn_size = 2

    def fake_recv_attn_output(*, metadata, ubatch_idx):
        assert ubatch_idx == 0
        return AFDA2FTransferPayload(
            hidden_states=_FakeTensorLike("hidden"),
            metadata=metadata,
            atten_batch_size=[
                _FakeScalar(7),
                _FakeScalar(0),
                _FakeScalar(11),
            ],
            group_list="groups",
            dynamic_scales=_FakeTensorLike("scales"),
            expand_x_shared=_FakeTensorLike("shared-hidden"),
            dynamic_scales_shared=_FakeTensorLike("shared-scales"),
            ep_recv_counts_shared=[_FakeScalar(2)],
            x_active_mask=_FakeTensorLike("active-mask"),
        )

    monkeypatch.setattr(connector, "recv_attn_output", fake_recv_attn_output)

    work_item = connector.recv_ffn_work_item(stage_idx=0, max_num_tokens=16)

    assert work_item.layer_idx == 11
    assert work_item.stage_idx == 0
    assert work_item.total_num_tokens == 7
    assert work_item.shared_num_tokens == 2
    assert work_item.num_tokens == 5
    assert work_item.hidden_states == "hidden[:5]"
    assert work_item.metadata.layer_idx == 11
    assert work_item.metadata.seq_lens == [5]
    assert work_item.recv_output.dynamic_scales == "scales[:5]"
    assert work_item.recv_output.expand_x_shared == "shared-hidden[:2]"
    assert work_item.recv_output.dynamic_scales_shared == "shared-scales[:2]"
    assert work_item.recv_output.x_active_mask == "active-mask[:5]"
    assert work_item.metadata.connector_data.layer_idx == 11


def test_async_ffn_work_item_uses_expert_counts_for_routed_tokens(monkeypatch):
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn"),
    )

    def fake_recv_attn_output(*, metadata, ubatch_idx):
        assert ubatch_idx == 0
        return AFDA2FTransferPayload(
            hidden_states=_FakeTensorLike("hidden"),
            metadata=metadata,
            atten_batch_size=[
                _FakeScalar(6),
                _FakeScalar(0),
                _FakeScalar(23),
            ],
            group_list=[
                _FakeScalar(1),
                _FakeScalar(0),
                _FakeScalar(0),
                _FakeScalar(1),
                _FakeScalar(1),
                _FakeScalar(0),
                _FakeScalar(0),
                _FakeScalar(1),
                _FakeScalar(0),
                _FakeScalar(1),
                _FakeScalar(1),
                _FakeScalar(0),
                _FakeScalar(0),
                _FakeScalar(0),
                _FakeScalar(0),
                _FakeScalar(0),
            ],
            expand_x_shared=_FakeTensorLike("shared-hidden"),
            dynamic_scales_shared=_FakeTensorLike("shared-scales"),
            ep_recv_counts_shared=[_FakeScalar(1)],
        )

    monkeypatch.setattr(connector, "recv_attn_output", fake_recv_attn_output)

    work_item = connector.recv_ffn_work_item(stage_idx=0, max_num_tokens=16)

    assert work_item.layer_idx == 23
    assert work_item.total_num_tokens == 6
    assert work_item.shared_num_tokens == 1
    assert work_item.num_tokens == 6
    assert work_item.hidden_states == "hidden[:6]"
    assert work_item.metadata.seq_lens == [6]
    assert work_item.recv_output.expand_x_shared == "shared-hidden[:1]"
    assert work_item.recv_output.dynamic_scales_shared == "shared-scales[:1]"


def test_async_cam_shared_token_count_uses_expert_tokens_shared_directly():
    metadata = AFDTransferMetadata.create_ffn_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_lens=[10],
    )
    payload = AFDA2FTransferPayload(
        hidden_states=_FakeTensorLike("hidden"),
        metadata=metadata,
        atten_batch_size=[
            _FakeScalar(10),
            _FakeScalar(0),
            _FakeScalar(7),
        ],
        ep_recv_counts=[_FakeScalar(12), _FakeScalar(15)],
        ep_recv_counts_shared=[_FakeScalar(9)],
    )

    assert async_cam_module._cam_shared_token_count(payload, fallback=10) == 9


def test_async_slice_cam_payload_shared_tensors_fallback_to_100_tokens():
    payload = AFDA2FTransferPayload(
        hidden_states=_FakeTensorLike("hidden"),
        metadata=AFDTransferMetadata.create_ffn_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_lens=[10],
        ),
        dynamic_scales=_FakeTensorLike("scales"),
        expand_x_shared=_FakeTensorLike("shared-hidden"),
        dynamic_scales_shared=_FakeTensorLike("shared-scales"),
    )

    async_cam_module._slice_cam_payload_to_actual_tokens(
        payload.hidden_states,
        payload,
        num_tokens=5,
        shared_num_tokens=0,
    )

    assert payload.dynamic_scales == "scales[:5]"
    assert payload.expand_x_shared == "shared-hidden[:100]"
    assert payload.dynamic_scales_shared == "shared-scales[:100]"


def test_async_send_ffn_work_item_output_preserves_all_shared_passthrough(
    monkeypatch,
):
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="ffn"),
    )
    sent_outputs = []

    def fake_send_ffn_output(ffn_output, metadata, **kwargs):
        sent_outputs.append((ffn_output, metadata, kwargs))

    monkeypatch.setattr(connector, "send_ffn_output", fake_send_ffn_output)

    def fake_recv_attn_output(*, metadata, ubatch_idx):
        return AFDA2FTransferPayload(
            hidden_states=_FakeTensorLike("hidden"),
            metadata=metadata,
            atten_batch_size=[
                _FakeScalar(5),
                _FakeScalar(0),
                _FakeScalar(7),
            ],
            expand_x_shared=_FakeTensorLike("shared-hidden"),
            dynamic_scales_shared=_FakeTensorLike("shared-scales"),
            ep_recv_counts_shared=[_FakeScalar(5)],
        )

    monkeypatch.setattr(connector, "recv_attn_output", fake_recv_attn_output)

    work_item = connector.recv_ffn_work_item(stage_idx=0, max_num_tokens=16)
    sent_output = connector.send_ffn_work_item_output(
        work_item,
        AFDF2ATransferPayload(
            routed_output="computed-routed",
            shared_output="computed-shared",
        ),
    )

    assert work_item.num_tokens == 0
    assert work_item.hidden_states == "hidden[:0]"
    assert work_item.metadata.seq_lens == [5]
    assert sent_output == AFDF2ATransferPayload(
        routed_output=work_item.recv_output.hidden_states,
        shared_output="computed-shared",
    )
    assert sent_outputs == [
        (
            work_item.recv_output.hidden_states,
            work_item.metadata,
            {"ubatch_idx": 0, "expand_x_shared": "computed-shared"},
        ),
    ]


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
    connector = CAMAsyncAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
    )

    result = connector.select_experts(router_logits="logits", global_num_experts=8)

    assert result == ("weights", "ids")
    assert calls == [(8, {"router_logits": "logits"})]
