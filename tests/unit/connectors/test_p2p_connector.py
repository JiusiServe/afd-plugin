from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig, afd_config_from_mapping
from afd_plugin.connectors import AFDConnectorFactory, AFDDPMetadata
from afd_plugin.distributed import build_rank_mapping


def _fake_vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype="bf16",
            enforce_eager=True,
            hf_config=SimpleNamespace(hidden_size=16, num_hidden_layers=2),
        ),
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
    )


def _tolist(value):
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return list(value)


def test_p2p_connector_is_registered():
    sys.modules.pop("afd_plugin.connectors.gpu.p2p", None)

    cls = AFDConnectorFactory.get_connector_class("p2pconnector")

    assert cls.__name__ == "P2PAFDConnector"


def test_p2p_connector_can_be_constructed_without_runtime_initialization():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            enabled=True,
            role="attention",
            connector="p2pconnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )

    assert connector.is_initialized is False
    assert connector.world_rank == 1
    assert connector.dst_list == [0]


def test_p2p_connector_uses_dp_rank_as_role_rank_for_native_dp():
    connector = AFDConnectorFactory.create_connector(
        1,
        1,
        SimpleNamespace(
            model_config=SimpleNamespace(
                dtype="bf16",
                enforce_eager=True,
                hf_config=SimpleNamespace(hidden_size=16, num_hidden_layers=2),
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                data_parallel_rank=1,
            ),
        ),
        AFDConfig(
            enabled=True,
            role="attention",
            connector="p2pconnector",
            num_attention_ranks=2,
            num_ffn_ranks=2,
        ),
    )

    assert connector.mapping.role_rank == 1
    assert connector.world_rank == 3
    assert connector.p2p_rank == 3


@pytest.mark.parametrize(
    ("attention_size", "ffn_size", "role", "role_rank", "subgroup_ranks", "dsts"),
    [
        (2, 2, "attention", 1, (1, 3), (1,)),
        (2, 1, "attention", 0, (0, 1, 2), (0,)),
        (4, 2, "attention", 2, (1, 4, 5), ()),
        (4, 2, "ffn", 1, (1, 4, 5), ()),
    ],
)
def test_p2p_topology_supports_equal_and_integer_multiple_attention_counts(
    attention_size,
    ffn_size,
    role,
    role_rank,
    subgroup_ranks,
    dsts,
):
    mapping = build_rank_mapping(
        AFDConfig(
            enabled=True,
            role=role,
            connector="p2pconnector",
            num_attention_ranks=attention_size,
            num_ffn_ranks=ffn_size,
            afd_role_rank=role_rank,
        ),
    )

    assert mapping.ratio == attention_size // ffn_size
    assert mapping.subgroup_ranks == subgroup_ranks
    assert mapping.dp_metadata_destinations == dsts


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {
                "connector": "p2pconnector",
                "num_attention_ranks": 1,
                "num_ffn_ranks": 2,
            },
            "num_attention_ranks >= num_ffn_ranks",
        ),
        (
            {
                "connector": "p2pconnector",
                "num_attention_ranks": 3,
                "num_ffn_ranks": 2,
            },
            "multiple of num_ffn_ranks",
        ),
    ],
)
def test_p2p_topology_validation_errors_are_clear(raw, message):
    with pytest.raises(ValueError, match=message):
        afd_config_from_mapping(raw)


def test_p2p_module_exports_connector_class():
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")

    assert module.P2PAFDConnector.__module__ == "afd_plugin.connectors.gpu.p2p"


def test_p2p_dp_metadata_serialization_uses_json_payload():
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    metadata = SimpleNamespace(
        num_tokens_across_dp_cpu=[3, 5],
        max_tokens_across_dp_cpu=5,
    )

    payload = module._encode_dp_metadata_payload(
        {7: metadata},
        is_graph_capturing=True,
        is_warmup=False,
    )
    decoded, is_graph_capturing, is_warmup = module._decode_dp_metadata_payload(
        payload,
    )

    assert payload.startswith(b"{")
    assert isinstance(decoded[7], AFDDPMetadata)
    assert _tolist(decoded[7].num_tokens_across_dp_cpu) == [3, 5]
    assert int(decoded[7].max_tokens_across_dp_cpu) == 5
    with decoded[7].sp_local_sizes(sequence_parallel_size=1):
        assert decoded[7].get_chunk_sizes_across_dp_rank() == [3, 5]
    assert _tolist(decoded[7].cu_tokens_across_sp(1)) == [3, 8]
    assert is_graph_capturing is True
    assert is_warmup is False


def test_p2p_custom_ops_register_send_recv_with_fake_impls(monkeypatch):
    module = importlib.import_module("afd_plugin.connectors.gpu.p2p")
    calls = []

    torch_module = types.ModuleType("torch")
    torch_module.Tensor = object

    vllm_module = types.ModuleType("vllm")
    utils_module = types.ModuleType("vllm.utils")
    torch_utils_module = types.ModuleType("vllm.utils.torch_utils")

    def direct_register_custom_op(**kwargs):
        calls.append(kwargs)

    torch_utils_module.direct_register_custom_op = direct_register_custom_op
    utils_module.torch_utils = torch_utils_module
    vllm_module.utils = utils_module

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils_module)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils_module)
    monkeypatch.setattr(module, "_AFD_CUSTOM_OPS_REGISTERED", False)

    module._register_p2p_custom_ops()

    assert [call["op_name"] for call in calls] == [
        "afd_p2p_send",
        "afd_p2p_recv",
    ]
    assert calls[0]["mutates_args"] == ["tensor"]
    assert calls[1]["mutates_args"] == ["out"]
    assert callable(calls[0]["fake_impl"])
    assert callable(calls[1]["fake_impl"])


def test_p2p_hidden_state_send_uses_registered_custom_op(monkeypatch):
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            enabled=True,
            role="attention",
            connector="p2pconnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )
    communicator = object()
    connector.a2e_pynccl = communicator
    connector.a2e_comm_id = 17

    calls = []
    torch_module = types.ModuleType("torch")
    torch_module.ops = SimpleNamespace(
        vllm=SimpleNamespace(
            afd_p2p_send=lambda tensor, dst, comm_id: (
                calls.append((tensor, dst, comm_id)) or None
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    hidden_states = SimpleNamespace(
        is_cpu=False,
        device="cuda:0",
        shape=(4, 16),
        dtype="bf16",
    )
    output = connector._send_hidden_states(
        hidden_states,
        1,
        SimpleNamespace(world_size=2, rank=0),
        communicator,
    )

    assert calls == [(hidden_states, 1, 17)]
    assert output is None


def test_p2p_recv_preserves_dynamic_ref_tensor_first_dim(monkeypatch):
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _fake_vllm_config(),
        AFDConfig(
            enabled=True,
            role="attention",
            connector="p2pconnector",
            num_attention_ranks=2,
            num_ffn_ranks=1,
        ),
    )
    communicator = object()
    connector.e2a_pynccl = communicator
    connector.e2a_comm_id = 23

    calls = []
    torch_module = types.ModuleType("torch")
    torch_module.ops = SimpleNamespace(
        vllm=SimpleNamespace(
            afd_p2p_recv=lambda tensor, src, comm_id: (
                calls.append((tensor, src, comm_id)) or None
            ),
        ),
    )
    torch_module.empty = lambda *_args, **_kwargs: pytest.fail(
        "recv should reuse the dynamic ref tensor",
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    ref_tensor = SimpleNamespace(
        is_cpu=False,
        device="cuda:0",
        shape=(7, 16),
        dtype="bf16",
    )
    tensor_metadata = SimpleNamespace(
        device="cuda:0",
        dtype="bf16",
        size=(64, 16),
    )

    output = connector._recv_hidden_states(
        0,
        SimpleNamespace(world_size=2, rank=1),
        communicator,
        tensor_metadata,
        ref_tensor=ref_tensor,
    )

    assert output is ref_tensor
    assert calls == [(ref_tensor, 0, 23)]
