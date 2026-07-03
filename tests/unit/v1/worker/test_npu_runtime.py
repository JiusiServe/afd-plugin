from __future__ import annotations

import importlib
import logging
import sys
from collections import deque
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("torch")

from afd_plugin.compat.ascend import (
    fail_if_unsupported_npu_afd_features,
    npu_afd_num_ubatches,
)
from afd_plugin.compat.ascend import runtime as ascend_runtime
from afd_plugin.connectors import (
    AFDConnectorMetadata,
    AFDMetadata,
    AFDRecvOutput,
)


class _RecordingConnector:
    world_rank = 0

    def __init__(self):
        self.dp_metadata_updates = []
        self.sent_dp_metadata_lists = []

    def is_attn_top_min_size_rank(self, world_rank):
        return world_rank == self.world_rank

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list,
        *,
        is_graph_capturing=False,
        is_warmup=False,
    ):
        self.dp_metadata_updates.append(
            (dp_metadata_list, is_graph_capturing, is_warmup),
        )

    def send_dp_metadata_list(
        self,
        dp_metadata_list,
        *,
        is_graph_capturing=False,
        is_warmup=False,
    ):
        self.sent_dp_metadata_lists.append(
            (dp_metadata_list, is_graph_capturing, is_warmup),
        )


class _FakeFFNConnector:
    def __init__(self, *, attn_size=1, ffn_size=1, role_rank=0, world_rank=0):
        self.dp_metadata_list = {}
        self.attn_outputs = deque()
        self.ffn_outputs = []
        self.updates = []
        self.metadata_updates = []
        self.attn_size = attn_size
        self.ffn_size = ffn_size
        self.world_rank = world_rank
        self.topology = SimpleNamespace(role_rank=role_rank)

    def update_state_from_dp_metadata(self, dp_metadata_list, **kwargs):
        self.dp_metadata_list = dict(dp_metadata_list)
        self.updates.append((dict(dp_metadata_list), kwargs))

    def recv_attn_output(self, metadata=None, ubatch_idx=None):
        for item in tuple(self.attn_outputs):
            if item[1].stage_idx == ubatch_idx:
                self.attn_outputs.remove(item)
                return AFDRecvOutput(hidden_states=item[0], metadata=item[1])
        raise IndexError(ubatch_idx)

    def create_recv_metadata(self, **kwargs):
        return AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=kwargs["layer_idx"],
            stage_idx=kwargs["ubatch_idx"],
            seq_lens=[1],
        )

    def send_ffn_output(self, ffn_output, metadata, **kwargs):
        self.ffn_outputs.append((ffn_output, metadata, kwargs))

    def update_metadata(self, metadata, recv_output):
        self.metadata_updates.append((metadata, recv_output))

    def close(self):
        return None


class _FakeModel:
    def compute_ffn_output(self, hidden_states, layer_idx, **kwargs):
        del kwargs
        return f"npu-ffn({hidden_states}, layer={layer_idx})"


class _FakeDPMetadata:
    def __init__(self, values):
        self.num_tokens_across_dp_cpu = values


def _parallel_config(**overrides):
    values = {
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
        "enable_dbo": False,
        "use_ubatching": False,
        "num_ubatches": 1,
        "ubatch_size": 0,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "dbo_decode_token_threshold": 1,
        "dbo_prefill_token_threshold": 1,
        "worker_cls": "unused",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _vllm_config(*, role="attention", extra_config=None, **parallel_overrides):
    return SimpleNamespace(
        additional_config={
            "afd": {
                "enabled": True,
                "role": role,
                "connector": "camp2pconnector",
                "extra_config": extra_config or {},
            },
        },
        parallel_config=_parallel_config(**parallel_overrides),
        model_config=SimpleNamespace(enforce_eager=True),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL"),
        ),
    )


def _require_npu_runtime():
    pytest.importorskip("vllm", reason="NPU runtime tests require vLLM")
    pytest.importorskip("vllm_ascend", reason="NPU runtime tests require vLLM-Ascend")
    pytest.importorskip("torch_npu", reason="NPU runtime tests require torch-npu")


def _new_attention_runner():
    _require_npu_runtime()
    from afd_plugin.v1.worker.ascend.attention_model_runner import (
        AFDNPUAttentionModelRunner,
    )

    return object.__new__(AFDNPUAttentionModelRunner)


def _new_ffn_runner():
    _require_npu_runtime()
    from afd_plugin.v1.worker.ascend.ffn_model_runner import AFDNPUFFNModelRunner

    return object.__new__(AFDNPUFFNModelRunner)


def _new_ffn_worker():
    _require_npu_runtime()
    from afd_plugin.v1.worker.ascend.ffn_worker import AFDNPUFFNWorker

    return object.__new__(AFDNPUFFNWorker)


def test_npu_attention_runner_builds_and_mirrors_metadata():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(role="attention")
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=[1]),
        ubatch_slices=None,
        batch_descriptor=SimpleNamespace(num_tokens=5),
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    metadata = forward_context.additional_kwargs["afd_metadata"]
    assert forward_context.afd_metadata is metadata
    assert metadata.afd_tokens_lens == [1]
    assert len(runner.afd_connector.dp_metadata_updates) == 1
    assert len(runner.afd_connector.sent_dp_metadata_lists) == 1


def test_npu_attention_runner_builds_dp_fallback():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(role="attention")
    runner.afd_connector = object()
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 7)

    dp_metadata = runner._ensure_dp_metadata(None)

    tokens = dp_metadata.num_tokens_across_dp_cpu
    if not isinstance(tokens, list):
        tokens = tokens.tolist()
    assert tokens == [7]


def test_npu_attention_runner_sends_graph_flags():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(role="attention")
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = True
    runner._afd_is_graph_capturing = True
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 3)

    runner._send_dp_metadata(SimpleNamespace(num_tokens_across_dp_cpu=[3]), None)

    assert runner.afd_connector.dp_metadata_updates[0][1:] == (True, True)
    assert runner.afd_connector.sent_dp_metadata_lists[0][1:] == (True, True)


def test_npu_attention_runner_sends_per_ubatch_dp_metadata():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(
        role="attention",
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False

    ubatch_slices = [
        SimpleNamespace(
            request_slice=slice(0, 2),
            token_slice=slice(0, 4),
            num_tokens=4,
        ),
        SimpleNamespace(
            request_slice=slice(2, 3),
            token_slice=slice(4, 7),
            num_tokens=3,
        ),
    ]

    runner._send_dp_metadata(None, ubatch_slices)

    dp_metadata_list = runner.afd_connector.dp_metadata_updates[0][0]
    assert sorted(dp_metadata_list) == [0, 1]
    assert _tokens(dp_metadata_list[0]) == [4]
    assert _tokens(dp_metadata_list[1]) == [3]
    assert runner.afd_connector.sent_dp_metadata_lists[0][0] == dp_metadata_list


def test_npu_attention_capture_microbatch_also_captures_single_stage():
    _require_npu_runtime()
    from vllm.config import CUDAGraphMode

    runner = _new_attention_runner()
    runner.compilation_config = SimpleNamespace(cudagraph_num_of_warmups=1)
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_suppress_metadata_send = False
    runner._afd_pending_metadata = "original"
    dummy_calls = []
    sent_metadata = []

    def dummy_run(num_tokens, **kwargs):
        dummy_calls.append(
            (
                num_tokens,
                kwargs.copy(),
                runner._is_warmup,
                runner._afd_is_graph_capturing,
                runner._afd_suppress_metadata_send,
            ),
        )
        return kwargs["allow_microbatching"]

    runner._dummy_run = dummy_run
    runner._build_afd_metadata = lambda ubatch_slices, num_tokens: SimpleNamespace(
        ubatch_slices=ubatch_slices,
        num_tokens=num_tokens,
    )
    runner._build_capture_dp_metadata = lambda num_tokens: SimpleNamespace(
        num_tokens_across_dp_cpu=[num_tokens],
    )

    def send_dp_metadata(dp_metadata, ubatch_slices):
        sent_metadata.append(
            (
                dp_metadata,
                ubatch_slices,
                runner._afd_is_graph_capturing,
                runner._is_warmup,
            ),
        )

    runner._send_dp_metadata = send_dp_metadata
    desc = SimpleNamespace(num_tokens=12, uniform=True, num_active_loras=0)

    result = runner._warmup_and_capture(
        desc,
        CUDAGraphMode.FULL,
        allow_microbatching=True,
    )

    assert result is True
    assert [call[1]["allow_microbatching"] for call in dummy_calls] == [
        False,
        False,
        True,
        True,
    ]
    assert [call[1]["cudagraph_runtime_mode"] for call in dummy_calls] == [
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL,
    ]
    assert [call[1].get("is_graph_capturing", False) for call in dummy_calls] == [
        False,
        True,
        False,
        True,
    ]
    assert [call[2] for call in dummy_calls] == [True, False, True, False]
    assert [call[3] for call in dummy_calls] == [False, True, False, True]
    assert len(sent_metadata) == 1
    dp_metadata, ubatch_slices, is_graph_capturing, is_warmup = sent_metadata[0]
    assert _tokens(dp_metadata) == [12]
    assert ubatch_slices is None
    assert is_graph_capturing is True
    assert is_warmup is False
    assert runner._is_warmup is False
    assert runner._afd_is_graph_capturing is False
    assert runner._afd_suppress_metadata_send is False
    assert runner._afd_pending_metadata == "original"


def test_npu_attention_metadata_positional_args_and_padded_slices():
    _require_npu_runtime()
    from afd_plugin.v1.worker.ascend.ubatch_utils import (
        UBatchSlice,
        pad_out_ubatch_slices,
    )

    ubatch_slices = [
        UBatchSlice(slice(0, 1), slice(0, 4)),
        UBatchSlice(slice(1, 2), slice(4, 8)),
    ]

    normalized = pad_out_ubatch_slices(ubatch_slices, 8, 4)

    assert normalized[-1].request_slice == slice(1, 4)
    assert normalized[-1].token_slice == slice(4, 8)


def test_npu_create_ascend_forward_context_marks_current_ubatch(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.ascend import forward_context as forward_context_module

    monkeypatch.setattr(
        forward_context_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        forward_context_module,
        "get_dp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(
        forward_context_module,
        "get_moe_comm_method",
        lambda moe_comm_type: f"method:{moe_comm_type}",
    )
    afd_metadata = AFDMetadata(
        afd_tokens_start_loc=[0, 4],
        afd_reqs_start_loc=[0, 1],
        afd_stage_idx=0,
        afd_connector=object(),
        afd_tokens_lens=[4, 3],
        num_of_stages=2,
        afd_tokens_unpadded_lens=[4, 3],
    )
    cur_forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": afd_metadata},
        all_moe_layers={},
        moe_comm_type="mc2",
        in_profile_run=False,
        capturing=False,
        mmrs_fusion=False,
        flash_comm_v1_enabled=False,
        flashcomm_v2_enabled=False,
        is_first_layer=True,
        layer_idx=0,
        prefetch_mlp_gate_up_proj=False,
        prefetch_mlp_down_proj=False,
        model_instance=None,
        is_draft_model=False,
        is_draft_model_prefill=False,
        draft_attn_metadatas=None,
        max_tokens_across_pcp=None,
        mc2_mask=None,
    )
    ubatch_slices = [
        SimpleNamespace(
            request_slice=slice(0, 1),
            token_slice=slice(0, 4),
            num_tokens=4,
        ),
        SimpleNamespace(
            request_slice=slice(1, 2),
            token_slice=slice(4, 7),
            num_tokens=3,
        ),
    ]
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
    )

    new_forward_context = forward_context_module.create_ascend_forward_context(
        cur_forward_context,
        attn_metadata=None,
        vllm_config=vllm_config,
        ubatch_slices=ubatch_slices,
        ubatch_num=1,
    )

    child_metadata = new_forward_context.additional_kwargs["afd_metadata"]
    assert new_forward_context.ubatch_idx == 1
    assert new_forward_context.num_ubatches == 2
    assert new_forward_context.num_tokens == 3
    assert child_metadata.ubatch_idx == 1
    assert child_metadata.afd_stage_idx == 1
    assert new_forward_context.afd_metadata is child_metadata


def test_npu_ffn_runner_executes_eager_ffn_step():
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.updates == [
        ({0: runner.connector.dp_metadata_list[0]}, {"is_graph_capturing": False}),
    ]
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]
    assert runner.connector.metadata_updates == [
        (metadata, AFDRecvOutput(hidden_states="hidden", metadata=metadata)),
    ]


class _FakeGraph:
    def __init__(self):
        self.replay_count = 0

    def replay(self):
        self.replay_count += 1


def test_npu_ffn_runner_replays_acl_graph_when_key_exists(caplog):
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    dp_metadata = {0: _FakeDPMetadata([1])}
    graph = _FakeGraph()
    runner._acl_graphs = {runner._make_graph_key(dp_metadata): {"graph": graph}}

    with caplog.at_level(
        logging.INFO,
        logger="afd_plugin.v1.worker.ascend.ffn_model_runner",
    ):
        runner.execute_model(dp_metadata_list=dp_metadata)

    assert graph.replay_count == 1
    assert runner.connector.ffn_outputs == []
    assert "AFD NPU FFN ACL graph key hit" in caplog.text


def test_npu_ffn_runner_graph_key_uses_ffn_aggregated_token_counts():
    runner = _new_ffn_runner()
    runner.connector = _FakeFFNConnector(attn_size=8, ffn_size=4)
    runner.max_num_tokens = 24

    assert runner._make_graph_key({0: _FakeDPMetadata([12] * 8)}) == (
        (0, (24, 24, 24, 24)),
    )


def test_npu_ffn_runner_logs_acl_graph_miss_and_falls_back_to_eager(caplog):
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    runner._acl_graphs = {}
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    with caplog.at_level(
        logging.WARNING,
        logger="afd_plugin.v1.worker.ascend.ffn_model_runner",
    ):
        runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert "AFD NPU FFN ACL graph key miss" in caplog.text
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_warmup_uses_eager_forward_without_graph():
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    runner._acl_graphs = {}
    capture_flags = []

    def fail_graph_capture_context():
        raise AssertionError("warmup must not enter graph_capture context")

    runner._graph_capture_context = fail_graph_capture_context
    runner._set_cudagraph_capturing_enabled = capture_flags.append
    runner._npu_free_memory = lambda: 0
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_ffn_step(
        dp_metadata_list={0: _FakeDPMetadata([1])},
        is_warmup=True,
    )

    assert runner._acl_graphs == {}
    assert capture_flags == [True, False]
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_capture_stores_acl_graph_and_skips_duplicate_state_update(
    monkeypatch,
):
    _require_npu_runtime()
    from afd_plugin.v1.worker.ascend import ffn_model_runner

    monkeypatch.setattr(ffn_model_runner, "_full_aclgraph_runtime_mode", lambda: "FULL")
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    runner._acl_graphs = {}
    runner.graph_pool = None
    runner._graph_capture_context = lambda: nullcontext()
    runner._npu_graph_context = lambda graph: nullcontext()
    runner._new_npu_graph = _FakeGraph
    runner._set_cudagraph_capturing_enabled = lambda enabled: None
    runner._npu_free_memory = lambda: 0
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    dp_metadata = {0: _FakeDPMetadata([1])}
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_ffn_step(
        dp_metadata_list=dp_metadata,
        is_graph_capturing=True,
    )

    assert runner._make_graph_key(dp_metadata) in runner._acl_graphs
    assert runner.connector.updates == [
        (dp_metadata, {"is_graph_capturing": True}),
    ]


def test_npu_ffn_runner_requires_compute_hook():
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = SimpleNamespace()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    metadata = AFDConnectorMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    with pytest.raises(AttributeError, match="compute_ffn_output"):
        runner.execute_ffn_step(dp_metadata_list={0: _FakeDPMetadata([1])})


def test_npu_ffn_worker_scheduler_execute_model_fails_fast():
    worker = _new_ffn_worker()

    with pytest.raises(RuntimeError, match="connector-driven"):
        worker.execute_model(scheduler_output=object())


def test_npu_ffn_worker_loop_error_is_propagated(caplog):
    worker = _new_ffn_worker()
    worker._ffn_thread = None
    worker._ffn_shutdown_event = None
    worker._ffn_loop_error = None
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(is_initialized=True),
    )

    expected_error = RuntimeError("boom")

    def fail_loop():
        raise expected_error

    worker._run_ffn_server_loop = fail_loop

    with caplog.at_level(
        logging.ERROR,
        logger="afd_plugin.v1.worker.ascend.ffn_worker",
    ):
        worker.start_ffn_server_loop()
        assert worker._ffn_thread is not None
        worker._ffn_thread.join(timeout=5)

    with pytest.raises(RuntimeError, match="AFD NPU FFN worker loop failed") as exc:
        worker.raise_ffn_loop_error_if_any()

    assert exc.value.__cause__ is expected_error
    assert "AFD NPU FFN worker loop failed" in caplog.text


def test_npu_feature_validation_rejects_unsupported_switches():
    for extra_config, message in [
        ({"compute_gate_on_attention": True}, "compute_gate_on_attention"),
        ({"quant_mode": 1}, "quant_mode=0"),
        ({"is_attn_multistream": True}, "multistream"),
        ({"multistream_info": {"attn_enable": "True"}}, "multistream_info"),
    ]:
        with pytest.raises(RuntimeError, match=message):
            fail_if_unsupported_npu_afd_features(
                _vllm_config(extra_config=extra_config),
            )


def test_npu_feature_validation_allows_two_ubatches_only():
    config = _vllm_config(
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )
    fail_if_unsupported_npu_afd_features(config)
    assert npu_afd_num_ubatches(config) == 2

    with pytest.raises(RuntimeError, match="exactly two ubatches"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                enable_dbo=True,
                use_ubatching=True,
                num_ubatches=4,
                ubatch_size=4,
            ),
        )

    config = _vllm_config()
    config.model_config.enforce_eager = False
    fail_if_unsupported_npu_afd_features(config)


def test_npu_ubatch_allows_mc2_comm_when_thresholds_are_met(monkeypatch):
    fake_numpy = ModuleType("numpy")
    fake_numpy.ndarray = object
    fake_torch = ModuleType("torch")
    fake_torch.Tensor = object
    fake_vllm = ModuleType("vllm")
    fake_vllm_config = ModuleType("vllm.config")
    fake_vllm_config.VllmConfig = object
    fake_vllm_v1 = ModuleType("vllm.v1")
    fake_vllm_worker = ModuleType("vllm.v1.worker")
    fake_vllm_ubatch_utils = ModuleType("vllm.v1.worker.ubatch_utils")
    fake_vllm_ubatch_utils.UBatchSlice = object
    fake_vllm_ubatch_utils.UBatchSlices = list

    def check_ubatch_thresholds(config, num_tokens, uniform_decode):
        if not config.use_ubatching:
            return False
        if uniform_decode:
            return num_tokens >= config.dbo_decode_token_threshold
        return num_tokens >= config.dbo_prefill_token_threshold

    fake_vllm_ubatch_utils.check_ubatch_thresholds = check_ubatch_thresholds

    fake_vllm_ascend = ModuleType("vllm_ascend")
    fake_forward_context = ModuleType("vllm_ascend.ascend_forward_context")

    class MoECommType:
        MC2 = object()
        FUSED_MC2 = object()

    fake_forward_context.MoECommType = MoECommType
    fake_attention = ModuleType("vllm_ascend.attention")
    fake_attention_utils = ModuleType("vllm_ascend.attention.utils")
    fake_attention_utils.AscendCommonAttentionMetadata = object

    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", fake_vllm_config)
    monkeypatch.setitem(sys.modules, "vllm.v1", fake_vllm_v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", fake_vllm_worker)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.ubatch_utils",
        fake_vllm_ubatch_utils,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_vllm_ascend)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        fake_forward_context,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", fake_attention)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.utils",
        fake_attention_utils,
    )

    module_name = "afd_plugin.v1.worker.ascend.ubatch_utils"
    original_module = sys.modules.pop(module_name, None)
    try:
        ubatch_utils = importlib.import_module(module_name)
        config = _vllm_config(
            enable_dbo=True,
            use_ubatching=True,
            num_ubatches=2,
            ubatch_size=4,
            dbo_decode_token_threshold=2,
            dbo_prefill_token_threshold=12,
        )

        assert ubatch_utils.check_enable_ubatch(
            num_tokens_unpadded=12,
            num_tokens_padded=12,
            uniform_decode=True,
            vllm_config=config,
            moe_comm_type=ubatch_utils.MoECommType.MC2,
        )
        assert ubatch_utils.check_enable_ubatch(
            num_tokens_unpadded=12,
            num_tokens_padded=12,
            uniform_decode=True,
            vllm_config=config,
            moe_comm_type=ubatch_utils.MoECommType.FUSED_MC2,
        )
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module


def test_npu_afd_config_patch_restores_dbo_for_afd(monkeypatch):
    fake_package = ModuleType("vllm_ascend")
    fake_package.__path__ = []
    fake_platform = ModuleType("vllm_ascend.platform")

    class FakeParallelConfig:
        def __init__(self, *, enable_dbo, ubatch_size):
            self.enable_dbo = enable_dbo
            self.ubatch_size = ubatch_size

        @property
        def use_ubatching(self):
            return self.enable_dbo or self.ubatch_size > 1

        @property
        def num_ubatches(self):
            return 2 if self.enable_dbo else self.ubatch_size

    class NPUPlatform:
        @staticmethod
        def _fix_incompatible_config(vllm_config):
            parallel_config = vllm_config.parallel_config
            parallel_config.enable_dbo = False
            parallel_config.ubatch_size = 0
            return "fixed"

    fake_platform.NPUPlatform = NPUPlatform
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", fake_platform)
    monkeypatch.setattr(ascend_runtime, "_PATCHES_APPLIED", False)

    ascend_runtime.apply_afd_ascend_patches_if_needed()

    config = _vllm_config()
    config.parallel_config = FakeParallelConfig(enable_dbo=True, ubatch_size=4)
    assert NPUPlatform._fix_incompatible_config(config) == "fixed"
    assert config.parallel_config.enable_dbo is True
    assert config.parallel_config.use_ubatching is True
    assert config.parallel_config.num_ubatches == 2
    assert config.parallel_config.ubatch_size == 4

    disabled_config = _vllm_config()
    disabled_config.parallel_config = FakeParallelConfig(
        enable_dbo=True,
        ubatch_size=4,
    )
    disabled_config.additional_config["afd"]["enabled"] = False
    assert NPUPlatform._fix_incompatible_config(disabled_config) == "fixed"
    assert disabled_config.parallel_config.enable_dbo is False
    assert disabled_config.parallel_config.use_ubatching is False


def _tokens(dp_metadata):
    values = dp_metadata.num_tokens_across_dp_cpu
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)
