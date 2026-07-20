from __future__ import annotations

import logging
import threading
from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDTransferMetadata,
    Trigger,
)
from afd_plugin.v1.worker.cuda_graph import make_ffn_graph_key
from afd_plugin.v1.worker.ffn_model_runner import (
    GPUFFNModelRunner,
    _set_moe_layer_index,
)
from afd_plugin.v1.worker.ffn_worker import AFDFFNWorker


class _FakeConnector:
    def __init__(self):
        self.attn_outputs = deque()
        self.ffn_outputs = []
        self.dp_metadata_updates = []
        self.closed = False
        # The runners reach the control plane through connector.control_plane;
        # the fake serves as both.
        self.control_plane = self

    def update_state_from_dp_metadata(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.dp_metadata_updates.append(
            (
                dict(payload.dp_metadata_list),
                payload.is_graph_capturing,
                payload.is_warmup,
            ),
        )

    def recv_attn_output(self, ubatch_idx=None):
        if ubatch_idx is None:
            return self.attn_outputs.popleft()
        for item in tuple(self.attn_outputs):
            if item.metadata.stage_idx == ubatch_idx:
                self.attn_outputs.remove(item)
                return item
        raise IndexError(ubatch_idx)

    def send_ffn_output(self, ffn_output, metadata):
        self.ffn_outputs.append((ffn_output, metadata))

    def close(self):
        self.closed = True


class _ConnectorDrivenFakeConnector(_FakeConnector):
    ffn_step_trigger = Trigger.CONNECTOR

    def __init__(self):
        super().__init__()
        self.control_plane = None


class _FakeModel:
    def compute_ffn_output(self, hidden_states, layer_idx):
        return f"ffn({hidden_states}, layer={layer_idx})"


class _StepProfiler:
    def __init__(self):
        self.steps = 0
        self.stopped = False

    def step(self):
        self.steps += 1

    def stop(self):
        self.stopped = True


def _metadata():
    return AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )


def _metadata_for_stage(stage_idx):
    return AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=stage_idx,
        seq_len=1,
    )


def _payload(hidden_states, metadata):
    return AFDA2FTransferPayload(hidden_states=hidden_states, metadata=metadata)


def _runner_with_connector_and_model(model, *, num_layers=1):
    runner = object.__new__(GPUFFNModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            is_moe_model=True,
        ),
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False,
            static_forward_context={},
        ),
    )
    runner.connector = _FakeConnector()
    runner.model = model
    runner.num_layers = num_layers
    runner.use_cuda_graph = False
    runner._cuda_graphs = {}
    runner.prof = None
    return runner


class _FakeDPMetadata:
    def __init__(self, values):
        self.num_tokens_across_dp_cpu = values


def _tokens(dp_metadata):
    values = dp_metadata.num_tokens_across_dp_cpu
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


class _FakeGraph:
    def __init__(self):
        self.replay_count = 0

    def replay(self):
        self.replay_count += 1


def test_ffn_runner_executes_model_compute_ffn_output():
    runner = _runner_with_connector_and_model(_FakeModel())
    metadata = _metadata()
    runner.connector.attn_outputs.append(_payload("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert len(runner.connector.dp_metadata_updates) == 1
    dp_metadata_update, is_graph_capturing, is_warmup = (
        runner.connector.dp_metadata_updates[0]
    )
    assert sorted(dp_metadata_update) == [0]
    assert _tokens(dp_metadata_update[0]) == [1]
    assert is_graph_capturing is False
    assert is_warmup is False
    assert runner.connector.ffn_outputs == [
        ("ffn(hidden, layer=0)", metadata),
    ]
    assert metadata.layer_idx == 0


def test_ffn_runner_passthrough_without_model_compute_hook():
    runner = _runner_with_connector_and_model(SimpleNamespace())
    metadata = _metadata()
    runner.connector.attn_outputs.append(_payload("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.ffn_outputs == [("hidden", metadata)]


def test_ffn_runner_processes_each_ubatch_for_each_layer():
    runner = _runner_with_connector_and_model(_FakeModel(), num_layers=2)
    metadata_0_layer_0 = _metadata_for_stage(0)
    metadata_1_layer_0 = _metadata_for_stage(1)
    metadata_0_layer_1 = _metadata_for_stage(0)
    metadata_1_layer_1 = _metadata_for_stage(1)
    runner.connector.attn_outputs.extend(
        [
            _payload("hidden-1-l0", metadata_1_layer_0),
            _payload("hidden-0-l0", metadata_0_layer_0),
            _payload("hidden-1-l1", metadata_1_layer_1),
            _payload("hidden-0-l1", metadata_0_layer_1),
        ],
    )

    runner.execute_model(
        dp_metadata_list={
            0: _FakeDPMetadata([1]),
            1: _FakeDPMetadata([1]),
        },
    )

    assert runner.connector.ffn_outputs == [
        ("ffn(hidden-0-l0, layer=0)", metadata_0_layer_0),
        ("ffn(hidden-1-l0, layer=0)", metadata_1_layer_0),
        ("ffn(hidden-0-l1, layer=1)", metadata_0_layer_1),
        ("ffn(hidden-1-l1, layer=1)", metadata_1_layer_1),
    ]


def test_ffn_runner_requires_dp_metadata_list():
    runner = object.__new__(GPUFFNModelRunner)
    runner.prof = None

    with pytest.raises(RuntimeError, match="requires dp_metadata_list"):
        runner.execute_model()


def test_ffn_runner_makes_original_style_graph_key():
    key = make_ffn_graph_key(
        {
            1: _FakeDPMetadata([5, 7]),
            0: _FakeDPMetadata([2, 3]),
        },
    )

    assert key == ((0, (2, 3)), (1, (5, 7)))


def test_ffn_runner_replays_cuda_graph_when_key_exists():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.use_cuda_graph = True
    graph = _FakeGraph()
    dp_metadata = {0: _FakeDPMetadata([1])}
    runner._cuda_graphs = {
        make_ffn_graph_key(dp_metadata): {"graph": graph},
    }

    runner.execute_model(dp_metadata_list=dp_metadata)

    assert graph.replay_count == 1
    assert runner.connector.ffn_outputs == []


def test_ffn_runner_cuda_graph_miss_falls_back_to_eager():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.use_cuda_graph = True
    metadata = _metadata()
    runner.connector.attn_outputs.append(_payload("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.ffn_outputs == [
        ("ffn(hidden, layer=0)", metadata),
    ]


def test_ffn_runner_steps_gpu_profiler():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.prof = _StepProfiler()
    runner.connector.attn_outputs.append(_payload("hidden", _metadata()))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.prof.steps == 1


def test_ffn_runner_stops_gpu_profiler_on_shutdown():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.prof = _StepProfiler()

    runner.shutdown()

    assert runner.prof.stopped is True
    assert runner.connector.closed is True


def test_ffn_forward_can_skip_connector_state_update_for_capture():
    runner = _runner_with_connector_and_model(_FakeModel())
    metadata = _metadata()
    runner.connector.attn_outputs.append(_payload("hidden", metadata))

    runner._ffn_forward(
        dp_metadata_list={0: _FakeDPMetadata([1])},
        is_graph_capturing=True,
        update_connector_state=False,
    )

    assert runner.connector.dp_metadata_updates == []
    assert runner.connector.ffn_outputs == [
        ("ffn(hidden, layer=0)", metadata),
    ]


def test_ffn_runner_connector_driven_step_uses_payload_layer_metadata():
    runner = _runner_with_connector_and_model(_FakeModel(), num_layers=2)
    runner.connector = _ConnectorDrivenFakeConnector()
    metadata_layer_0 = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    metadata_layer_1 = AFDTransferMetadata.create_attention_metadata(
        layer_idx=1,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.extend(
        [
            _payload("hidden-l0", metadata_layer_0),
            _payload("hidden-l1", metadata_layer_1),
        ],
    )

    runner.execute_connector_driven_step()

    assert runner.connector.dp_metadata_updates == []
    assert runner.connector.ffn_outputs == [
        ("ffn(hidden-l0, layer=0)", metadata_layer_0),
        ("ffn(hidden-l1, layer=1)", metadata_layer_1),
    ]


def test_ffn_runner_connector_driven_step_rejects_control_plane_connector():
    runner = _runner_with_connector_and_model(_FakeModel())

    with pytest.raises(RuntimeError, match="connector-driven"):
        runner.execute_connector_driven_step()


def test_set_moe_layer_index_resets_for_current_layer():
    forward_context = SimpleNamespace(
        all_moe_layers=[
            "model.layers.1.mlp.experts",
            "model.layers.2.mlp.experts",
            "model.layers.3.mlp.experts",
        ],
        moe_layer_index=99,
    )

    _set_moe_layer_index(forward_context, 2)

    assert forward_context.moe_layer_index == 1


def test_ffn_worker_scheduler_execute_model_fails_fast():
    worker = object.__new__(AFDFFNWorker)

    with pytest.raises(RuntimeError, match="connector-driven"):
        worker.execute_model(scheduler_output=object())


def test_ffn_worker_uses_connector_driven_loop_for_async_connector():
    worker = object.__new__(AFDFFNWorker)
    event = threading.Event()
    calls = []

    def execute_connector_driven_step():
        calls.append("step")
        event.set()

    worker._ffn_shutdown_event = event
    worker.device = SimpleNamespace(type="cpu")
    worker.model_runner = SimpleNamespace(
        connector=_ConnectorDrivenFakeConnector(),
        execute_connector_driven_step=execute_connector_driven_step,
    )

    worker._run_ffn_server_loop()

    assert calls == ["step"]


def test_ffn_worker_loop_logs_unexpected_thread_errors(caplog):
    worker = object.__new__(AFDFFNWorker)
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

    with caplog.at_level(logging.ERROR, logger="afd_plugin.v1.worker.ffn_worker"):
        worker.start_ffn_server_loop()
        assert worker._ffn_thread is not None
        worker._ffn_thread.join(timeout=5)

    assert worker._ffn_loop_error is expected_error
    assert "AFD FFN worker loop failed" in caplog.text
    with pytest.raises(RuntimeError, match="AFD FFN worker loop failed") as exc:
        worker.raise_ffn_loop_error_if_any()
    assert exc.value.__cause__ is expected_error
