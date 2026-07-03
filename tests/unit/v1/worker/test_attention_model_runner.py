from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig
from afd_plugin.model_executor.models.forward_context import (
    get_afd_metadata_from_forward_context,
)
from afd_plugin.v1.worker.attention_model_runner import (
    AFDAttentionModelRunner,
    _has_enough_tokens_for_ubatches,
    _is_ubatch_child_afd_context,
    _with_dp_derived_afd_rank,
    fail_if_cuda_graph_enabled,
    fail_if_unsupported_ubatching,
)
from afd_plugin.v1.worker.ubatch_wrapper import (
    build_ubatch_additional_kwargs,
    build_ubatch_afd_metadata,
)


class _UbatchSlice:
    def __init__(self, token_start, token_stop, request_start, request_stop):
        self.token_slice = slice(token_start, token_stop)
        self.request_slice = slice(request_start, request_stop)

    @property
    def num_tokens(self):
        return self.token_slice.stop - self.token_slice.start


class _RecordingConnector:
    world_rank = 1

    def __init__(self):
        self.dp_metadata_updates = []
        self.sent_dp_metadata_lists = []
        self.dp_metadata_update_flags = []
        self.sent_dp_metadata_flags = []
        self.closed = False

    def is_attn_top_min_size_rank(self, world_rank):
        return world_rank == self.world_rank

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list,
        *,
        is_graph_capturing=False,
        is_warmup=False,
    ):
        self.dp_metadata_updates.append(dp_metadata_list)
        self.dp_metadata_update_flags.append((is_graph_capturing, is_warmup))

    def send_dp_metadata_list(
        self,
        dp_metadata_list,
        *,
        is_graph_capturing=False,
        is_warmup=False,
    ):
        self.sent_dp_metadata_lists.append(dp_metadata_list)
        self.sent_dp_metadata_flags.append((is_graph_capturing, is_warmup))

    def close(self):
        self.closed = True


class _StepProfiler:
    def __init__(self):
        self.steps = 0
        self.stopped = False

    def step(self):
        self.steps += 1

    def stop(self):
        self.stopped = True


def _install_fake_vllm_forward_context(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_forward_context = ModuleType("vllm.forward_context")

    def create_forward_context():
        return SimpleNamespace(
            additional_kwargs={},
            dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=[1]),
            ubatch_slices=None,
            batch_descriptor=SimpleNamespace(num_tokens=1),
        )

    fake_forward_context.create_forward_context = create_forward_context
    fake_vllm.forward_context = fake_forward_context
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", fake_forward_context)
    return fake_forward_context, create_forward_context


def _parallel_config(**overrides):
    values = {
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
        "use_ubatching": False,
        "num_ubatches": 1,
        "dbo_decode_token_threshold": 32,
        "dbo_prefill_token_threshold": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_attention_runner_builds_single_stage_metadata():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_connector = object()
    runner._afd_transaction_counter = 0

    metadata = runner._build_afd_metadata(None, 7)

    assert metadata.afd_tokens_start_loc == [0]
    assert metadata.afd_reqs_start_loc == [0]
    assert metadata.afd_tokens_lens == [7]
    assert metadata.afd_tokens_unpadded_lens == [7]
    assert metadata.num_of_stages == 1
    assert metadata.afd_connector is runner.afd_connector


def test_attention_runner_installs_afd_metadata_on_forward_context():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 5)
    forward_context = SimpleNamespace(
        additional_kwargs={"platform_key": "platform_value"},
        dp_metadata="dp",
        ubatch_slices=None,
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    assert forward_context.additional_kwargs["platform_key"] == "platform_value"
    assert forward_context.additional_kwargs["afd_metadata"].afd_tokens_lens == [5]
    assert runner.afd_connector.dp_metadata_updates == [{0: "dp"}]
    assert runner.afd_connector.sent_dp_metadata_lists == [{0: "dp"}]
    assert runner.afd_connector.sent_dp_metadata_flags == [(False, False)]


def test_attention_runner_initializes_missing_forward_context_kwargs():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_suppress_metadata_send = True
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 5)
    forward_context = SimpleNamespace(
        additional_kwargs=None,
        dp_metadata="dp",
        ubatch_slices=None,
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    assert forward_context.additional_kwargs["afd_metadata"].afd_tokens_lens == [5]


def test_attention_runner_uses_padded_full_graph_tokens_for_afd_metadata():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 1)
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=[1]),
        ubatch_slices=None,
        batch_descriptor=SimpleNamespace(num_tokens=64),
        cudagraph_runtime_mode=SimpleNamespace(name="FULL"),
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    metadata = forward_context.additional_kwargs["afd_metadata"]
    assert metadata.afd_tokens_lens == [1]
    sent_metadata = runner.afd_connector.sent_dp_metadata_lists[0][0]
    tokens = sent_metadata.num_tokens_across_dp_cpu
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    assert tokens == [64]


def test_attention_runner_sends_per_ubatch_dp_metadata():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = None
    ubatch_slices = [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)]

    runner._send_dp_metadata(None, ubatch_slices)

    assert set(runner.afd_connector.dp_metadata_updates[0]) == {0, 1}
    assert set(runner.afd_connector.sent_dp_metadata_lists[0]) == {0, 1}


def test_attention_runner_skips_dp_metadata_send_for_ubatch_child_context():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = None

    parent = runner._build_afd_metadata(
        [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)],
        8,
    )
    child = build_ubatch_afd_metadata(
        parent,
        [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)],
        1,
    )
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": child},
        dp_metadata="child-dp",
        ubatch_slices=None,
    )

    assert _is_ubatch_child_afd_context(forward_context, child)

    runner._install_afd_metadata_on_forward_context(forward_context)

    assert forward_context.additional_kwargs["afd_metadata"] is child
    assert runner.afd_connector.dp_metadata_updates == []
    assert runner.afd_connector.sent_dp_metadata_lists == []


def test_attention_runner_does_not_skip_single_stage_context():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 5)
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata="dp",
        ubatch_slices=None,
    )

    assert not _is_ubatch_child_afd_context(
        forward_context,
        runner._afd_pending_metadata,
    )

    runner._install_afd_metadata_on_forward_context(forward_context)

    assert runner.afd_connector.sent_dp_metadata_lists == [{0: "dp"}]


def test_ubatch_metadata_clones_parent_and_preserves_additional_kwargs():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_connector = object()
    runner._afd_transaction_counter = 0
    parent = runner._build_afd_metadata(
        [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)],
        8,
    )
    parent.afd_tokens_unpadded_lens = [3, 4]

    first = build_ubatch_afd_metadata(
        parent, [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)], 0
    )
    second = build_ubatch_afd_metadata(
        parent, [_UbatchSlice(0, 3, 0, 1), _UbatchSlice(3, 8, 1, 2)], 1
    )
    child_kwargs = build_ubatch_additional_kwargs(
        {"platform_key": "platform_value", "afd_metadata": parent},
        second,
    )

    assert first is not parent
    assert second is not parent
    assert first is not second
    assert first.ubatch_idx == 0
    assert first.afd_stage_idx == 0
    assert first.afd_tokens_lens == [3]
    assert second.ubatch_idx == 1
    assert second.afd_stage_idx == 1
    assert second.afd_tokens_start_loc == [3]
    assert second.afd_reqs_start_loc == [1]
    assert second.afd_tokens_lens == [5]
    assert second.afd_tokens_unpadded_lens == [4]
    assert child_kwargs["platform_key"] == "platform_value"
    assert child_kwargs["afd_metadata"] is second


def test_phase5_allows_two_way_ubatching_but_rejects_other_counts():
    fail_if_unsupported_ubatching(
        SimpleNamespace(
            parallel_config=_parallel_config(use_ubatching=True, num_ubatches=2),
        ),
    )

    with pytest.raises(RuntimeError, match="exactly two"):
        fail_if_unsupported_ubatching(
            SimpleNamespace(
                parallel_config=_parallel_config(use_ubatching=True, num_ubatches=4),
            ),
        )


def test_attention_runner_enables_ubatching_for_afd_dp1_thresholds():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            use_ubatching=True,
            num_ubatches=2,
            dbo_decode_token_threshold=2,
            dbo_prefill_token_threshold=4,
        ),
    )
    runner.uniform_decode_query_len = 1
    runner._is_uniform_decode = lambda **_kwargs: False

    assert runner._should_ubatch_without_vllm_dp(
        num_tokens=4,
        num_reqs=1,
        num_scheduled_tokens_np=[4],
        max_num_scheduled_tokens=4,
        use_cascade_attn=False,
    )

    assert not runner._should_ubatch_without_vllm_dp(
        num_tokens=3,
        num_reqs=1,
        num_scheduled_tokens_np=[3],
        max_num_scheduled_tokens=3,
        use_cascade_attn=False,
    )


def test_attention_runner_enables_decode_ubatching_for_afd_dp1_thresholds():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            use_ubatching=True,
            num_ubatches=2,
            dbo_decode_token_threshold=2,
            dbo_prefill_token_threshold=4,
        ),
    )
    runner.uniform_decode_query_len = 1
    runner._is_uniform_decode = lambda **_kwargs: True

    assert runner._should_ubatch_without_vllm_dp(
        num_tokens=2,
        num_reqs=2,
        num_scheduled_tokens_np=[1, 1],
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
    )

    assert not runner._should_ubatch_without_vllm_dp(
        num_tokens=1,
        num_reqs=1,
        num_scheduled_tokens_np=[1],
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
    )


def test_attention_runner_rejects_empty_native_ubatches():
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(num_ubatches=2),
    )

    assert not _has_enough_tokens_for_ubatches(vllm_config, 1)
    assert _has_enough_tokens_for_ubatches(vllm_config, 2)


def test_attention_runner_inherits_native_dummy_run_microbatching():
    assert "_dummy_run" in AFDAttentionModelRunner.__dict__


def test_attention_runner_steps_gpu_profiler(monkeypatch):
    runner = object.__new__(AFDAttentionModelRunner)
    runner.prof = _StepProfiler()

    def execute_model(_self, *args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(
        AFDAttentionModelRunner.__mro__[1],
        "execute_model",
        execute_model,
    )

    result = runner.execute_model("scheduler", flag=True)

    assert runner.prof.steps == 1
    assert result == (("scheduler",), {"flag": True})


def test_attention_runner_stops_gpu_profiler_on_shutdown(monkeypatch):
    runner = object.__new__(AFDAttentionModelRunner)
    runner.prof = _StepProfiler()
    runner.afd_connector = _RecordingConnector()
    called = []

    def shutdown(_self):
        called.append(True)

    monkeypatch.setattr(AFDAttentionModelRunner.__mro__[1], "shutdown", shutdown)

    runner.shutdown()

    assert runner.prof.stopped is True
    assert runner.afd_connector.closed is True
    assert called == [True]


def test_forward_context_provider_installs_metadata_before_model_forward(monkeypatch):
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    fake_forward_context, original_create = _install_fake_vllm_forward_context(
        monkeypatch,
    )

    from afd_plugin.model_executor.models.forward_context import (
        use_afd_metadata_provider,
    )

    with use_afd_metadata_provider(runner):
        forward_context = fake_forward_context.create_forward_context()
        metadata = get_afd_metadata_from_forward_context(forward_context)

    assert metadata is not None
    assert metadata.afd_tokens_lens == [1]
    assert forward_context.additional_kwargs["afd_metadata"] is metadata
    assert runner.afd_connector.sent_dp_metadata_lists
    assert fake_forward_context.create_forward_context is original_create


def test_forward_context_provider_can_install_without_sending_metadata(monkeypatch):
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = True
    runner._afd_suppress_metadata_send = True
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner._build_afd_metadata(None, 1)
    fake_forward_context, original_create = _install_fake_vllm_forward_context(
        monkeypatch,
    )

    from afd_plugin.model_executor.models.forward_context import (
        use_afd_metadata_provider,
    )

    with use_afd_metadata_provider(runner):
        forward_context = fake_forward_context.create_forward_context()
        metadata = get_afd_metadata_from_forward_context(forward_context)

    assert metadata is runner._afd_pending_metadata
    assert runner.afd_connector.sent_dp_metadata_lists == []
    assert fake_forward_context.create_forward_context is original_create


def test_attention_runtime_rejects_unsupported_cuda_graph_modes():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(cudagraph_mode="PIECEWISE"),
        parallel_config=_parallel_config(),
    )

    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        fail_if_cuda_graph_enabled(vllm_config)


def test_attention_runtime_allows_full_decode_only_cuda_graph():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY"),
        parallel_config=_parallel_config(),
    )

    fail_if_cuda_graph_enabled(vllm_config)


def test_attention_runner_forwards_capture_and_warmup_flags():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_config = AFDConfig(enabled=True, role="attention")
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(),
    )
    runner.afd_connector = _RecordingConnector()
    runner._is_warmup = True
    runner._afd_is_graph_capturing = True
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = None

    runner._send_dp_metadata("dp", None)

    assert runner.afd_connector.dp_metadata_update_flags == [(True, True)]
    assert runner.afd_connector.sent_dp_metadata_flags == [(True, True)]


def test_attention_runner_builds_capture_dp_metadata_for_native_dp():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(data_parallel_size=2),
    )

    metadata = runner._build_capture_dp_metadata(64)

    tokens = metadata.num_tokens_across_dp_cpu
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    assert tokens == [64, 64]
    assert int(metadata.max_tokens_across_dp_cpu) == 64


def test_afd_rank_derives_from_data_parallel_rank():
    config = AFDConfig(
        enabled=True,
        role="attention",
        connector="p2pconnector",
        num_attention_ranks=2,
        num_ffn_ranks=2,
        afd_role_rank=0,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(data_parallel_size=2, data_parallel_rank=1),
    )

    ranked = _with_dp_derived_afd_rank(vllm_config, config)

    assert ranked.afd_role_rank == 1
    assert config.afd_role_rank == 0


# --- TP rank derivation tests ---


def _parallel_config_with_tp(
    *,
    dp_size=1,
    dp_rank=0,
    tp_size=1,
    **overrides,
):
    values = {
        "data_parallel_size": dp_size,
        "data_parallel_rank": dp_rank,
        "tensor_parallel_size": tp_size,
        "use_ubatching": False,
        "num_ubatches": 1,
        "dbo_decode_token_threshold": 32,
        "dbo_prefill_token_threshold": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_afd_rank_derives_from_tp_rank_dp1_tp2(monkeypatch):
    """DP=1, TP=2: each TP worker gets a unique role_rank."""
    monkeypatch.setattr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_tensor_model_parallel_rank",
        lambda: 1,
    )
    config = AFDConfig(
        enabled=True,
        role="attention",
        connector="p2pconnector",
        num_attention_ranks=4,
        num_ffn_ranks=4,
        afd_role_rank=0,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config_with_tp(dp_size=1, tp_size=2),
    )

    ranked = _with_dp_derived_afd_rank(vllm_config, config)

    # role_rank = base(0) + dp_rank(0) * tp_size(2) + tp_rank(1) = 1
    assert ranked.afd_role_rank == 1


def test_afd_rank_derives_from_dp_and_tp_ranks_dp2_tp2(monkeypatch):
    """DP=2, TP=2: role_rank = dp_rank * tp_size + tp_rank."""
    monkeypatch.setattr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_tensor_model_parallel_rank",
        lambda: 1,
    )
    config = AFDConfig(
        enabled=True,
        role="attention",
        connector="p2pconnector",
        num_attention_ranks=4,
        num_ffn_ranks=4,
        afd_role_rank=0,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config_with_tp(dp_size=2, dp_rank=1, tp_size=2),
    )

    ranked = _with_dp_derived_afd_rank(vllm_config, config)

    # role_rank = base(0) + dp_rank(1) * tp_size(2) + tp_rank(1) = 3
    assert ranked.afd_role_rank == 3


def test_afd_rank_unchanged_when_dp1_tp1():
    """DP=1, TP=1: no derivation needed, config returned as-is."""
    config = AFDConfig(
        enabled=True,
        role="attention",
        connector="p2pconnector",
        num_attention_ranks=1,
        num_ffn_ranks=1,
        afd_role_rank=0,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config_with_tp(dp_size=1, tp_size=1),
    )

    ranked = _with_dp_derived_afd_rank(vllm_config, config)

    assert ranked is config


def test_afd_rank_raises_for_out_of_range_dp2_tp2(monkeypatch):
    """role_rank must stay within role_size."""
    monkeypatch.setattr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    config = AFDConfig(
        enabled=True,
        role="attention",
        connector="p2pconnector",
        num_attention_ranks=2,
        num_ffn_ranks=2,
        afd_role_rank=0,
    )
    vllm_config = SimpleNamespace(
        parallel_config=_parallel_config_with_tp(dp_size=2, dp_rank=1, tp_size=2),
    )

    with pytest.raises(ValueError, match="out of range"):
        _with_dp_derived_afd_rank(vllm_config, config)
