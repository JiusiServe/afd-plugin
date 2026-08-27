# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)
pytest.importorskip("vllm", exc_type=ImportError)

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.v1.worker.gpu.cudagraph_utils import (  # noqa: E402
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers  # noqa: E402

from afd_plugin.compat.backports.vllm_v026_mrv2_dbo import (  # noqa: E402
    AFDBatchExecutionDescriptor,
    create_ubatch_slices,
    dispatch_afd_dbo_and_sync_dp,
    prepare_attn_for_ubatch,
    runtime,  # noqa: E402
    slice_input_batch,
)


def _parallel_config(*, decode_threshold=4, prefill_threshold=8):
    return SimpleNamespace(
        use_ubatching=True,
        num_ubatches=2,
        dbo_decode_token_threshold=decode_threshold,
        dbo_prefill_token_threshold=prefill_threshold,
    )


def test_runtime_abi_matches_full_graph_dispatch_signature():
    parameters = inspect.signature(dispatch_afd_dbo_and_sync_dp).parameters

    assert runtime.AFD_MRV2_DBO_RUNTIME_ABI == 3
    assert {"cudagraph_manager", "need_eager"} <= parameters.keys()


def test_dispatch_selects_two_ubatches_and_uniform_padding(monkeypatch):
    def all_reduce(tensor, group):
        assert group == "cpu-group"
        tensor[0] = torch.tensor([8, 12], dtype=torch.int32)
        tensor[1] = torch.tensor([1, 1], dtype=torch.int32)
        tensor[2] = torch.tensor([1, 1], dtype=torch.int32)

    monkeypatch.setattr(runtime.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(
        runtime,
        "get_dp_group",
        lambda: SimpleNamespace(cpu_group="cpu-group"),
    )

    descriptor, counts = dispatch_afd_dbo_and_sync_dp(
        num_reqs=8,
        num_tokens=8,
        uniform_token_count=1,
        dp_size=2,
        dp_rank=0,
        parallel_config=_parallel_config(),
        decode_query_len=1,
        allow_ubatching=True,
    )

    assert isinstance(descriptor, AFDBatchExecutionDescriptor)
    assert descriptor.cg_mode is CUDAGraphMode.NONE
    assert descriptor.num_tokens == 12
    assert descriptor.num_ubatches == 2
    assert counts.tolist() == [12, 12]


def test_dispatch_uses_single_batch_when_one_rank_is_below_threshold(monkeypatch):
    def all_reduce(tensor, group):
        del group
        tensor[0] = torch.tensor([3, 12], dtype=torch.int32)
        tensor[1] = torch.tensor([1, 1], dtype=torch.int32)
        tensor[2] = torch.tensor([1, 1], dtype=torch.int32)

    monkeypatch.setattr(runtime.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(
        runtime,
        "get_dp_group",
        lambda: SimpleNamespace(cpu_group=None),
    )

    descriptor, counts = dispatch_afd_dbo_and_sync_dp(
        num_reqs=3,
        num_tokens=3,
        uniform_token_count=1,
        dp_size=2,
        dp_rank=0,
        parallel_config=_parallel_config(),
        decode_query_len=1,
        allow_ubatching=True,
    )

    assert not isinstance(descriptor, AFDBatchExecutionDescriptor)
    assert descriptor.num_tokens == 3
    assert counts.tolist() == [3, 12]


def test_slice_input_batch_keeps_all_padding_trailing_stage_well_formed():
    buffers = InputBuffers(2, 8, torch.device("cpu"))
    batch = InputBatch.make_dummy(1, 3, buffers)
    buffers.is_padding[:3].fill_(False)
    buffers.is_padding[3:8].fill_(True)
    batch = replace(
        batch,
        num_tokens_after_padding=8,
        input_ids=buffers.input_ids[:8],
        positions=buffers.positions[:8],
        is_padding=buffers.is_padding[:8],
    )
    stages = create_ubatch_slices(batch, 2)

    trailing = slice_input_batch(
        batch,
        stages[1],
        torch.zeros(3, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )

    assert [stage.num_tokens for stage in stages] == [4, 4]
    assert trailing.num_reqs == 1
    assert trailing.num_tokens == 0
    assert trailing.num_tokens_after_padding == 4
    assert trailing.num_scheduled_tokens.tolist() == [0]
    assert trailing.is_padding.tolist() == [True, True, True, True]


def test_prepare_attn_for_second_ubatch_restores_builder_order():
    first_builder = object()
    second_builder = object()

    class Group:
        metadata_builders = [first_builder, second_builder]

    group = Group()

    class ModelState:
        def prepare_attn(self, *_args, **_kwargs):
            return group.metadata_builders[0]

    selected = prepare_attn_for_ubatch(
        ModelState(),
        input_batch=object(),
        block_tables=(),
        slot_mappings=object(),
        attn_groups=[[group]],
        kv_cache_config=object(),
        ubatch_index=1,
    )

    assert selected is second_builder
    assert group.metadata_builders == [first_builder, second_builder]


def test_dispatch_requests_captured_two_ubatch_descriptor(monkeypatch):
    descriptor = AFDBatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=12,
        num_reqs=8,
        num_ubatches=2,
    )
    dispatch_calls = []

    class Manager:
        def dispatch(self, *args, **kwargs):
            dispatch_calls.append(("dispatch", args, kwargs))
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=args[1],
                num_reqs=args[0],
                uniform_token_count=args[2],
            )

        def dispatch_ubatches(self, base, num_ubatches):
            dispatch_calls.append(("dispatch_ubatches", base, num_ubatches))
            assert base.num_tokens == 12
            assert num_ubatches == 2
            return descriptor

    def all_reduce(tensor, group):
        del group
        tensor[0] = torch.tensor([8, 12], dtype=torch.int32)
        tensor[1] = torch.tensor([1, 1], dtype=torch.int32)
        tensor[2] = torch.tensor([1, 1], dtype=torch.int32)
        tensor[3] = torch.tensor(
            [CUDAGraphMode.FULL.value, CUDAGraphMode.FULL.value],
            dtype=torch.int32,
        )

    monkeypatch.setattr(runtime.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(
        runtime,
        "get_dp_group",
        lambda: SimpleNamespace(cpu_group=None),
    )

    selected, counts = dispatch_afd_dbo_and_sync_dp(
        num_reqs=8,
        num_tokens=8,
        uniform_token_count=1,
        dp_size=2,
        dp_rank=0,
        parallel_config=_parallel_config(),
        decode_query_len=1,
        allow_ubatching=True,
        cudagraph_manager=Manager(),
    )

    assert selected is descriptor
    assert counts.tolist() == [12, 12]
    assert dispatch_calls[:2] == [
        ("dispatch", (8, 8, 1), {"num_active_loras": 0}),
        ("dispatch", (8, 12, 1), {"num_active_loras": 0}),
    ]
    assert dispatch_calls[2][0] == "dispatch_ubatches"


def test_dispatch_accepts_upstream_manager_without_ubatch_keyword(monkeypatch):
    dispatch_calls = []

    class Manager:
        def dispatch(
            self,
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
        ):
            dispatch_calls.append(
                (num_reqs, num_tokens, uniform_token_count, num_active_loras)
            )
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                uniform_token_count=uniform_token_count,
                num_active_loras=num_active_loras,
            )

    def all_reduce(tensor, group):
        del group
        tensor[0] = torch.tensor([8, 12], dtype=torch.int32)
        tensor[1] = torch.tensor([1, 1], dtype=torch.int32)
        tensor[2] = torch.tensor([1, 1], dtype=torch.int32)

    monkeypatch.setattr(runtime.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(
        runtime,
        "get_dp_group",
        lambda: SimpleNamespace(cpu_group=None),
    )

    selected, counts = dispatch_afd_dbo_and_sync_dp(
        num_reqs=8,
        num_tokens=8,
        uniform_token_count=1,
        dp_size=2,
        dp_rank=0,
        parallel_config=_parallel_config(),
        decode_query_len=1,
        allow_ubatching=True,
        cudagraph_manager=Manager(),
    )

    assert isinstance(selected, AFDBatchExecutionDescriptor)
    assert selected.cg_mode is CUDAGraphMode.NONE
    assert selected.num_tokens == 12
    assert selected.num_ubatches == 2
    assert counts.tolist() == [12, 12]
    assert dispatch_calls == [(8, 8, 1, 0)]


def test_dispatch_avoids_empty_second_ubatch_after_graph_padding(monkeypatch):
    dispatch_calls = []

    class Manager:
        def dispatch(
            self,
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
        ):
            dispatch_calls.append((num_reqs, num_tokens, uniform_token_count))
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=8,
                num_reqs=8,
                uniform_token_count=1,
                num_active_loras=num_active_loras,
            )

        def dispatch_ubatches(self, base, num_ubatches):
            return AFDBatchExecutionDescriptor(
                **vars(base),
                num_ubatches=num_ubatches,
            )

    def all_reduce(tensor, group):
        del group
        tensor[0] = torch.tensor([4, 7], dtype=torch.int32)
        tensor[1] = torch.tensor([1, 1], dtype=torch.int32)
        tensor[2] = torch.tensor([1, 1], dtype=torch.int32)
        tensor[3] = torch.tensor(
            [CUDAGraphMode.FULL.value, CUDAGraphMode.FULL.value],
            dtype=torch.int32,
        )

    monkeypatch.setattr(runtime.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(
        runtime,
        "get_dp_group",
        lambda: SimpleNamespace(cpu_group=None),
    )

    selected, counts = dispatch_afd_dbo_and_sync_dp(
        num_reqs=4,
        num_tokens=4,
        uniform_token_count=1,
        dp_size=2,
        dp_rank=0,
        parallel_config=_parallel_config(),
        decode_query_len=1,
        allow_ubatching=True,
        cudagraph_manager=Manager(),
    )

    assert not isinstance(selected, AFDBatchExecutionDescriptor)
    assert selected.cg_mode is CUDAGraphMode.FULL
    assert selected.num_tokens == 8
    assert counts.tolist() == [8, 8]
    assert dispatch_calls == [(4, 4, 1), (4, 7, 1), (4, 7, 1)]


def test_dispatch_uses_eager_dbo_when_any_rank_misses_graph(monkeypatch):
    dispatch_calls = []

    class Manager:
        def dispatch(
            self,
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
        ):
            dispatch_calls.append(
                (num_reqs, num_tokens, uniform_token_count, num_active_loras)
            )
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                uniform_token_count=uniform_token_count,
                num_active_loras=num_active_loras,
            )

        def dispatch_ubatches(self, base, num_ubatches):
            raise AssertionError((base, num_ubatches))

    def all_reduce(tensor, group):
        del group
        tensor[0] = torch.tensor([8, 12], dtype=torch.int32)
        tensor[1] = torch.tensor([1, 1], dtype=torch.int32)
        tensor[2] = torch.tensor([1, 1], dtype=torch.int32)
        tensor[3] = torch.tensor(
            [CUDAGraphMode.FULL.value, CUDAGraphMode.NONE.value],
            dtype=torch.int32,
        )

    monkeypatch.setattr(runtime.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(
        runtime,
        "get_dp_group",
        lambda: SimpleNamespace(cpu_group=None),
    )

    selected, counts = dispatch_afd_dbo_and_sync_dp(
        num_reqs=8,
        num_tokens=8,
        uniform_token_count=1,
        dp_size=2,
        dp_rank=0,
        parallel_config=_parallel_config(),
        decode_query_len=1,
        allow_ubatching=True,
        cudagraph_manager=Manager(),
    )

    assert isinstance(selected, AFDBatchExecutionDescriptor)
    assert selected.cg_mode is CUDAGraphMode.NONE
    assert selected.num_tokens == 12
    assert selected.num_ubatches == 2
    assert counts.tolist() == [12, 12]
    assert dispatch_calls == [(8, 8, 1, 0)]


def test_merge_ubatch_outputs_preserves_ascend_auxiliary_structure():
    outputs = [
        (torch.tensor([[1]]), [torch.tensor([[2]]), torch.tensor([[3]])]),
        (torch.tensor([[4]]), [torch.tensor([[5]]), torch.tensor([[6]])]),
    ]

    hidden_states, auxiliary = runtime.merge_ubatch_outputs(outputs)

    assert hidden_states.tolist() == [[1], [4]]
    assert auxiliary[0].tolist() == [[2], [5]]
    assert auxiliary[1].tolist() == [[3], [6]]
