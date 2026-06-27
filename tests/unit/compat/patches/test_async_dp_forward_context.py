from __future__ import annotations

import importlib
import logging
import sys
import time
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def _config(*, connector: str = "afdasyncconnector"):
    return SimpleNamespace(
        additional_config={
            "afd": {
                "enabled": True,
                "connector": connector,
                "role": "attention",
            },
        },
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False,
            static_all_moe_layers=None,
            static_forward_context={},
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            is_moe_model=True,
        ),
    )


def _install_fake_vllm_forward_context(monkeypatch: pytest.MonkeyPatch):
    vllm_module = types.ModuleType("vllm")
    config_module = types.ModuleType("vllm.config")
    forward_module = types.ModuleType("vllm.forward_context")
    vllm_v1_module = types.ModuleType("vllm.v1")
    worker_module = types.ModuleType("vllm.v1.worker")
    dp_utils_module = types.ModuleType("vllm.v1.worker.dp_utils")

    class CUDAGraphMode:
        NONE = "NONE"

    config_module.CUDAGraphMode = CUDAGraphMode

    class DPMetadata:
        @staticmethod
        def make(parallel_config, num_tokens, num_tokens_across_dp):
            return SimpleNamespace(
                parallel_config=parallel_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
            )

    class BatchDescriptor:
        def __init__(self, num_tokens):
            self.num_tokens = num_tokens

    @contextmanager
    def original_set_forward_context(*args, **kwargs):
        del args, kwargs
        forward_module.original_set_forward_context_called = True
        yield

    def original_coordinate_batch_across_dp(*args, **kwargs):
        del args, kwargs
        forward_module.original_coordinate_called = True
        return True, "tokens", CUDAGraphMode.NONE

    @contextmanager
    def override_forward_context(forward_context):
        previous = forward_module.current_forward_context
        forward_module.current_forward_context = forward_context
        try:
            yield
        finally:
            forward_module.current_forward_context = previous

    def create_forward_context(
        attn_metadata,
        vllm_config,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
        slot_mapping,
        additional_kwargs,
        skip_compiled,
    ):
        return SimpleNamespace(
            attn_metadata=attn_metadata,
            vllm_config=vllm_config,
            dp_metadata=dp_metadata,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            ubatch_slices=ubatch_slices,
            slot_mapping=slot_mapping,
            additional_kwargs=additional_kwargs,
            skip_compiled=skip_compiled,
        )

    forward_module.DPMetadata = DPMetadata
    forward_module.BatchDescriptor = BatchDescriptor
    forward_module.set_forward_context = original_set_forward_context
    forward_module.coordinate_batch_across_dp = original_coordinate_batch_across_dp
    forward_module.override_forward_context = override_forward_context
    forward_module.create_forward_context = create_forward_context
    forward_module.current_forward_context = None
    forward_module.original_set_forward_context_called = False
    forward_module.original_coordinate_called = False
    forward_module.track_batchsize = False
    forward_module.forward_start_time = 0.0
    forward_module.last_logging_time = 0.0
    forward_module.batchsize_logging_interval = 1000.0
    forward_module.batchsize_forward_time = {}
    forward_module.time = time
    forward_module.current_platform = SimpleNamespace(
        set_additional_forward_context=lambda **kwargs: {"platform": kwargs},
        synchronize=None,
    )
    forward_module.logger = logging.getLogger("fake-forward-context")

    dp_utils_module.coordinate_batch_across_dp = original_coordinate_batch_across_dp

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", worker_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.dp_utils", dp_utils_module)
    return forward_module, dp_utils_module, CUDAGraphMode


def _load_patch_module(monkeypatch: pytest.MonkeyPatch):
    _install_fake_vllm_forward_context(monkeypatch)
    module_name = "afd_plugin.compat.patches.async_dp_forward_context"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def test_async_dp_forward_context_skips_dp_metadata(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    forward_module = sys.modules["vllm.forward_context"]
    patch_module.apply_async_dp_forward_context_patch()
    patch_module.apply_async_dp_forward_context_patch()

    with forward_module.set_forward_context(
        attn_metadata=object(),
        vllm_config=_config(),
        num_tokens=4,
    ):
        context = forward_module.current_forward_context
        assert context.dp_metadata is None
        assert context.additional_kwargs["platform"]["dp_metadata"] is None

    assert forward_module.original_set_forward_context_called is False


def test_async_dp_forward_context_preserves_non_async_path(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    forward_module = sys.modules["vllm.forward_context"]
    patch_module.apply_async_dp_forward_context_patch()

    with forward_module.set_forward_context(
        attn_metadata=object(),
        vllm_config=_config(connector="camp2pconnector"),
        num_tokens=4,
    ):
        pass

    assert forward_module.original_set_forward_context_called is True


def test_async_dp_coordinate_batch_across_dp_early_returns(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    dp_utils_module = sys.modules["vllm.v1.worker.dp_utils"]
    patch_module.apply_async_dp_forward_context_patch()
    config = _config()
    config.parallel_config.async_dp = True

    result = dp_utils_module.coordinate_batch_across_dp(
        num_tokens_unpadded=8,
        parallel_config=config.parallel_config,
    )

    assert result == (False, None, "NONE")


def test_async_dp_coordinate_batch_across_dp_preserves_non_async(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    dp_utils_module = sys.modules["vllm.v1.worker.dp_utils"]
    forward_module = sys.modules["vllm.forward_context"]
    patch_module.apply_async_dp_forward_context_patch()

    result = dp_utils_module.coordinate_batch_across_dp(
        num_tokens_unpadded=8,
        parallel_config=SimpleNamespace(),
    )

    assert result == (True, "tokens", "NONE")
    assert forward_module.original_coordinate_called is True
