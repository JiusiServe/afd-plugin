from __future__ import annotations

import importlib
import logging
import sys
import types
from enum import IntEnum
from types import SimpleNamespace

import pytest


class _EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1


def _install_fake_vllm_core(monkeypatch: pytest.MonkeyPatch):
    vllm_module = types.ModuleType("vllm")
    vllm_v1_module = types.ModuleType("vllm.v1")
    vllm_engine_module = types.ModuleType("vllm.v1.engine")
    core_module = types.ModuleType("vllm.v1.engine.core")
    plugins_module = types.ModuleType("vllm.plugins")

    def load_general_plugins():
        return None

    plugins_module.load_general_plugins = load_general_plugins

    class EngineCore:
        def __init__(
            self,
            vllm_config,
            executor_class,
            log_stats,
            executor_fail_callback=None,
            include_finished_set=False,
        ):
            self.vllm_config = vllm_config
            self.original_init_called = True

        def shutdown(self):
            self.original_shutdown_called = True

        def _initialize_kv_caches(self, vllm_config):
            del vllm_config
            self.original_initialize_kv_caches_called = True

        def step(self):
            return None

        def step_with_batch_queue(self):
            return None

    class EngineCoreProc(EngineCore):
        def run_busy_loop(self):
            self.original_run_busy_loop_called = True

    class DPEngineCoreProc(EngineCoreProc):
        pass

    class _StructuredOutputManager:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config

        def clear_backend(self):
            self.backend_cleared = True

    class _MMRegistry:
        def engine_receiver_cache_from_config(self, vllm_config):
            return ("mm-cache", vllm_config)

    def get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory):
        del vllm_config, available_gpu_memory
        return kv_cache_specs

    def generate_scheduler_kv_cache_config(kv_cache_configs):
        del kv_cache_configs
        return SimpleNamespace(num_blocks=0, kv_cache_groups=[])

    def get_hash_fn_by_name(name):
        return name

    def init_none_hash(_hash_fn):
        return None

    def get_request_block_hasher(block_size, hash_fn):
        return block_size, hash_fn

    core_module.EngineCore = EngineCore
    core_module.EngineCoreProc = EngineCoreProc
    core_module.DPEngineCoreProc = DPEngineCoreProc
    core_module.EngineShutdownState = _EngineShutdownState
    core_module.VLLM_VERSION = "0.19.1"
    core_module.logger = logging.getLogger("fake-vllm-core")
    core_module.logger.info_once = lambda *args, **kwargs: None
    core_module.envs = SimpleNamespace(VLLM_ELASTIC_EP_SCALE_UP_LAUNCH=False)
    core_module.StructuredOutputManager = _StructuredOutputManager
    core_module.MULTIMODAL_REGISTRY = _MMRegistry()
    core_module.get_kv_cache_configs = get_kv_cache_configs
    core_module.generate_scheduler_kv_cache_config = generate_scheduler_kv_cache_config
    core_module.get_hash_fn_by_name = get_hash_fn_by_name
    core_module.init_none_hash = init_none_hash
    core_module.get_request_block_hasher = get_request_block_hasher
    core_module.freeze_gc_heap = lambda: None
    core_module.maybe_attach_gc_debug_callback = lambda: None
    core_module.enable_envs_cache = lambda: None
    core_module.EngineCoreOutputs = lambda **kwargs: SimpleNamespace(**kwargs)

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", vllm_engine_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core_module)
    monkeypatch.setitem(sys.modules, "vllm.plugins", plugins_module)
    return core_module


def _load_patch_module() -> types.ModuleType:
    module_name = "afd_plugin.compat.patches.engine_core"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _config(role: str):
    class Scheduler:
        connector = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_kv_connector(self):
            return None

        def shutdown(self):
            self.shutdown_called = True

    scheduler_config = SimpleNamespace(
        async_scheduling=False,
        enable_chunked_prefill=False,
        get_scheduler_cls=lambda: Scheduler,
    )
    cache_config = SimpleNamespace(
        block_size=16,
        enable_prefix_caching=False,
        prefix_caching_hash_algo="builtin",
        num_gpu_blocks=None,
    )
    parallel_config = SimpleNamespace(
        data_parallel_rank_local=0,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    model_config = SimpleNamespace(max_model_len=8, runner_type="generate")

    def validate_block_size():
        cache_config.validated = True

    return SimpleNamespace(
        additional_config={"afd": {"role": role}},
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        model_config=model_config,
        speculative_config=None,
        ec_transfer_config=None,
        validate_block_size=validate_block_size,
    )


def test_engine_core_patch_skips_kv_scheduler_init_for_ffn(monkeypatch):
    core_module = _install_fake_vllm_core(monkeypatch)
    patch_module = _load_patch_module()
    importlib.reload(patch_module)

    class Executor:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.calls = []

        def register_failure_callback(self, callback):
            self.callback = callback

        def collective_rpc(self, method):
            self.calls.append(method)

        def shutdown(self):
            self.calls.append("shutdown")

    engine = core_module.EngineCore(_config("ffn"), Executor, log_stats=True)

    assert not hasattr(engine, "original_init_called")
    assert engine.afd_config.role == "ffn"
    assert engine.scheduler is None
    assert engine.structured_output_manager is None
    assert isinstance(engine.model_executor, Executor)


def test_engine_core_patch_leaves_non_ffn_path_untouched(monkeypatch):
    core_module = _install_fake_vllm_core(monkeypatch)
    _load_patch_module()

    class Executor:
        max_concurrent_batches = 1

        def __init__(self, vllm_config):
            self.vllm_config = vllm_config

        def get_kv_cache_specs(self):
            return []

        def initialize_from_config(self, kv_cache_configs):
            self.kv_cache_configs = kv_cache_configs

        def shutdown(self):
            self.shutdown_called = True

    engine = core_module.EngineCore(_config("attention"), Executor, log_stats=False)

    assert not hasattr(engine, "original_init_called")
    assert isinstance(engine.model_executor, Executor)
    assert engine.scheduler is not None
    assert engine.available_gpu_memory_for_kv_cache == -1


def test_engine_core_patch_runs_and_stops_ffn_loop(monkeypatch):
    core_module = _install_fake_vllm_core(monkeypatch)
    _load_patch_module()

    class Executor:
        def __init__(self, vllm_config):
            self.calls = []

        def collective_rpc(self, method):
            self.calls.append(method)

        def shutdown(self):
            self.calls.append("shutdown")

    engine = core_module.EngineCoreProc(_config("ffn"), Executor, log_stats=True)
    engine.shutdown_state = _EngineShutdownState.RUNNING

    from afd_plugin.compat.patches import engine_core as engine_core_patch

    def request_shutdown(_seconds):
        engine.shutdown_state = _EngineShutdownState.REQUESTED

    monkeypatch.setattr(engine_core_patch.time, "sleep", request_shutdown)

    with pytest.raises(SystemExit):
        engine.run_busy_loop()

    assert engine.model_executor.calls == [
        "start_ffn_server_loop",
        "raise_ffn_loop_error_if_any",
        "stop_ffn_server_loop",
    ]
