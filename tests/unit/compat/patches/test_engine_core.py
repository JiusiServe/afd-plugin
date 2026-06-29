from __future__ import annotations

import logging
import sys
import types
from enum import IntEnum
from types import SimpleNamespace

import pytest

from afd_plugin.compat.patches.engine_core import apply_engine_core_patch


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
            del executor_class, log_stats, executor_fail_callback, include_finished_set
            self.vllm_config = vllm_config
            self.original_init_called = True

        def shutdown(self):
            self.original_shutdown_called = True

    class EngineCoreProc(EngineCore):
        def run_busy_loop(self):
            self.original_run_busy_loop_called = True

    class DPEngineCoreProc(EngineCoreProc):
        pass

    core_module.EngineCore = EngineCore
    core_module.EngineCoreProc = EngineCoreProc
    core_module.DPEngineCoreProc = DPEngineCoreProc
    core_module.EngineShutdownState = _EngineShutdownState
    core_module.VLLM_VERSION = "0.23.0"
    core_module.logger = logging.getLogger("fake-vllm-core")

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", vllm_engine_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core_module)
    monkeypatch.setitem(sys.modules, "vllm.plugins", plugins_module)
    return core_module


def _config(role: str):
    return SimpleNamespace(
        additional_config={"afd": {"enabled": True, "role": role}},
        parallel_config=SimpleNamespace(data_parallel_rank_local=0),
    )


def test_engine_core_patch_skips_kv_scheduler_init_for_ffn(monkeypatch):
    core_module = _install_fake_vllm_core(monkeypatch)
    apply_engine_core_patch()
    apply_engine_core_patch()

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
    apply_engine_core_patch()

    engine = core_module.EngineCore(_config("attention"), object, log_stats=False)

    assert engine.original_init_called is True


def test_engine_core_patch_runs_and_stops_ffn_loop(monkeypatch):
    core_module = _install_fake_vllm_core(monkeypatch)
    apply_engine_core_patch()

    class Executor:
        def __init__(self, vllm_config):
            del vllm_config
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
