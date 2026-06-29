from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from afd_plugin.compat.patches.config_validation import apply_config_validation_patch


def _install_fake_vllm_config(monkeypatch):
    vllm_module = types.ModuleType("vllm")
    vllm_module.__version__ = "0.23.0"
    config_package = types.ModuleType("vllm.config")
    config_module = types.ModuleType("vllm.config.vllm")
    engine_package = types.ModuleType("vllm.engine")
    arg_utils_module = types.ModuleType("vllm.engine.arg_utils")

    class VllmConfig:
        def __post_init__(self):
            if self.parallel_config.use_ubatching:
                assert self.parallel_config.all2all_backend in {
                    "deepep_low_latency",
                    "deepep_high_throughput",
                }, "native all2all backend assertion"
            self.post_init_backend = self.parallel_config.all2all_backend

    class EngineArgs:
        def create_engine_config(self):
            if self.enable_dbo:
                assert self.all2all_backend in {
                    "deepep_low_latency",
                    "deepep_high_throughput",
                }, "native all2all backend assertion"
            cfg = VllmConfig()
            cfg.additional_config = self.additional_config
            cfg.parallel_config = SimpleNamespace(
                use_ubatching=self.enable_dbo or self.ubatch_size > 1,
                all2all_backend=self.all2all_backend,
            )
            cfg.__post_init__()
            return cfg

    config_module.VllmConfig = VllmConfig
    arg_utils_module.EngineArgs = EngineArgs
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_package)
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", config_module)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_package)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_module)
    return arg_utils_module, config_module


def _engine_args(*, enabled):
    args = sys.modules["vllm.engine.arg_utils"].EngineArgs()
    args.additional_config = {"afd": {"enabled": enabled, "role": "attention"}}
    args.enable_dbo = True
    args.ubatch_size = 1
    args.all2all_backend = "allgather_reducescatter"
    return args


def test_config_validation_patch_relaxes_backend_for_afd_ubatching(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    apply_config_validation_patch()
    apply_config_validation_patch()
    args = _engine_args(enabled=True)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert args.all2all_backend == "allgather_reducescatter"
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"


def test_config_validation_patch_preserves_non_afd_validation(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    apply_config_validation_patch()
    args = _engine_args(enabled=False)

    try:
        arg_utils_module.EngineArgs.create_engine_config(args)
    except AssertionError as exc:
        assert "native all2all" in str(exc)
    else:
        raise AssertionError("expected native all2all backend assertion")


def test_config_validation_patch_allows_vllm_dev_checkout(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    sys.modules["vllm"].__version__ = "0.1.dev14230+g68b0c3135"
    apply_config_validation_patch()
    args = _engine_args(enabled=True)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"


def test_config_validation_patch_relaxes_repeated_vllm_post_init(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    apply_config_validation_patch()
    args = _engine_args(enabled=True)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)
    cfg.additional_config = args.additional_config
    cfg.parallel_config.use_ubatching = True
    cfg.__post_init__()

    assert cfg.post_init_backend == "deepep_low_latency"
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"
