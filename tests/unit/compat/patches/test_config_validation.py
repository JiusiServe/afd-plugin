from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from afd_plugin.validation import (
    ATTENTION_WORKER_FQCN,
    FFN_WORKER_FQCN,
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_WORKER_FQCN,
    VLLM_ASCEND_310P_WORKER_FQCN,
    VLLM_ASCEND_NPU_WORKER_FQCN,
    VLLM_ASCEND_XLITE_WORKER_FQCN,
    VLLM_GPU_WORKER_FQCN,
)


def _install_fake_vllm_config(monkeypatch):
    vllm_module = types.ModuleType("vllm")
    vllm_module.__version__ = "0.26.0"
    config_package = types.ModuleType("vllm.config")
    config_module = types.ModuleType("vllm.config.vllm")
    engine_package = types.ModuleType("vllm.engine")
    arg_utils_module = types.ModuleType("vllm.engine.arg_utils")
    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(
        is_cuda=lambda: True,
        device_type="cuda",
    )

    class VllmConfig:
        platform_worker_cls = VLLM_GPU_WORKER_FQCN

        def __post_init__(self):
            if self.parallel_config.use_ubatching:
                assert self.parallel_config.all2all_backend in {
                    "deepep_low_latency",
                    "deepep_high_throughput",
                }, "native all2all backend assertion"
            if self.parallel_config.worker_cls == "auto":
                self.parallel_config.worker_cls = self.platform_worker_cls
            self.post_init_backend = self.parallel_config.all2all_backend

    class EngineArgs:
        def create_engine_config(self, usage_context=None, headless=False):
            del usage_context, headless
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
                worker_cls=self.worker_cls,
            )
            cfg.__post_init__()
            return cfg

    config_module.VllmConfig = VllmConfig
    config_module.logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
    arg_utils_module.EngineArgs = EngineArgs
    arg_utils_module.logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_package)
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", config_module)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_package)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_module)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)
    return arg_utils_module, config_module


def _load_patch_module():
    module_name = "afd_plugin.compat.patches.config_validation"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _engine_args(*, active, role="attention", worker_cls="auto"):
    args = sys.modules["vllm.engine.arg_utils"].EngineArgs()
    args.additional_config = {"afd": {"role": role}} if active else {}
    args.enable_dbo = True
    args.ubatch_size = 1
    args.all2all_backend = "allgather_reducescatter"
    args.worker_cls = worker_cls
    return args


def _set_fake_platform(*, is_cuda, device_type):
    sys.modules["vllm.platforms"].current_platform = SimpleNamespace(
        is_cuda=lambda: is_cuda,
        device_type=device_type,
    )


def test_config_validation_patch_relaxes_backend_for_afd_ubatching(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    patch_module = _load_patch_module()
    importlib.reload(patch_module)
    args = _engine_args(active=True)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert args.all2all_backend == "allgather_reducescatter"
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"


def test_config_validation_patch_preserves_non_afd_validation(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=False)

    try:
        arg_utils_module.EngineArgs.create_engine_config(args)
    except AssertionError as exc:
        assert "native all2all" in str(exc)
    else:
        raise AssertionError("expected native all2all backend assertion")


def test_config_validation_patch_allows_vllm_dev_checkout(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    sys.modules["vllm"].__version__ = "0.1.dev14230+g68b0c3135"
    _load_patch_module()
    args = _engine_args(active=True)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"


def test_config_validation_patch_selects_worker_after_upstream_post_init(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=True)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)
    assert cfg.post_init_backend == "deepep_low_latency"
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"
    assert cfg.parallel_config.worker_cls == ATTENTION_WORKER_FQCN


def test_config_validation_patch_relaxes_explicit_post_init_revalidation(monkeypatch):
    arg_utils_module, _config_module = _install_fake_vllm_config(monkeypatch)
    _load_patch_module()
    args = _engine_args(active=True)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)
    cfg.__post_init__()

    assert cfg.post_init_backend == "deepep_low_latency"
    assert cfg.parallel_config.all2all_backend == "allgather_reducescatter"


@pytest.mark.parametrize(
    (
        "role",
        "platform_worker_cls",
        "is_cuda",
        "device_type",
        "expected_worker_cls",
    ),
    [
        (
            "attention",
            VLLM_GPU_WORKER_FQCN,
            True,
            "cuda",
            ATTENTION_WORKER_FQCN,
        ),
        ("ffn", VLLM_GPU_WORKER_FQCN, True, "cuda", FFN_WORKER_FQCN),
        (
            "attention",
            VLLM_ASCEND_NPU_WORKER_FQCN,
            False,
            "npu",
            NPU_ATTENTION_WORKER_FQCN,
        ),
        (
            "ffn",
            VLLM_ASCEND_NPU_WORKER_FQCN,
            False,
            "npu",
            NPU_FFN_WORKER_FQCN,
        ),
    ],
)
def test_config_validation_patch_auto_selects_afd_worker(
    monkeypatch,
    role,
    platform_worker_cls,
    is_cuda,
    device_type,
    expected_worker_cls,
):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    config_module.VllmConfig.platform_worker_cls = platform_worker_cls
    _set_fake_platform(is_cuda=is_cuda, device_type=device_type)
    _load_patch_module()
    args = _engine_args(active=True, role=role)

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert cfg.parallel_config.worker_cls == expected_worker_cls


def test_config_validation_patch_preserves_explicit_worker(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    config_module.VllmConfig.platform_worker_cls = VLLM_GPU_WORKER_FQCN
    _load_patch_module()
    args = _engine_args(
        active=True,
        role="attention",
        worker_cls=ATTENTION_WORKER_FQCN,
    )

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert cfg.parallel_config.worker_cls == ATTENTION_WORKER_FQCN


def test_config_validation_patch_auto_selects_without_ubatching(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    config_module.VllmConfig.platform_worker_cls = VLLM_GPU_WORKER_FQCN
    _load_patch_module()
    args = _engine_args(active=True, role="ffn")
    args.enable_dbo = False

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert cfg.parallel_config.worker_cls == FFN_WORKER_FQCN


def test_config_validation_installs_ascend_patch_only_on_npu(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    import afd_plugin.compat.npu as npu_compat

    calls = []
    monkeypatch.setattr(
        npu_compat,
        "apply_afd_ascend_patches_if_needed",
        lambda: calls.append("npu"),
    )
    patch_module = _load_patch_module()
    importlib.reload(patch_module)

    cuda_args = _engine_args(active=True)
    arg_utils_module.EngineArgs.create_engine_config(cuda_args)
    assert calls == []

    config_module.VllmConfig.platform_worker_cls = VLLM_ASCEND_NPU_WORKER_FQCN
    _set_fake_platform(is_cuda=False, device_type="npu")
    npu_args = _engine_args(active=True)
    arg_utils_module.EngineArgs.create_engine_config(npu_args)
    assert calls == ["npu"]


def test_config_validation_patch_preserves_non_afd_platform_default(monkeypatch):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    config_module.VllmConfig.platform_worker_cls = VLLM_GPU_WORKER_FQCN
    _load_patch_module()
    args = _engine_args(active=False)
    args.enable_dbo = False

    cfg = arg_utils_module.EngineArgs.create_engine_config(args)

    assert cfg.parallel_config.worker_cls == VLLM_GPU_WORKER_FQCN


@pytest.mark.parametrize(
    ("platform_worker_cls", "is_cuda", "device_type"),
    [
        (VLLM_ASCEND_310P_WORKER_FQCN, False, "npu"),
        (VLLM_ASCEND_XLITE_WORKER_FQCN, False, "npu"),
        ("other_platform.worker.Worker", False, "xpu"),
        (VLLM_GPU_WORKER_FQCN, False, "cuda"),
    ],
)
def test_config_validation_patch_rejects_unsupported_auto_platform(
    monkeypatch,
    platform_worker_cls,
    is_cuda,
    device_type,
):
    arg_utils_module, config_module = _install_fake_vllm_config(monkeypatch)
    config_module.VllmConfig.platform_worker_cls = platform_worker_cls
    _set_fake_platform(is_cuda=is_cuda, device_type=device_type)
    _load_patch_module()
    args = _engine_args(active=True)

    with pytest.raises(ValueError, match="automatic worker selection"):
        arg_utils_module.EngineArgs.create_engine_config(args)
