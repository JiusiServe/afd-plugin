# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from afd_plugin.compat.ascend import runtime as ascend_runtime
from afd_plugin.compat.ascend.runtime import fix_all2all_backend_for_afd


def _vllm_config(*, enable_sp=False, all2all_backend="allgather_reducescatter"):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_sp),
        ),
        parallel_config=SimpleNamespace(
            all2all_backend=all2all_backend,
        ),
    )


def test_fix_all2all_backend_overrides_to_flashinfer_when_sp_disabled():
    config = _vllm_config(enable_sp=False, all2all_backend="allgather_reducescatter")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "flashinfer_all2allv"


def test_fix_all2all_backend_skips_when_sp_enabled():
    config = _vllm_config(enable_sp=True, all2all_backend="allgather_reducescatter")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "allgather_reducescatter"


def test_fix_all2all_backend_skips_when_already_flashinfer():
    config = _vllm_config(enable_sp=False, all2all_backend="flashinfer_all2allv")

    fix_all2all_backend_for_afd(config)

    assert config.parallel_config.all2all_backend == "flashinfer_all2allv"


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

    class NPUPlatform:
        @staticmethod
        def _fix_incompatible_config(vllm_config):
            parallel_config = vllm_config.parallel_config
            parallel_config.enable_dbo = False
            parallel_config.ubatch_size = 0
            return "fixed"

    def afd_vllm_config(*, enabled=True):
        config = _vllm_config()
        config.additional_config = {
            "afd": {
                "enabled": enabled,
                "role": "attention",
                "connector": "camp2pconnector",
            },
        }
        config.parallel_config = FakeParallelConfig(enable_dbo=True, ubatch_size=4)
        return config

    fake_platform.NPUPlatform = NPUPlatform
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", fake_platform)
    monkeypatch.setattr(ascend_runtime, "_PATCHES_APPLIED", False)

    ascend_runtime.apply_afd_ascend_patches_if_needed()

    config = afd_vllm_config()
    assert NPUPlatform._fix_incompatible_config(config) == "fixed"
    assert config.parallel_config.enable_dbo is True
    assert config.parallel_config.use_ubatching is True
    assert config.parallel_config.ubatch_size == 4

    disabled_config = afd_vllm_config(enabled=False)
    assert NPUPlatform._fix_incompatible_config(disabled_config) == "fixed"
    assert disabled_config.parallel_config.enable_dbo is False
    assert disabled_config.parallel_config.use_ubatching is False
