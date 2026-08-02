# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.compat.npu import runtime as ascend_runtime
from afd_plugin.compat.npu.runtime import fix_all2all_backend_for_afd


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


def test_ascend_forward_context_installs_afd_metadata(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_config = ModuleType("vllm.config")
    fake_forward_context_module = ModuleType("vllm.forward_context")
    fake_vllm_ascend = ModuleType("vllm_ascend")
    fake_vllm_ascend.__path__ = []
    fake_ascend_forward_context = ModuleType(
        "vllm_ascend.ascend_forward_context",
    )
    forward_context = SimpleNamespace(additional_kwargs=None)
    calls = []

    class CUDAGraphMode:
        NONE = "none"

    @contextmanager
    def set_ascend_forward_context(
        attn_metadata,
        vllm_config,
        *,
        batch_descriptor,
        aclgraph_runtime_mode,
        model_instance,
        num_tokens,
        num_tokens_across_dp,
    ):
        calls.append(
            {
                "attn_metadata": attn_metadata,
                "vllm_config": vllm_config,
                "batch_descriptor": batch_descriptor,
                "aclgraph_runtime_mode": aclgraph_runtime_mode,
                "model_instance": model_instance,
                "num_tokens": num_tokens,
                "num_tokens_across_dp": num_tokens_across_dp,
            },
        )
        yield

    fake_config.CUDAGraphMode = CUDAGraphMode
    fake_forward_context_module.get_forward_context = lambda: forward_context
    fake_ascend_forward_context.set_ascend_forward_context = set_ascend_forward_context
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", fake_config)
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        fake_forward_context_module,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_vllm_ascend)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        fake_ascend_forward_context,
    )

    vllm_config = SimpleNamespace()
    afd_metadata = SimpleNamespace()
    model_instance = SimpleNamespace()
    with ascend_runtime.ascend_forward_context(
        vllm_config=vllm_config,
        afd_metadata=afd_metadata,
        model_instance=model_instance,
        num_tokens=3,
    ) as current_forward_context:
        assert current_forward_context is forward_context
        assert forward_context.additional_kwargs["afd_metadata"] is afd_metadata

    assert calls == [
        {
            "attn_metadata": None,
            "vllm_config": vllm_config,
            "batch_descriptor": None,
            "aclgraph_runtime_mode": CUDAGraphMode.NONE,
            "model_instance": model_instance,
            "num_tokens": 3,
            "num_tokens_across_dp": None,
        },
    ]


def test_npu_afd_config_patch_restores_dbo_for_afd(monkeypatch):
    fake_package = ModuleType("vllm_ascend")
    fake_package.__path__ = []
    fake_platform = ModuleType("vllm_ascend.platform")

    class FakeParallelConfig:
        def __init__(self, *, enable_dbo, ubatch_size):
            self.enable_dbo = enable_dbo
            self.ubatch_size = ubatch_size
            self.all2all_backend = "deepep_low_latency"

        @property
        def use_ubatching(self):
            return self.enable_dbo or self.ubatch_size > 1

    class NPUPlatform:
        @staticmethod
        def _fix_incompatible_config(vllm_config):
            parallel_config = vllm_config.parallel_config
            parallel_config.enable_dbo = False
            parallel_config.ubatch_size = 0

        @classmethod
        def check_and_update_config(cls, vllm_config):
            cls._fix_incompatible_config(vllm_config)
            parallel_config = vllm_config.parallel_config
            parallel_config.all2all_backend = "flashinfer_all2allv"
            if getattr(vllm_config, "fail_update", False):
                raise RuntimeError("upstream config failure")

    def afd_vllm_config(*, active=True):
        config = _vllm_config()
        config.additional_config = (
            {
                "afd": {
                    "role": "attention",
                    "connector": "CAMP2pAFDConnector",
                },
            }
            if active
            else {}
        )
        config.parallel_config = FakeParallelConfig(enable_dbo=True, ubatch_size=4)
        return config

    fake_platform.NPUPlatform = NPUPlatform
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", fake_platform)
    monkeypatch.setattr(ascend_runtime, "_PATCHES_APPLIED", False)

    ascend_runtime.apply_afd_ascend_patches_if_needed()

    config = afd_vllm_config()
    assert NPUPlatform.check_and_update_config(config) is None
    assert config.parallel_config.enable_dbo is True
    assert config.parallel_config.use_ubatching is True
    assert config.parallel_config.ubatch_size == 4
    assert config.parallel_config.all2all_backend == "deepep_low_latency"

    failing_config = afd_vllm_config()
    failing_config.fail_update = True
    with pytest.raises(RuntimeError, match="upstream config failure"):
        NPUPlatform.check_and_update_config(failing_config)
    assert failing_config.parallel_config.enable_dbo is True
    assert failing_config.parallel_config.ubatch_size == 4
    assert failing_config.parallel_config.all2all_backend == "deepep_low_latency"

    inactive_config = afd_vllm_config(active=False)
    assert NPUPlatform.check_and_update_config(inactive_config) is None
    assert inactive_config.parallel_config.enable_dbo is False
    assert inactive_config.parallel_config.use_ubatching is False
    assert inactive_config.parallel_config.all2all_backend == "flashinfer_all2allv"
