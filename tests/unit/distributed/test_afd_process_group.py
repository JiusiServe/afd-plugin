# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.distributed.afd_process_group import (  # noqa: E402
    create_hccl_process_group_options,
)


def test_create_hccl_process_group_options_is_scoped(monkeypatch):
    class FakeOptions:
        hccl_config = None

    class FakeTorchNPU(ModuleType):
        _C: SimpleNamespace

    fake_torch_npu = FakeTorchNPU("torch_npu")
    fake_torch_npu._C = SimpleNamespace(
        _distributed_c10d=SimpleNamespace(
            ProcessGroupHCCL=SimpleNamespace(Options=FakeOptions),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    assert create_hccl_process_group_options(None) is None
    options = create_hccl_process_group_options(4096)
    other_options = create_hccl_process_group_options(4096)

    assert options.hccl_config == {"hccl_buffer_size": 4096}
    assert options is not other_options
