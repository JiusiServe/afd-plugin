# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only Phase 0 baselines for current Ascend uBatch behavior."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@dataclass
class _UBatchSlice:
    request_slice: slice
    token_slice: slice

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start

    def is_empty(self) -> bool:
        return self.num_tokens == 0 or (
            self.request_slice.start == self.request_slice.stop
        )


def _check_ubatch_thresholds(config, num_tokens, *, uniform_decode):
    if not config.use_ubatching:
        return False
    threshold = (
        config.dbo_decode_token_threshold
        if uniform_decode
        else config.dbo_prefill_token_threshold
    )
    return num_tokens >= threshold


@pytest.fixture
def ubatch_utils(monkeypatch):
    """Load the production helper with only its import contracts stubbed."""

    modules = {
        "torch": types.ModuleType("torch"),
        "vllm": types.ModuleType("vllm"),
        "vllm.config": types.ModuleType("vllm.config"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.ubatch_utils": types.ModuleType("vllm.v1.worker.ubatch_utils"),
        "vllm_ascend": types.ModuleType("vllm_ascend"),
        "vllm_ascend.ascend_forward_context": types.ModuleType(
            "vllm_ascend.ascend_forward_context"
        ),
        "vllm_ascend.attention": types.ModuleType("vllm_ascend.attention"),
        "vllm_ascend.attention.utils": types.ModuleType("vllm_ascend.attention.utils"),
    }
    modules["torch"].Tensor = object
    modules["vllm.config"].VllmConfig = object
    modules["vllm.v1.worker.ubatch_utils"].UBatchSlice = _UBatchSlice
    modules["vllm.v1.worker.ubatch_utils"].UBatchSlices = list[_UBatchSlice]
    modules[
        "vllm.v1.worker.ubatch_utils"
    ].check_ubatch_thresholds = _check_ubatch_thresholds
    modules["vllm_ascend.ascend_forward_context"].MoECommType = object
    modules["vllm_ascend.attention.utils"].AscendCommonAttentionMetadata = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = (
        Path(__file__).resolve().parents[4] / "afd_plugin/v1/worker/npu/ubatch_utils.py"
    )
    module_name = "_afd_phase0_ubatch_utils"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _vllm_config(**overrides):
    values = {
        "dbo_decode_token_threshold": 8,
        "dbo_prefill_token_threshold": 16,
        "decode_context_parallel_size": 1,
        "enable_dbo": True,
        "num_ubatches": 2,
        "prefill_context_parallel_size": 1,
        "use_ubatching": True,
    }
    values.update(overrides)
    return SimpleNamespace(parallel_config=SimpleNamespace(**values))


@pytest.mark.parametrize(
    ("config", "num_unpadded", "num_padded", "uniform_decode", "expected"),
    [
        (_vllm_config(), 16, 16, False, True),
        (_vllm_config(enable_dbo=False), 16, 16, False, False),
        (_vllm_config(), 15, 16, False, False),
        (_vllm_config(prefill_context_parallel_size=2), 16, 16, False, False),
        (_vllm_config(num_ubatches=4), 16, 16, False, False),
        (_vllm_config(), 1, 2, False, False),
        (_vllm_config(), 8, 8, True, True),
        (_vllm_config(), 7, 8, True, False),
    ],
)
def test_native_dbo_decision_matrix(
    ubatch_utils,
    config,
    num_unpadded,
    num_padded,
    uniform_decode,
    expected,
) -> None:
    assert (
        ubatch_utils.check_enable_ubatch(
            num_unpadded,
            num_padded,
            uniform_decode,
            config,
            None,
        )
        is expected
    )


def test_native_token_split_and_padding_contract(ubatch_utils) -> None:
    scheduled_tokens = np.array([6, 6, 2], dtype=np.int32)

    slices, padded_slices = ubatch_utils.maybe_create_ubatch_slices(
        True,
        scheduled_tokens,
        num_tokens_padded=16,
        num_reqs_padded=4,
        vllm_config=_vllm_config(),
    )

    assert slices == [
        _UBatchSlice(slice(0, 2), slice(0, 8)),
        _UBatchSlice(slice(1, 3), slice(8, 14)),
    ]
    assert padded_slices == [
        _UBatchSlice(slice(0, 2), slice(0, 8)),
        _UBatchSlice(slice(1, 4), slice(8, 16)),
    ]
    assert sum(item.num_tokens for item in padded_slices) == 16


@pytest.mark.parametrize(
    ("scheduled_tokens", "expected"),
    [
        (
            [2, 3, 5, 7],
            [
                _UBatchSlice(slice(0, 3), slice(0, 10)),
                _UBatchSlice(slice(3, 4), slice(10, 17)),
            ],
        ),
        (
            [824, 846, 16],
            [
                _UBatchSlice(slice(0, 1), slice(0, 824)),
                _UBatchSlice(slice(1, 3), slice(824, 1686)),
            ],
        ),
        ([5], None),
        ([0, 5], None),
    ],
)
def test_request_boundary_split_contract(
    ubatch_utils,
    scheduled_tokens,
    expected,
) -> None:
    actual = ubatch_utils.create_request_boundary_ubatch_slices(
        np.asarray(scheduled_tokens, dtype=np.int32)
    )

    assert actual == expected
