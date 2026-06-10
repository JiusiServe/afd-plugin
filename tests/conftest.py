# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared pytest configuration and fixtures for afd-plugin tests."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    for marker in [
        "gpu: GPU-gated AFD integration tests.",
        "npu: NPU-gated AFD integration tests.",
        "e2e: full end-to-end tests requiring model weights and hardware.",
        "eval: accuracy/evaluation tests requiring datasets.",
        "slow: tests that take more than 120 seconds.",
    ]:
        config.addinivalue_line("markers", marker)


# ---------------------------------------------------------------------------
# Session-scoped fixtures for GPU E2E tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def afd_e2e_model() -> str:
    """Return the model path/ID for E2E tests, or skip if unset."""
    model = os.environ.get("AFD_GPU_E2E_MODEL")
    if not model:
        pytest.skip("Set AFD_GPU_E2E_MODEL to run GPU E2E tests")
    return model


@pytest.fixture(scope="session")
def afd_gpu_list() -> list[str]:
    """Return the list of GPUs available for E2E tests."""
    raw = os.environ.get("AFD_GPU_E2E_GPUS", "0,1,2,3")
    return [g.strip() for g in raw.split(",") if g.strip()]


@pytest.fixture(scope="session")
def afd_vllm_bin() -> str:
    """Return the vllm binary path for E2E tests."""
    return os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm")


# ---------------------------------------------------------------------------
# Session-scoped fixtures for NPU E2E tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def npu_available() -> bool:
    """Skip if torch_npu is not available."""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pytest.skip("torch_npu not available; skipping NPU tests")
    return True


@pytest.fixture(scope="session")
def npu_e2e_model() -> str:
    """Return the model path for NPU E2E tests, or skip."""
    model = os.environ.get("AFD_NPU_E2E_MODEL")
    if not model:
        pytest.skip("Set AFD_NPU_E2E_MODEL to run NPU E2E tests")
    return model


@pytest.fixture(scope="session")
def npu_attn_device() -> str:
    return os.environ.get("AFD_NPU_ATTN_DEVICES", "0")


@pytest.fixture(scope="session")
def npu_ffn_device() -> str:
    return os.environ.get("AFD_NPU_FFN_DEVICES", "1")


@pytest.fixture(scope="session")
def npu_vllm_bin() -> str:
    return os.environ.get("AFD_NPU_VLLM_BIN", "vllm")
