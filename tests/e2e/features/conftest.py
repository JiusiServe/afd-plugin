# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Module-scoped server fixtures for feature tests."""

from __future__ import annotations

import os

import pytest

from tests.e2e.conftest import AFDServer, _launch_afd_server


@pytest.fixture(scope="module")
def afd_server_1a1f(
    afd_e2e_model: str,
    afd_gpu_list: list[str],
    afd_vllm_bin: str,
) -> AFDServer:
    """Launch 1A1F GPU AFD servers, yield AFDServer, cleanup on teardown."""
    if len(afd_gpu_list) < 2:
        pytest.skip("1A1F requires at least 2 GPUs")

    api_port = int(os.environ.get("AFD_E2E_API_PORT", "19000"))
    afd_port = int(os.environ.get("AFD_E2E_AFD_PORT", "6390"))

    server = _launch_afd_server(
        backend="gpu",
        model=afd_e2e_model,
        vllm_bin=afd_vllm_bin,
        attention_devices=afd_gpu_list[:1],
        ffn_devices=afd_gpu_list[1:2],
        api_port_base=api_port,
        afd_port=afd_port,
        common_vllm_args=["--trust-remote-code"],
    )
    yield server
    server.shutdown()


@pytest.fixture(scope="module")
def npu_server_1a1f(
    npu_available: bool,
    npu_e2e_model: str,
    npu_attn_device: str,
    npu_ffn_device: str,
    npu_vllm_bin: str,
) -> AFDServer:
    """Launch 1A1F NPU AFD servers, yield AFDServer, cleanup on teardown."""
    api_port = int(os.environ.get("AFD_NPU_API_PORT", "19800"))
    afd_port = int(os.environ.get("AFD_NPU_AFD_PORT", "6397"))

    server = _launch_afd_server(
        backend="npu",
        model=npu_e2e_model,
        vllm_bin=npu_vllm_bin,
        attention_devices=[npu_attn_device],
        ffn_devices=[npu_ffn_device],
        api_port_base=api_port,
        afd_port=afd_port,
    )
    yield server
    server.shutdown()
