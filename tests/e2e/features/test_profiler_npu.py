# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU profiler E2E tests: verify TensorBoard trace files are generated.

Covers both eager and cuda-graph (FULL_DECODE_ONLY) 1A1F topologies.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.conftest import AFDServer, _launch_afd_server


def _run_profiler(
    *,
    graph: bool,
    npu_e2e_model: str,
    npu_attn_device: str,
    npu_ffn_device: str,
    npu_vllm_bin: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch NPU AFD 1A1F with profiler enabled and assert trace files appear."""
    profiler_dir = tmp_path / "profiler_logs"
    attn_dir = profiler_dir / "attn"
    ffn_dir = profiler_dir / "ffn"

    # Enable profiler with minimal schedule so traces appear quickly.
    for role, role_dir in [("ATTENTION", attn_dir), ("FFN", ffn_dir)]:
        prefix = f"AFD_NPU_{role}_PROFILER"
        monkeypatch.setenv(f"{prefix}_ENABLE", "true")
        monkeypatch.setenv(f"{prefix}_WAIT", "0")
        monkeypatch.setenv(f"{prefix}_WARMUP", "1")
        monkeypatch.setenv(f"{prefix}_ACTIVE", "1")
        monkeypatch.setenv(f"{prefix}_REPEAT", "1")
        monkeypatch.setenv(f"{prefix}_SKIP_FIRST", "0")
        monkeypatch.setenv(f"{prefix}_DIR", str(role_dir))

    if graph:
        api_port = int(os.environ.get("AFD_NPU_PROFILER_GRAPH_API_PORT", "19811"))
        afd_port = int(os.environ.get("AFD_NPU_PROFILER_GRAPH_AFD_PORT", "6409"))
    else:
        api_port = int(os.environ.get("AFD_NPU_PROFILER_API_PORT", "19801"))
        afd_port = int(os.environ.get("AFD_NPU_PROFILER_AFD_PORT", "6399"))

    launch_kwargs: dict[str, object] = {
        "backend": "npu",
        "model": npu_e2e_model,
        "vllm_bin": npu_vllm_bin,
        "attention_devices": [npu_attn_device],
        "ffn_devices": [npu_ffn_device],
        "api_port_base": api_port,
        "afd_port": afd_port,
    }
    if graph:
        launch_kwargs["cuda_graph_full_decode_only"] = True
        launch_kwargs["cudagraph_capture_size"] = 8

    server: AFDServer | None = None
    try:
        server = _launch_afd_server(**launch_kwargs)  # type: ignore[arg-type]

        # Send a few requests to trigger profiler steps.
        for _ in range(3):
            body = server.request_completion("Hello", max_tokens=4)
            assert "choices" in body

    finally:
        if server is not None:
            server.shutdown()

    # Verify trace files were written by tensorboard_trace_handler.
    attn_traces = (
        [f for f in attn_dir.rglob("*") if f.is_file()] if attn_dir.exists() else []
    )
    ffn_traces = (
        [f for f in ffn_dir.rglob("*") if f.is_file()] if ffn_dir.exists() else []
    )

    assert len(attn_traces) > 0, f"No profiler trace files in attention dir {attn_dir}"
    assert len(ffn_traces) > 0, f"No profiler trace files in FFN dir {ffn_dir}"


@pytest.mark.npu
@pytest.mark.e2e
def test_npu_profiler_produces_traces_eager(
    npu_available: bool,
    npu_e2e_model: str,
    npu_attn_device: str,
    npu_ffn_device: str,
    npu_vllm_bin: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NPU AFD 1A1F (eager) with profiler enabled produces TensorBoard traces."""
    _run_profiler(
        graph=False,
        npu_e2e_model=npu_e2e_model,
        npu_attn_device=npu_attn_device,
        npu_ffn_device=npu_ffn_device,
        npu_vllm_bin=npu_vllm_bin,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.slow
def test_npu_profiler_produces_traces_graph(
    npu_available: bool,
    npu_e2e_model: str,
    npu_attn_device: str,
    npu_ffn_device: str,
    npu_vllm_bin: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NPU AFD 1A1F (cuda graph) + profiler produces TensorBoard traces."""
    _run_profiler(
        graph=True,
        npu_e2e_model=npu_e2e_model,
        npu_attn_device=npu_attn_device,
        npu_ffn_device=npu_ffn_device,
        npu_vllm_bin=npu_vllm_bin,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
