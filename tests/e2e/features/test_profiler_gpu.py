# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU profiler E2E tests: verify TensorBoard trace files are generated.

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
    afd_e2e_model: str,
    afd_gpu_list: list[str],
    afd_vllm_bin: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch GPU AFD 1A1F with profiler enabled and assert trace files appear."""
    if len(afd_gpu_list) < 2:
        pytest.skip("1A1F requires at least 2 GPUs")

    profiler_dir = tmp_path / "profiler_logs"
    attn_dir = profiler_dir / "attn"
    ffn_dir = profiler_dir / "ffn"

    # Enable profiler with minimal schedule so traces appear quickly.
    for role, role_dir in [("ATTENTION", attn_dir), ("FFN", ffn_dir)]:
        prefix = f"AFD_GPU_{role}_PROFILER"
        monkeypatch.setenv(f"{prefix}_ENABLE", "true")
        monkeypatch.setenv(f"{prefix}_WAIT", "0")
        monkeypatch.setenv(f"{prefix}_WARMUP", "1")
        monkeypatch.setenv(f"{prefix}_ACTIVE", "1")
        monkeypatch.setenv(f"{prefix}_REPEAT", "1")
        monkeypatch.setenv(f"{prefix}_SKIP_FIRST", "0")
        monkeypatch.setenv(f"{prefix}_DIR", str(role_dir))

    if graph:
        api_port = int(os.environ.get("AFD_PROFILER_GRAPH_API_PORT", "19810"))
        afd_port = int(os.environ.get("AFD_PROFILER_GRAPH_AFD_PORT", "6408"))
    else:
        api_port = int(os.environ.get("AFD_PROFILER_API_PORT", "19800"))
        afd_port = int(os.environ.get("AFD_PROFILER_AFD_PORT", "6398"))

    launch_kwargs: dict[str, object] = {
        "backend": "gpu",
        "model": afd_e2e_model,
        "vllm_bin": afd_vllm_bin,
        "attention_devices": afd_gpu_list[:1],
        "ffn_devices": afd_gpu_list[1:2],
        "api_port_base": api_port,
        "afd_port": afd_port,
        "common_vllm_args": ["--trust-remote-code"],
    }
    if graph:
        launch_kwargs["cuda_graph_full_decode_only"] = True
        launch_kwargs["cudagraph_capture_size"] = 8

    server: AFDServer | None = None
    try:
        server = _launch_afd_server(**launch_kwargs)  # type: ignore[arg-type]

        for _ in range(3):
            body = server.request_completion("Hello", max_tokens=4)
            assert "choices" in body

    finally:
        if server is not None:
            server.shutdown()

    attn_traces = (
        [f for f in attn_dir.rglob("*") if f.is_file()] if attn_dir.exists() else []
    )
    ffn_traces = (
        [f for f in ffn_dir.rglob("*") if f.is_file()] if ffn_dir.exists() else []
    )

    assert len(attn_traces) > 0, f"No profiler trace files in attention dir {attn_dir}"
    assert len(ffn_traces) > 0, f"No profiler trace files in FFN dir {ffn_dir}"


@pytest.mark.gpu
@pytest.mark.e2e
def test_gpu_profiler_produces_traces_eager(
    afd_e2e_model: str,
    afd_gpu_list: list[str],
    afd_vllm_bin: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU AFD 1A1F (eager) with profiler enabled produces TensorBoard traces."""
    _run_profiler(
        graph=False,
        afd_e2e_model=afd_e2e_model,
        afd_gpu_list=afd_gpu_list,
        afd_vllm_bin=afd_vllm_bin,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_gpu_profiler_produces_traces_graph(
    afd_e2e_model: str,
    afd_gpu_list: list[str],
    afd_vllm_bin: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU AFD 1A1F (cuda graph) + profiler produces TensorBoard traces."""
    _run_profiler(
        graph=True,
        afd_e2e_model=afd_e2e_model,
        afd_gpu_list=afd_gpu_list,
        afd_vllm_bin=afd_vllm_bin,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
