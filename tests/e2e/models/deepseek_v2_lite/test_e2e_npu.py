# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in NPU (Ascend) E2E tests for DeepSeekV2 AFD — 2A2F eager/graph matrix.

All tests use a 2A2F topology (2 attention + 2 FFN servers, needs >=4 NPUs) and
shell out to the manual runner so the command line stays close to production
usage. The matrix exercises:

  base / +TP / +DBO / +profile / +TP+DBO+profile   ×   {eager, graph}

= 10 tests. DBO variants self-skip on NPU (DBO is not supported there yet).
Profiler is enabled purely through AFD_NPU_{ATTENTION,FFN}_PROFILER_* env vars,
which leak through runner.py's os.environ.copy() into the vllm worker — no
runner/source change required.

Skipped unless AFD_NPU_E2E_MODEL is set to a local DeepSeekV2-Lite model path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "tests" / "e2e" / "runner.py"


def _npu_list() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get("AFD_NPU_E2E_DEVICES", "0,1,2,3").split(",")
        if item.strip()
    ]


def _model_path() -> str:
    model = os.environ.get("AFD_NPU_E2E_MODEL")
    if not model:
        pytest.skip("set AFD_NPU_E2E_MODEL to run DeepSeekV2 AFD NPU E2E tests")
    return model


def _skip_dbo_on_npu() -> None:
    """DBO is not supported on NPU yet — skip cleanly instead of failing."""
    pytest.skip("DBO is not supported on NPU yet")


def _graph_capture_size() -> int:
    return int(os.environ.get("AFD_NPU_E2E_GRAPH_CAPTURE_SIZE", "8"))


def _run_e2e(
    *,
    npus: list[str],
    api_port_base: int,
    afd_port: int,
    graph: bool = False,
    tp: int = 1,
    dbo: bool = False,
) -> None:
    """Run a 2A2F AFD E2E via runner.py with the requested feature mix."""
    model = _model_path()
    if len(npus) < 4:
        pytest.skip(f"2A2F requires 4 NPUs; got {len(npus)}")

    capture_size = _graph_capture_size()

    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        model,
        "--vllm-bin",
        os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm"),
        "--num-attention-ranks",
        "2",
        "--num-ffn-ranks",
        "2",
        "--attention-gpus",
        ",".join(npus[:2]),
        "--ffn-gpus",
        ",".join(npus[2:4]),
        "--api-port-base",
        str(api_port_base),
        "--afd-port",
        str(afd_port),
        "--tp-size",
        str(tp),
        "--device-backend",
        "npu",
        "--max-tokens",
        os.environ.get("AFD_NPU_E2E_MAX_TOKENS", "8"),
        "--startup-timeout",
        os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "900"),
        "--common-vllm-arg=--trust-remote-code",
    ]

    if graph:
        requests = capture_size
        command.extend(
            [
                "--cuda-graph-full-decode-only",
                "--cudagraph-capture-size",
                str(capture_size),
                "--num-requests",
                str(requests),
                "--request-concurrency",
                str(requests),
            ],
        )

    if dbo:
        command.extend(
            [
                "--enable-dbo",
                "--dbo-decode-token-threshold",
                os.environ.get("AFD_NPU_E2E_DBO_DECODE_THRESHOLD", "1"),
                "--dbo-prefill-token-threshold",
                os.environ.get(
                    "AFD_NPU_E2E_DBO_PREFILL_THRESHOLD",
                    str(capture_size),
                ),
            ],
        )
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _enable_profiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Enable the worker profiler via env vars; return (attn_dir, ffn_dir).

    These env vars leak through runner.py's build_env (os.environ.copy()) into
    the vllm worker, where afd_plugin/compat/ascend/profiler.py reads them.
    """
    profiler_dir = tmp_path / "profiler_logs"
    attn_dir = profiler_dir / "attn"
    ffn_dir = profiler_dir / "ffn"
    for role, role_dir in [("ATTENTION", attn_dir), ("FFN", ffn_dir)]:
        prefix = f"AFD_NPU_{role}_PROFILER"
        monkeypatch.setenv(f"{prefix}_ENABLE", "true")
        monkeypatch.setenv(f"{prefix}_WAIT", "0")
        monkeypatch.setenv(f"{prefix}_WARMUP", "1")
        monkeypatch.setenv(f"{prefix}_ACTIVE", "1")
        monkeypatch.setenv(f"{prefix}_REPEAT", "1")
        monkeypatch.setenv(f"{prefix}_SKIP_FIRST", "0")
        monkeypatch.setenv(f"{prefix}_DIR", str(role_dir))
    return attn_dir, ffn_dir


def _assert_profiler_traces(attn_dir: Path, ffn_dir: Path) -> None:
    attn = [f for f in attn_dir.rglob("*") if f.is_file()] if attn_dir.exists() else []
    ffn = [f for f in ffn_dir.rglob("*") if f.is_file()] if ffn_dir.exists() else []
    assert len(attn) > 0, f"No profiler trace files in attention dir {attn_dir}"
    assert len(ffn) > 0, f"No profiler trace files in FFN dir {ffn_dir}"


# ---------------------------------------------------------------------------
# Base 2A2F: eager + graph
# ---------------------------------------------------------------------------


@pytest.mark.npu
def test_deepseek_v2_2a2f_eager():
    _run_e2e(
        npus=_npu_list(),
        api_port_base=int(os.environ.get("AFD_NPU_E2E_2A2F_EAGER_API_PORT", "18000")),
        afd_port=int(os.environ.get("AFD_NPU_E2E_2A2F_EAGER_AFD_PORT", "6239")),
    )


@pytest.mark.npu
@pytest.mark.slow
def test_deepseek_v2_2a2f_graph():
    _run_e2e(
        npus=_npu_list(),
        api_port_base=int(os.environ.get("AFD_NPU_E2E_2A2F_GRAPH_API_PORT", "18100")),
        afd_port=int(os.environ.get("AFD_NPU_E2E_2A2F_GRAPH_AFD_PORT", "6249")),
        graph=True,
    )


# ---------------------------------------------------------------------------
# 2A2F + TP=2: eager + graph
# ---------------------------------------------------------------------------


@pytest.mark.npu
def test_deepseek_v2_2a2f_tp_eager():
    _run_e2e(
        npus=_npu_list(),
        api_port_base=int(
            os.environ.get("AFD_NPU_E2E_2A2F_TP_EAGER_API_PORT", "18200")
        ),
        afd_port=int(os.environ.get("AFD_NPU_E2E_2A2F_TP_EAGER_AFD_PORT", "6259")),
        tp=2,
    )


@pytest.mark.npu
@pytest.mark.slow
def test_deepseek_v2_2a2f_tp_graph():
    _run_e2e(
        npus=_npu_list(),
        api_port_base=int(
            os.environ.get("AFD_NPU_E2E_2A2F_TP_GRAPH_API_PORT", "18300")
        ),
        afd_port=int(os.environ.get("AFD_NPU_E2E_2A2F_TP_GRAPH_AFD_PORT", "6269")),
        graph=True,
        tp=2,
    )


# ---------------------------------------------------------------------------
# 2A2F + DBO: eager + graph (NPU self-skips — DBO unsupported)
# ---------------------------------------------------------------------------


@pytest.mark.npu
def test_deepseek_v2_2a2f_dbo_eager():
    _skip_dbo_on_npu()
    _run_e2e(
        npus=_npu_list(),
        api_port_base=int(
            os.environ.get("AFD_NPU_E2E_2A2F_DBO_EAGER_API_PORT", "18400")
        ),
        afd_port=int(os.environ.get("AFD_NPU_E2E_2A2F_DBO_EAGER_AFD_PORT", "6279")),
        dbo=True,
    )


@pytest.mark.npu
@pytest.mark.slow
def test_deepseek_v2_2a2f_dbo_graph():
    _skip_dbo_on_npu()
    _run_e2e(
        npus=_npu_list(),
        api_port_base=int(
            os.environ.get("AFD_NPU_E2E_2A2F_DBO_GRAPH_API_PORT", "18500")
        ),
        afd_port=int(os.environ.get("AFD_NPU_E2E_2A2F_DBO_GRAPH_AFD_PORT", "6289")),
        graph=True,
        dbo=True,
    )


# ---------------------------------------------------------------------------
# 2A2F + profiler: eager + graph
# ---------------------------------------------------------------------------


@pytest.mark.npu
def test_deepseek_v2_2a2f_profile_eager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    attn_dir, ffn_dir = _enable_profiler(tmp_path, monkeypatch)
    _run_e2e(
        npus=_npu_list(),
        api_port_base=int(
            os.environ.get("AFD_NPU_E2E_2A2F_PROFILE_EAGER_API_PORT", "18600")
        ),
        afd_port=int(os.environ.get("AFD_NPU_E2E_2A2F_PROFILE_EAGER_AFD_PORT", "6299")),
    )
    _assert_profiler_traces(attn_dir, ffn_dir)


@pytest.mark.npu
@pytest.mark.slow
def test_deepseek_v2_2a2f_profile_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    attn_dir, ffn_dir = _enable_profiler(tmp_path, monkeypatch)
    _run_e2e(
        npus=_npu_list(),
        api_port_base=int(
            os.environ.get("AFD_NPU_E2E_2A2F_PROFILE_GRAPH_API_PORT", "18700")
        ),
        afd_port=int(os.environ.get("AFD_NPU_E2E_2A2F_PROFILE_GRAPH_AFD_PORT", "6309")),
        graph=True,
    )
    _assert_profiler_traces(attn_dir, ffn_dir)


# ---------------------------------------------------------------------------
# 2A2F + TP=2 + DBO + profiler: eager + graph (NPU self-skips — DBO unsupported)
# ---------------------------------------------------------------------------


@pytest.mark.npu
def test_deepseek_v2_2a2f_tp_dbo_profile_eager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _skip_dbo_on_npu()
    attn_dir, ffn_dir = _enable_profiler(tmp_path, monkeypatch)
    _run_e2e(
        npus=_npu_list(),
        api_port_base=int(
            os.environ.get("AFD_NPU_E2E_2A2F_TP_DBO_PROFILE_EAGER_API_PORT", "18800"),
        ),
        afd_port=int(
            os.environ.get("AFD_NPU_E2E_2A2F_TP_DBO_PROFILE_EAGER_AFD_PORT", "6319"),
        ),
        tp=2,
        dbo=True,
    )
    _assert_profiler_traces(attn_dir, ffn_dir)


@pytest.mark.npu
@pytest.mark.slow
def test_deepseek_v2_2a2f_tp_dbo_profile_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _skip_dbo_on_npu()
    attn_dir, ffn_dir = _enable_profiler(tmp_path, monkeypatch)
    _run_e2e(
        npus=_npu_list(),
        api_port_base=int(
            os.environ.get("AFD_NPU_E2E_2A2F_TP_DBO_PROFILE_GRAPH_API_PORT", "18900"),
        ),
        afd_port=int(
            os.environ.get("AFD_NPU_E2E_2A2F_TP_DBO_PROFILE_GRAPH_AFD_PORT", "6329"),
        ),
        graph=True,
        tp=2,
        dbo=True,
    )
    _assert_profiler_traces(attn_dir, ffn_dir)
