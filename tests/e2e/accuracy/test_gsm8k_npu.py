# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GSM8K accuracy evaluation via lm-eval against NPU AFD servers.

Covers both eager and cuda-graph (FULL_DECODE_ONLY) 1A1F topologies.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.conftest import AFDServer, _launch_afd_server
from tests.e2e.helpers_gsm8k import (
    _extract_gsm8k_accuracy,
    _run_lm_eval,
)


def _default_tasks_dir() -> Path:
    """Resolve the bundled offline gsm8k task dir relative to this file."""
    return (
        Path(__file__).resolve().parents[4]
        / "lm-evaluation-harness"
        / "lm_eval"
        / "tasks"
        / "gsm8k"
    )


def _run_gsm8k(
    *,
    graph: bool,
    npu_e2e_model: str,
    npu_attn_device: str,
    npu_ffn_device: str,
    npu_vllm_bin: str,
    tmp_path: Path,
) -> None:
    """Run GSM8K via lm-eval against an NPU AFD 1A1F server (eager or graph)."""
    pytest.importorskip("lm_eval", reason="lm-eval not installed")

    # ALWAYS use the local offline task config (gsm8k.yaml -> dataset_path:
    # parquet -> staged /root/.cache/gsm8k/*.parquet). Resolved relative to this
    # file so it works regardless of cwd; env var overrides. If the dir is
    # missing, skip rather than fall back to the built-in openai/gsm8k task,
    # which would try to download from HuggingFace and fails on offline NPU hosts.
    tasks_dir = os.environ.get("AFD_NPU_GSM8K_TASK_DIR") or str(_default_tasks_dir())
    if not Path(tasks_dir).is_dir():
        pytest.skip(
            f"offline gsm8k task dir not found: {tasks_dir}. "
            "Stage gsm8k parquet + gsm8k.yaml, or set AFD_NPU_GSM8K_TASK_DIR.",
        )

    threshold = float(os.environ.get("AFD_NPU_GSM8K_THRESHOLD", "0.20"))
    tolerance = float(os.environ.get("AFD_NPU_GSM8K_TOLERANCE", "0.05"))
    _limit = os.environ.get("AFD_NPU_GSM8K_LIMIT")
    limit = int(_limit) if _limit else None

    if graph:
        api_port = int(os.environ.get("AFD_NPU_GSM8K_GRAPH_API_PORT", "19610"))
        afd_port = int(os.environ.get("AFD_NPU_GSM8K_GRAPH_AFD_PORT", "6402"))
    else:
        api_port = int(os.environ.get("AFD_NPU_GSM8K_API_PORT", "19600"))
        afd_port = int(os.environ.get("AFD_NPU_GSM8K_AFD_PORT", "6392"))

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
        # Real graph mode: cuda_graph_full_decode_only=True drives
        # FULL_DECODE_ONLY in runner.build_vllm_command. (Note: the
        # `enforce_eager` kwarg is a no-op in _launch_afd_server, so graph mode
        # MUST be requested via cuda_graph_full_decode_only.)
        launch_kwargs["cuda_graph_full_decode_only"] = True
        launch_kwargs["cudagraph_capture_size"] = 8

    afd_server: AFDServer | None = None
    try:
        afd_server = _launch_afd_server(**launch_kwargs)  # type: ignore[arg-type]

        results = _run_lm_eval(
            base_url=afd_server.base_url,
            model_name=afd_server.served_model,
            output_path=str(
                tmp_path / f"lm_eval_output_{'graph' if graph else 'eager'}"
            ),
            tokenizer=npu_e2e_model,
            tasks_dir=tasks_dir,
            limit=limit,
        )

        accuracy = _extract_gsm8k_accuracy(results)
        effective_threshold = threshold - tolerance

        mode = "graph" if graph else "eager"
        print(
            f"\n[GSM8K NPU {mode}] accuracy={accuracy:.4f} threshold={threshold} "
            f"tolerance={tolerance} effective_min={effective_threshold:.4f}"
        )

        assert accuracy >= effective_threshold, (
            f"GSM8K ({mode}) accuracy {accuracy:.4f} < "
            f"effective threshold {effective_threshold:.4f} "
            f"(threshold={threshold}, tolerance={tolerance})"
        )

    finally:
        if afd_server is not None:
            afd_server.shutdown()


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.eval
@pytest.mark.slow
def test_gsm8k_lm_eval_1a1f_eager(
    npu_available: bool,
    npu_e2e_model: str,
    npu_attn_device: str,
    npu_ffn_device: str,
    npu_vllm_bin: str,
    tmp_path: Path,
) -> None:
    """GSM8K accuracy via lm-eval against NPU AFD 1A1F (eager)."""
    _run_gsm8k(
        graph=False,
        npu_e2e_model=npu_e2e_model,
        npu_attn_device=npu_attn_device,
        npu_ffn_device=npu_ffn_device,
        npu_vllm_bin=npu_vllm_bin,
        tmp_path=tmp_path,
    )


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.eval
@pytest.mark.slow
def test_gsm8k_lm_eval_1a1f_graph(
    npu_available: bool,
    npu_e2e_model: str,
    npu_attn_device: str,
    npu_ffn_device: str,
    npu_vllm_bin: str,
    tmp_path: Path,
) -> None:
    """GSM8K accuracy via lm-eval against NPU AFD 1A1F (cuda graph FULL_DECODE_ONLY)."""
    _run_gsm8k(
        graph=True,
        npu_e2e_model=npu_e2e_model,
        npu_attn_device=npu_attn_device,
        npu_ffn_device=npu_ffn_device,
        npu_vllm_bin=npu_vllm_bin,
        tmp_path=tmp_path,
    )
