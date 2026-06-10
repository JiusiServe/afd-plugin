# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GSM8K accuracy evaluation via lm-eval against GPU AFD servers.

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


def _run_gsm8k(
    *,
    graph: bool,
    afd_e2e_model: str,
    afd_gpu_list: list[str],
    afd_vllm_bin: str,
    tmp_path: Path,
) -> None:
    """Run GSM8K via lm-eval against a GPU AFD 1A1F server (eager or graph)."""
    if len(afd_gpu_list) < 2:
        pytest.skip("1A1F requires at least 2 GPUs")

    pytest.importorskip("lm_eval", reason="lm-eval not installed")

    threshold = float(os.environ.get("AFD_GSM8K_THRESHOLD", "0.20"))
    tolerance = float(os.environ.get("AFD_GSM8K_TOLERANCE", "0.05"))
    _limit = os.environ.get("AFD_GSM8K_LIMIT")
    limit = int(_limit) if _limit else None

    if graph:
        api_port = int(os.environ.get("AFD_GSM8K_GRAPH_API_PORT", "19610"))
        afd_port = int(os.environ.get("AFD_GSM8K_GRAPH_AFD_PORT", "6402"))
    else:
        api_port = int(os.environ.get("AFD_GSM8K_API_PORT", "19600"))
        afd_port = int(os.environ.get("AFD_GSM8K_AFD_PORT", "6392"))

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

    afd_server: AFDServer | None = None
    try:
        afd_server = _launch_afd_server(**launch_kwargs)  # type: ignore[arg-type]

        results = _run_lm_eval(
            base_url=afd_server.base_url,
            model_name=afd_server.served_model,
            output_path=str(
                tmp_path / f"lm_eval_output_{'graph' if graph else 'eager'}"
            ),
            tokenizer=afd_e2e_model,
            limit=limit,
        )

        accuracy = _extract_gsm8k_accuracy(results)
        effective_threshold = threshold - tolerance

        mode = "graph" if graph else "eager"
        print(
            f"\n[GSM8K {mode}] accuracy={accuracy:.4f} threshold={threshold} "
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


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.eval
@pytest.mark.slow
def test_gsm8k_lm_eval_1a1f_eager(
    afd_e2e_model: str,
    afd_gpu_list: list[str],
    afd_vllm_bin: str,
    tmp_path: Path,
) -> None:
    """GSM8K accuracy via lm-eval against GPU AFD 1A1F (eager)."""
    _run_gsm8k(
        graph=False,
        afd_e2e_model=afd_e2e_model,
        afd_gpu_list=afd_gpu_list,
        afd_vllm_bin=afd_vllm_bin,
        tmp_path=tmp_path,
    )


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.eval
@pytest.mark.slow
def test_gsm8k_lm_eval_1a1f_graph(
    afd_e2e_model: str,
    afd_gpu_list: list[str],
    afd_vllm_bin: str,
    tmp_path: Path,
) -> None:
    """GSM8K accuracy via lm-eval against GPU AFD 1A1F (cuda graph FULL_DECODE_ONLY)."""
    _run_gsm8k(
        graph=True,
        afd_e2e_model=afd_e2e_model,
        afd_gpu_list=afd_gpu_list,
        afd_vllm_bin=afd_vllm_bin,
        tmp_path=tmp_path,
    )
