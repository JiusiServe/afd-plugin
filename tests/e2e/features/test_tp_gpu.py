# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU E2E correctness tests: DeepSeekV2-Lite with TP=2 (eager + CUDA graph).

Each test runs a 2A2F topology with tensor-parallel size 2 (DP=1) and asserts
the runner-produced completion contains the expected text.

Skipped unless AFD_GPU_E2E_MODEL is set to a local model path.
Requires 4 GPUs (2 attention + 2 FFN).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "tests" / "e2e" / "runner.py"


def _gpu_list() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get("AFD_GPU_E2E_GPUS", "0,1,2,3").split(",")
        if item.strip()
    ]


def _model_path() -> str:
    model = os.environ.get("AFD_GPU_E2E_MODEL")
    if not model:
        pytest.skip("set AFD_GPU_E2E_MODEL to run TP=2 GPU E2E tests")
    return model


def _run_tp2(*, graph: bool) -> None:
    gpus = _gpu_list()
    if len(gpus) < 4:
        pytest.skip(f"requires 4 GPUs; got {len(gpus)}")

    model = _model_path()
    capture_size = os.environ.get("AFD_GPU_E2E_GRAPH_CAPTURE_SIZE", "8")

    if graph:
        api_port = os.environ.get("AFD_GPU_E2E_TP2_GRAPH_API_PORT", "18016")
        afd_port = os.environ.get("AFD_GPU_E2E_TP2_GRAPH_AFD_PORT", "6256")
    else:
        api_port = os.environ.get("AFD_GPU_E2E_TP2_EAGER_API_PORT", "18006")
        afd_port = os.environ.get("AFD_GPU_E2E_TP2_EAGER_AFD_PORT", "6246")

    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        model,
        "--vllm-bin",
        os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm"),
        "--num-attention-ranks",
        "2",
        "--num-ffn-ranks",
        "2",
        "--attention-gpus",
        ",".join(gpus[:2]),
        "--ffn-gpus",
        ",".join(gpus[2:4]),
        "--api-port-base",
        api_port,
        "--afd-port",
        afd_port,
        "--tp-size",
        "2",
        "--max-tokens",
        "7",
        "--prompt",
        "San Francisco is a",
        "--startup-timeout",
        os.environ.get("AFD_GPU_E2E_STARTUP_TIMEOUT", "900"),
        "--common-vllm-arg=--trust-remote-code",
        "--expect-text",
        "city of neighborhoods, and each one",
    ]
    if graph:
        command.extend(
            ["--cuda-graph-full-decode-only", "--cudagraph-capture-size", capture_size],
        )

    subprocess.run(command, cwd=REPO_ROOT, check=True)


@pytest.mark.gpu
def test_deepseek_v2_tp2_eager_e2e():
    """TP=2, DP=1 eager E2E correctness test on GPU."""
    _run_tp2(graph=False)


@pytest.mark.gpu
@pytest.mark.slow
def test_deepseek_v2_tp2_graph_e2e():
    """TP=2, DP=1 CUDA graph FULL_DECODE_ONLY E2E correctness test on GPU."""
    _run_tp2(graph=True)
