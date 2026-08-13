# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Backend-neutral DeepSeek-V2-Lite E2E scenarios."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest

from tests.conftest import download_dataset, download_model

REPO_ROOT = Path(__file__).resolve().parents[4]
GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
DEEPSEEK_V2_LITE_REPO_ID = "deepseek-ai/DeepSeek-V2-Lite"
SCENARIOS = (
    "baseline-graph",
    "afd-eager",
    "afd-graph",
    "afd-graph-dbo",
)
# Covers the roughly 64-second nested lm-eval/vLLM cleanup bound with buffer.
RUNNER_CLEANUP_TIMEOUT_S = 90


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def _devices(name: str, expected_count: int) -> list[str]:
    devices = [item.strip() for item in _required_env(name).split(",") if item.strip()]
    if len(devices) != expected_count:
        raise RuntimeError(f"{name} must contain exactly {expected_count} devices")
    if len(devices) != len(set(devices)):
        raise RuntimeError(f"{name} devices must be unique")
    return devices


def prepare_e2e_assets() -> None:
    """Ensure GSM8K and DeepSeek-V2-Lite are available for the runner."""
    download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)

    backend = _required_env("AFD_E2E_BACKEND")
    if backend == "gpu":
        env_name = "AFD_GPU_E2E_MODEL"
    elif backend == "npu":
        env_name = "AFD_NPU_E2E_MODEL"
    else:
        raise RuntimeError("AFD_E2E_BACKEND must be 'gpu' or 'npu'")

    existing = os.environ.get(env_name)
    if existing:
        print(
            f"[e2e] Using existing model path {Path(existing).expanduser()}",
            flush=True,
        )
        return

    model_path = download_model(DEEPSEEK_V2_LITE_REPO_ID)
    os.environ[env_name] = str(model_path)


def build_runner_command(scenario: str, gsm8k_output_path: Path) -> list[str]:
    backend = _required_env("AFD_E2E_BACKEND")
    if backend == "gpu":
        model = _required_env("AFD_GPU_E2E_MODEL")
        vllm_bin = os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm")
    elif backend == "npu":
        model = _required_env("AFD_NPU_E2E_MODEL")
        vllm_bin = os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm")
    else:
        raise RuntimeError("AFD_E2E_BACKEND must be 'gpu' or 'npu'")

    devices = _devices("AFD_E2E_DEVICES", 3)
    attention_devices = devices[:2]
    ffn_devices = devices[2:]
    if scenario == "baseline-graph":
        attention_devices = attention_devices[:1]

    command = [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        model,
        "--vllm-bin",
        vllm_bin,
        "--device-backend",
        backend,
        "--attention-devices",
        ",".join(attention_devices),
    ]
    if scenario != "baseline-graph":
        command.extend(["--ffn-devices", ",".join(ffn_devices)])
    command.extend(
        [
            "--scenario",
            scenario,
            "--gsm8k-output-path",
            str(gsm8k_output_path),
        ],
    )
    return command


def _run_runner(command: list[str]) -> None:
    handled_signals = (signal.SIGTERM, signal.SIGINT)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
    process: subprocess.Popen | None = None
    received_signal: int | None = None
    forwarded = False

    def forward_received_signal() -> None:
        nonlocal forwarded
        if process is None or received_signal is None or forwarded:
            return
        forwarded = True
        with suppress(ProcessLookupError):
            os.killpg(process.pid, received_signal)
        raise SystemExit(128 + received_signal)

    def forward_cancellation(signum, _frame) -> None:
        nonlocal received_signal
        if received_signal is not None:
            return
        received_signal = signum
        forward_received_signal()

    for signum in handled_signals:
        signal.signal(signum, forward_cancellation)

    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            start_new_session=True,
        )
        forward_received_signal()
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
    finally:
        try:
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=RUNNER_CLEANUP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    try:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    finally:
                        process.wait()
        finally:
            for signum, previous_handler in previous_handlers.items():
                signal.signal(signum, previous_handler)


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> None:
    """Prepare shared E2E assets once for this test module."""
    prepare_e2e_assets()


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIOS)
def test_deepseek_v2_lite(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    _run_runner(command)
