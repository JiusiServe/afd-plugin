# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Backend-neutral DeepSeek-V2-Lite E2E scenarios."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.conftest import download_dataset, download_model, run_runner

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
DEEPSEEK_V2_LITE_REPO_ID = "deepseek-ai/DeepSeek-V2-Lite"
DEEPSEEK_V2_LITE_MAX_MODEL_LEN = 4096
DEFAULT_DEVICE_IDS = ("0", "1", "2", "3")
ATTENTION_DEVICE_COUNT = 2
AFD_FFN_DEVICE_COUNT = 1
BASELINE_DEVICE_COUNT = 4
SCENARIOS = (
    "baseline-graph",
    "afd-eager",
    "afd-graph",
    "afd-graph-dbo",
    "afd-graph-dbo-2a2f",
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


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

    env_devices = os.environ.get("AFD_E2E_DEVICES")
    devices = (
        [item.strip() for item in env_devices.split(",") if item.strip()]
        if env_devices
        else list(DEFAULT_DEVICE_IDS)
    )
    if scenario == "baseline-graph":
        attention_devices = devices[:BASELINE_DEVICE_COUNT]
        ffn_devices = []
    else:
        attention_devices = devices[:ATTENTION_DEVICE_COUNT]
        ffn_end = (
            BASELINE_DEVICE_COUNT
            if scenario == "afd-graph-dbo-2a2f"
            else ATTENTION_DEVICE_COUNT + AFD_FFN_DEVICE_COUNT
        )
        ffn_devices = devices[ATTENTION_DEVICE_COUNT:ffn_end]

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
        f"--common-vllm-arg=--max-model-len={DEEPSEEK_V2_LITE_MAX_MODEL_LEN}",
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


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> None:
    """Prepare shared E2E assets once for this test module."""
    prepare_e2e_assets()


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIOS)
def test_deepseek_v2_lite(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_runner(command)
