# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in CUDA Qwen3.5-122B-A10B E2E coverage."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import download_dataset, run_runner
from tests.e2e.runner import AFD_EAGER_4A4F_SCENARIO, BASELINE_EAGER_SCENARIO

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
LARGE_MODEL_OPT_IN_ENV = "AFD_E2E_LARGE_MODEL"
DEVICE_COUNT = 8
ROLE_DEVICE_COUNT = 4
SCENARIOS = (BASELINE_EAGER_SCENARIO, AFD_EAGER_4A4F_SCENARIO)
COMMON_VLLM_ARGS = (
    "--dtype=bfloat16",
    "--language-model-only",
    "--max-model-len=4096",
    "--max-num-seqs=1",
    "--max-num-batched-tokens=4096",
    "--mamba-cache-mode=align",
    "--all2all-backend=allgather_reducescatter",
    "--seed=0",
)
CONTROLLED_ROUTING_ENV_VARS = (
    "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
    "AFD_BENCHMARK_FORCE_LB_TOPN_PER_RANK",
)
FLASHINFER_SAMPLER_ENV = "VLLM_USE_FLASHINFER_SAMPLER"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def _devices() -> list[str]:
    raw_devices = _required_env("AFD_E2E_DEVICES")
    devices = [item.strip() for item in raw_devices.split(",") if item.strip()]
    if len(devices) != DEVICE_COUNT:
        raise RuntimeError(
            f"AFD_E2E_DEVICES must contain exactly {DEVICE_COUNT} devices",
        )
    if len(devices) != len(set(devices)):
        raise RuntimeError("AFD_E2E_DEVICES must contain unique devices")
    return devices


def prepare_e2e_assets() -> None:
    """Validate the explicit large-model contract, then cache GSM8K."""
    if _required_env("AFD_E2E_BACKEND") != "gpu":
        raise RuntimeError("Qwen3.5-122B E2E supports only the 'gpu' backend")
    if _required_env(LARGE_MODEL_OPT_IN_ENV) != "1":
        raise RuntimeError(f"{LARGE_MODEL_OPT_IN_ENV} must be set to 1")
    model_path = Path(_required_env("AFD_GPU_E2E_MODEL")).expanduser()
    if not model_path.is_dir():
        raise RuntimeError(
            f"AFD_GPU_E2E_MODEL must be an existing directory: {model_path}"
        )
    _devices()
    download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)


def build_runner_command(scenario: str, gsm8k_output_path: Path) -> list[str]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported Qwen3.5-122B scenario: {scenario}")
    if _required_env("AFD_E2E_BACKEND") != "gpu":
        raise RuntimeError("Qwen3.5-122B E2E supports only the 'gpu' backend")

    devices = _devices()
    command = [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        _required_env("AFD_GPU_E2E_MODEL"),
        "--vllm-bin",
        os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        "gpu",
        "--attention-devices",
        ",".join(devices[:ROLE_DEVICE_COUNT]),
    ]
    command.extend(f"--common-vllm-arg={arg}" for arg in COMMON_VLLM_ARGS)
    if scenario == AFD_EAGER_4A4F_SCENARIO:
        command.extend(
            ["--ffn-devices", ",".join(devices[ROLE_DEVICE_COUNT:])],
        )
    command.extend(
        [
            "--scenario",
            scenario,
            "--gsm8k-output-path",
            str(gsm8k_output_path),
            "--served-model-name-prefix",
            "qwen3-5-122b-afd",
        ],
    )
    return command


def natural_routing_env() -> dict[str, str]:
    """Return the natural-routing environment validated by this profile."""
    env = os.environ.copy()
    for name in CONTROLLED_ROUTING_ENV_VARS:
        env.pop(name, None)
    # vLLM 0.26.0's FlashInfer sampler rejects Blackwell SM12 during device
    # capability detection. Greedy GSM8K does not require that sampler.
    env[FLASHINFER_SAMPLER_ENV] = "0"
    return env


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> Iterator[None]:
    prepare_e2e_assets()
    yield


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIOS)
def test_qwen3_5_122b(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_runner(command, env=natural_routing_env())
