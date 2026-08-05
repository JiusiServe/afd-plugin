# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in NPU E2E smoke test for DeepSeekV2-Lite async CAM.

The smoke limits vLLM memory utilization so the CAM HCCL buffer retains
device-memory headroom. Skipped unless ``AFD_NPU_ASYNC_CAM_E2E_MODEL`` (or the
shared ``AFD_NPU_E2E_MODEL`` fallback) points to a local model path.
The smoke topology uses 4 NPUs:

  Attention: DP=1, TP=2
  FFN:       DP=2, EP=2
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.runner import ASYNC_AFD_CONNECTOR

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "tests" / "e2e" / "runner.py"
CAM_VENDOR_PATH = Path(
    "/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM",
)
CAM_OP_API_PATH = CAM_VENDOR_PATH / "op_api"
CAM_OP_API_LIB_PATH = CAM_OP_API_PATH / "lib"
CAM_ATTENTION_RANKS = 2
CAM_FFN_RANKS = 2
CAM_REQUIRED_NPUS = CAM_ATTENTION_RANKS + CAM_FFN_RANKS
CAM_HCCL_BUFFSIZE = "4096"
CAM_GPU_MEMORY_UTILIZATION = "0.75"
CAM_ACCOUNTING_PROMPT = "\n".join(
    [
        "<|im_start|>system",
        "You are a professional accountant. Answer questions using accounting "
        "knowledge, output only the option letter (A/B/C/D).<|im_end|>",
        "<|im_start|>user",
        "Question: A company's balance sheet as of December 31, 2023 shows:",
        "  Current assets: Cash and equivalents 5 million yuan, Accounts "
        "receivable 8 million yuan, Inventory 6 million yuan",
        "  Non-current assets: Net fixed assets 12 million yuan",
        "  Current liabilities: Short-term loans 4 million yuan, Accounts "
        "payable 3 million yuan",
        "  Non-current liabilities: Long-term loans 9 million yuan",
        "  Owner's equity: Paid-in capital 10 million yuan, Retained earnings ?",
        "Requirement: Calculate the company's Asset-Liability Ratio and Current "
        "Ratio (round to two decimal places).",
        "Options:",
        "A. Asset-Liability Ratio=58.33%, Current Ratio=1.90",
        "B. Asset-Liability Ratio=62.50%, Current Ratio=2.17",
        "C. Asset-Liability Ratio=65.22%, Current Ratio=1.75",
        "D. Asset-Liability Ratio=68.00%, Current Ratio=2.50<|im_end|>",
        "<|im_start|>assistant",
        "",
    ],
)


def _npu_list() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get(
            "AFD_NPU_ASYNC_CAM_E2E_DEVICES",
            "0,1,2,3",
        ).split(",")
        if item.strip()
    ]


def _model_path() -> str:
    model = os.environ.get("AFD_NPU_ASYNC_CAM_E2E_MODEL") or os.environ.get(
        "AFD_NPU_E2E_MODEL",
    )
    if not model:
        pytest.skip("set AFD_NPU_ASYNC_CAM_E2E_MODEL to run async CAM NPU E2E tests")
    return model


def _prepend_env_paths(
    env: dict[str, str],
    name: str,
    *paths: Path,
) -> None:
    existing_paths = [path for path in env.get(name, "").split(os.pathsep) if path]
    ordered_paths = [str(path) for path in paths]
    env[name] = os.pathsep.join(dict.fromkeys([*ordered_paths, *existing_paths]))


def _async_cam_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "1")
    env["HCCL_BUFFSIZE"] = CAM_HCCL_BUFFSIZE
    env.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    env.setdefault("VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL", "1")
    env.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")
    _prepend_env_paths(
        env,
        "LD_LIBRARY_PATH",
        CAM_OP_API_PATH,
        CAM_OP_API_LIB_PATH,
    )
    _prepend_env_paths(
        env,
        "ASCEND_CUSTOM_OPP_PATH",
        CAM_VENDOR_PATH,
    )
    python_paths = [
        str(path)
        for path in (
            Path("/vllm-workspace/vllm"),
            Path("/vllm-workspace/vllm-ascend"),
        )
        if path.exists()
    ]
    if python_paths:
        current_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            os.pathsep.join(python_paths)
            if not current_pythonpath
            else os.pathsep.join([*python_paths, current_pythonpath])
        )
    return env


def _async_cam_extra_config(model_path: str) -> str:
    dynamic_quant = int(
        os.environ.get(
            "AFD_NPU_ASYNC_CAM_E2E_DYNAMIC_QUANT",
            (
                "1"
                if (Path(model_path) / "quant_model_description.json").is_file()
                else "0"
            ),
        ),
    )
    return json.dumps(
        {
            "dynamicQuant": dynamic_quant,
            "async_moe_ubatching": False,
            "async_moe_num_ubatches": 2,
            "async_moe_split": "request",
            "attn_ranks_per_dp": 2,
        },
        separators=(",", ":"),
    )


def _async_cam_common_vllm_args() -> list[str]:
    return [
        "--trust-remote-code",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "8000",
        "--gpu-memory-utilization",
        CAM_GPU_MEMORY_UTILIZATION,
        "--no-enable-prefix-caching",
    ]


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.slow
def test_deepseek_v2_lite_async_cam_attn_dp1tp2_ffn_dp2ep2_smoke():
    npus = _npu_list()
    if len(npus) < CAM_REQUIRED_NPUS:
        pytest.skip(
            f"async CAM smoke requires {CAM_REQUIRED_NPUS} NPUs; got {len(npus)}",
        )

    model_path = _model_path()
    common_vllm_args = [
        f"--common-vllm-arg={argument}" for argument in _async_cam_common_vllm_args()
    ]
    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        model_path,
        "--served-model-name-prefix",
        "cam-async",
        "--vllm-bin",
        os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        "npu",
        "--afd-connector",
        ASYNC_AFD_CONNECTOR,
        "--afd-async",
        "--compute-gate-on-attention",
        "--afd-connector-extra-config",
        _async_cam_extra_config(model_path),
        "--num-attention-ranks",
        str(CAM_ATTENTION_RANKS),
        "--num-ffn-ranks",
        str(CAM_FFN_RANKS),
        "--attention-tp-size",
        "2",
        "--ffn-tp-size",
        "1",
        "--attention-gpus",
        ",".join(npus[:CAM_ATTENTION_RANKS]),
        "--ffn-gpus",
        ",".join(npus[CAM_ATTENTION_RANKS:CAM_REQUIRED_NPUS]),
        "--api-port-base",
        os.environ.get("AFD_NPU_ASYNC_CAM_E2E_API_PORT", "19080"),
        "--afd-port",
        os.environ.get("AFD_NPU_ASYNC_CAM_E2E_AFD_PORT", "6453"),
        "--startup-timeout",
        os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "900"),
        "--prompt",
        CAM_ACCOUNTING_PROMPT,
        "--max-tokens",
        os.environ.get("AFD_NPU_ASYNC_CAM_E2E_MAX_TOKENS", "32"),
        "--temperature",
        "0",
        "--num-requests",
        "1",
        "--request-concurrency",
        "1",
        *common_vllm_args,
    ]

    max_model_len = os.environ.get("AFD_NPU_ASYNC_CAM_E2E_MAX_MODEL_LEN")
    if max_model_len:
        command.extend(
            [
                "--common-vllm-arg=--max-model-len",
                f"--common-vllm-arg={max_model_len}",
            ],
        )

    subprocess.run(command, cwd=REPO_ROOT, env=_async_cam_env(), check=True)
