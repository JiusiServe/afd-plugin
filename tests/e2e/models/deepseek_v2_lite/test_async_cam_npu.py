# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in NPU E2E smoke test for DeepSeekV2-Lite async CAM.

Skipped unless AFD_NPU_E2E_MODEL is set to a local DeepSeekV2-Lite model path.
The smoke topology uses 4 NPUs:

  Attention: DP=1, TP=2
  FFN:       DP=2, EP=2
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "tests" / "e2e" / "runner.py"
ASYNC_CAM_EXTRA_CONFIG = (
    '{"quant_mode":0,"dynamicQuant":0,"async_moe_ubatching":false,'
    '"async_moe_num_ubatches":2,"async_moe_split":"request",'
    '"attn_ranks_per_dp":2}'
)
DSV2_ACCOUNTING_PROMPT = "\n".join(
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
        for item in os.environ.get("AFD_NPU_ASYNC_CAM_E2E_DEVICES", "0,1,2,3").split(
            ",",
        )
        if item.strip()
    ]


def _model_path() -> str:
    model = os.environ.get("AFD_NPU_E2E_MODEL")
    if not model:
        pytest.skip("set AFD_NPU_E2E_MODEL to run async CAM NPU E2E tests")
    return model


def _async_cam_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("HCCL_BUFFSIZE", "6144")
    env.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    env.setdefault("VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL", "1")
    env.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")
    cam_op_lib = Path("/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib")
    if cam_op_lib.exists():
        current_ld_library_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            str(cam_op_lib)
            if not current_ld_library_path
            else f"{cam_op_lib}:{current_ld_library_path}"
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


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.slow
def test_deepseek_v2_lite_async_cam_attn_dp1tp2_ffn_dp2ep2_smoke():
    npus = _npu_list()
    if len(npus) < 4:
        pytest.skip(f"async CAM smoke requires 4 NPUs; got {len(npus)}")

    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        _model_path(),
        "--vllm-bin",
        os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        "npu",
        "--afd-connector",
        "afdasyncconnector",
        "--afd-async",
        "--compute-gate-on-attention",
        "--afd-extra-config",
        ASYNC_CAM_EXTRA_CONFIG,
        "--num-attention-ranks",
        "2",
        "--num-ffn-ranks",
        "2",
        "--attention-tp-size",
        "2",
        "--ffn-tp-size",
        "1",
        "--attention-gpus",
        ",".join(npus[:2]),
        "--ffn-gpus",
        ",".join(npus[2:4]),
        "--api-port-base",
        os.environ.get("AFD_NPU_ASYNC_CAM_E2E_API_PORT", "19080"),
        "--afd-port",
        os.environ.get("AFD_NPU_ASYNC_CAM_E2E_AFD_PORT", "6453"),
        "--startup-timeout",
        os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "900"),
        "--prompt",
        DSV2_ACCOUNTING_PROMPT,
        "--max-tokens",
        os.environ.get("AFD_NPU_ASYNC_CAM_E2E_MAX_TOKENS", "32"),
        "--temperature",
        "0",
        "--num-requests",
        "1",
        "--request-concurrency",
        "1",
        "--common-vllm-arg=--trust-remote-code",
        "--common-vllm-arg=--max-num-seqs",
        "--common-vllm-arg=8",
        "--common-vllm-arg=--max-num-batched-tokens",
        "--common-vllm-arg=8000",
        "--common-vllm-arg=--no-enable-prefix-caching",
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
