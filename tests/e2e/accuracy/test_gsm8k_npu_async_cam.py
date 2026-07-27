# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GSM8K accuracy evaluation for CAMAsyncAFDConnector on Ascend NPU.

Mirrors ``test_gsm8k_npu.py`` but targets the asynchronous CAM connector.
Covers both a minimal 1A1F smoke-like accuracy baseline and the 8-NPU
non-PCP DP+TP/SP topology used to validate ``async_moe_split="token"``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.e2e.conftest import AFDServer, _launch_afd_server
from tests.e2e.helpers_gsm8k import (
    _extract_gsm8k_accuracy,
    _run_lm_eval,
)


def _default_tasks_dir() -> Path:
    """Resolve the bundled offline gsm8k task dir relative to this repo.

    Mirrors the helper in ``test_gsm8k_npu.py``.
    """
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root.parent / "lm-evaluation-harness" / "lm_eval" / "tasks" / "gsm8k"


def _npu_list() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get(
            "AFD_NPU_ASYNC_CAM_E2E_DEVICES", "0,1,2,3,4,5,6,7"
        ).split(",")
        if item.strip()
    ]


def _run_gsm8k_async_cam(
    *,
    split: str,
    npu_e2e_model: str,
    npu_vllm_bin: str,
    tmp_path: Path,
    attention_devices: list[str],
    ffn_devices: list[str],
    attention_tp_size: int,
    ffn_tp_size: int,
    attn_ranks_per_dp: int,
    api_port_base: int,
    afd_port: int,
    batch_size: int = 1,
) -> None:
    """Run GSM8K via lm-eval against a CAMAsync server."""
    pytest.importorskip("lm_eval", reason="lm-eval not installed")

    tasks_dir = os.environ.get("AFD_NPU_GSM8K_TASK_DIR") or str(_default_tasks_dir())
    if not Path(tasks_dir).is_dir():
        pytest.skip(
            f"offline gsm8k task dir not found: {tasks_dir}. "
            "Stage gsm8k parquet + gsm8k.yaml, or set AFD_NPU_GSM8K_TASK_DIR.",
        )

    threshold = float(os.environ.get("AFD_GSM8K_THRESHOLD", "0.20"))
    tolerance = float(os.environ.get("AFD_GSM8K_TOLERANCE", "0.05"))
    _limit = os.environ.get("AFD_GSM8K_LIMIT")
    limit = int(_limit) if _limit else None
    _env_batch_size = os.environ.get("AFD_GSM8K_BATCH_SIZE")
    effective_batch_size = (
        int(_env_batch_size) if _env_batch_size is not None else batch_size
    )

    connector_extra_config = json.dumps(
        {
            "dynamicQuant": 0,
            "async_moe_ubatching": True,
            "async_moe_num_ubatches": 2,
            "async_moe_split": split,
            "attn_ranks_per_dp": attn_ranks_per_dp,
        },
        separators=(",", ":"),
    )

    common_vllm_args = [
        "--trust-remote-code",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "8000",
        "--no-enable-prefix-caching",
    ]
    max_model_len = os.environ.get("AFD_NPU_ASYNC_CAM_E2E_MAX_MODEL_LEN")
    if max_model_len:
        common_vllm_args.extend(["--max-model-len", max_model_len])

    afd_server: AFDServer | None = None
    try:
        afd_server = _launch_afd_server(
            model=npu_e2e_model,
            backend="npu",
            vllm_bin=npu_vllm_bin,
            connector="CAMAsyncAFDConnector",
            attention_devices=attention_devices,
            ffn_devices=ffn_devices,
            attention_tp_size=attention_tp_size,
            ffn_tp_size=ffn_tp_size,
            afd_async=True,
            compute_gate_on_attention=True,
            afd_connector_extra_config=[connector_extra_config],
            api_port_base=api_port_base,
            afd_port=afd_port,
            startup_timeout=float(os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "900")),
            served_model_name_prefix="deepseek-v2-lite-afd",
            common_vllm_args=common_vllm_args,
        )

        results = _run_lm_eval(
            base_url=afd_server.base_url,
            model_name=afd_server.served_model,
            output_path=str(tmp_path / f"lm_eval_output_{split}"),
            tokenizer=npu_e2e_model,
            tasks_dir=tasks_dir,
            limit=limit,
            batch_size=effective_batch_size,
        )

        accuracy = _extract_gsm8k_accuracy(results)
        effective_threshold = threshold - tolerance

        print(
            f"\n[GSM8K NPU async CAM split={split}] "
            f"accuracy={accuracy:.4f} threshold={threshold} "
            f"tolerance={tolerance} effective_min={effective_threshold:.4f} "
            f"batch_size={effective_batch_size}"
        )

        assert accuracy >= effective_threshold, (
            f"GSM8K async CAM ({split=}) accuracy {accuracy:.4f} < "
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
def test_gsm8k_lm_eval_async_cam_1a1f(
    npu_available: bool,
    npu_e2e_model: str,
    npu_vllm_bin: str,
    tmp_path: Path,
) -> None:
    """GSM8K accuracy for CAMAsync 1A1F baseline."""
    npus = _npu_list()
    if len(npus) < 2:
        pytest.skip(f"async CAM 1A1F GSM8K test requires 2 NPUs; got {len(npus)}")

    _run_gsm8k_async_cam(
        split="request",
        npu_e2e_model=npu_e2e_model,
        npu_vllm_bin=npu_vllm_bin,
        tmp_path=tmp_path,
        attention_devices=npus[:1],
        ffn_devices=npus[1:2],
        attention_tp_size=1,
        ffn_tp_size=1,
        attn_ranks_per_dp=1,
        api_port_base=int(os.environ.get("AFD_NPU_ASYNC_CAM_1A1F_API_PORT", "19090")),
        afd_port=int(os.environ.get("AFD_NPU_ASYNC_CAM_1A1F_AFD_PORT", "6463")),
    )


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.eval
@pytest.mark.slow
@pytest.mark.parametrize("split", ["request", "token"])
@pytest.mark.skipif(
    os.environ.get("AFD_NPU_ASYNC_CAM_RUN_DP3TP2") != "1",
    reason=(
        "DP3TP2+EP2 GSM8K test is opt-in; set AFD_NPU_ASYNC_CAM_RUN_DP3TP2=1 to enable"
    ),
)
def test_gsm8k_lm_eval_async_cam_dp3tp2_ep2(
    npu_available: bool,
    npu_e2e_model: str,
    npu_vllm_bin: str,
    tmp_path: Path,
    split: str,
) -> None:
    """GSM8K accuracy for CAMAsync DP3TP2+EP2 with request/token split."""
    npus = _npu_list()
    if len(npus) < 8:
        pytest.skip(f"async CAM DP3TP2+EP2 GSM8K test requires 8 NPUs; got {len(npus)}")

    _run_gsm8k_async_cam(
        split=split,
        npu_e2e_model=npu_e2e_model,
        npu_vllm_bin=npu_vllm_bin,
        tmp_path=tmp_path,
        attention_devices=npus[:6],
        ffn_devices=npus[6:8],
        attention_tp_size=2,
        ffn_tp_size=1,
        attn_ranks_per_dp=2,
        api_port_base=int(os.environ.get("AFD_NPU_ASYNC_CAM_E2E_API_PORT", "19080")),
        afd_port=int(os.environ.get("AFD_NPU_ASYNC_CAM_E2E_AFD_PORT", "6453")),
        batch_size=8,
    )
