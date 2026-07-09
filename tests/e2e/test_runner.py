from __future__ import annotations

import argparse
import io
import json
import urllib.error

import pytest

from tests.e2e import runner
from tests.e2e.runner import build_vllm_command


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model="/models/DeepSeek-V2-Lite",
        vllm_bin="vllm",
        num_attention_ranks=2,
        num_ffn_ranks=2,
        api_host="127.0.0.1",
        api_port_base=18100,
        afd_host="127.0.0.1",
        afd_port=6249,
        startup_timeout=900,
        served_model_name_prefix="deepseek-v2-lite-afd",
        prompt="San Francisco is a",
        max_tokens=16,
        temperature=0.0,
        num_requests=None,
        request_concurrency=None,
        cuda_graph_full_decode_only=False,
        cudagraph_capture_size=64,
        enable_dbo=False,
        dbo_decode_token_threshold=1,
        dbo_prefill_token_threshold=None,
        use_decode_bench_connector=False,
        common_vllm_arg=["--trust-remote-code"],
        attention_vllm_arg=[],
        ffn_vllm_arg=[],
        tp_size=1,
        attention_tp_size=None,
        ffn_tp_size=None,
        afd_connector=None,
        afd_async=False,
        compute_gate_on_attention=False,
        afd_extra_config=[],
        device_backend="gpu",
    )


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_runner_uses_native_dp_for_attention_topology():
    command = build_vllm_command(_args(), role="attention")

    assert command.count("serve") == 1
    assert _arg_value(command, "--data-parallel-size") == "2"
    assert _arg_value(command, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in command
    assert _arg_value(command, "--worker-cls") == (
        "afd_plugin.v1.worker.AFDAttentionWorker"
    )
    assert _arg_value(command, "--port") == "18100"
    assert _arg_value(command, "--served-model-name") == (
        "deepseek-v2-lite-afd-attention"
    )

    additional_config = json.loads(_arg_value(command, "--additional-config"))
    assert additional_config["afd"]["num_attention_ranks"] == 2
    assert additional_config["afd"]["num_ffn_ranks"] == 2
    assert "extra_config" not in additional_config["afd"]
    assert "afd_role_rank" not in additional_config["afd"]


def test_runner_uses_native_dp_for_ffn_topology():
    command = build_vllm_command(_args(), role="ffn")

    assert _arg_value(command, "--data-parallel-size") == "2"
    assert _arg_value(command, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in command
    assert _arg_value(command, "--worker-cls") == "afd_plugin.v1.worker.AFDFFNWorker"
    assert _arg_value(command, "--port") == "18101"
    assert _arg_value(command, "--served-model-name") == "deepseek-v2-lite-afd-ffn"


def test_runner_uses_plugin_decode_bench_connector():
    args = _args()
    args.use_decode_bench_connector = True

    command = build_vllm_command(args, role="attention")
    kv_transfer_config = json.loads(_arg_value(command, "--kv-transfer-config"))

    assert kv_transfer_config["kv_connector"] == "AFDDecodeBenchConnector"
    assert kv_transfer_config["kv_connector_module_path"] == (
        "tools.benchmarks.decode_bench"
    )


def test_runner_builds_npu_async_cam_role_specific_topology():
    args = _args()
    args.device_backend = "npu"
    args.num_attention_ranks = 2
    args.num_ffn_ranks = 2
    args.attention_tp_size = 2
    args.ffn_tp_size = 1
    args.afd_connector = runner.ASYNC_AFD_CONNECTOR
    args.afd_async = True
    args.compute_gate_on_attention = True
    args.afd_extra_config = [
        (
            '{"quant_mode":0,"dynamicQuant":0,'
            '"async_moe_ubatching":false,"async_moe_num_ubatches":2,'
            '"async_moe_split":"request","attn_ranks_per_dp":2}'
        ),
    ]

    attention_command = build_vllm_command(args, role="attention")
    ffn_command = build_vllm_command(args, role="ffn")

    assert _arg_value(attention_command, "--data-parallel-size") == "1"
    assert _arg_value(attention_command, "--tensor-parallel-size") == "2"
    assert _arg_value(ffn_command, "--data-parallel-size") == "2"
    assert _arg_value(ffn_command, "--tensor-parallel-size") == "1"
    assert _arg_value(attention_command, "--worker-cls") == (
        "afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker"
    )
    assert _arg_value(ffn_command, "--worker-cls") == (
        "afd_plugin.v1.worker.ascend.AFDNPUFFNWorker"
    )

    additional_config = json.loads(_arg_value(attention_command, "--additional-config"))
    afd_config = additional_config["afd"]
    assert afd_config["connector"] == runner.ASYNC_AFD_CONNECTOR
    assert afd_config["async"] is True
    assert afd_config["compute_gate_on_attention"] is True
    assert afd_config["num_attention_ranks"] == 2
    assert afd_config["num_ffn_ranks"] == 2
    assert afd_config["extra_config"]["attn_ranks_per_dp"] == 2
    assert afd_config["extra_config"]["async_moe_split"] == "request"


def test_runner_rejects_role_rank_count_not_divisible_by_tp():
    args = _args()
    args.attention_tp_size = 3

    with pytest.raises(ValueError, match="attention rank count"):
        runner.validate_topology(args, ["0", "1"], ["2", "3"])


def test_runner_drops_flashcomm_for_npu_role_without_tp(monkeypatch):
    args = _args()
    args.device_backend = "npu"
    args.ffn_tp_size = 1
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")

    env = runner.build_env("2,3", args, role="ffn")

    assert "VLLM_ASCEND_ENABLE_FLASHCOMM1" not in env


def test_runner_fails_fast_when_server_exits_before_api_is_ready():
    args = _args()
    process = argparse.Namespace(
        args=["vllm", "serve"],
        poll=lambda: 1,
    )

    with pytest.raises(RuntimeError, match="exited before Attention API was ready"):
        runner.wait_for_openai_api(args, [process])


def test_runner_sends_one_concurrent_request_per_attention_dp_rank(monkeypatch):
    args = _args()
    calls = []

    def fake_request_completion(_args):
        calls.append(_args)
        return {"id": len(calls)}

    monkeypatch.setattr(runner, "request_completion", fake_request_completion)

    responses = runner.request_completions(args)

    assert len(calls) == 2
    assert len(responses) == 2


def test_request_completion_includes_http_error_body(monkeypatch):
    args = _args()
    error = urllib.error.HTTPError(
        url="http://127.0.0.1:18100/v1/completions",
        code=500,
        msg="Internal Server Error",
        hdrs=None,
        fp=io.BytesIO(b'{"error":"CUDA out of memory"}'),
    )

    def fake_urlopen(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        runner.request_completion(args)


def test_runner_keeps_successful_concurrent_responses(monkeypatch, capsys):
    args = _args()
    args.num_requests = 3
    calls = []

    def fake_request_completion(_args):
        calls.append(_args)
        if len(calls) == 2:
            raise RuntimeError("transient request failure")
        return {"id": len(calls)}

    monkeypatch.setattr(runner, "request_completion", fake_request_completion)

    responses = runner.request_completions(args)

    assert responses == [{"id": 1}, {"id": 3}]
    assert "transient request failure" in capsys.readouterr().err
