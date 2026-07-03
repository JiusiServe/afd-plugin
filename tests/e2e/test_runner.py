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
        "afd_plugin.connectors.decode_bench"
    )


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
