#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run manual DeepSeekV2 AFD end-to-end smoke tests.

This script is intentionally opt-in and is not collected by pytest. It starts
one FFN-side ``vllm serve`` process and one Attention-side ``vllm serve``
OpenAI-compatible API process. XAYF topologies are represented as native vLLM
data parallelism: Attention runs with ``DP=X, TP=1`` and FFN runs with
``DP=Y, TP=1``. By default the runner keeps the eager baseline behavior; CUDA
graph and DBO coverage are opt-in flags.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    args = parse_args()
    attention_gpus = parse_csv(args.attention_gpus)
    ffn_gpus = parse_csv(args.ffn_gpus)
    validate_topology(args, attention_gpus, ffn_gpus)

    processes: list[subprocess.Popen[str]] = []
    log_threads: list[threading.Thread] = []

    try:
        ffn_cmd = build_vllm_command(args, role="ffn")
        ffn_cuda_visible_devices = ",".join(ffn_gpus)
        print_command("FFN", ffn_cmd, ffn_cuda_visible_devices)
        ffn_proc = start_process(
            "ffn",
            ffn_cmd,
            build_env(ffn_cuda_visible_devices, args),
        )
        processes.append(ffn_proc)
        log_threads.append(stream_output("ffn", ffn_proc))

        ensure_alive(ffn_proc, "FFN process exited during startup")

        attention_cmd = build_vllm_command(args, role="attention")
        attention_cuda_visible_devices = ",".join(attention_gpus)
        print_command("ATTN", attention_cmd, attention_cuda_visible_devices)
        attention_proc = start_process(
            "attention",
            attention_cmd,
            build_env(attention_cuda_visible_devices, args),
        )
        processes.append(attention_proc)
        log_threads.append(stream_output("attention", attention_proc))

        ensure_alive(attention_proc, "Attention process exited during startup")
        wait_for_openai_api(args)

        responses = request_completions(args)
        for request_idx, response in enumerate(responses):
            print(f"\n=== Completion response: request {request_idx} ===")
            print(json.dumps(response, ensure_ascii=False, indent=2))

        if args.expect_text:
            for request_idx, response in enumerate(responses):
                choices = response.get("choices", [])
                assert choices, f"request {request_idx}: no choices in response"
                text = choices[0].get("text", "")
                assert args.expect_text in text, (
                    f"request {request_idx}: expected {args.expect_text!r} in {text!r}"
                )
                print(f"  ✓ request {request_idx}: output contains expected text")

        return 0
    finally:
        terminate_processes(processes)
        for thread in log_threads:
            thread.join(timeout=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manual DeepSeekV2 AFD E2E smoke test.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="DeepSeekV2-Lite model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--vllm-bin",
        default="vllm",
        help="vLLM executable to run. Defaults to 'vllm'.",
    )
    parser.add_argument(
        "--num-attention-ranks",
        dest="num_attention_ranks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-ffn-ranks",
        dest="num_ffn_ranks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--attention-gpus",
        default="0",
        help=(
            "Comma-separated CUDA_VISIBLE_DEVICES values for the Attention "
            "serve process. The number of GPUs must match Attention DP size."
        ),
    )
    parser.add_argument(
        "--ffn-gpus",
        default="1",
        help=(
            "Comma-separated CUDA_VISIBLE_DEVICES values for the FFN serve "
            "process. The number of GPUs must match FFN DP size."
        ),
    )
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port-base", type=int, default=8000)
    parser.add_argument("--afd-host", default="127.0.0.1")
    parser.add_argument("--afd-port", type=int, default=1239)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--prompt", default="San Francisco is a")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help=(
            "Number of completion requests to send. Defaults to the number of "
            "Attention ranks."
        ),
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent completion requests. Defaults to --num-requests.",
    )
    parser.add_argument(
        "--served-model-name-prefix",
        default="deepseek-v2-lite-afd",
        help="Prefix used for role-specific served model names.",
    )
    parser.add_argument(
        "--cuda-graph-full-decode-only",
        action="store_true",
        help="Run without --enforce-eager and set cudagraph_mode=FULL_DECODE_ONLY.",
    )
    parser.add_argument(
        "--cudagraph-capture-size",
        type=int,
        default=64,
        help=(
            "Capture size used for max-num-seqs, max-num-batched-tokens, "
            "max-cudagraph-capture-size, and cudagraph-capture-sizes."
        ),
    )
    parser.add_argument(
        "--enable-dbo",
        action="store_true",
        help="Enable vLLM DBO/ubatching for both AFD roles.",
    )
    parser.add_argument(
        "--dbo-decode-token-threshold",
        type=int,
        default=1,
        help="Value passed to --dbo-decode-token-threshold when DBO is enabled.",
    )
    parser.add_argument(
        "--dbo-prefill-token-threshold",
        type=int,
        default=None,
        help=(
            "Value passed to --dbo-prefill-token-threshold when DBO is enabled. "
            "Defaults to --cudagraph-capture-size."
        ),
    )
    parser.add_argument(
        "--use-decode-bench-connector",
        action="store_true",
        help="Pass an AFDDecodeBenchConnector kv-transfer-config to Attention.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size. Defaults to 1.",
    )
    parser.add_argument(
        "--expect-text",
        default=None,
        help="If set, assert that every completion response contains this text.",
    )
    parser.add_argument(
        "--device-backend",
        choices=["gpu", "npu"],
        default="gpu",
        help="Device backend. 'gpu' uses CUDA workers, 'npu' uses Ascend workers.",
    )
    parser.add_argument(
        "--common-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added to all processes.",
    )
    parser.add_argument(
        "--attention-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added only to Attention processes.",
    )
    parser.add_argument(
        "--ffn-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added only to FFN processes.",
    )
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_topology(
    args: argparse.Namespace,
    attention_gpus: list[str],
    ffn_gpus: list[str],
) -> None:
    if args.num_attention_ranks != len(attention_gpus):
        raise ValueError(
            "--num-attention-ranks must match the number of --attention-gpus",
        )
    if args.num_ffn_ranks != len(ffn_gpus):
        raise ValueError("--num-ffn-ranks must match the number of --ffn-gpus")
    if args.num_attention_ranks < 1 or args.num_ffn_ranks < 1:
        raise ValueError("AFD E2E requires at least one Attention and FFN rank")


def build_vllm_command(
    args: argparse.Namespace,
    *,
    role: str,
) -> list[str]:
    tp_size = args.tp_size
    role_total_ranks = (
        args.num_attention_ranks if role == "attention" else args.num_ffn_ranks
    )
    role_dp_size = max(1, role_total_ranks // tp_size)
    is_npu = args.device_backend == "npu"

    afd_config = {
        "afd": {
            "enabled": True,
            "role": role,
            "connector": "camp2pconnector" if is_npu else "p2pconnector",
            "host": args.afd_host,
            "port": args.afd_port,
            "num_attention_ranks": args.num_attention_ranks,
            "num_ffn_ranks": args.num_ffn_ranks,
        },
    }
    if is_npu:
        worker_cls = (
            "afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker"
            if role == "attention"
            else "afd_plugin.v1.worker.ascend.AFDNPUFFNWorker"
        )
    else:
        worker_cls = (
            "afd_plugin.v1.worker.AFDAttentionWorker"
            if role == "attention"
            else "afd_plugin.v1.worker.AFDFFNWorker"
        )
    cmd = [
        args.vllm_bin,
        "serve",
        args.model,
        "--worker-cls",
        worker_cls,
        "--served-model-name",
        served_model_name(args, role),
        "--data-parallel-size",
        str(role_dp_size),
        "--tensor-parallel-size",
        str(tp_size),
        "--enable-expert-parallel",
        "--additional-config",
        json.dumps(afd_config, separators=(",", ":")),
    ]
    if args.cuda_graph_full_decode_only:
        capture_size = str(args.cudagraph_capture_size)
        cmd.extend(
            [
                "--max-num-seqs",
                capture_size,
                "--max-num-batched-tokens",
                capture_size,
                "--max-cudagraph-capture-size",
                capture_size,
                "--cudagraph-capture-sizes",
                capture_size,
                "--compilation-config",
                json.dumps(
                    {"cudagraph_mode": "FULL_DECODE_ONLY"},
                    separators=(",", ":"),
                ),
            ],
        )
    else:
        cmd.append("--enforce-eager")

    if args.enable_dbo:
        prefill_threshold = (
            args.dbo_prefill_token_threshold
            if args.dbo_prefill_token_threshold is not None
            else args.cudagraph_capture_size
        )
        cmd.extend(
            [
                "--enable-dbo",
                "--dbo-decode-token-threshold",
                str(args.dbo_decode_token_threshold),
                "--dbo-prefill-token-threshold",
                str(prefill_threshold),
            ],
        )

    if role == "attention":
        cmd.extend(
            ["--host", args.api_host, "--port", str(attention_api_port(args))],
        )
        if args.use_decode_bench_connector:
            cmd.extend(["--kv-transfer-config", decode_bench_connector_config()])
        cmd.extend(args.attention_vllm_arg)
    else:
        cmd.extend(
            ["--host", args.api_host, "--port", str(ffn_api_port(args))],
        )
        cmd.extend(args.ffn_vllm_arg)
    cmd.extend(args.common_vllm_arg)
    return cmd


def decode_bench_connector_config() -> str:
    return json.dumps(
        {
            "kv_connector": "AFDDecodeBenchConnector",
            "kv_connector_module_path": "afd_plugin.connectors.decode_bench",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "fill_mean": 0.015,
                "fill_std": 0.0,
            },
        },
        separators=(",", ":"),
    )


def served_model_name(args: argparse.Namespace, role: str) -> str:
    return f"{args.served_model_name_prefix}-{role}"


def attention_api_port(args: argparse.Namespace) -> int:
    return args.api_port_base


def ffn_api_port(args: argparse.Namespace) -> int:
    return args.api_port_base + 1


def build_env(visible_devices: str, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.device_backend == "npu":
        env["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    else:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["VLLM_PLUGINS"] = "ascend,afd" if args.device_backend == "npu" else "afd"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("AFD_PLUGIN_EARLY_ENGINE_PATCH", None)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not current_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{current_pythonpath}"
    )
    return env


def start_process(
    name: str,
    command: list[str],
    env: dict[str, str],
) -> subprocess.Popen[str]:
    del name
    return subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )


def stream_output(
    name: str,
    process: subprocess.Popen[str],
) -> threading.Thread:
    def worker() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{name}] {line}", end="")

    thread = threading.Thread(target=worker, name=f"{name}-log-stream", daemon=True)
    thread.start()
    return thread


def wait_for_openai_api(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.startup_timeout
    url = f"http://{args.api_host}:{attention_api_port(args)}/v1/models"
    last_error: BaseException | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print(f"\nAttention API is ready at {url}")
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(2)

    raise TimeoutError(
        f"Timed out waiting for Attention API at {url}; last error={last_error!r}",
    )


def request_completion(args: argparse.Namespace) -> dict[str, Any]:
    url = f"http://{args.api_host}:{attention_api_port(args)}/v1/completions"
    payload = {
        "model": served_model_name(args, "attention"),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = read_http_error_body(exc)
        raise RuntimeError(
            f"Completion request to {url} failed with HTTP {exc.code} "
            f"{exc.reason}: {body}",
        ) from exc
    return json.loads(body)


def read_http_error_body(error: urllib.error.HTTPError) -> str:
    body = error.read().decode("utf-8", errors="replace")
    return body if body else "<empty response body>"


def request_completions(args: argparse.Namespace) -> list[dict[str, Any]]:
    request_count = (
        int(args.num_requests)
        if args.num_requests is not None
        else max(int(args.num_attention_ranks), 1)
    )
    if request_count == 1:
        return [request_completion(args)]

    responses: list[dict[str, Any] | None] = [None] * request_count
    concurrency = (
        int(args.request_concurrency)
        if args.request_concurrency is not None
        else request_count
    )
    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as executor:
        futures = {
            executor.submit(request_completion, args): request_idx
            for request_idx in range(request_count)
        }
        for future in as_completed(futures):
            request_idx = futures[future]
            try:
                responses[request_idx] = future.result()
            except Exception as exc:
                print(
                    f"Completion request {request_idx} failed: {exc!r}",
                    file=sys.stderr,
                )

    completed_responses = [response for response in responses if response is not None]
    if not completed_responses:
        raise RuntimeError("All completion requests failed")
    return completed_responses


def ensure_alive(process: subprocess.Popen[str], message: str) -> None:
    returncode = process.poll()
    if returncode is not None:
        raise RuntimeError(f"{message} (returncode={returncode})")


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    for process in reversed(processes):
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)


def print_command(name: str, command: list[str], cuda_visible_devices: str) -> None:
    printable = " ".join(shell_quote(token) for token in command)
    print(f"\n=== Starting {name} (CUDA_VISIBLE_DEVICES={cuda_visible_devices}) ===")
    print(printable)


def shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "@%_+=:,./-" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    sys.exit(main())
