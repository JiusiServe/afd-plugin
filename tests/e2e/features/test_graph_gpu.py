# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA graph correctness test for GPU AFD.

Verifies that cuda-graph FULL_DECODE_ONLY decoding produces output identical to
eager decoding on a 1A1F topology.
"""

from __future__ import annotations

import os

import pytest

from tests.e2e.conftest import AFDServer, _launch_afd_server

PROMPT = "The answer to life, the universe, and everything is"
MAX_TOKENS = 16


@pytest.mark.gpu
@pytest.mark.e2e
def test_graph_output_matches_eager_1a1f(
    afd_e2e_model: str,
    afd_gpu_list: list[str],
    afd_vllm_bin: str,
) -> None:
    """CUDA graph (FULL_DECODE_ONLY) output must match eager output."""
    if len(afd_gpu_list) < 2:
        pytest.skip("1A1F requires at least 2 GPUs")

    eager_api_port = int(os.environ.get("AFD_GRAPH_EAGER_PORT", "19700"))
    graph_api_port = int(os.environ.get("AFD_GRAPH_CG_PORT", "19701"))
    eager_afd_port = int(os.environ.get("AFD_GRAPH_EAGER_AFD_PORT", "6393"))
    graph_afd_port = int(os.environ.get("AFD_GRAPH_CG_AFD_PORT", "6394"))

    eager_server: AFDServer | None = None
    graph_server: AFDServer | None = None

    try:
        # Phase 1: Eager mode
        eager_server = _launch_afd_server(
            backend="gpu",
            model=afd_e2e_model,
            vllm_bin=afd_vllm_bin,
            attention_devices=afd_gpu_list[:1],
            ffn_devices=afd_gpu_list[1:2],
            api_port_base=eager_api_port,
            afd_port=eager_afd_port,
            enforce_eager=True,
            common_vllm_args=["--trust-remote-code"],
        )
        eager_body = eager_server.request_completion(
            PROMPT,
            max_tokens=MAX_TOKENS,
        )
        eager_text = eager_body["choices"][0]["text"]
        eager_server.shutdown()
        eager_server = None

        # Phase 2: CUDA graph mode (FULL_DECODE_ONLY)
        graph_server = _launch_afd_server(
            backend="gpu",
            model=afd_e2e_model,
            vllm_bin=afd_vllm_bin,
            attention_devices=afd_gpu_list[:1],
            ffn_devices=afd_gpu_list[1:2],
            api_port_base=graph_api_port,
            afd_port=graph_afd_port,
            cuda_graph_full_decode_only=True,
            cudagraph_capture_size=8,
            common_vllm_args=["--trust-remote-code"],
        )

        # Send enough requests to trigger graph replay
        graph_texts: list[str] = []
        for _ in range(3):
            body = graph_server.request_completion(
                PROMPT,
                max_tokens=MAX_TOKENS,
            )
            graph_texts.append(body["choices"][0]["text"])

        for i, graph_text in enumerate(graph_texts):
            assert graph_text == eager_text, (
                f"CUDA graph output {i} diverges from eager:\n"
                f"  eager: {eager_text!r}\n"
                f"  graph: {graph_text!r}"
            )

    finally:
        if eager_server is not None:
            eager_server.shutdown()
        if graph_server is not None:
            graph_server.shutdown()
