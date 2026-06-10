# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""ACL graph (Ascend CUDA graph equivalent) correctness test for NPU AFD.

Verifies that cuda-graph FULL_DECODE_ONLY decoding produces output identical to
eager decoding on a 1A1F topology.
"""

from __future__ import annotations

import os

import pytest

from tests.e2e.conftest import AFDServer, _launch_afd_server

PROMPT = "The answer to life, the universe, and everything is"
MAX_TOKENS = 16


@pytest.mark.npu
@pytest.mark.e2e
def test_graph_output_matches_eager_1a1f(
    npu_available: bool,
    npu_e2e_model: str,
    npu_attn_device: str,
    npu_ffn_device: str,
    npu_vllm_bin: str,
) -> None:
    """ACL graph (FULL_DECODE_ONLY) output must match eager output."""
    eager_api_port = int(os.environ.get("AFD_NPU_ACL_EAGER_PORT", "19700"))
    graph_api_port = int(os.environ.get("AFD_NPU_ACL_GRAPH_PORT", "19701"))
    eager_afd_port = int(os.environ.get("AFD_NPU_ACL_EAGER_AFD_PORT", "6393"))
    graph_afd_port = int(os.environ.get("AFD_NPU_ACL_GRAPH_AFD_PORT", "6394"))

    eager_server: AFDServer | None = None
    graph_server: AFDServer | None = None

    try:
        # Phase 1: Eager mode
        eager_server = _launch_afd_server(
            backend="npu",
            model=npu_e2e_model,
            vllm_bin=npu_vllm_bin,
            attention_devices=[npu_attn_device],
            ffn_devices=[npu_ffn_device],
            api_port_base=eager_api_port,
            afd_port=eager_afd_port,
        )
        eager_body = eager_server.request_completion(
            PROMPT,
            max_tokens=MAX_TOKENS,
        )
        eager_text = eager_body["choices"][0]["text"]
        eager_server.shutdown()
        eager_server = None

        # Phase 2: ACL graph mode (FULL_DECODE_ONLY). Request real graph mode via
        # cuda_graph_full_decode_only=True — NOT enforce_eager=False (which is a
        # no-op in _launch_afd_server and would silently run eager).
        graph_server = _launch_afd_server(
            backend="npu",
            model=npu_e2e_model,
            vllm_bin=npu_vllm_bin,
            attention_devices=[npu_attn_device],
            ffn_devices=[npu_ffn_device],
            api_port_base=graph_api_port,
            afd_port=graph_afd_port,
            cuda_graph_full_decode_only=True,
            cudagraph_capture_size=8,
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
                f"ACL graph output {i} diverges from eager:\n"
                f"  eager: {eager_text!r}\n"
                f"  graph: {graph_text!r}"
            )

    finally:
        if eager_server is not None:
            eager_server.shutdown()
        if graph_server is not None:
            graph_server.shutdown()
