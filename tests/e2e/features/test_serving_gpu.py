# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""OpenAI-compatible API serving tests for GPU AFD 1A1F topology."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from tests.e2e.conftest import AFDServer

# ---------------------------------------------------------------------------
# Basic completions
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
def test_completions_basic(afd_server_1a1f: AFDServer) -> None:
    """Basic /v1/completions returns a well-formed response."""
    body = afd_server_1a1f.request_completion(
        "Hello, world",
        max_tokens=16,
        temperature=0.0,
    )

    assert "choices" in body
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert "text" in choice
    assert isinstance(choice["text"], str)
    assert len(choice["text"]) > 0
    assert choice["finish_reason"] in ("stop", "length")


@pytest.mark.gpu
@pytest.mark.e2e
def test_completions_usage_stats(afd_server_1a1f: AFDServer) -> None:
    """Verify usage field reports prompt and completion tokens."""
    body = afd_server_1a1f.request_completion(
        "The capital of France is",
        max_tokens=8,
        temperature=0.0,
    )

    assert "usage" in body
    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.gpu
@pytest.mark.e2e
def test_completions_invalid_model(afd_server_1a1f: AFDServer) -> None:
    """Request with wrong model name returns an error."""
    url = f"{afd_server_1a1f.base_url}/v1/completions"
    payload = json.dumps(
        {
            "model": "nonexistent-model",
            "prompt": "test",
            "max_tokens": 1,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(request, timeout=30)
