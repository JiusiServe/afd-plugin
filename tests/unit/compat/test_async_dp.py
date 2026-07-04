from __future__ import annotations

from types import SimpleNamespace

from afd_plugin.compat.async_dp import (
    is_afd_async_attention_dp,
    is_afd_async_dp,
)


def _config(
    *,
    connector: str = "afdasyncconnector",
    role: str = "attention",
    async_dp: bool = True,
):
    return SimpleNamespace(
        additional_config={
            "afd": {
                "enabled": True,
                "connector": connector,
                "role": role,
                "async": async_dp,
            },
        },
        parallel_config=SimpleNamespace(),
    )


def test_async_dp_helper_detects_async_connector():
    config = _config()

    assert is_afd_async_dp(config) is True
    assert is_afd_async_attention_dp(config) is True


def test_async_dp_helper_rejects_missing_async_flag():
    config = _config(async_dp=False)

    assert is_afd_async_dp(config) is False
    assert is_afd_async_attention_dp(config) is False


def test_async_dp_helper_rejects_non_async_connector():
    config = _config(connector="camp2pconnector", async_dp=False)

    assert is_afd_async_dp(config) is False


def test_async_dp_helper_distinguishes_ffn_role():
    config = _config(role="ffn")

    assert is_afd_async_dp(config) is True
    assert is_afd_async_attention_dp(config) is False
