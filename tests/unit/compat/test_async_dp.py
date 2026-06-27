from __future__ import annotations

from types import SimpleNamespace

from afd_plugin.compat.async_dp import (
    ensure_async_dp_compat_attr,
    is_afd_async_attention_dp,
    is_afd_async_dp,
    parallel_config_async_dp,
)


def _config(*, connector: str = "afdasyncconnector", role: str = "attention"):
    return SimpleNamespace(
        additional_config={
            "afd": {
                "enabled": True,
                "connector": connector,
                "role": role,
            },
        },
        parallel_config=SimpleNamespace(),
    )


def test_async_dp_helper_detects_async_connector():
    config = _config()

    assert is_afd_async_dp(config) is True
    assert is_afd_async_attention_dp(config) is True
    assert ensure_async_dp_compat_attr(config) is True
    assert parallel_config_async_dp(config.parallel_config) is True


def test_async_dp_helper_rejects_non_async_connector():
    config = _config(connector="camp2pconnector")

    assert is_afd_async_dp(config) is False
    assert ensure_async_dp_compat_attr(config) is False
    assert parallel_config_async_dp(config.parallel_config) is False


def test_async_dp_helper_distinguishes_ffn_role():
    config = _config(role="ffn")

    assert is_afd_async_dp(config) is True
    assert is_afd_async_attention_dp(config) is False
