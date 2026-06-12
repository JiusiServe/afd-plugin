from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.config import (
    AFDConfig,
    afd_config_from_mapping,
    async_moe_num_ubatches,
    async_moe_split,
    async_moe_ubatching_enabled,
    parse_afd_config,
)


def test_parse_empty_additional_config_returns_disabled_default():
    config = parse_afd_config({})

    assert config == AFDConfig()
    assert not config.enabled
    assert config.is_attention_server


def test_parse_canonical_additional_config_namespace():
    config = parse_afd_config(
        {
            "afd": {
                "enabled": True,
                "role": "ffn",
                "connector": "p2pconnector",
                "num_attention_ranks": 2,
                "num_ffn_ranks": 2,
                "afd_role_rank": 1,
            },
        },
        expected_role="ffn",
    )

    assert config.enabled
    assert config.role == "ffn"
    assert config.afd_role == "ffn"
    assert config.is_ffn_server
    assert config.afd_role_rank == 1


def test_parse_vllm_like_config_object():
    vllm_config = SimpleNamespace(
        additional_config={
            "afd": {
                "enabled": True,
                "role": "attention",
                "connector": "p2pconnector",
            },
        },
    )

    config = parse_afd_config(vllm_config, expected_role="attention")

    assert config.enabled
    assert config.is_attention_server


def test_compute_gate_on_attention_can_come_from_extra_config():
    config = parse_afd_config(
        {
            "afd": {
                "enabled": True,
                "role": "ffn",
                "extra_config": {"compute_gate_on_attention": "true"},
            },
        },
        expected_role="ffn",
    )

    assert config.compute_gate_on_attention is True


def test_async_moe_ubatching_helpers_read_extra_config():
    config = parse_afd_config(
        {
            "afd": {
                "enabled": True,
                "connector": "afdasyncconnector",
                "role": "attention",
                "extra_config": {
                    "async_moe_ubatching": "true",
                    "async_moe_num_ubatches": "2",
                    "async_moe_split": "Request",
                    "compute_gate_on_attention": True,
                },
            },
        },
        expected_role="attention",
    )

    assert async_moe_ubatching_enabled(config) is True
    assert async_moe_num_ubatches(config) == 2
    assert async_moe_split(config) == "request"


def test_original_afd_field_aliases_are_supported():
    config = afd_config_from_mapping(
        {
            "enabled": "true",
            "afd_role": "ffn",
            "afd_connector": "p2pconnector",
            "afd_host": "localhost",
            "afd_port": 2345,
            "afd_extra_config": {"rank_map": "env"},
        },
    )

    assert config.role == "ffn"
    assert config.connector == "p2pconnector"
    assert config.afd_host == "localhost"
    assert config.afd_port == 2345
    assert config.afd_extra_config == {"rank_map": "env"}


def test_integer_like_config_values_are_coerced():
    class IntLike:
        def __int__(self) -> int:
            return 2

    config = afd_config_from_mapping(
        {
            "num_attention_ranks": IntLike(),
            "num_ffn_ranks": IntLike(),
            "afd_role_rank": "1",
        },
    )

    assert config.num_attention_ranks == 2
    assert config.num_ffn_ranks == 2
    assert config.afd_role_rank == 1


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"enabled": "maybe"}, "enabled must be a boolean"),
        ({"role": "decode"}, "AFD role must be one of"),
        ({"connector": "tcp"}, "AFD connector must be one of"),
        ({"afd_role_rank": 2, "num_attention_ranks": 2}, "afd_role_rank"),
        ({"num_attention_servers": 2}, "unknown AFD config field"),
        ({"num_ffn_servers": 2}, "unknown AFD config field"),
        ({"afd_server_rank": 0}, "unknown AFD config field"),
        ({"unknown": True}, "unknown AFD config field"),
    ],
)
def test_validation_errors_are_clear(raw, message):
    with pytest.raises((TypeError, ValueError), match=message):
        afd_config_from_mapping(raw)


def test_role_mismatch_fails_fast():
    with pytest.raises(ValueError, match="AFD role mismatch"):
        afd_config_from_mapping(
            {"enabled": True, "role": "ffn"},
            expected_role="attention",
        )


def test_compute_hash_changes_for_graph_affecting_fields():
    attention = AFDConfig(enabled=True, role="attention")
    ffn = AFDConfig(enabled=True, role="ffn")

    assert attention.compute_hash() != ffn.compute_hash()
