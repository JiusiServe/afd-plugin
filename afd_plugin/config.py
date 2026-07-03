# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD configuration parsed from vLLM ``additional_config``."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal

AFD_ADDITIONAL_CONFIG_KEY: Final[str] = "afd"
AFDRole = Literal["attention", "ffn"]

SUPPORTED_AFD_ROLES: Final[tuple[str, ...]] = ("attention", "ffn")
SUPPORTED_AFD_CONNECTORS: Final[tuple[str, ...]] = (
    "p2pconnector",
    "camp2pconnector",
)

_ALIASES: Final[dict[str, str]] = {
    "afd_connector": "connector",
    "afd_role": "role",
    "afd_port": "port",
    "afd_host": "host",
    "afd_extra_config": "extra_config",
}


@dataclass(frozen=True)
class AFDConfig:
    """Plugin-owned AFD configuration.

    The plugin uses shorter keys inside ``additional_config["afd"]`` while
    preserving read-only compatibility aliases for legacy ``afd_*`` field
    names.
    """

    enabled: bool = False
    extra_config: dict[str, Any] = field(default_factory=dict)
    connector: str = "p2pconnector"
    role: AFDRole = "attention"
    port: int = 1239
    host: str = "127.0.0.1"
    num_afd_stages: int = 3
    num_attention_servers: int = 1
    num_ffn_servers: int = 1
    afd_server_rank: int = 0
    compute_gate_on_attention: bool = False

    @property
    def afd_extra_config(self) -> dict[str, Any]:
        return self.extra_config

    @property
    def afd_connector(self) -> str:
        return self.connector

    @property
    def afd_role(self) -> AFDRole:
        return self.role

    @property
    def afd_port(self) -> int:
        return self.port

    @property
    def afd_host(self) -> str:
        return self.host

    @property
    def is_attention_server(self) -> bool:
        return self.role == "attention"

    @property
    def is_ffn_server(self) -> bool:
        return self.role == "ffn"

    def compute_hash(self) -> str:
        """Return a stable hash for graph-affecting AFD settings."""

        factors: list[Any] = [
            self.enabled,
            self.connector,
            self.role,
            self.num_afd_stages,
            self.num_attention_servers,
            self.num_ffn_servers,
        ]
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def validate(self, *, expected_role: str | None = None) -> None:
        validate_afd_config(self, expected_role=expected_role)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"{field_name} must be a boolean, got {value!r}")


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer, got {value!r}") from exc


def _normalize_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    valid_fields = set(AFDConfig.__dataclass_fields__)  # type: ignore[attr-defined]

    for key, value in raw.items():
        normalized_key = _ALIASES.get(key, key)
        if normalized_key not in valid_fields:
            raise ValueError(
                "unknown AFD config field "
                f"{key!r}; put connector-specific values under 'extra_config'",
            )
        if normalized_key in normalized:
            raise ValueError(
                f"duplicate AFD config field for {normalized_key!r}: "
                f"both alias and canonical key were provided",
            )
        normalized[normalized_key] = value

    if "enabled" in normalized:
        normalized["enabled"] = _coerce_bool(
            normalized["enabled"],
            field_name="enabled",
        )

    for field_name in (
        "port",
        "num_afd_stages",
        "num_attention_servers",
        "num_ffn_servers",
        "afd_server_rank",
    ):
        if field_name in normalized:
            normalized[field_name] = _coerce_int(
                normalized[field_name],
                field_name=field_name,
            )

    if "extra_config" in normalized and not isinstance(
        normalized["extra_config"],
        dict,
    ):
        raise TypeError("extra_config must be a dict")

    return normalized


def afd_config_from_mapping(
    raw: Mapping[str, Any] | None,
    *,
    validate: bool = True,
    expected_role: str | None = None,
) -> AFDConfig:
    if raw is None:
        config = AFDConfig()
    elif not isinstance(raw, Mapping):
        raise TypeError(
            f"AFD config must be a mapping, got {type(raw).__name__}",
        )
    else:
        config = AFDConfig(**_normalize_mapping(raw))

    if validate:
        config.validate(expected_role=expected_role)
    return config


def parse_afd_config(
    source: Mapping[str, Any] | object | None,
    *,
    validate: bool = True,
    expected_role: str | None = None,
) -> AFDConfig:
    """Parse ``additional_config["afd"]`` or a vLLM-like config object."""

    if source is None:
        return afd_config_from_mapping(
            None,
            validate=validate,
            expected_role=expected_role,
        )

    additional_config = (
        source if isinstance(source, Mapping) else source.additional_config
    )
    if additional_config is None:
        afd_raw = None
    elif not isinstance(additional_config, Mapping):
        raise TypeError(
            "additional_config must be a mapping when parsing AFD config, "
            f"got {type(additional_config).__name__}",
        )
    else:
        afd_raw = additional_config.get(AFD_ADDITIONAL_CONFIG_KEY)

    return afd_config_from_mapping(
        afd_raw,
        validate=validate,
        expected_role=expected_role,
    )


def validate_afd_config(
    config: AFDConfig,
    *,
    expected_role: str | None = None,
) -> None:
    """Validate AFD config values without importing vLLM or CUDA modules."""

    if config.role not in SUPPORTED_AFD_ROLES:
        raise ValueError(
            f"AFD role must be one of {SUPPORTED_AFD_ROLES!r}, got {config.role!r}",
        )
    if expected_role is not None and config.role != expected_role:
        raise ValueError(
            f"AFD role mismatch: expected {expected_role!r}, got {config.role!r}",
        )
    if config.connector not in SUPPORTED_AFD_CONNECTORS:
        raise ValueError(
            "AFD connector must be one of "
            f"{SUPPORTED_AFD_CONNECTORS!r}, got {config.connector!r}",
        )
    p2p_sizes: tuple[int, int] | None = None
    if config.connector == "p2pconnector":
        from afd_plugin.distributed import (
            topology_from_config,
            validate_p2p_topology,
        )

        validate_p2p_topology(config)
        p2p_sizes = topology_from_config(config)
    if not config.host:
        raise ValueError("AFD host must be non-empty")
    if not 0 < config.port < 65536:
        raise ValueError(f"AFD port must be in 1..65535, got {config.port}")
    if config.num_afd_stages <= 0:
        raise ValueError(
            f"num_afd_stages must be positive, got {config.num_afd_stages}",
        )
    if config.num_attention_servers <= 0:
        raise ValueError(
            "num_attention_servers must be positive, "
            f"got {config.num_attention_servers}",
        )
    if config.num_ffn_servers <= 0:
        raise ValueError(
            f"num_ffn_servers must be positive, got {config.num_ffn_servers}",
        )

    if config.role == "attention":
        server_count = p2p_sizes[0] if p2p_sizes else config.num_attention_servers
    else:
        server_count = p2p_sizes[1] if p2p_sizes else config.num_ffn_servers
    if not 0 <= config.afd_server_rank < server_count:
        raise ValueError(
            "afd_server_rank must be within this role's server count "
            f"(rank={config.afd_server_rank}, count={server_count})",
        )


__all__ = [
    "AFDConfig",
    "afd_config_from_mapping",
    "AFD_ADDITIONAL_CONFIG_KEY",
    "AFDRole",
    "SUPPORTED_AFD_CONNECTORS",
    "SUPPORTED_AFD_ROLES",
    "parse_afd_config",
    "validate_afd_config",
]
