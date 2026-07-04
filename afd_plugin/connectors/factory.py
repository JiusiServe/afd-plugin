# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Factory for plugin-owned AFD connectors."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors.base import AFDConnectorBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class AFDConnectorFactory:
    _registry: dict[str, Callable[[], type[AFDConnectorBase]]] = {}

    @classmethod
    def register_connector(
        cls,
        name: str,
        module_path: str,
        class_name: str,
        *,
        replace: bool = False,
    ) -> None:
        if name in cls._registry and not replace:
            raise ValueError(f"connector {name!r} is already registered")

        def loader() -> type[AFDConnectorBase]:
            module = importlib.import_module(module_path)
            connector_cls = vars(module)[class_name]
            if not issubclass(connector_cls, AFDConnectorBase):
                raise TypeError(
                    f"{module_path}.{class_name} is not an AFDConnectorBase",
                )
            return connector_cls

        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig | None = None,
    ) -> AFDConnectorBase:
        config = afd_config or parse_afd_config(vllm_config)
        if config.connector not in cls._registry:
            raise ValueError(f"unsupported AFD connector type: {config.connector}")
        connector_cls = cls._registry[config.connector]()
        return connector_cls(rank, local_rank, vllm_config, config)

    @classmethod
    def get_connector_class(cls, connector_name: str) -> type[AFDConnectorBase]:
        if connector_name not in cls._registry:
            raise ValueError(f"unsupported AFD connector type: {connector_name}")
        return cls._registry[connector_name]()


AFDConnectorFactory.register_connector(
    "p2pconnector",
    "afd_plugin.connectors.gpu.p2p",
    "P2PAFDConnector",
)
AFDConnectorFactory.register_connector(
    "camp2pconnector",
    "afd_plugin.connectors.npu.camp2p",
    "CAMP2PAFDConnector",
)


__all__ = ["AFDConnectorFactory"]
