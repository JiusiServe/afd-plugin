# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD connector namespace."""

from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.connectors.factory import AFDConnectorFactory
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDDPMetadata,
    AFDF2ATransferPayload,
    AFDForwardContextMetadata,
    AFDSingleDPMetadata,
    AFDTransferMetadata,
    AFDTransferState,
)

__all__ = [
    "AFDConnectorBase",
    "ConnectorExtraInfo",
    "AFDTransferState",
    "AFDConnectorFactory",
    "AFDTransferMetadata",
    "AFDDPMetadata",
    "AFDControlPayload",
    "AFDF2ATransferPayload",
    "AFDForwardContextMetadata",
    "AFDA2FTransferPayload",
    "AFDSingleDPMetadata",
]
