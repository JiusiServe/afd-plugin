# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD connector namespace."""

from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.factory import AFDConnectorFactory
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDTransferState,
    AFDTransferMetadata,
    AFDDPMetadata,
    AFDControlPayload,
    AFDF2ATransferPayload,
    AFDMetadata,
    AFDSingleDPMetadata,
)

__all__ = [
    "AFDConnectorBase",
    "AFDTransferState",
    "AFDConnectorFactory",
    "AFDTransferMetadata",
    "AFDDPMetadata",
    "AFDControlPayload",
    "AFDF2ATransferPayload",
    "AFDMetadata",
    "AFDA2FTransferPayload",
    "AFDSingleDPMetadata",
]
