# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD connector namespace."""

from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.factory import AFDConnectorFactory
from afd_plugin.connectors.metadata import (
    AFDAttnOutput,
    AFDConnectorData,
    AFDConnectorMetadata,
    AFDDPMetadata,
    AFDDPMetadataPayload,
    AFDFFNOutput,
    AFDMetadata,
    AFDSingleDPMetadata,
)

__all__ = [
    "AFDConnectorBase",
    "AFDConnectorData",
    "AFDConnectorFactory",
    "AFDConnectorMetadata",
    "AFDDPMetadata",
    "AFDDPMetadataPayload",
    "AFDFFNOutput",
    "AFDMetadata",
    "AFDAttnOutput",
    "AFDSingleDPMetadata",
]
