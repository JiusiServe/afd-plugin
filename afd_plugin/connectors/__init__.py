# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD connector namespace."""

from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.factory import AFDConnectorFactory
from afd_plugin.connectors.metadata import (
    AFDConnectorLike,
    AFDConnectorMetadata,
    AFDDPMetadata,
    AFDMetadata,
    AFDRecvOutput,
    AFDSingleDPMetadata,
    DPMetadataLike,
    WorkHandleLike,
)

__all__ = [
    "AFDConnectorBase",
    "AFDConnectorFactory",
    "AFDConnectorLike",
    "AFDConnectorMetadata",
    "AFDDPMetadata",
    "AFDMetadata",
    "AFDRecvOutput",
    "AFDSingleDPMetadata",
    "DPMetadataLike",
    "WorkHandleLike",
]
