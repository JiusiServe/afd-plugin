# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD connector namespace."""

from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.factory import AFDConnectorFactory
from afd_plugin.connectors.metadata import (
    AFDConnectorMetadata,
    AFDDPMetadata,
    AFDMetadata,
    AFDRecvOutput,
    AFDSingleDPMetadata,
)

__all__ = [
    "AFDConnectorBase",
    "AFDConnectorFactory",
    "AFDConnectorMetadata",
    "AFDDPMetadata",
    "AFDMetadata",
    "AFDRecvOutput",
    "AFDSingleDPMetadata",
]
