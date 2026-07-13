# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU-specific AFD connector implementations."""

from afd_plugin.connectors.npu.camp2p import (
    CAMP2PAFDConnectorMetadata,
    CAMP2pConnector,
    build_camp2p_topology,
)

__all__ = [
    "CAMP2pConnector",
    "CAMP2PAFDConnectorMetadata",
    "build_camp2p_topology",
]
