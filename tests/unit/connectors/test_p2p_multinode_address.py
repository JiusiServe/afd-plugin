# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Subgroup rendezvous address selection for multi-node P2P deployments.

Each subgroup's StatelessProcessGroup is served by its first member binding
host:port, so members on other nodes must dial that member's own address.
The connector derives every rank's address from the interface that routes
to the configured rendezvous host. On a single host that address equals the
configured one, which keeps the historical single-host layouts unchanged.
"""

from __future__ import annotations

import socket

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.gpu.p2p import _local_address_toward  # noqa: E402


def test_loopback_host_yields_loopback_address():
    # The recipes use host=127.0.0.1: every rank must keep binding/dialing
    # loopback exactly as before the multi-node change.
    assert _local_address_toward("127.0.0.1", 6239) == "127.0.0.1"


def test_own_lan_address_yields_itself():
    # host = this machine's own routable address (single host, LAN IP in the
    # config): the probe must return that same address.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-3, nothing is sent
        own = probe.getsockname()[0]
    if own.startswith("127."):
        pytest.skip("no routable interface on this machine")
    assert _local_address_toward(own, 6239) == own


def test_probe_sends_nothing_and_needs_no_listener():
    # A closed port on loopback still resolves (UDP connect only sets the
    # route), so init does not depend on the master being up yet.
    assert _local_address_toward("127.0.0.1", 1) == "127.0.0.1"
