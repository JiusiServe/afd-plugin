from __future__ import annotations

import pytest

pytest.importorskip("torch")

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import (
    AFDConnectorBase,
    AFDConnectorFactory,
    AFDConnectorMetadata,
    AFDDPMetadata,
    AFDRecvOutput,
)


def test_dummy_connector_is_not_registered():
    with pytest.raises(ValueError, match="unsupported AFD connector type"):
        AFDConnectorFactory.get_connector_class("dummy")


def test_backend_connector_modules_are_registered_by_backend_package():
    pytest.importorskip("vllm")
    pytest.importorskip("torch_npu")

    assert (
        AFDConnectorFactory.get_connector_class("p2pconnector").__module__
        == "afd_plugin.connectors.gpu.p2p"
    )
    assert (
        AFDConnectorFactory.get_connector_class("camp2pconnector").__module__
        == "afd_plugin.connectors.npu.camp2p"
    )


def test_connector_metadata_validates_sequence_lengths():
    with pytest.raises(ValueError, match="sequence lengths"):
        AFDConnectorMetadata(
            layer_idx=0,
            stage_idx=0,
            seq_lens=[0],
        )


def test_recv_output_carries_connector_payload_fields():
    metadata = AFDConnectorMetadata.create_ffn_metadata(
        layer_idx=1,
        stage_idx=2,
        seq_lens=[3],
    )
    output = AFDRecvOutput(
        hidden_states="hidden",
        metadata=metadata,
        topk_ids="ids",
        cam_p2p_ep_name="ep",
    )

    assert output.hidden_states == "hidden"
    assert output.metadata is metadata
    assert output.topk_ids == "ids"
    assert output.cam_p2p_ep_name == "ep"


def test_connector_base_builds_default_recv_metadata_from_dp_metadata():
    connector = _MinimalConnector(
        0,
        0,
        object(),
        AFDConfig(enabled=True, connector="camp2pconnector"),
    )

    metadata = connector.create_recv_metadata(
        dp_metadata_list={1: AFDDPMetadata([4])},
        ubatch_idx=1,
        layer_idx=3,
    )

    assert metadata.layer_idx == 3
    assert metadata.stage_idx == 1
    assert metadata.seq_lens == [4]


class _MinimalConnector(AFDConnectorBase):
    @property
    def is_initialized(self):
        return True

    def close(self):
        return None

    def init_afd_connector(self):
        return None

    def send_attn_output(self, hidden_states, metadata):
        return None

    def recv_ffn_output(self, handle=None, **kwargs):
        return None

    def recv_attn_output(self, timeout_ms=None, ubatch_idx=None):
        return AFDRecvOutput(
            hidden_states=None,
            metadata=AFDConnectorMetadata.create_ffn_metadata(
                layer_idx=0,
                stage_idx=0,
                seq_lens=[1],
            ),
        )

    def send_ffn_output(self, ffn_output, metadata):
        return None
