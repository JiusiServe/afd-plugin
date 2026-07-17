from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from afd_plugin.connectors import (
    AFDA2FTransferPayload,
    AFDConnectorBase,
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferMetadata,
)


def test_dummy_connector_is_not_registered():
    with pytest.raises(ValueError, match="unsupported AFD connector type"):
        AFDConnectorFactory.get_connector_class("dummy")


def test_p2p_extra_info_rejects_unknown_fields():
    source = SimpleNamespace(
        additional_config={
            "afd": {
                "role": "attention",
                "connector_extra_config": {"core_num": 8},
            },
        },
    )

    with pytest.raises(ValueError, match="does not support connector_extra_config"):
        AFDConnectorFactory.parse_connector_extra_info(
            "P2pNcclAFDConnector",
            source,
        )


def test_backend_connector_modules_are_registered_by_backend_package():
    pytest.importorskip("vllm")
    pytest.importorskip("torch_npu")

    assert (
        AFDConnectorFactory.get_connector_class("P2pNcclAFDConnector").__module__
        == "afd_plugin.connectors.gpu.p2p"
    )
    assert (
        AFDConnectorFactory.get_connector_class("CAMP2pAFDConnector").__module__
        == "afd_plugin.connectors.npu.camp2p"
    )


def test_connector_metadata_validates_sequence_lengths():
    with pytest.raises(ValueError, match="sequence lengths"):
        AFDTransferMetadata(
            layer_idx=0,
            stage_idx=0,
            seq_lens=[0],
        )


def test_attn_output_carries_connector_payload_fields():
    metadata = AFDTransferMetadata.create_ffn_metadata(
        layer_idx=1,
        stage_idx=2,
        seq_lens=[3],
    )
    output = AFDA2FTransferPayload(
        hidden_states="hidden",
        metadata=metadata,
        topk_ids="ids",
        cam_p2p_ep_name="ep",
    )

    assert output.hidden_states == "hidden"
    assert output.metadata is metadata
    assert output.topk_ids == "ids"
    assert output.cam_p2p_ep_name == "ep"
    assert repr(output).startswith("AFDA2FTransferPayload(")


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

    def recv_ffn_output(self, **kwargs):
        return None

    def recv_attn_output(self, ubatch_idx=None):
        return AFDA2FTransferPayload(
            hidden_states=None,
            metadata=AFDTransferMetadata.create_ffn_metadata(
                layer_idx=0,
                stage_idx=0,
                seq_lens=[1],
            ),
        )

    def send_ffn_output(self, ffn_output, metadata):
        return None

    def update_state_from_dp_metadata(self, payload):
        return None

    def send_dp_metadata_list(self, payload):
        return None

    def recv_dp_metadata_list(self):
        return AFDControlPayload(
            dp_metadata_list={
                0: AFDDPMetadata(
                    _cpu_token_tensor([1]),
                ),
            },
            is_graph_capturing=False,
            is_warmup=False,
        )


def _cpu_token_tensor(values: list[int]):
    import torch

    return torch.tensor(values, dtype=torch.int32, device="cpu")
