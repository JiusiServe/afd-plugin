# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Connector contract for AFD Attention/FFN communication."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.metadata import (
    AFDConnectorMetadata,
    AFDRecvOutput,
    DPMetadataLike,
)


class AFDConnectorBase(ABC):
    """Abstract base class for plugin-owned AFD connectors."""

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: object,
        afd_config: AFDConfig,
    ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.vllm_config = vllm_config
        self.afd_config = afd_config

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_afd_connector(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        raise NotImplementedError

    def get_connector_rank(self) -> int:
        return self.rank

    def get_connector_local_rank(self) -> int:
        return self.local_rank

    @abstractmethod
    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
        **kwargs: object,
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def recv_ffn_output(self, handle: object = None, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def recv_attn_output(
        self,
        timeout_ms: int | None = None,
        ubatch_idx: int | None = None,
        **kwargs: object,
    ) -> AFDRecvOutput:
        raise NotImplementedError

    @abstractmethod
    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        metadata: AFDConnectorMetadata,
        **kwargs: object,
    ) -> None:
        raise NotImplementedError

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list: dict[int, DPMetadataLike],
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        raise NotImplementedError

    def send_dp_metadata_list(
        self,
        dp_metadata_list: dict[int, DPMetadataLike],
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        raise NotImplementedError

    def recv_dp_metadata_list(
        self,
        timeout_ms: int | None = None,
    ) -> tuple[dict[int, DPMetadataLike], bool, bool]:
        raise NotImplementedError

    def create_recv_metadata(self, **kwargs: object) -> AFDConnectorMetadata:
        dp_metadata_list = kwargs.get("dp_metadata_list") or {}
        ubatch_idx = int(kwargs.get("ubatch_idx", 0))
        layer_idx = int(kwargs.get("layer_idx", 0))
        seq_lens = kwargs.get("seq_lens")
        if seq_lens is None:
            seq_lens = [_num_tokens_for_stage(dp_metadata_list, ubatch_idx)]
        return AFDConnectorMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=ubatch_idx,
            seq_lens=list(seq_lens),
        )

    def configure_metadata(  # noqa: B027
        self,
        metadata: AFDConnectorMetadata,
        **kwargs: object,
    ) -> None:
        pass

    def update_metadata(
        self,
        metadata: AFDConnectorMetadata,
        recv_output: AFDRecvOutput,
    ) -> None:
        metadata.seq_lens = list(recv_output.metadata.seq_lens)


def _num_tokens_for_stage(
    dp_metadata_list: dict[int, DPMetadataLike],
    stage_idx: int,
) -> int:
    dp_metadata = dp_metadata_list.get(int(stage_idx))
    if dp_metadata is None:
        return 1
    token_counts = dp_metadata.num_tokens_across_dp_cpu
    item = token_counts[0]
    if not isinstance(item, (int, float)):
        item = item.item()
    return max(1, int(item))


__all__ = ["AFDConnectorBase"]
