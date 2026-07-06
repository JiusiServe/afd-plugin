# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Connector contract for AFD Attention/FFN communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.metadata import (
    AFDConnectorMetadata,
    AFDDPMetadata,
    AFDDPMetadataPayload,
    AFDRecvOutput,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import DPMetadata


class AFDConnectorBase(ABC):
    """Base class for plugin-owned AFD Attention/FFN connectors.

    An AFD connector owns the communication contract between the Attention
    runtime and the FFN runtime. Implementations are responsible for three
    kinds of work:

    1. Initializing and releasing backend communication resources.
    2. Moving hidden states between Attention and FFN ranks.
    3. Applying DP metadata control-plane payloads so later tensor transfers
       can use backend-specific shapes, buffers, or connector data.

    Implementations may use very different transport mechanisms. For example,
    the GPU P2P connector uses explicit point-to-point tensor transfers, while
    the NPU CAMP2P connector carries additional backend state through
    ``AFDConnectorMetadata.connector_data``. This base class documents the
    common runtime contract shared by those implementations.
    """

    uses_dp_metadata_control_plane = True
    ffn_step_trigger = "dp_metadata"

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
    ) -> None:
        """Initialize common connector context.

        Args:
            rank: The rank value passed by the owning vLLM worker at connector
                construction time. This is the process rank known to the caller
                and is not necessarily the same as a connector-specific
                ``world_rank`` computed by a backend topology.
            local_rank: Device-local rank for selecting the local accelerator.
                GPU implementations generally use this as the CUDA device
                index; NPU implementations use it as the NPU device index.
            vllm_config: Upstream vLLM ``VllmConfig`` for the current worker.
                Connectors use it to read model shape, dtype, parallelism, and
                graph/capture configuration.
            afd_config: Parsed AFD plugin configuration. This contains the AFD
                role, connector name, topology sizes, host/port, and
                connector-specific extra config.
        """
        self.rank = rank
        self.local_rank = local_rank
        self.vllm_config = vllm_config
        self.afd_config = afd_config

    # ==============================
    # Lifecycle methods
    # ==============================

    @abstractmethod
    def close(self) -> None:
        """Release connector-owned communication resources.

        Implementations should destroy process groups, communicators, cached
        handles, and backend-specific state owned by the connector. The method
        should be safe to call during worker shutdown and should leave
        ``is_initialized`` false after successful cleanup.
        """
        raise NotImplementedError

    @abstractmethod
    def init_afd_connector(self) -> None:
        """Initialize backend communication resources.

        This method is called once the owning runtime is ready to create AFD
        communication state. Implementations typically create process groups,
        register custom ops, initialize backend communicators, and precompute
        topology-derived rank lists.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Return whether backend communication resources are initialized."""
        raise NotImplementedError

    # ==============================
    # Attention-side data path
    # ==============================

    @abstractmethod
    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
        **kwargs: Any,
    ) -> None:
        """Send Attention hidden states to the FFN runtime.

        This method is called on Attention-side ranks after an Attention block
        produces hidden states for one layer/stage.

        Args:
            hidden_states: Tensor to send to the FFN side. The leading
                dimension should match ``metadata.total_tokens`` unless the
                backend explicitly supports a different compiled/captured
                calling convention.
            metadata: Per-transfer metadata describing layer, stage, and token
                layout. Backends may also consume ``metadata.connector_data``.
            **kwargs: Optional backend-specific arguments.

        Raises:
            ValueError: If tensor shape does not match metadata.
            RuntimeError: If the connector is not initialized or required
                backend-specific metadata is missing.
        """
        raise NotImplementedError

    @abstractmethod
    def recv_ffn_output(self, **kwargs: Any) -> torch.Tensor:
        """Receive FFN output on the Attention side.

        This method returns the tensor that should continue through the
        Attention-side model execution after FFN computation completes.

        Args:
            **kwargs: Optional backend-specific receive arguments.

        Returns:
            The FFN output tensor for the current layer/stage.

        Raises:
            RuntimeError: If required backend state or arguments are missing.
        """
        raise NotImplementedError

    # ==============================
    # FFN-side data path
    # ==============================

    @abstractmethod
    def recv_attn_output(
        self,
        ubatch_idx: int | None = None,
        **kwargs: Any,
    ) -> AFDRecvOutput:
        """Receive Attention hidden states on the FFN side.

        This method is called by FFN-side execution before running the FFN
        compute for a layer/stage. It returns both the hidden states and the
        metadata needed by later FFN-side calls.

        Args:
            ubatch_idx: Optional microbatch/stage index to receive. ``None``
                means the backend should use its default stage, usually stage
                ``0``.
            **kwargs: Optional backend-specific receive arguments.

        Returns:
            ``AFDRecvOutput`` containing received hidden states, transfer
            metadata, and optional backend-specific fields produced by the
            receive operation.
        """
        raise NotImplementedError

    @abstractmethod
    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        metadata: AFDConnectorMetadata,
        **kwargs: Any,
    ) -> None:
        """Send FFN output back to the Attention runtime.

        Args:
            ffn_output: Tensor produced by the FFN computation. The leading
                dimension should match ``metadata.total_tokens`` unless the
                backend explicitly supports a different compiled/captured
                calling convention.
            metadata: Metadata associated with the matching
                ``recv_attn_output`` call. Backends may depend on
                ``metadata.connector_data`` that was prepared before receive or
                updated after receive.
            **kwargs: Optional backend-specific send arguments such as
                ``ubatch_idx`` or op-specific metadata.

        Raises:
            ValueError: If tensor shape does not match metadata.
            RuntimeError: If required backend state or connector data is
                missing.
        """
        raise NotImplementedError

    # ==============================
    # DP metadata control plane
    # ==============================

    @abstractmethod
    def update_state_from_dp_metadata(
        self,
        payload: AFDDPMetadataPayload,
    ) -> None:
        """Apply a DP metadata payload to local connector state.

        This is a local state update, not a network send. Implementations use it
        to store DP metadata and flags, derive tensor metadata, and prepare
        backend buffers or connector-specific state before data-path calls.

        Args:
            payload: Structured DP metadata control-plane payload. It contains
                per-stage DP token metadata plus graph-capturing and warmup
                flags.
        """
        raise NotImplementedError

    @abstractmethod
    def send_dp_metadata_list(
        self,
        payload: AFDDPMetadataPayload,
    ) -> None:
        """Submit DP metadata control-plane payload from Attention to FFN.

        The caller does not need to decide whether the current rank is a sender.
        Implementations should encapsulate their topology-specific sender-rank
        checks and no-op on ranks that should not transmit DP metadata.

        Args:
            payload: Structured DP metadata payload to make available to the
                FFN-side connector loop.
        """
        raise NotImplementedError

    @abstractmethod
    def recv_dp_metadata_list(
        self,
        timeout_ms: int | None = None,
    ) -> AFDDPMetadataPayload:
        """Receive a DP metadata control-plane payload on the FFN side.

        Args:
            timeout_ms: Optional timeout in milliseconds. Current backends may
                still use blocking communication primitives; callers should not
                assume all implementations enforce this timeout precisely.

        Returns:
            Structured DP metadata payload received from the Attention side.

        Raises:
            TimeoutError: If an implementation supports timeout and no payload
                arrives before the deadline.
            RuntimeError: If the control-plane communication group is not
                initialized or unsupported by this connector.
        """
        raise NotImplementedError

    # ==============================
    # Metadata helpers
    # ==============================

    def create_recv_metadata(self, **kwargs: Any) -> AFDConnectorMetadata:
        """Create metadata for an FFN-side receive operation.

        This helper is used before ``recv_attn_output`` when the runner needs a
        metadata object to describe the expected receive. Backends may override
        it to compute role-specific batch sizes or attach connector-specific
        data.

        Args:
            **kwargs: Optional metadata inputs. The base implementation
                recognizes:

                * ``dp_metadata_list``: Mapping from stage index to DP metadata.
                * ``ubatch_idx``: Stage/microbatch index. Defaults to ``0``.
                * ``layer_idx``: Layer index. Defaults to ``0``.
                * ``seq_lens``: Explicit sequence lengths. If omitted, the base
                  implementation derives one length from ``dp_metadata_list``.

        Returns:
            ``AFDConnectorMetadata`` for the receive operation.
        """
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
        **kwargs: Any,
    ) -> None:
        """Attach or update backend-specific data on existing metadata.

        The base implementation is a no-op. Backends override this when they
        need to populate ``metadata.connector_data`` before a send or receive
        path consumes it.

        Args:
            metadata: Metadata object to mutate in place.
            **kwargs: Backend-specific values used to build connector data, such
                as ``batch_size`` or layer/op parameters.
        """
        pass

    def update_metadata(
        self,
        metadata: AFDConnectorMetadata,
        recv_output: AFDRecvOutput,
    ) -> None:
        """Update metadata after an Attention-to-FFN receive completes.

        This hook lets a backend copy runtime values from ``recv_output`` into
        ``metadata`` or ``metadata.connector_data`` so that the following
        ``send_ffn_output`` call has the backend state it needs.

        Args:
            metadata: Metadata object associated with the receive. It is mutated
                in place.
            recv_output: Payload returned by ``recv_attn_output``.
        """
        metadata.seq_lens = list(recv_output.metadata.seq_lens)


def _num_tokens_for_stage(
    dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
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
