# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pinned instance-scoped attention metadata builder fanout for NPU uBatch."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from afd_plugin.v1.worker.npu.ubatch_utils import split_attn_metadata

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import torch

    from afd_plugin.v1.worker.npu.attention_model_runner import (
        AFDNPUAttentionModelRunner,
    )


@dataclass
class _FanoutMetadata:
    stages: list[Any]

    @property
    def mm_prefix_range(self) -> Any:
        return self.stages[0].mm_prefix_range

    @mm_prefix_range.setter
    def mm_prefix_range(self, value: Any) -> None:
        for stage_metadata in self.stages:
            stage_metadata.mm_prefix_range = value


class _FanoutMetadataBuilder:
    """Temporary builder that preserves each real stage builder's identity."""

    def __init__(
        self,
        builders: list[Any],
        ubatch_slices: UBatchSlices,
        max_num_tokens: int,
    ) -> None:
        self.builders = builders
        self.ubatch_slices = ubatch_slices
        self.max_num_tokens = max_num_tokens

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        **extra_attn_metadata_args: Any,
    ) -> _FanoutMetadata:
        stage_common_metadata = split_attn_metadata(
            self.ubatch_slices,
            common_attn_metadata,
            self.max_num_tokens,
        )
        stage_metadata = []
        for stage, builder in enumerate(self.builders):
            common = stage_common_metadata[stage]
            stage_metadata.append(
                builder.build(
                    common_prefix_len=common_prefix_len,
                    common_attn_metadata=common,
                    **extra_attn_metadata_args,
                )
            )
        return _FanoutMetadata(stage_metadata)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> _FanoutMetadata:
        stage_common_metadata = split_attn_metadata(
            self.ubatch_slices,
            common_attn_metadata,
            self.max_num_tokens,
        )
        stage_metadata = []
        for stage, builder in enumerate(self.builders):
            common = stage_common_metadata[stage]
            stage_metadata.append(builder.build_for_cudagraph_capture(common))
        return _FanoutMetadata(stage_metadata)


def _validate_builders(builders: list[Any]) -> None:
    supported_builder_types = (
        AscendAttentionMetadataBuilder,
        AscendMLAMetadataBuilder,
    )
    for builder in builders:
        if not isinstance(builder, supported_builder_types):
            raise NotImplementedError(
                "NPU uBatch metadata fanout has not validated builder "
                f"{type(builder).__module__}.{type(builder).__name__}"
            )


@contextmanager
def _intercept_metadata_builders(
    runner: AFDNPUAttentionModelRunner,
    ubatch_slices: UBatchSlices,
    max_num_tokens: int,
) -> Iterator[None]:
    intercepted: list[tuple[Any, Any]] = []
    try:
        for attn_groups in runner.attn_groups:
            for attn_group in attn_groups:
                stage_builders = [
                    attn_group.get_metadata_builder(stage)
                    for stage in range(len(ubatch_slices))
                ]
                _validate_builders(stage_builders)
                original_builder = attn_group.metadata_builders[0]
                attn_group.metadata_builders[0] = _FanoutMetadataBuilder(
                    stage_builders,
                    ubatch_slices,
                    max_num_tokens,
                )
                intercepted.append((attn_group, original_builder))
        yield
    finally:
        for attn_group, original_builder in intercepted:
            attn_group.metadata_builders[0] = original_builder


def _transpose_metadata(
    metadata_by_layer: dict[str, _FanoutMetadata],
    num_stages: int,
) -> list[dict[str, Any]]:
    metadata_by_stage: list[dict[str, Any]] = [dict() for _ in range(num_stages)]
    for layer_name, fanout_metadata in metadata_by_layer.items():
        if not isinstance(fanout_metadata, _FanoutMetadata):
            raise TypeError(
                f"layer {layer_name} bypassed the NPU uBatch metadata fanout"
            )
        for stage, stage_metadata in enumerate(fanout_metadata.stages):
            metadata_by_stage[stage][layer_name] = stage_metadata
    return metadata_by_stage


class AttentionMetadataFanoutV0191rc1:
    """Reuse the pinned upstream common-metadata path and fan out builders."""

    # Upstream source: vllm_ascend/worker/model_runner_v1.py.
    # Upstream ref: v0.19.1rc1 (da421afad7192dac64e39ae1d32305d57344f3cf).
    # Patch reason: pinned upstream accepts slices but never fans out builders.
    # Patch functionality: intercept builders on this runner instance only.
    # Signature: mirrors the upstream metadata hook after the explicit runner
    # parameter required by this unbound adapter.
    @staticmethod
    def build(
        runner: AFDNPUAttentionModelRunner,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        num_scheduled_tokens_np: np.ndarray | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
    ) -> tuple[list[dict[str, Any]], CommonAttentionMetadata | None]:
        # ### PATCH START: per-stage Ascend metadata builder fanout
        if ubatch_slices is None:
            raise ValueError("NPU uBatch metadata fanout requires stage slices")
        max_num_tokens = num_tokens_padded or num_tokens
        with _intercept_metadata_builders(
            runner,
            ubatch_slices,
            max_num_tokens,
        ):
            metadata_by_layer, spec_decode_common = (
                NPUModelRunner._build_attention_metadata(
                    runner,
                    num_tokens,
                    num_reqs,
                    max_query_len,
                    num_tokens_padded,
                    num_reqs_padded,
                    None,
                    logits_indices,
                    use_spec_decode,
                    for_cudagraph_capture,
                    num_scheduled_tokens,
                    num_scheduled_tokens_np,
                    cascade_attn_prefix_lens,
                )
            )
        # ### PATCH END: per-stage Ascend metadata builder fanout
        return (
            _transpose_metadata(metadata_by_layer, len(ubatch_slices)),
            spec_decode_common,
        )


__all__ = ["AttentionMetadataFanoutV0191rc1"]
