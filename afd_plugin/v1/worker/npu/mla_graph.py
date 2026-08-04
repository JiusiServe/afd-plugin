# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Graph-local MLA parameter helpers for Ascend ubatch execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, TypeVar

from afd_plugin.compat.patches.npu.mla_graph import (
    AFD_MLA_GRAPH_PARAMS_KEY,
)

if TYPE_CHECKING:
    import torch
    from vllm.forward_context import ForwardContext
    from vllm_ascend.compilation.acl_graph import GraphParams

MetadataT = TypeVar("MetadataT")


def new_mla_graph_params(
    num_tokens: int,
    workspace: torch.Tensor,
) -> GraphParams:
    """Create an empty MLA registry for one token shape and FIA workspace."""
    # Delay the NPU-only import so this helper remains CPU-import-safe.
    from vllm_ascend.compilation.acl_graph import GraphParams

    return GraphParams(
        events={num_tokens: []},
        workspaces={num_tokens: workspace},
        handles={num_tokens: []},
        attn_params={num_tokens: []},
    )


def merge_mla_graph_params(
    attn_metadata: list[dict[str, MetadataT]],
    graph_params: tuple[GraphParams, GraphParams],
    num_tokens: int,
) -> tuple[dict[tuple[str, int], MetadataT], GraphParams]:
    """Validate and merge two stage-local MLA registries for graph replay."""
    if len(attn_metadata) != 2 or len(graph_params) != 2:
        raise RuntimeError(
            "MLA DBO FULL graph requires exactly two metadata stages; "
            f"got metadata={len(attn_metadata)}, params={len(graph_params)}",
        )

    layer_keys = tuple(attn_metadata[0])
    if tuple(attn_metadata[1]) != layer_keys:
        raise RuntimeError("MLA DBO FULL graph layer order differs by stage")

    expected_records = len(layer_keys)
    for stage_index, params in enumerate(graph_params):
        record_counts = (
            len(params.events[num_tokens]),
            len(params.handles[num_tokens]),
            len(params.attn_params[num_tokens]),
        )
        if record_counts != (
            expected_records,
            expected_records,
            expected_records,
        ):
            raise RuntimeError(
                "MLA DBO FULL graph record count mismatch for "
                f"stage {stage_index}: metadata={expected_records}, "
                f"events={record_counts[0]}, handles={record_counts[1]}, "
                f"params={record_counts[2]}",
            )

    workspace = graph_params[0].workspaces[num_tokens]
    if graph_params[1].workspaces[num_tokens] is not workspace:
        raise RuntimeError(
            "MLA DBO FULL graph requires one shared FIA workspace",
        )

    merged_metadata: dict[tuple[str, int], MetadataT] = {}
    merged = new_mla_graph_params(num_tokens, workspace)

    # Preserve the updater's layer-major, stage-minor record order.
    for layer_index, layer_key in enumerate(layer_keys):
        for stage_index in range(2):
            params = graph_params[stage_index]
            merged_key = (layer_key, stage_index)
            merged_metadata[merged_key] = attn_metadata[stage_index][layer_key]
            merged.events[num_tokens].append(
                params.events[num_tokens][layer_index],
            )
            merged.handles[num_tokens].append(
                params.handles[num_tokens][layer_index],
            )
            merged.attn_params[num_tokens].append(
                params.attn_params[num_tokens][layer_index],
            )

    return merged_metadata, merged


@contextmanager
def override_mla_graph_params(
    forward_context: ForwardContext,
    attn_metadata: dict[tuple[str, int], MetadataT],
    graph_params: GraphParams,
) -> Iterator[None]:
    """Temporarily expose merged MLA state to the upstream graph updater."""
    original_metadata = forward_context.attn_metadata
    original_additional_kwargs = forward_context.additional_kwargs
    # Do not mutate the parent context's shared additional_kwargs mapping.
    temporary_kwargs = dict(original_additional_kwargs or {})
    temporary_kwargs[AFD_MLA_GRAPH_PARAMS_KEY] = graph_params
    forward_context.attn_metadata = attn_metadata
    forward_context.additional_kwargs = temporary_kwargs
    try:
        yield
    finally:
        forward_context.attn_metadata = original_metadata
        forward_context.additional_kwargs = original_additional_kwargs


__all__ = [
    "merge_mla_graph_params",
    "new_mla_graph_params",
    "override_mla_graph_params",
]
