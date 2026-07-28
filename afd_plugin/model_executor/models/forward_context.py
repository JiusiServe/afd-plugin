# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Helpers for plugin-owned model wrappers to read AFD forward metadata."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, Final, TypedDict

import vllm.forward_context as forward_context_module
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.worker.ubatch_utils import UBatchSlices

from afd_plugin.connectors import AFDForwardContextMetadata

ASYNC_MOE_UBATCH_METADATA_KEY: Final[str] = "afd_async_moe_ubatch_metadata"


class _AsyncMoeUbatchMetadataOptional(TypedDict, total=False):
    """Optional SP-local fields for ``AsyncMoeUbatchMetadata``.

    A separate ``total=False`` base class is used instead of
    ``typing.NotRequired`` because the plugin supports Python 3.10, where
    ``NotRequired`` is only available via the external ``typing_extensions``
    package. Keeping the base class avoids adding a new runtime dependency.
    """

    # SP-local stage layout: when SP shards the full batch across TP ranks,
    # transpose it into equal per-stage rank shards before attention and
    # restore the original layout after the async MoE pipeline.
    use_sp_stage_resharding: bool
    sp_local_stage_slices: UBatchSlices
    stage_actual_token_counts: list[int]
    sp_local_stage_actual_token_counts: list[int]


class AsyncMoeUbatchMetadata(_AsyncMoeUbatchMetadataOptional):
    """Async MoE ubatch sidecar metadata carried by the forward context.

    The required fields (``attn_metadata``, ``ubatch_slices``) are always
    populated by the attention model runner before the model forward starts.
    The optional layout fields describe padded stage inputs and their real
    token coverage:

    - ``use_sp_stage_resharding``: set to ``True`` by the attention model
      runner when Ascend sequence parallelism is active and the applied split
      is TP-aligned.
    - ``sp_local_stage_slices``: set by
      ``build_async_moe_stage_inputs`` during model forward; its lengths
      describe the equal per-rank stage shards, not ranges into the original
      full-batch local shard.
    - ``stage_actual_token_counts`` and
      ``sp_local_stage_actual_token_counts`` distinguish real token rows from
      DP/SP padding globally and on the current TP rank.
    """

    attn_metadata: object
    ubatch_slices: UBatchSlices


def get_afd_metadata_from_forward_context(
    forward_context: ForwardContext | None = None,
) -> AFDForwardContextMetadata | None:
    """Return AFD metadata from vLLM ``ForwardContext.additional_kwargs``.

    Model wrappers use this helper so AFD metadata stays outside vLLM's
    ``ForwardContext`` schema.
    """

    if forward_context is None:
        forward_context = get_forward_context()

    additional_kwargs = forward_context.additional_kwargs or {}
    # Keep the type refinement static: torch.compile traces this helper and
    # cannot wrap the runtime ``types.UnionType`` created by typing.cast.
    metadata: AFDForwardContextMetadata | None = additional_kwargs.get("afd_metadata")
    return metadata


def get_async_moe_ubatch_metadata_from_forward_context(
    forward_context: ForwardContext | None = None,
) -> AsyncMoeUbatchMetadata | None:
    """Return async MoE ubatch sidecar metadata from the current context."""

    if forward_context is None:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()

    additional_kwargs = forward_context.additional_kwargs or {}
    metadata: AsyncMoeUbatchMetadata | None = additional_kwargs.get(
        ASYNC_MOE_UBATCH_METADATA_KEY,
    )
    return metadata


@contextmanager
def use_afd_metadata_provider(provider: Any) -> Iterator[None]:
    """Install AFD metadata as vLLM creates a forward context.

    Native vLLM dummy runs call the model directly, bypassing
    ``AFDAttentionModelRunner._model_forward()``. Out-of-tree plugins cannot
    extend the ``set_forward_context()`` signature, so during dummy runs we
    temporarily wrap ``create_forward_context()`` and mutate
    ``additional_kwargs`` immediately after vLLM creates the context. Model code
    can then do a simple metadata read, which keeps ``torch.compile`` away from
    provider lookups.
    """

    original_create = forward_context_module.create_forward_context
    install = provider._install_afd_metadata_on_forward_context

    @wraps(original_create)
    def create_forward_context_with_afd(*args: Any, **kwargs: Any) -> ForwardContext:
        forward_context = original_create(*args, **kwargs)
        install(forward_context)
        return forward_context

    forward_context_module.create_forward_context = create_forward_context_with_afd
    try:
        yield
    finally:
        forward_context_module.create_forward_context = original_create


__all__ = [
    "ASYNC_MOE_UBATCH_METADATA_KEY",
    "get_afd_metadata_from_forward_context",
    "get_async_moe_ubatch_metadata_from_forward_context",
    "use_afd_metadata_provider",
]
