# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Helpers for plugin-owned model wrappers to read AFD forward metadata."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

import vllm.forward_context as forward_context_module
from vllm.forward_context import get_forward_context


def get_afd_metadata_from_forward_context(forward_context: object | None = None) -> Any:
    """Return AFD metadata from vLLM ``ForwardContext.additional_kwargs``.

    The out-of-tree plugin deliberately avoids adding ``ForwardContext`` fields
    in Phase 2. Model wrappers should use this helper instead of reading a
    patched ``forward_ctx.afd_metadata`` attribute.
    """

    if forward_context is None:
        forward_context = get_forward_context()

    additional_kwargs = forward_context.additional_kwargs or {}
    metadata = additional_kwargs.get("afd_metadata")
    if metadata is not None:
        return metadata
    return getattr(forward_context, "afd_metadata", None)


@contextmanager
def use_afd_metadata_provider(provider: Any) -> Iterator[None]:
    """Install AFD metadata as vLLM creates a forward context.

    Native vLLM dummy runs call the model directly, bypassing
    ``AFDAttentionModelRunner._model_forward()``.  The original in-tree AFD
    implementation passed ``afd_metadata`` into ``set_forward_context()``
    before the compiled model was entered.  Out-of-tree we cannot extend that
    signature, so during dummy runs we temporarily wrap
    ``create_forward_context()`` and mutate ``additional_kwargs`` immediately
    after vLLM creates the context.  Model code can then do a simple metadata
    read, which keeps ``torch.compile`` away from provider lookups.
    """

    original_create = forward_context_module.create_forward_context
    install = provider._install_afd_metadata_on_forward_context

    @wraps(original_create)
    def create_forward_context_with_afd(*args: Any, **kwargs: Any) -> Any:
        forward_context = original_create(*args, **kwargs)
        install(forward_context)
        return forward_context

    forward_context_module.create_forward_context = create_forward_context_with_afd
    try:
        yield
    finally:
        forward_context_module.create_forward_context = original_create


__all__ = [
    "get_afd_metadata_from_forward_context",
    "use_afd_metadata_provider",
]
