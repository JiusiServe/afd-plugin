# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend forward-context helpers for connector-driven AFD steps."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def mirror_afd_metadata_on_forward_context(
    forward_context: object,
    afd_metadata: object,
) -> None:
    """Store AFD metadata in canonical kwargs and Ascend's mirrored attribute."""

    if forward_context.additional_kwargs is None:
        forward_context.additional_kwargs = {}
    forward_context.additional_kwargs["afd_metadata"] = afd_metadata
    forward_context.afd_metadata = afd_metadata


@contextmanager
def ascend_forward_context(
    *,
    vllm_config: VllmConfig,
    afd_metadata: object | None = None,
    model_instance: object | None = None,
    num_tokens: int = 0,
    num_tokens_across_dp: object | None = None,
    aclgraph_runtime_mode: object | None = None,
) -> Iterator[object | None]:
    """Create the minimal forward context needed by connector-driven FFN steps."""

    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    except Exception:
        yield None
        return

    if aclgraph_runtime_mode is None:
        aclgraph_runtime_mode = CUDAGraphMode.NONE

    context_kwargs = {
        "attn_metadata": None,
        "vllm_config": vllm_config,
        "batch_descriptor": None,
        "aclgraph_runtime_mode": aclgraph_runtime_mode,
        "model_instance": model_instance,
        "afd_metadata": afd_metadata,
        "num_tokens": int(num_tokens),
        "num_tokens_across_dp": num_tokens_across_dp,
    }
    signature = inspect.signature(set_ascend_forward_context)
    if not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        context_kwargs = {
            key: value
            for key, value in context_kwargs.items()
            if key in signature.parameters
        }

    with set_ascend_forward_context(**context_kwargs):
        forward_context = get_forward_context()
        if afd_metadata is not None:
            mirror_afd_metadata_on_forward_context(forward_context, afd_metadata)
        yield forward_context


__all__ = ["ascend_forward_context", "mirror_afd_metadata_on_forward_context"]
