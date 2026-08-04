# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Route vLLM-Ascend MLA graph parameters through AFD forward contexts.

Upstream source: ``vllm_ascend/attention/mla_v1.py``.
"""

from __future__ import annotations

AFD_MLA_GRAPH_PARAMS_KEY = "afd_mla_graph_params"
_MLA_GRAPH_PATCH_ATTR = "_afd_plugin_mla_graph_patch_state"


def apply_afd_mla_graph_patch() -> bool:
    """Install the MLA resolver, returning whether it is available."""

    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
        from vllm_ascend.attention import mla_v1
    except ImportError:
        return False

    if hasattr(mla_v1, _MLA_GRAPH_PATCH_ATTR):
        return True

    original_get_graph_params = mla_v1.get_graph_params

    # Patch reason: AFD aggregate graph capture owns one GraphParams per ubatch.
    # Patch functionality: resolve graph params from the active AFD forward
    # context while preserving upstream process-global behavior otherwise.
    # Signature: matches upstream; no added parameters.
    def get_graph_params():
        # ### PATCH START: AFD MLA graph registry
        if is_forward_context_available():
            forward_context = get_forward_context()
            additional_kwargs = forward_context.additional_kwargs or {}
            graph_params = additional_kwargs.get(AFD_MLA_GRAPH_PARAMS_KEY)
            if graph_params is not None:
                return graph_params
        # ### PATCH END: AFD MLA graph registry
        return original_get_graph_params()

    mla_v1.get_graph_params = get_graph_params
    setattr(
        mla_v1,
        _MLA_GRAPH_PATCH_ATTR,
        original_get_graph_params,
    )
    return True


__all__ = [
    "AFD_MLA_GRAPH_PARAMS_KEY",
    "apply_afd_mla_graph_patch",
]
