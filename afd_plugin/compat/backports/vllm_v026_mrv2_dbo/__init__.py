# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Temporary vLLM 0.26 ModelRunnerV2 DBO backport."""

from .runtime import (
    AFDBatchExecutionDescriptor,
    assert_backport_required,
    create_ubatch_slices,
    dispatch_afd_dbo_and_sync_dp,
    prepare_attn_for_ubatch,
    share_metadata_builder_workspaces,
    slice_input_batch,
    slice_model_inputs,
    use_two_metadata_builders,
)

__all__ = [
    "AFDBatchExecutionDescriptor",
    "assert_backport_required",
    "create_ubatch_slices",
    "dispatch_afd_dbo_and_sync_dp",
    "prepare_attn_for_ubatch",
    "share_metadata_builder_workspaces",
    "slice_input_batch",
    "slice_model_inputs",
    "use_two_metadata_builders",
]
