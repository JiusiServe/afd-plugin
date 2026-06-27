# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patches for AFD async-DP forward-context coordination.

This module patches:
1. ``vllm.forward_context.set_forward_context``
2. ``vllm.forward_context.coordinate_batch_across_dp``
3. ``vllm.v1.worker.dp_utils.coordinate_batch_across_dp``

Why:
    vLLM 0.19.1 constructs ``DPMetadata`` and coordinates token counts across
    MoE DP ranks whenever DP size is greater than one. AFD async-DP uses the
    connector data flow instead of vLLM's DP metadata control plane, so those
    all-reduce and metadata paths must be skipped for the AFD async connector.

How:
    Non-AFD configs delegate to vLLM unchanged. AFD async configs create the
    normal ``ForwardContext`` with ``dp_metadata=None`` and make DP batch
    coordination return the same early-exit shape as DP=1.

Future plan:
    Remove this patch when vLLM exposes a plugin-owned way to opt out of MoE
    DP metadata coordination per engine role.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any

import vllm.forward_context as forward_context_module
import vllm.v1.worker.dp_utils as dp_utils_module
from vllm.config import CUDAGraphMode
from vllm.forward_context import DPMetadata

from afd_plugin.compat.async_dp import (
    ensure_async_dp_compat_attr,
    is_afd_async_dp,
    parallel_config_async_dp,
)
from afd_plugin.compat.vllm import TARGET_VLLM_VERSION

_PATCH_APPLIED = False

_original_set_forward_context = forward_context_module.set_forward_context
_original_coordinate_batch_across_dp = dp_utils_module.coordinate_batch_across_dp

_FORWARD_CONTEXT_IMPORT_MODULES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.kv_connector_model_runner_mixin",
    "vllm_ascend.ascend_forward_context",
)


@contextmanager
def _patched_set_forward_context(
    attn_metadata: Any,
    vllm_config: Any,
    num_tokens: int | None = None,
    num_tokens_across_dp: Any | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: Any | None = None,
    ubatch_slices: Any | None = None,
    slot_mapping: dict[str, Any] | list[dict[str, Any]] | None = None,
    skip_compiled: bool = False,
):
    """Create a forward context without DP metadata for AFD async-DP."""

    ensure_async_dp_compat_attr(vllm_config)
    if not is_afd_async_dp(vllm_config):
        with _original_set_forward_context(
            attn_metadata,
            vllm_config,
            num_tokens,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            batch_descriptor,
            ubatch_slices,
            slot_mapping,
            skip_compiled,
        ):
            yield
        return

    need_to_track_batchsize = (
        forward_context_module.track_batchsize and attn_metadata is not None
    )
    if need_to_track_batchsize:
        forward_context_module.forward_start_time = (
            forward_context_module.time.perf_counter()
        )

    dp_metadata: DPMetadata | None = None

    if (
        cudagraph_runtime_mode != CUDAGraphMode.NONE
        and num_tokens is not None
        and batch_descriptor is None
    ):
        batch_descriptor = forward_context_module.BatchDescriptor(
            num_tokens=num_tokens,
        )

    additional_kwargs = (
        forward_context_module.current_platform.set_additional_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=vllm_config,
            dp_metadata=dp_metadata,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            ubatch_slices=ubatch_slices,
        )
    )

    forward_context = forward_context_module.create_forward_context(
        attn_metadata,
        vllm_config,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
        slot_mapping,
        additional_kwargs,
        skip_compiled,
    )

    try:
        with forward_context_module.override_forward_context(forward_context):
            yield
    finally:
        if need_to_track_batchsize:
            batchsize = num_tokens
            synchronize = forward_context_module.current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = forward_context_module.time.perf_counter()
            forward_context_module.batchsize_forward_time[batchsize].append(
                (now - forward_context_module.forward_start_time) * 1000,
            )
            if (
                now - forward_context_module.last_logging_time
                > forward_context_module.batchsize_logging_interval
            ):
                forward_context_module.last_logging_time = now
                forward_stats = []
                for bs, times in (
                    forward_context_module.batchsize_forward_time.items()
                ):
                    if len(times) <= 1:
                        continue
                    medium = forward_context_module.torch.quantile(
                        forward_context_module.torch.tensor(times),
                        q=0.5,
                    ).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    forward_context_module.logger.info(
                        (
                            "Batchsize forward time stats "
                            "(batchsize, count, median_time(ms)): %s"
                        ),
                        forward_stats,
                    )


def _patched_coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    parallel_config: Any,
    uniform_decode: bool | None = None,
    num_scheduled_tokens_per_request: dict[str, int] | None = None,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    allow_microbatching: bool = True,
) -> tuple[bool, Any | None, CUDAGraphMode]:
    """Skip native DP batch coordination for mirrored AFD async-DP."""

    if parallel_config_async_dp(parallel_config):
        return False, None, cudagraph_mode
    return _original_coordinate_batch_across_dp(
        num_tokens_unpadded,
        parallel_config,
        uniform_decode,
        num_scheduled_tokens_per_request,
        cudagraph_mode,
        allow_microbatching,
    )


def apply_async_dp_forward_context_patch() -> None:
    """Apply AFD async-DP forward-context patches once per process."""

    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    if not _is_target_vllm_compatible():
        return
    _PATCH_APPLIED = True

    forward_context_module.set_forward_context = _patched_set_forward_context
    forward_context_module.coordinate_batch_across_dp = (
        _patched_coordinate_batch_across_dp
    )
    dp_utils_module.coordinate_batch_across_dp = _patched_coordinate_batch_across_dp
    _patch_loaded_forward_context_imports()
    forward_context_module.logger.debug(
        "AFD async-DP forward-context patch applied",
    )


def _patch_loaded_forward_context_imports() -> None:
    for module_name in _FORWARD_CONTEXT_IMPORT_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            module.set_forward_context = _patched_set_forward_context


def _is_target_vllm_compatible() -> bool:
    try:
        import vllm

        version_value = vllm.__version__
    except (AttributeError, ImportError):
        return True
    version_text = str(version_value)
    if "dev" in version_text:
        return True
    return version_text.startswith(TARGET_VLLM_VERSION)


__all__ = ["apply_async_dp_forward_context_patch"]
