# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patch vLLM-Ascend platform config normalization for AFD-owned DBO.

Upstream source: ``vllm_ascend/platform.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from afd_plugin.config import parse_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_ASCEND_PLATFORM_PATCH_ATTR = "_afd_plugin_ascend_platform_patch_state"


def apply_afd_ascend_dbo_config_patch() -> None:
    """Preserve AFD-owned DBO settings during vLLM-Ascend config normalization.

    vLLM-Ascend's platform compatibility pass disables DBO/ubatching fields for
    ordinary NPU runs. AFD owns its NPU ubatching path, so this patch snapshots
    those fields for AFD-enabled configs, lets upstream normalization run, then
    restores the AFD DBO values. The patch is a no-op when vLLM-Ascend is not
    importable or when this process has already installed the wrapper.
    """

    try:
        from vllm_ascend.platform import NPUPlatform
    except Exception:
        return

    if hasattr(NPUPlatform, _ASCEND_PLATFORM_PATCH_ATTR):
        return

    original_fix_incompatible_config = NPUPlatform._fix_incompatible_config

    # Patch reason: vLLM-Ascend resets DBO fields inside NPUPlatform config
    # normalization, while AFD now owns the Ascend DBO/ubatching path.
    # Patch functionality: preserves upstream normalization for non-AFD configs and
    # restores AFD DBO fields after upstream normalization for AFD-enabled configs.
    # Expansion exception: upstream _fix_incompatible_config is platform-owned
    # normalization; keep narrow original-function delegation so this patch only
    # owns the AFD DBO preservation.
    # Signature: matches upstream; no added parameters.
    def _fix_incompatible_config(vllm_config: VllmConfig) -> Any:
        # ### PATCH START: AFD DBO config preservation
        saved = _snapshot_afd_dbo_config(vllm_config)
        # ### PATCH END: AFD DBO config preservation
        result = original_fix_incompatible_config(vllm_config)
        # ### PATCH START: AFD DBO config preservation
        if saved is not None:
            _restore_afd_dbo_config(vllm_config, saved)
        # ### PATCH END: AFD DBO config preservation
        return result

    NPUPlatform._fix_incompatible_config = staticmethod(_fix_incompatible_config)
    setattr(
        NPUPlatform,
        _ASCEND_PLATFORM_PATCH_ATTR,
        original_fix_incompatible_config,
    )


def _snapshot_afd_dbo_config(vllm_config: VllmConfig) -> dict[str, bool | int] | None:
    if not _is_afd_config_enabled(vllm_config):
        return None
    parallel_config = vllm_config.parallel_config
    return {
        "enable_dbo": parallel_config.enable_dbo,
        "use_ubatching": parallel_config.use_ubatching,
        "ubatch_size": parallel_config.ubatch_size,
    }


def _restore_afd_dbo_config(
    vllm_config: VllmConfig,
    saved: dict[str, bool | int],
) -> None:
    parallel_config = vllm_config.parallel_config
    if not (
        saved["enable_dbo"]
        or saved["use_ubatching"]
        or int(saved["ubatch_size"] or 0) != 0
    ):
        return
    parallel_config.enable_dbo = saved["enable_dbo"]
    parallel_config.ubatch_size = saved["ubatch_size"]


def _is_afd_config_enabled(vllm_config: VllmConfig) -> bool:
    try:
        return parse_afd_config(vllm_config, validate=False).enabled
    except Exception:
        return False


__all__ = ["apply_afd_ascend_dbo_config_patch"]
