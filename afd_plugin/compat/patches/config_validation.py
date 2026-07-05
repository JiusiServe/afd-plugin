# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Config validation shim for AFD-owned ubatching.

vLLM 0.19.1 validates native microbatching by requiring a DeepEP all2all
backend. AFD ubatching uses plugin connectors instead, so this patch only
relaxes that assertion for configs with ``additional_config["afd"].enabled``.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from afd_plugin.compat.vllm import TARGET_VLLM_VERSION
from afd_plugin.config import parse_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

logger = logging.getLogger(__name__)


@dataclass
class _PatchState:
    engine_args_create_engine_config: Callable[..., Any]
    vllm_config_post_init: Callable[..., Any] | None


_PATCH_ATTR = "_afd_plugin_config_validation_patch_state"
_AFD_TEMP_BACKEND = "deepep_low_latency"


def apply_config_validation_patch() -> None:
    """Apply the narrow AFD ubatching config-validation patch."""

    try:
        arg_utils_module = importlib.import_module("vllm.engine.arg_utils")
    except Exception:
        logger.debug("AFD config validation patch skipped: vLLM args unavailable")
        return
    try:
        config_module = importlib.import_module("vllm.config.vllm")
    except Exception:
        config_module = None

    if hasattr(arg_utils_module, _PATCH_ATTR):
        return

    engine_args_cls = getattr(arg_utils_module, "EngineArgs", None)
    if engine_args_cls is None:
        return

    state = _PatchState(
        engine_args_create_engine_config=engine_args_cls.create_engine_config,
        vllm_config_post_init=(
            getattr(getattr(config_module, "VllmConfig", None), "__post_init__", None)
            if config_module is not None
            else None
        ),
    )

    def patched_create_engine_config(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not _should_relax_engine_args_backend(self):
            return state.engine_args_create_engine_config(self, *args, **kwargs)

        original_backend = self.all2all_backend
        self.all2all_backend = _AFD_TEMP_BACKEND
        try:
            config = state.engine_args_create_engine_config(self, *args, **kwargs)
        finally:
            self.all2all_backend = original_backend
        parallel_config = getattr(config, "parallel_config", None)
        if parallel_config is not None:
            parallel_config.all2all_backend = original_backend
        return config

    def patched_vllm_config_post_init(self) -> Any:
        assert state.vllm_config_post_init is not None
        if not _should_relax_vllm_config_backend(self):
            return state.vllm_config_post_init(self)

        parallel_config = self.parallel_config
        original_backend = parallel_config.all2all_backend
        parallel_config.all2all_backend = _AFD_TEMP_BACKEND
        try:
            return state.vllm_config_post_init(self)
        finally:
            parallel_config.all2all_backend = original_backend

    engine_args_cls.create_engine_config = patched_create_engine_config
    if state.vllm_config_post_init is not None:
        config_module.VllmConfig.__post_init__ = patched_vllm_config_post_init
    setattr(arg_utils_module, _PATCH_ATTR, state)
    if config_module is not None:
        setattr(config_module, _PATCH_ATTR, state)
    logger.debug("AFD config validation patch applied")


def _should_relax_engine_args_backend(engine_args: EngineArgs) -> bool:
    if not _is_target_vllm_compatible():
        return False
    try:
        afd_config = parse_afd_config(
            getattr(engine_args, "additional_config", None),
        )
    except Exception:
        return False
    if not afd_config.enabled:
        return False
    if (
        not bool(getattr(engine_args, "enable_dbo", False))
        and int(
            getattr(engine_args, "ubatch_size", 1),
        )
        <= 1
    ):
        return False

    backend = getattr(engine_args, "all2all_backend", None)
    return backend not in {"deepep_low_latency", "deepep_high_throughput"}


def _should_relax_vllm_config_backend(vllm_config: VllmConfig) -> bool:
    if not _is_target_vllm_compatible():
        return False
    try:
        afd_config = parse_afd_config(vllm_config)
    except Exception:
        return False
    if not afd_config.enabled:
        return False

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        return False
    if not bool(getattr(parallel_config, "use_ubatching", False)):
        return False

    backend = getattr(parallel_config, "all2all_backend", None)
    return backend not in {"deepep_low_latency", "deepep_high_throughput"}


def _is_target_vllm_compatible() -> bool:
    try:
        import vllm

        version_value = getattr(vllm, "__version__", None)
    except Exception:
        version_value = None
    if version_value is None:
        return True
    version_text = str(version_value)
    if "dev" in version_text:
        return True
    return version_text.startswith(TARGET_VLLM_VERSION)


__all__ = ["apply_config_validation_patch"]
