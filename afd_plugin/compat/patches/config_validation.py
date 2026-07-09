# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Config validation shim for AFD-owned ubatching.

vLLM 0.19.1 validates native microbatching by requiring a DeepEP all2all
backend. AFD ubatching uses plugin connectors instead, so this patch only
relaxes that assertion for configs with ``additional_config["afd"].enabled``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import vllm.config.vllm as config_module
import vllm.engine.arg_utils as arg_utils_module

from afd_plugin.compat.vllm import TARGET_VLLM_VERSION
from afd_plugin.config import parse_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

_ORIGINAL_CREATE_ENGINE_CONFIG_ATTR = "_afd_plugin_original_create_engine_config"
_ORIGINAL_VLLM_CONFIG_POST_INIT_ATTR = "_afd_plugin_original_vllm_config_post_init"
_AFD_TEMP_BACKEND = "deepep_low_latency"
_original_create_engine_config: Callable[..., Any] | None = None
_original_vllm_config_post_init: Callable[..., Any] | None = None


# Patch reason: vLLM validates native ubatching by requiring a DeepEP all2all
# backend, while AFD ubatching is implemented by plugin connectors.
# Patch functionality: temporarily uses a supported backend only during
# upstream EngineArgs-to-VllmConfig validation for AFD configs.
# Expansion exception: upstream create_engine_config is a large config builder;
# keep a narrow original-function delegation so this patch only owns the AFD
# validation bypass.
# Signature: matches upstream; no added parameters.
def create_engine_config(
    self: EngineArgs,
    usage_context: UsageContext | None = None,
    headless: bool = False,
) -> VllmConfig:
    """Create the VllmConfig."""

    assert _original_create_engine_config is not None
    if not _should_relax_engine_args_backend(self):
        return _original_create_engine_config(
            self,
            usage_context,
            headless,
        )

    # ### PATCH START: AFD ubatching all2all backend validation
    # vLLM validates native ubatching against DeepEP backends. AFD ubatching
    # uses plugin connectors, so temporarily present a supported backend only
    # while upstream builds and validates VllmConfig.
    original_backend = self.all2all_backend
    self.all2all_backend = _AFD_TEMP_BACKEND
    try:
        config = _original_create_engine_config(
            self,
            usage_context,
            headless,
        )
    finally:
        self.all2all_backend = original_backend
    config.parallel_config.all2all_backend = original_backend
    # ### PATCH END: AFD ubatching all2all backend validation
    return config


# Patch reason: VllmConfig validation can rerun the native ubatching all2all
# backend assertion after EngineArgs construction.
# Patch functionality: temporarily relaxes that assertion for AFD configs while
# restoring the real backend after upstream validation.
# Expansion exception: upstream VllmConfig.__post_init__ is a large validation
# pipeline; keep a narrow original-function delegation so this patch only owns
# the AFD validation bypass.
# Signature: matches upstream; no added parameters.
def __post_init__(self: VllmConfig):
    """Verify configs are valid & consistent with each other."""

    assert _original_vllm_config_post_init is not None
    if not _should_relax_vllm_config_backend(self):
        return _original_vllm_config_post_init(self)

    # ### PATCH START: AFD ubatching all2all backend validation
    # Repeated VllmConfig validation can run after EngineArgs construction.
    # Keep AFD's real all2all backend on the config, but use a temporary DeepEP
    # value while upstream performs its native ubatching assertion.
    parallel_config = self.parallel_config
    original_backend = parallel_config.all2all_backend
    parallel_config.all2all_backend = _AFD_TEMP_BACKEND
    try:
        result = _original_vllm_config_post_init(self)
    finally:
        parallel_config.all2all_backend = original_backend
    # ### PATCH END: AFD ubatching all2all backend validation
    return result


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


if _is_target_vllm_compatible():
    if not hasattr(arg_utils_module, _ORIGINAL_CREATE_ENGINE_CONFIG_ATTR):
        setattr(
            arg_utils_module,
            _ORIGINAL_CREATE_ENGINE_CONFIG_ATTR,
            arg_utils_module.EngineArgs.create_engine_config,
        )

    if not hasattr(config_module, _ORIGINAL_VLLM_CONFIG_POST_INIT_ATTR):
        setattr(
            config_module,
            _ORIGINAL_VLLM_CONFIG_POST_INIT_ATTR,
            config_module.VllmConfig.__post_init__,
        )

    _original_create_engine_config = getattr(
        arg_utils_module,
        _ORIGINAL_CREATE_ENGINE_CONFIG_ATTR,
    )
    _original_vllm_config_post_init = getattr(
        config_module,
        _ORIGINAL_VLLM_CONFIG_POST_INIT_ATTR,
    )

    arg_utils_module.EngineArgs.create_engine_config = create_engine_config
    config_module.VllmConfig.__post_init__ = __post_init__
    arg_utils_module.logger.debug("AFD config validation patch applied")


__all__: list[str] = []
