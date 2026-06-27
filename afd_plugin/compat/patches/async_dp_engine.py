# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patches for AFD async-DP engine scheduling.

This module patches:
1. ``vllm.v1.engine.core.EngineCoreProc.run_engine_core``
2. ``vllm.v1.engine.utils.launch_core_engines``
3. ``vllm.v1.engine.utils.DPCoordinator`` construction during launch
4. ``vllm.v1.engine.core_client.DPAsyncMPClient.add_request_async``

Why:
    vLLM 0.19.1's native MoE DP path uses ``DPEngineCoreProc`` and DP wave
    notifications. AFD async-DP Attention ranks are connector-driven and must
    step independently while keeping the original DP/EP topology for expert
    placement and weight loading.

How:
    AFD async configs are selected by plugin-owned
    ``additional_config["afd"]["async"]``. Attention-side MoE DP engine
    processes instantiate ``EngineCoreProc`` instead of ``DPEngineCoreProc``;
    coordinator stats remain enabled, but wave coordination and client
    ``FIRST_REQ`` wakeups are disabled.

Future plan:
    Remove this patch when vLLM exposes an external async-DP scheduling hook
    that can be selected by plugin-owned configuration.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import vllm.v1.engine.core as engine_core_module
import vllm.v1.engine.core_client as core_client_module
import vllm.v1.engine.utils as engine_utils_module
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import DPAsyncMPClient

from afd_plugin.compat.async_dp import (
    is_afd_async_attention_dp,
    is_afd_async_dp,
)
from afd_plugin.compat.vllm import TARGET_VLLM_VERSION

_PATCH_APPLIED = False

_original_run_engine_core = EngineCoreProc.run_engine_core
_original_launch_core_engines = engine_utils_module.launch_core_engines
_original_dp_coordinator = engine_utils_module.DPCoordinator
_original_add_request_async = DPAsyncMPClient.add_request_async


def _patched_run_engine_core(
    *args: Any,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs: Any,
) -> Any:
    """Replace MoE DP proc selection for AFD async Attention engines."""

    vllm_config = kwargs["vllm_config"]
    if not is_afd_async_attention_dp(vllm_config):
        return _original_run_engine_core(
            *args,
            dp_rank=dp_rank,
            local_dp_rank=local_dp_rank,
            **kwargs,
        )

    original_dp_engine_core_proc = engine_core_module.DPEngineCoreProc

    def build_async_attention_engine_core(
        *engine_args: Any,
        **engine_kwargs: Any,
    ) -> EngineCoreProc:
        return EngineCoreProc(*engine_args, engine_index=dp_rank, **engine_kwargs)

    engine_core_module.DPEngineCoreProc = build_async_attention_engine_core
    try:
        return _original_run_engine_core(
            *args,
            dp_rank=dp_rank,
            local_dp_rank=local_dp_rank,
            **kwargs,
        )
    finally:
        engine_core_module.DPEngineCoreProc = original_dp_engine_core_proc


@contextmanager
def _patched_launch_core_engines(
    vllm_config: Any,
    executor_class: type[Any],
    log_stats: bool,
    addresses: Any,
    num_api_servers: int = 1,
) -> Any:
    """Disable coordinator wave mode while launching AFD async-DP engines."""

    if not is_afd_async_dp(vllm_config):
        with _original_launch_core_engines(
            vllm_config,
            executor_class,
            log_stats,
            addresses,
            num_api_servers,
        ) as launch_result:
            yield launch_result
        return

    original_dp_coordinator = engine_utils_module.DPCoordinator

    def build_async_dp_coordinator(
        parallel_config: Any,
        enable_wave_coordination: bool = True,
    ) -> Any:
        """Replace launch-time coordinator wave behavior for AFD async-DP."""

        del enable_wave_coordination
        return _original_dp_coordinator(
            parallel_config,
            enable_wave_coordination=False,
        )

    engine_utils_module.DPCoordinator = build_async_dp_coordinator
    try:
        with _original_launch_core_engines(
            vllm_config,
            executor_class,
            log_stats,
            addresses,
            num_api_servers,
        ) as launch_result:
            yield launch_result
    finally:
        engine_utils_module.DPCoordinator = original_dp_coordinator


async def _patched_add_request_async(self: Any, request: Any) -> None:
    """Skip the DP wave ``FIRST_REQ`` notification for AFD async-DP."""

    if not is_afd_async_dp(self.vllm_config):
        return await _original_add_request_async(self, request)

    self._ensure_stats_update_task()

    request.current_wave = self.current_wave
    request.client_index = self.client_index

    chosen_engine = self.get_core_engine_for_request(request)
    to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
    await to_await

    self._ensure_output_queue_task()


def apply_async_dp_engine_patch() -> None:
    """Apply AFD async-DP engine patches once per process."""

    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    if not _is_target_vllm_compatible():
        return
    _PATCH_APPLIED = True

    EngineCoreProc.run_engine_core = staticmethod(_patched_run_engine_core)
    engine_utils_module.launch_core_engines = _patched_launch_core_engines
    core_client_module.launch_core_engines = _patched_launch_core_engines
    DPAsyncMPClient.add_request_async = _patched_add_request_async
    engine_core_module.logger.debug("AFD async-DP engine patch applied")


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


__all__ = ["apply_async_dp_engine_patch"]
