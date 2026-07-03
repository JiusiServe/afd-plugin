# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""EngineCore compatibility patch for AFD FFN daemon mode.

The AFD FFN side runs as a connector daemon, not as a normal request-scheduling
EngineCore. After constructing the model executor, FFN EngineCore
initialization returns before KV cache and scheduler setup. This keeps FFN
startup out of HybridKVCacheCoordinator.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import logging
import queue
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from afd_plugin.config import AFDConfig, parse_afd_config

logger = logging.getLogger(__name__)


@dataclass
class _PatchState:
    engine_core_init: Callable[..., Any]
    engine_core_initialize_kv_caches: Callable[..., Any] | None
    engine_core_shutdown: Callable[..., Any]
    engine_core_proc_run_busy_loop: Callable[..., Any] | None
    dp_engine_core_proc_run_busy_loop: Callable[..., Any] | None
    initialize_kv_caches_returns_tuple: bool


_PATCH_ATTR = "_afd_plugin_engine_core_patch_state"


def apply_engine_core_patch() -> None:
    """Apply the AFD FFN EngineCore patch if vLLM is importable.

    The patch is intentionally narrow: all non-FFN configs delegate to the
    original vLLM methods. For FFN configs, EngineCore stops after executor
    construction and the process busy loop starts the connector-driven worker
    loop.
    """

    try:
        core_module = importlib.import_module("vllm.v1.engine.core")
    except Exception:
        logger.debug("AFD EngineCore patch skipped: vLLM core is unavailable")
        return

    if hasattr(core_module, _PATCH_ATTR):
        return

    engine_core_cls = core_module.EngineCore
    engine_core_proc_cls = getattr(core_module, "EngineCoreProc", None)
    dp_engine_core_proc_cls = getattr(core_module, "DPEngineCoreProc", None)

    state = _PatchState(
        engine_core_init=engine_core_cls.__init__,
        engine_core_initialize_kv_caches=getattr(
            engine_core_cls,
            "_initialize_kv_caches",
            None,
        ),
        engine_core_shutdown=engine_core_cls.shutdown,
        engine_core_proc_run_busy_loop=(
            getattr(engine_core_proc_cls, "run_busy_loop", None)
            if engine_core_proc_cls is not None
            else None
        ),
        dp_engine_core_proc_run_busy_loop=(
            getattr(dp_engine_core_proc_cls, "run_busy_loop", None)
            if dp_engine_core_proc_cls is not None
            else None
        ),
        initialize_kv_caches_returns_tuple=_initialize_kv_caches_returns_tuple(
            engine_core_cls,
        ),
    )

    def patched_engine_core_init(
        self: Any,
        vllm_config: Any,
        executor_class: type[Any],
        log_stats: bool,
        executor_fail_callback: Callable[..., Any] | None = None,
        include_finished_set: bool = False,
    ) -> None:
        if not _is_afd_ffn_config(vllm_config):
            state.engine_core_init(
                self,
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                include_finished_set,
            )
            return

        _initialize_ffn_engine_core(
            self,
            core_module,
            vllm_config,
            executor_class,
            log_stats,
            executor_fail_callback,
        )

    def patched_engine_core_shutdown(self: Any) -> None:
        if not _is_afd_ffn_engine(self):
            state.engine_core_shutdown(self)
            return

        _stop_ffn_worker_loop(self)
        model_executor = getattr(self, "model_executor", None)
        if model_executor is not None:
            model_executor.shutdown()
        with suppress(Exception):
            gc.unfreeze()

    def patched_initialize_kv_caches(self: Any, vllm_config: Any) -> Any:
        if not _is_afd_ffn_config(vllm_config):
            assert state.engine_core_initialize_kv_caches is not None
            return state.engine_core_initialize_kv_caches(self, vllm_config)

        _prepare_late_loaded_ffn_engine_core(self, vllm_config)
        kv_cache_config = _AFDFFNKVCacheConfig()
        if state.initialize_kv_caches_returns_tuple:
            return 0, 0, kv_cache_config
        return kv_cache_config

    def patched_engine_core_proc_run_busy_loop(self: Any) -> Any:
        if not _is_afd_ffn_engine(self):
            assert state.engine_core_proc_run_busy_loop is not None
            return state.engine_core_proc_run_busy_loop(self)
        return _run_ffn_busy_loop(self, core_module)

    def patched_dp_engine_core_proc_run_busy_loop(self: Any) -> Any:
        if not _is_afd_ffn_engine(self):
            assert state.dp_engine_core_proc_run_busy_loop is not None
            return state.dp_engine_core_proc_run_busy_loop(self)
        return _run_ffn_busy_loop(self, core_module)

    engine_core_cls.__init__ = patched_engine_core_init
    if state.engine_core_initialize_kv_caches is not None:
        engine_core_cls._initialize_kv_caches = patched_initialize_kv_caches
    engine_core_cls.shutdown = patched_engine_core_shutdown
    if engine_core_proc_cls is not None and state.engine_core_proc_run_busy_loop:
        engine_core_proc_cls.run_busy_loop = patched_engine_core_proc_run_busy_loop
    if dp_engine_core_proc_cls is not None and state.dp_engine_core_proc_run_busy_loop:
        dp_engine_core_proc_cls.run_busy_loop = (
            patched_dp_engine_core_proc_run_busy_loop
        )

    setattr(core_module, _PATCH_ATTR, state)
    logger.debug("AFD EngineCore patch applied")


class _AFDFFNKVCacheConfig:
    kv_cache_groups: list[Any] = []


class _AFDFFNNoopScheduler:
    connector = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def get_kv_connector(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def has_requests(self) -> bool:
        return False

    def has_unfinished_requests(self) -> bool:
        return False

    def get_num_unfinished_requests(self) -> int:
        return 0

    def finish_requests(self, *args: Any, **kwargs: Any) -> list[Any]:
        del args, kwargs
        return []


def _initialize_ffn_engine_core(
    self: Any,
    core_module: Any,
    vllm_config: Any,
    executor_class: type[Any],
    log_stats: bool,
    executor_fail_callback: Callable[..., Any] | None,
) -> None:
    try:
        from vllm.plugins import load_general_plugins

        load_general_plugins()
    except Exception:
        logger.debug("AFD FFN EngineCore could not reload vLLM plugins", exc_info=True)

    afd_config = _get_afd_config(vllm_config)
    self.vllm_config = vllm_config
    self.afd_config = afd_config
    self.log_stats = log_stats

    with suppress(Exception):
        vllm_config.afd_config = afd_config

    parallel_config = getattr(vllm_config, "parallel_config", None)
    local_dp_rank = getattr(parallel_config, "data_parallel_rank_local", 0)
    if not local_dp_rank:
        version = getattr(core_module, "VLLM_VERSION", "unknown")
        core_logger = getattr(core_module, "logger", logger)
        core_logger.info(
            "Initializing an AFD FFN V1 engine (v%s) with config: %s",
            version,
            vllm_config,
        )

    self.model_executor = executor_class(vllm_config)
    if executor_fail_callback is not None:
        self.model_executor.register_failure_callback(executor_fail_callback)

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None:
        cache_config.num_gpu_blocks = 0
        cache_config.num_cpu_blocks = 0

    # These attributes let common shutdown/debug utility paths tolerate the
    # intentionally skipped KV/scheduler initialization.
    self.available_gpu_memory_for_kv_cache = -1
    self.structured_output_manager = None
    self.scheduler = None
    self.mm_receiver_cache = None
    self.batch_queue_size = 0
    self.batch_queue = None
    self.request_block_hasher = None
    self.aborts_queue = queue.Queue()
    self._idle_state_callbacks = []
    self.use_spec_decode = False
    self.is_pooling_model = False
    self.is_ec_consumer = True


def _prepare_late_loaded_ffn_engine_core(self: Any, vllm_config: Any) -> None:
    afd_config = _get_afd_config(vllm_config)
    self.afd_config = afd_config
    with suppress(Exception):
        vllm_config.afd_config = afd_config

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None:
        cache_config.num_gpu_blocks = 0
        cache_config.num_cpu_blocks = 0
        with suppress(Exception):
            cache_config.enable_prefix_caching = False

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if scheduler_config is not None:
        with suppress(Exception):
            scheduler_config.enable_chunked_prefill = False
        with suppress(Exception):
            scheduler_config.get_scheduler_cls = lambda: _AFDFFNNoopScheduler


def _run_ffn_busy_loop(self: Any, core_module: Any) -> None:
    core_logger = getattr(core_module, "logger", logger)
    core_logger.info("AFD FFN EngineCore started; workers run connector loop.")

    started = False
    try:
        self.model_executor.collective_rpc("start_ffn_server_loop")
        started = True
        while _is_running(self, core_module):
            self.model_executor.collective_rpc("raise_ffn_loop_error_if_any")
            time.sleep(0.5)
    except KeyboardInterrupt:
        core_logger.info("AFD FFN EngineCore shutting down after KeyboardInterrupt")
    except Exception:
        core_logger.exception("AFD FFN EngineCore encountered a fatal error")
        raise
    finally:
        if started:
            _stop_ffn_worker_loop(self)

    raise SystemExit


def _stop_ffn_worker_loop(self: Any) -> None:
    model_executor = getattr(self, "model_executor", None)
    if model_executor is None:
        return
    try:
        model_executor.collective_rpc("stop_ffn_server_loop")
    except Exception:
        logger.debug(
            "AFD FFN worker loop stop failed or was already stopped",
            exc_info=True,
        )


def _is_running(self: Any, core_module: Any) -> bool:
    shutdown_state = getattr(self, "shutdown_state", None)
    engine_shutdown_state = getattr(core_module, "EngineShutdownState", None)
    running_state = getattr(engine_shutdown_state, "RUNNING", None)
    if shutdown_state is None or running_state is None:
        return True
    return shutdown_state == running_state


def _is_afd_ffn_engine(self: Any) -> bool:
    return _is_afd_ffn_config(getattr(self, "vllm_config", None))


def _is_afd_ffn_config(vllm_config: Any) -> bool:
    config = _get_afd_config(vllm_config)
    return config.enabled and config.role == "ffn"


def _get_afd_config(vllm_config: Any) -> AFDConfig:
    existing = getattr(vllm_config, "afd_config", None)
    if isinstance(existing, AFDConfig):
        return existing
    try:
        return parse_afd_config(vllm_config, validate=False)
    except Exception:
        logger.debug("Unable to parse AFD config from vLLM config", exc_info=True)
        return AFDConfig()


def _initialize_kv_caches_returns_tuple(engine_core_cls: type[Any]) -> bool:
    try:
        source = inspect.getsource(engine_core_cls.__init__)
    except Exception:
        return False
    return "num_gpu_blocks, num_cpu_blocks" in source


__all__ = ["apply_engine_core_patch"]
