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
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from afd_plugin.config import AFDConfig, parse_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = logging.getLogger(__name__)


_INITIALIZE_KV_CACHES_RETURNS_TUPLE_ATTR = (
    "_afd_plugin_initialize_kv_caches_returns_tuple"
)

_initialize_kv_caches_returns_tuple_value = False


# Patch reason: AFD FFN ranks run as connector daemons instead of normal
# request-scheduling EngineCore instances.
# Patch functionality: returns after model executor construction for AFD FFN
# configs while preserving the target upstream tag's normal EngineCore startup
# logic for non-AFD configs.
# Signature: matches upstream; no added parameters.
def __init__(
    self,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    executor_fail_callback: Callable | None = None,
    include_finished_set: bool = False,
):
    # ### PATCH START: AFD FFN EngineCore daemon initialization
    # FFN ranks are connector daemons, so stop EngineCore initialization after
    # executor construction instead of setting up KV cache and scheduler state.
    if _is_afd_ffn_config(vllm_config):
        _initialize_ffn_engine_core(
            self,
            _CORE_MODULE,
            vllm_config,
            executor_class,
            log_stats,
            executor_fail_callback,
        )
        return
    # ### PATCH END: AFD FFN EngineCore daemon initialization

    core_module = _require_core_module()

    # plugins need to be loaded at the engine/scheduler level too
    from vllm.plugins import load_general_plugins

    load_general_plugins()

    self.vllm_config = vllm_config
    if not vllm_config.parallel_config.data_parallel_rank_local:
        core_module.logger.info(
            "Initializing a V1 LLM engine (v%s) with config: %s",
            core_module.VLLM_VERSION,
            vllm_config,
        )

    self.log_stats = log_stats

    # Setup Model.
    self.model_executor = executor_class(vllm_config)
    if executor_fail_callback is not None:
        self.model_executor.register_failure_callback(executor_fail_callback)

    self.available_gpu_memory_for_kv_cache = -1

    if core_module.envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
        self._eep_scale_up_before_kv_init()

    # Setup KV Caches and update CacheConfig after profiling.
    kv_cache_config = self._initialize_kv_caches(vllm_config)
    self.structured_output_manager = core_module.StructuredOutputManager(vllm_config)

    # Setup scheduler.
    Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

    if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
        # Encoder models without KV cache don't support
        # chunked prefill. But do SSM models?
        if vllm_config.scheduler_config.enable_chunked_prefill:
            core_module.logger.warning(
                "Disabling chunked prefill for model without KVCache"
            )
            vllm_config.scheduler_config.enable_chunked_prefill = False

    scheduler_block_size = (
        vllm_config.cache_config.block_size
        * vllm_config.parallel_config.decode_context_parallel_size
        * vllm_config.parallel_config.prefill_context_parallel_size
    )

    self.scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=self.structured_output_manager,
        include_finished_set=include_finished_set,
        log_stats=self.log_stats,
        block_size=scheduler_block_size,
    )
    self.use_spec_decode = vllm_config.speculative_config is not None
    if self.scheduler.connector is not None:  # type: ignore
        self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

    mm_registry = core_module.MULTIMODAL_REGISTRY
    self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
        vllm_config
    )

    # If a KV connector is initialized for scheduler, we want to collect
    # handshake metadata from all workers so the connector in the scheduler
    # will have the full context
    kv_connector = self.scheduler.get_kv_connector()
    if kv_connector is not None:
        # Collect and store KV connector xfer metadata from workers
        # (after KV cache registration)
        xfer_handshake_metadata = (
            self.model_executor.get_kv_connector_handshake_metadata()
        )

        if xfer_handshake_metadata:
            # xfer_handshake_metadata is list of dicts from workers
            # Each dict already has structure {tp_rank: metadata}
            # Merge all worker dicts into a single dict
            content: dict[int, Any] = {}
            for worker_dict in xfer_handshake_metadata:
                if worker_dict is not None:
                    content.update(worker_dict)
            kv_connector.set_xfer_handshake_metadata(content)

    # Setup batch queue for pipeline parallelism.
    # Batch queue for scheduled batches. This enables us to asynchronously
    # schedule and execute batches, and is required by pipeline parallelism
    # to eliminate pipeline bubbles.
    self.batch_queue_size = self.model_executor.max_concurrent_batches
    self.batch_queue = None
    if self.batch_queue_size > 1:
        core_module.logger.debug(
            "Batch queue is enabled with size %d",
            self.batch_queue_size,
        )
        self.batch_queue = deque(maxlen=self.batch_queue_size)

    self.is_ec_consumer = (
        vllm_config.ec_transfer_config is None
        or vllm_config.ec_transfer_config.is_ec_consumer
    )
    self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

    self.request_block_hasher = None
    if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
        caching_hash_fn = core_module.get_hash_fn_by_name(
            vllm_config.cache_config.prefix_caching_hash_algo
        )
        core_module.init_none_hash(caching_hash_fn)

        self.request_block_hasher = core_module.get_request_block_hasher(
            scheduler_block_size, caching_hash_fn
        )

    self.step_fn = (
        self.step if self.batch_queue is None else self.step_with_batch_queue
    )
    self.async_scheduling = vllm_config.scheduler_config.async_scheduling

    self.aborts_queue = queue.Queue()

    self._idle_state_callbacks: list[Callable] = []

    # Mark the startup heap as static so that it's ignored by GC.
    # Reduces pause times of oldest generation collections.
    core_module.freeze_gc_heap()
    # If enable, attach GC debugger after static variable freeze.
    core_module.maybe_attach_gc_debug_callback()
    # Enable environment variable cache (e.g. assume no more
    # environment variable overrides after this point)
    core_module.enable_envs_cache()


# Patch reason: AFD FFN daemon engines skip scheduler/KV setup, so upstream
# shutdown can touch attributes that were intentionally not initialized.
# Patch functionality: stops the connector worker loop and shuts down the model
# executor for AFD FFN engines while preserving upstream shutdown for non-AFD
# engines.
# Signature: matches upstream; no added parameters.
def shutdown(self):
    # ### PATCH START: AFD FFN EngineCore shutdown
    # Stop the connector-driven worker loop before shutting down the executor;
    # scheduler/KV state may not exist for FFN daemon engines.
    if _is_afd_ffn_engine(self):
        _stop_ffn_worker_loop(self)
        model_executor = getattr(self, "model_executor", None)
        if model_executor is not None:
            model_executor.shutdown()
        with suppress(Exception):
            gc.unfreeze()
        return
    # ### PATCH END: AFD FFN EngineCore shutdown

    self.structured_output_manager.clear_backend()
    if self.model_executor:
        self.model_executor.shutdown()
    if self.scheduler:
        self.scheduler.shutdown()


# Patch reason: late-loaded AFD FFN EngineCore paths may ask for KV cache setup
# even though FFN daemons do not own request KV blocks.
# Patch functionality: returns a minimal KV-cache-shaped result for AFD FFN
# configs while preserving upstream KV cache initialization for non-AFD configs.
# Signature: matches upstream; no added parameters.
def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
    # ### PATCH START: AFD FFN late-loaded KV cache bypass
    # FFN daemon engines do not schedule requests or own KV cache blocks, but
    # late-loaded paths still need a minimal KV cache config-shaped result.
    if _is_afd_ffn_config(vllm_config):
        _prepare_late_loaded_ffn_engine_core(self, vllm_config)
        kv_cache_config = _AFDFFNKVCacheConfig()
        if _initialize_kv_caches_returns_tuple_value:
            result = 0, 0, kv_cache_config
        else:
            result = kv_cache_config
        return result
    # ### PATCH END: AFD FFN late-loaded KV cache bypass

    core_module = _require_core_module()
    start = time.time()

    # Get all kv cache needed by the model
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
    if has_kv_cache:
        if core_module.envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            # NOTE(yongji): should already be set
            # during _eep_scale_up_before_kv_init
            assert self.available_gpu_memory_for_kv_cache > 0
            available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                kv_cache_specs
            )
        else:
            # Profiles the peak memory usage of the model to determine how
            # much memory can be allocated for kv cache.
            available_gpu_memory = self.model_executor.determine_available_memory()
            self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
    else:
        # Attention free models don't need memory for kv cache
        available_gpu_memory = [0] * len(kv_cache_specs)

    assert len(kv_cache_specs) == len(available_gpu_memory)

    # Track max_model_len before KV cache config to detect auto-fit changes
    max_model_len_before = vllm_config.model_config.max_model_len

    kv_cache_configs = core_module.get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_gpu_memory
    )

    # If auto-fit reduced max_model_len, sync the new value to workers.
    # This is needed because workers were spawned before memory profiling
    # and have the original (larger) max_model_len cached.
    max_model_len_after = vllm_config.model_config.max_model_len
    if max_model_len_after != max_model_len_before:
        self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

    scheduler_kv_cache_config = core_module.generate_scheduler_kv_cache_config(
        kv_cache_configs
    )
    vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
    kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
    if kv_cache_groups:
        vllm_config.cache_config.block_size = min(
            g.kv_cache_spec.block_size for g in kv_cache_groups
        )

    vllm_config.validate_block_size()

    # Initialize kv cache and warmup the execution
    self.model_executor.initialize_from_config(kv_cache_configs)

    elapsed = time.time() - start
    core_module.logger.info_once(
        "init engine (profile, create kv cache, warmup model) took %.2f seconds",
        elapsed,
        scope="local",
    )
    return scheduler_kv_cache_config


# Patch reason: AFD FFN ranks must run the connector server loop rather than
# vLLM's normal request scheduling busy loop.
# Patch functionality: starts and monitors the FFN connector loop for AFD FFN
# engines while preserving upstream busy loops for non-AFD engine processes.
# Signature: matches upstream; no added parameters.
def run_busy_loop(self):
    # ### PATCH START: AFD FFN connector busy loop
    # FFN ranks run the connector server loop and poll worker-side failures
    # instead of executing vLLM's normal request scheduling loop.
    if _is_afd_ffn_engine(self):
        result = _run_ffn_busy_loop(self, _CORE_MODULE)
        return result
    # ### PATCH END: AFD FFN connector busy loop

    if isinstance(self, _DP_ENGINE_CORE_PROC_CLS):
        core_module = _require_core_module()

        # Loop until process is sent a SIGINT or SIGTERM
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            if self.eep_scaling_state is not None:
                _ = self.eep_scaling_state.progress()
                if self.eep_scaling_state.is_complete():
                    if self.eep_scaling_state.worker_type == "removing":
                        raise SystemExit
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None

            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    core_module.logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            core_module.EngineCoreOutputs(
                                wave_complete=self.current_wave
                            ),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

        raise SystemExit

    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()
        # 2) Step the engine core and return the outputs.
        self._process_engine_step()

    raise SystemExit


_CORE_MODULE: Any | None = None
_DP_ENGINE_CORE_PROC_CLS: type[Any] = type(None)


def _require_core_module() -> Any:
    assert _CORE_MODULE is not None
    return _CORE_MODULE


def _patch_engine_core() -> None:
    """Patch AFD FFN EngineCore behavior when this module is imported."""

    global _CORE_MODULE
    global _DP_ENGINE_CORE_PROC_CLS
    global _initialize_kv_caches_returns_tuple_value
    try:
        core_module = importlib.import_module("vllm.v1.engine.core")
    except Exception:
        logger.debug("AFD EngineCore patch skipped: vLLM core is unavailable")
        return

    _CORE_MODULE = core_module
    engine_core_cls = core_module.EngineCore
    engine_core_proc_cls = getattr(core_module, "EngineCoreProc", None)
    dp_engine_core_proc_cls = getattr(core_module, "DPEngineCoreProc", None)
    _DP_ENGINE_CORE_PROC_CLS = (
        dp_engine_core_proc_cls if dp_engine_core_proc_cls is not None else type(None)
    )

    if not hasattr(core_module, _INITIALIZE_KV_CACHES_RETURNS_TUPLE_ATTR):
        setattr(
            core_module,
            _INITIALIZE_KV_CACHES_RETURNS_TUPLE_ATTR,
            _initialize_kv_caches_returns_tuple(engine_core_cls),
        )

    _initialize_kv_caches_returns_tuple_value = getattr(
        core_module,
        _INITIALIZE_KV_CACHES_RETURNS_TUPLE_ATTR,
    )

    engine_core_cls.__init__ = __init__
    if hasattr(engine_core_cls, "_initialize_kv_caches"):
        engine_core_cls._initialize_kv_caches = _initialize_kv_caches
    engine_core_cls.shutdown = shutdown
    if engine_core_proc_cls is not None:
        engine_core_proc_cls.run_busy_loop = run_busy_loop
    if dp_engine_core_proc_cls is not None:
        dp_engine_core_proc_cls.run_busy_loop = run_busy_loop

    logger.debug("AFD EngineCore patch applied")


class _AFDFFNKVCacheConfig:
    kv_cache_groups: list[Any] = []


class _AFDFFNNoopScheduler:
    connector = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

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
        return []


def _initialize_ffn_engine_core(
    self,
    core_module: Any,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    executor_fail_callback: Callable | None,
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


def _prepare_late_loaded_ffn_engine_core(
    self,
    vllm_config: VllmConfig,
) -> None:
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


def _run_ffn_busy_loop(self, core_module: Any) -> None:
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


def _stop_ffn_worker_loop(self) -> None:
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


def _is_running(self, core_module: Any) -> bool:
    shutdown_state = getattr(self, "shutdown_state", None)
    engine_shutdown_state = getattr(core_module, "EngineShutdownState", None)
    running_state = getattr(engine_shutdown_state, "RUNNING", None)
    if shutdown_state is None or running_state is None:
        return True
    return shutdown_state == running_state


def _is_afd_ffn_engine(self) -> bool:
    return _is_afd_ffn_config(getattr(self, "vllm_config", None))


def _is_afd_ffn_config(vllm_config: VllmConfig | None) -> bool:
    config = _get_afd_config(vllm_config)
    return config.enabled and config.role == "ffn"


def _get_afd_config(vllm_config: VllmConfig | None) -> AFDConfig:
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


_patch_engine_core()


__all__: list[str] = []
