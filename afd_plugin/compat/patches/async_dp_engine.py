# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Patches for AFD async-DP engine scheduling.

This module patches:
1. ``vllm.v1.engine.core.EngineCoreProc.run_engine_core``
2. ``vllm.v1.engine.utils.launch_core_engines``
3. ``vllm.v1.engine.utils.DPCoordinator`` construction during launch
4. ``vllm.v1.engine.core_client.DPAsyncMPClient.add_request_async``

Why:
    vLLM 0.26.0's native MoE DP path uses ``DPEngineCoreProc`` and DP wave
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

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import vllm.v1.engine.core as engine_core_module
import vllm.v1.engine.core_client as core_client_module
import vllm.v1.engine.utils as engine_utils_module
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import DPAsyncMPClient

from afd_plugin.compat.vllm import TARGET_VLLM_VERSION
from afd_plugin.config import is_afd_async_dp, parse_optional_afd_config

if TYPE_CHECKING:
    from multiprocessing.queues import Queue

    from vllm.config import VllmConfig
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.coordinator import DPCoordinator
    from vllm.v1.engine.utils import (
        CoreEngineActorManager,
        CoreEngineProcManager,
        EngineZmqAddresses,
    )
    from vllm.v1.executor import Executor


# Patch reason: vLLM's MoE DP engine process uses DPEngineCoreProc, but AFD
# async Attention ranks are connector-driven and must not run DP wave logic.
# Patch functionality: keep upstream startup flow while selecting EngineCoreProc
# for AFD async Attention configs.
# Signature: matches upstream; no added parameters.
def run_engine_core(
    *args,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs,
):
    """Replace MoE DP proc selection for AFD async Attention engines."""

    engine_core_module.maybe_register_config_serialize_by_value()

    engine_core = None
    signal_callback = None
    try:
        vllm_config = kwargs["vllm_config"]
        parallel_config = vllm_config.parallel_config
        data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
        if data_parallel:
            parallel_config.data_parallel_rank_local = local_dp_rank
            process_title = f"EngineCore_DP{dp_rank}"
        else:
            process_title = "EngineCore"
        engine_core_module.set_process_title(process_title)
        engine_core_module.maybe_init_worker_tracer(
            "vllm.engine_core",
            "engine_core",
            process_title,
        )
        engine_core_module.decorate_logs()
        if parallel_config.numa_bind:
            engine_core_module.numa_utils.log_current_affinity_state(process_title)

        if data_parallel and vllm_config.kv_transfer_config is not None:
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
            )
            engine_core_module.logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        parallel_config.data_parallel_index = dp_rank
        if data_parallel and vllm_config.model_config.is_moe:
            parallel_config.data_parallel_rank = dp_rank
            # ### PATCH START: AFD async-DP Attention engine selection
            # Async-DP Attention ranks are connector-driven, so use the regular
            # EngineCoreProc instead of DPEngineCoreProc while keeping the
            # original DP rank metadata.
            if _is_afd_async_attention_config(vllm_config):
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
            else:
                engine_core = engine_core_module.DPEngineCoreProc(*args, **kwargs)
            # ### PATCH END: AFD async-DP Attention engine selection
        else:
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

        assert engine_core is not None

        def wakeup_engine() -> None:
            # Wakes up idle engine via input_queue when shutdown is requested
            # Not safe in a signal handler - we may interrupt the main thread
            # while it is holding the non-reentrant input_queue.mutex
            engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

        signal_callback = engine_core_module.SignalCallback(wakeup_engine)

        def signal_handler(signum, frame):
            signal_name = engine_core_module.signal.Signals(signum).name
            engine_core_module.logger.info(
                "[shutdown] EngineCore: trigger received signal=%s",
                signal_name,
            )
            engine_core.shutdown_state = (
                engine_core_module.EngineShutdownState.REQUESTED
            )
            signal_callback.trigger()

        engine_core_module.signal.signal(
            engine_core_module.signal.SIGTERM,
            signal_handler,
        )
        engine_core_module.signal.signal(
            engine_core_module.signal.SIGINT,
            signal_handler,
        )

        engine_core.run_busy_loop()

    except SystemExit:
        engine_core_module.logger.info_once("[shutdown] EngineCore: exiting busy loop")
        raise
    except Exception as exc:
        if engine_core is None:
            engine_core_module.logger.exception("EngineCore failed to start.")
        else:
            engine_core_module.logger.exception("EngineCore encountered a fatal error.")
            engine_core._send_engine_dead()
        raise exc
    finally:
        engine_core_module.signal.signal(
            engine_core_module.signal.SIGTERM,
            engine_core_module.signal.SIG_DFL,
        )
        engine_core_module.signal.signal(
            engine_core_module.signal.SIGINT,
            engine_core_module.signal.SIG_DFL,
        )
        if signal_callback is not None:
            signal_callback.stop()
        if engine_core is not None:
            engine_core.shutdown()


# Patch reason: vLLM enables MoE DP wave coordination when launching DP cores,
# while AFD async-DP only needs coordinator stats.
# Patch functionality: preserve upstream engine launch behavior but disable wave
# coordination for AFD async-DP configs.
# Signature: matches upstream; no added parameters.
@contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
    num_api_servers: int = 1,
) -> Iterator[
    tuple[
        CoreEngineProcManager | CoreEngineActorManager | None,
        DPCoordinator | None,
        EngineZmqAddresses,
        Queue | None,
    ]
]:
    """Disable coordinator wave mode while launching AFD async-DP engines."""

    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_rank = parallel_config.data_parallel_rank
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    offline_mode = local_start_index is not None

    tensor_queue: Queue | None = None
    multimodal_config = vllm_config.model_config.multimodal_config
    if multimodal_config is not None and multimodal_config.mm_tensor_ipc == "torch_shm":
        tensor_queue = engine_utils_module.get_mp_context().Queue()

    run_coordinator = (
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )

    if run_coordinator:
        coordinator = engine_utils_module.DPCoordinator(
            parallel_config,
            # ### PATCH START: AFD async-DP coordinator wave mode
            # Keep DP coordinator stats, but disable wave coordination for
            # connector-driven async-DP.
            enable_wave_coordination=(
                vllm_config.model_config.is_moe and not is_afd_async_dp(vllm_config)
            ),
            # ### PATCH END: AFD async-DP coordinator wave mode
        )

        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )

        engine_utils_module.logger.info(
            "Started DP Coordinator process (PID: %d)",
            coordinator.proc.pid,
        )
    else:
        coordinator = None

    if parallel_config.data_parallel_backend == "ray":
        engine_utils_module.logger.info("Starting ray-based data parallel backend")

        engine_actor_manager = engine_utils_module.CoreEngineActorManager(
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        yield engine_actor_manager, coordinator, addresses, tensor_queue
        return

    if offline_mode:
        assert local_engine_count == 1
        engines_to_handshake = [
            engine_utils_module.CoreEngine(index=dp_rank, local=True),
        ]
    elif dp_rank == 0:
        engines_to_handshake = [
            engine_utils_module.CoreEngine(index=i, local=(i < local_engine_count))
            for i in range(dp_size)
        ]
    else:
        assert local_engines_only, (
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible."
        )
        engines_to_handshake = [
            engine_utils_module.CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]

    handshake_local_only = offline_mode or local_engine_count == dp_size
    if parallel_config.enable_elastic_ep:
        handshake_local_only = False

    rpc_port = (
        parallel_config.data_parallel_rpc_port or engine_utils_module.get_open_port()
    )
    handshake_address = engine_utils_module.get_engine_client_zmq_addr(
        handshake_local_only,
        host,
        rpc_port,
    )

    if local_engines_only and dp_rank > 0:
        assert not handshake_local_only
        local_handshake_address = engine_utils_module.get_open_zmq_ipc_path()
        client_handshake_address = local_handshake_address
    else:
        local_handshake_address = handshake_address
        client_handshake_address = None

    with engine_utils_module.zmq_socket_ctx(
        local_handshake_address,
        engine_utils_module.zmq.ROUTER,
        bind=True,
    ) as handshake_socket:
        if local_engine_count:
            local_engine_manager = engine_utils_module.CoreEngineProcManager(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
                tensor_queue=tensor_queue,
            )
        else:
            local_engine_manager = None

        yield local_engine_manager, coordinator, addresses, tensor_queue

        engine_utils_module.wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and vllm_config.model_config.is_moe,
            vllm_config.cache_config,
            local_engine_manager,
            coordinator.proc if coordinator else None,
        )


# Patch reason: vLLM sends FIRST_REQ wakeups to coordinate DP waves, which AFD
# async-DP engines intentionally do not use.
# Patch functionality: preserve request routing and stats updates while skipping
# FIRST_REQ for AFD async-DP configs.
# Signature: matches upstream; no added parameters.
async def add_request_async(
    self,
    request: EngineCoreRequest,
) -> None:
    """Skip the DP wave ``FIRST_REQ`` notification for AFD async-DP."""

    self._ensure_stats_update_task()

    request.current_wave = self.current_wave
    request.client_index = self.client_index

    chosen_engine = self.get_core_engine_for_request(request)
    to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
    # ### PATCH START: AFD async-DP request wakeup
    # Async-DP engines step independently, so skip the coordinator FIRST_REQ
    # wakeup while preserving normal routing.
    if not self.engines_running and not is_afd_async_dp(self.vllm_config):
        req_msg = core_client_module.msgspec.msgpack.encode(
            ("FIRST_REQ", chosen_engine),
        )
        await self.first_req_send_socket.send(req_msg)
    # ### PATCH END: AFD async-DP request wakeup

    await to_await

    self._ensure_output_queue_task()


def _is_afd_async_attention_config(vllm_config: VllmConfig) -> bool:
    afd_config = parse_optional_afd_config(vllm_config, validate=False)
    return (
        afd_config is not None
        and is_afd_async_dp(vllm_config)
        and afd_config.role == "attention"
    )


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


if _is_target_vllm_compatible():
    EngineCoreProc.run_engine_core = staticmethod(run_engine_core)
    engine_utils_module.launch_core_engines = launch_core_engines
    core_client_module.launch_core_engines = launch_core_engines
    DPAsyncMPClient.add_request_async = add_request_async
    engine_core_module.logger.debug("AFD async-DP engine patch applied")


__all__: list[str] = []
