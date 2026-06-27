from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def _config(
    *,
    connector: str = "afdasyncconnector",
    role: str = "attention",
    async_dp: bool = True,
    is_moe: bool = True,
    data_parallel_size: int = 2,
):
    return SimpleNamespace(
        additional_config={
            "afd": {
                "enabled": True,
                "connector": connector,
                "role": role,
                "async": async_dp,
            },
        },
        model_config=SimpleNamespace(is_moe=is_moe),
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_size_local=data_parallel_size,
            data_parallel_rank=0,
            data_parallel_rank_local=0,
        ),
    )


def _install_fake_vllm_engine(monkeypatch: pytest.MonkeyPatch):
    vllm_module = types.ModuleType("vllm")
    vllm_v1_module = types.ModuleType("vllm.v1")
    engine_module = types.ModuleType("vllm.v1.engine")
    core_module = types.ModuleType("vllm.v1.engine.core")
    utils_module = types.ModuleType("vllm.v1.engine.utils")
    client_module = types.ModuleType("vllm.v1.engine.core_client")

    engine_module.EngineCoreRequestType = SimpleNamespace(ADD="ADD")

    class EngineCoreProc:
        def __init__(self, *args, engine_index=0, **kwargs):
            del args, kwargs
            self.kind = "engine"
            self.engine_index = engine_index

        @staticmethod
        def run_engine_core(*args, dp_rank=0, local_dp_rank=0, **kwargs):
            del local_dp_rank
            vllm_config = kwargs["vllm_config"]
            if (
                vllm_config.parallel_config.data_parallel_size > 1
                and vllm_config.model_config.is_moe
            ):
                return core_module.DPEngineCoreProc(*args, **kwargs)
            return EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

    class DPEngineCoreProc(EngineCoreProc):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.kind = "dp"

    core_module.EngineCoreProc = EngineCoreProc
    core_module.DPEngineCoreProc = DPEngineCoreProc
    core_module.logger = logging.getLogger("fake-async-dp-engine")

    class DPCoordinator:
        def __init__(self, parallel_config, enable_wave_coordination=True):
            self.parallel_config = parallel_config
            self.enable_wave_coordination = enable_wave_coordination

    @contextmanager
    def launch_core_engines(vllm_config, executor_class, log_stats, addresses,
                            num_api_servers=1):
        del executor_class, log_stats, addresses, num_api_servers
        yield utils_module.DPCoordinator(
            vllm_config.parallel_config,
            enable_wave_coordination=True,
        )

    utils_module.DPCoordinator = DPCoordinator
    utils_module.launch_core_engines = launch_core_engines
    client_module.launch_core_engines = launch_core_engines

    class DPAsyncMPClient:
        async def add_request_async(self, request):
            self._ensure_stats_update_task()
            request.current_wave = self.current_wave
            request.client_index = self.client_index
            chosen_engine = self.get_core_engine_for_request(request)
            to_await = self._send_input("ADD", request, chosen_engine)
            if not self.engines_running:
                await self.first_req_send_socket.send(("FIRST_REQ", chosen_engine))
            await to_await
            self._ensure_output_queue_task()

    client_module.DPAsyncMPClient = DPAsyncMPClient

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.utils", utils_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core_client", client_module)
    return core_module, utils_module, client_module


def _load_patch_module(monkeypatch: pytest.MonkeyPatch):
    _install_fake_vllm_engine(monkeypatch)
    module_name = "afd_plugin.compat.patches.async_dp_engine"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def test_async_dp_attention_uses_regular_engine_core(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    core_module = sys.modules["vllm.v1.engine.core"]
    patch_module.apply_async_dp_engine_patch()
    patch_module.apply_async_dp_engine_patch()

    engine = core_module.EngineCoreProc.run_engine_core(
        vllm_config=_config(),
        dp_rank=1,
    )

    assert engine.kind == "engine"
    assert engine.engine_index == 1


def test_async_dp_engine_patch_preserves_non_async_moe_dp(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    core_module = sys.modules["vllm.v1.engine.core"]
    patch_module.apply_async_dp_engine_patch()

    engine = core_module.EngineCoreProc.run_engine_core(
        vllm_config=_config(connector="camp2pconnector", async_dp=False),
        dp_rank=1,
    )

    assert engine.kind == "dp"


def test_async_dp_coordinator_disables_wave_coordination(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    utils_module = sys.modules["vllm.v1.engine.utils"]
    client_module = sys.modules["vllm.v1.engine.core_client"]
    patch_module.apply_async_dp_engine_patch()

    with utils_module.launch_core_engines(
        _config(),
        object,
        False,
        object(),
    ) as coordinator:
        assert coordinator.enable_wave_coordination is False

    assert client_module.launch_core_engines is utils_module.launch_core_engines


def test_async_dp_client_skips_first_req(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    client_module = sys.modules["vllm.v1.engine.core_client"]
    patch_module.apply_async_dp_engine_patch()

    class Sender:
        def __init__(self):
            self.messages = []

        async def send(self, message):
            self.messages.append(message)

    client = client_module.DPAsyncMPClient()
    client.vllm_config = _config()
    client.current_wave = 7
    client.client_index = 3
    client.engines_running = False
    client.first_req_send_socket = Sender()
    client.stats_ready = False
    client.output_ready = False
    client.sent_inputs = []
    client._ensure_stats_update_task = lambda: setattr(client, "stats_ready", True)
    client._ensure_output_queue_task = lambda: setattr(client, "output_ready", True)
    client.get_core_engine_for_request = lambda request: 1

    async def send_input(request_type, request, engine):
        client.sent_inputs.append((request_type, request, engine))

    client._send_input = send_input
    request = SimpleNamespace()

    asyncio.run(client.add_request_async(request))

    assert request.current_wave == 7
    assert request.client_index == 3
    assert client.first_req_send_socket.messages == []
    assert client.stats_ready is True
    assert client.output_ready is True
