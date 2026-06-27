# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from afd_plugin.offline_schedule import OfflineCsvSchedule


class _Request:
    resumable = False

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id


class _Scheduler:
    log_stats = False

    def __init__(self) -> None:
        self.requests = {}
        self.enqueued_request_ids = []

    def _enqueue_waiting_request(self, request: _Request) -> None:
        self.enqueued_request_ids.append(request.request_id)


@pytest.fixture
def offline_scheduler_module(monkeypatch):
    _install_vllm_scheduler_stub(monkeypatch)
    sys.modules.pop("afd_plugin.v1.offline_scheduler", None)
    module = importlib.import_module("afd_plugin.v1.offline_scheduler")
    yield module
    sys.modules.pop("afd_plugin.v1.offline_scheduler", None)


def _install_vllm_scheduler_stub(monkeypatch) -> None:
    for module_name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
    ):
        module = types.ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)

    scheduler_module = types.ModuleType("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = object
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.scheduler", scheduler_module)


def test_offline_admission_gate_releases_current_step_atomically(
    offline_scheduler_module,
    tmp_path,
):
    schedule_csv = tmp_path / "schedule.csv"
    schedule_csv.write_text("10\n20\n30\n0\n40\n", encoding="utf-8")
    schedule = OfflineCsvSchedule.from_csv(str(schedule_csv), dp_size=1, dp_rank=0)
    scheduler = _Scheduler()
    gate = offline_scheduler_module.OfflineAdmissionGate(
        scheduler,
        schedule,
        request_indices=(0, 1, 2, 3),
    )

    gate.add_request(_Request("r0"))
    gate.add_request(_Request("r1"))

    assert scheduler.enqueued_request_ids == []

    gate.add_request(_Request("r2"))

    assert scheduler.enqueued_request_ids == ["r0", "r1", "r2"]

    gate.add_request(_Request("r3"))

    assert scheduler.enqueued_request_ids == ["r0", "r1", "r2"]

    gate.after_schedule(
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id="r0"),
                SimpleNamespace(req_id="r1"),
                SimpleNamespace(req_id="r2"),
            ]
        )
    )

    assert scheduler.enqueued_request_ids == ["r0", "r1", "r2", "r3"]
