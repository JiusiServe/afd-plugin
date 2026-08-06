# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import signal

from tests.e2e import process_utils


def test_terminate_process_groups_signals_the_leader_before_the_group(
    monkeypatch,
):
    class FakeProcess:
        pid = 101

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    group_alive = True
    leader_signal_calls = []
    group_signal_calls = []

    def fake_kill(pid, sig):
        nonlocal group_alive
        leader_signal_calls.append((pid, sig))
        group_alive = False

    def fake_killpg(pid, sig):
        nonlocal group_alive
        group_signal_calls.append((pid, sig))
        if sig == signal.SIGTERM:
            group_alive = False
        if sig == 0 and not group_alive:
            raise ProcessLookupError

    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [FakeProcess()],
        termination_timeout_s=20,
        poll_interval_s=0.2,
        reap_timeout_s=5,
    )

    assert failures == []
    assert leader_signal_calls == [(101, signal.SIGTERM)]
    assert group_signal_calls == [(101, 0)]


def test_terminate_process_groups_cleans_children_after_leader_exits(monkeypatch):
    class ExitedLeader:
        pid = 101

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    group_alive = True

    def leader_is_gone(*_args):
        raise ProcessLookupError

    def fake_killpg(_pid, sig):
        nonlocal group_alive
        if sig == signal.SIGTERM:
            group_alive = False
        elif not group_alive:
            raise ProcessLookupError

    monkeypatch.setattr(process_utils.os, "kill", leader_is_gone, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [ExitedLeader()],
        termination_timeout_s=20,
        poll_interval_s=0.2,
        reap_timeout_s=5,
    )

    assert failures == []
    assert group_alive is False


def test_terminate_process_groups_reports_force_kill_escalation(monkeypatch):
    class FakeProcess:
        pid = 101

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    group_alive = True

    def fake_kill(_pid, _sig):
        return None

    def fake_killpg(_pid, sig):
        nonlocal group_alive
        if sig == 0 and not group_alive:
            raise ProcessLookupError
        if sig == 9:
            group_alive = False

    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [FakeProcess()],
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
    )

    assert any("forced SIGKILL" in failure for failure in failures)
    assert not any("still alive after SIGKILL" in failure for failure in failures)


def test_terminate_process_groups_reports_a_group_that_survives_sigkill(
    monkeypatch,
):
    class FakeProcess:
        pid = 101

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    group_liveness_checks = 0

    def fake_kill(_pid, _sig):
        return None

    def fake_killpg(_pid, sig):
        nonlocal group_liveness_checks
        if sig == 0:
            group_liveness_checks += 1

    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: 100.0)

    failures = process_utils.terminate_process_groups(
        [FakeProcess()],
        termination_timeout_s=0,
        poll_interval_s=0,
        reap_timeout_s=0,
    )

    assert any("forced SIGKILL" in failure for failure in failures)
    assert any("still alive after SIGKILL" in failure for failure in failures)
    assert group_liveness_checks >= 2


def test_terminate_process_groups_uses_one_deadline_and_reaps_every_process(
    monkeypatch,
):
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.returncode = None
            self.wait_calls = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            self.returncode = 0

    first_process = FakeProcess(101)
    second_process = FakeProcess(102)
    leader_signal_calls = []
    group_signal_calls = []

    def fake_kill(pid, sig):
        leader_signal_calls.append((pid, sig))
        if pid == first_process.pid and sig == signal.SIGTERM:
            first_process.returncode = 0

    def fake_killpg(pid, sig):
        group_signal_calls.append((pid, sig))
        process = first_process if pid == first_process.pid else second_process
        if sig == 0 and process.returncode is not None:
            raise ProcessLookupError
        if pid == second_process.pid and sig == signal.SIGKILL:
            second_process.returncode = 0

    monotonic_values = iter([100.0, 121.0, 200.0, 201.0, 204.0])
    monkeypatch.setattr(process_utils.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(process_utils.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        process_utils.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    failures = process_utils.terminate_process_groups(
        [first_process, second_process],
        termination_timeout_s=20,
        poll_interval_s=0.2,
        reap_timeout_s=5,
    )

    assert len(failures) == 1
    assert "forced SIGKILL" in failures[0]
    assert leader_signal_calls == [
        (101, signal.SIGTERM),
        (102, signal.SIGTERM),
    ]
    assert group_signal_calls == [
        (101, 0),
        (102, 0),
        (102, 9),
        (102, 0),
    ]
    assert first_process.wait_calls == [4]
    assert second_process.wait_calls == [1]
