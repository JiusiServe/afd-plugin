# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared process-group cleanup for E2E subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Sequence


def terminate_process_groups(
    processes: Sequence[subprocess.Popen[str]],
    *,
    termination_timeout_s: float,
    poll_interval_s: float,
    reap_timeout_s: float,
    process_name: str = "",
) -> list[str]:
    """Terminate process groups with one deadline and reap their leaders."""
    failures: list[str] = []
    live_pgids: list[int] = []
    group_description = f"{process_name} process group".strip()
    process_description = f"{process_name} process".strip()

    for process in processes:
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except OSError as exc:
                failures.append(
                    f"SIGTERM failed for {group_description} {process.pid}: {exc}",
                )
        except OSError as exc:
            failures.append(
                f"SIGTERM failed for {process_description} {process.pid}: {exc}",
            )
        live_pgids.append(process.pid)

    deadline = time.monotonic() + termination_timeout_s
    leaders_to_poll = list(processes)
    while live_pgids:
        unreaped_leaders: list[subprocess.Popen[str]] = []
        for process in leaders_to_poll:
            try:
                returncode = process.poll()
            except Exception as exc:
                failures.append(
                    f"poll failed for {process_description} {process.pid}: {exc}",
                )
                continue
            if returncode is None:
                unreaped_leaders.append(process)
        leaders_to_poll = unreaped_leaders

        surviving_pgids: list[int] = []
        for pgid in live_pgids:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                continue
            except OSError as exc:
                failures.append(
                    f"liveness check failed for {group_description} {pgid}: {exc}",
                )
            surviving_pgids.append(pgid)
        live_pgids = surviving_pgids
        if not live_pgids or time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)

    forced_pgids: list[int] = []
    for pgid in live_pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError as exc:
            failures.append(
                f"SIGKILL failed for {group_description} {pgid}: {exc}",
            )
            forced_pgids.append(pgid)
        else:
            failures.append(
                f"forced SIGKILL for {group_description} {pgid} after "
                f"{termination_timeout_s}s timeout",
            )
            forced_pgids.append(pgid)

    reap_deadline = time.monotonic() + reap_timeout_s
    for process in processes:
        try:
            process.wait(timeout=max(reap_deadline - time.monotonic(), 0))
        except Exception as exc:
            failures.append(
                f"wait failed for {process_description} {process.pid}: {exc}",
            )

    surviving_pgids = forced_pgids
    while surviving_pgids:
        still_alive: list[int] = []
        for pgid in surviving_pgids:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                continue
            except OSError as exc:
                failures.append(
                    f"post-SIGKILL liveness check failed for "
                    f"{group_description} {pgid}: {exc}",
                )
            still_alive.append(pgid)
        surviving_pgids = still_alive
        if not surviving_pgids or time.monotonic() >= reap_deadline:
            break
        time.sleep(poll_interval_s)

    for pgid in surviving_pgids:
        failures.append(
            f"{group_description} {pgid} still alive after SIGKILL",
        )

    return failures
