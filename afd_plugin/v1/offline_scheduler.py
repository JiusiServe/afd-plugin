# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Offline CSV admission scheduler for vLLM V1."""

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Iterable
from typing import Any

from afd_plugin import envs
from afd_plugin.offline_schedule import OfflineCsvSchedule
from vllm.v1.core.sched.scheduler import Scheduler

logger = logging.getLogger(__name__)


class OfflineAdmissionGate:
    def __init__(
        self,
        scheduler: "OfflineCsvScheduler",
        schedule: OfflineCsvSchedule,
        request_indices: tuple[int, ...] | None,
    ) -> None:
        self.scheduler = scheduler
        self.schedule = schedule
        self.request_indices = request_indices
        self.next_arrival_index = 0
        self.current_step = 0
        self.future_requests: dict[int, Any] = {}
        self.request_id_to_index: dict[str, int] = {}
        self.admitted_current_step_indices: set[int] = set()

        self.release_current_step()

    def add_request(self, request: Any) -> bool:
        request_index = self.next_request_index()
        self.request_id_to_index[request.request_id] = request_index

        step = self.schedule.index_to_step[request_index]
        if step == self.current_step:
            return True

        if request.resumable:
            request.streaming_queue = deque()
        self.scheduler.requests[request.request_id] = request
        if self.scheduler.log_stats:
            from vllm.v1.engine import EngineCoreEventType

            request.record_event(EngineCoreEventType.QUEUED)
        self.future_requests[request_index] = request
        return False

    def after_schedule(self, scheduler_output: Any) -> None:
        current_indices = self.current_step_indices()
        if not current_indices:
            self.advance_empty_steps()
            return

        scheduled_indices = {
            self.request_id_to_index[data.req_id]
            for data in scheduler_output.scheduled_new_reqs
            if data.req_id in self.request_id_to_index
        }
        self.admitted_current_step_indices.update(
            idx for idx in scheduled_indices if idx in current_indices
        )

        if current_indices <= self.admitted_current_step_indices:
            logger.info(
                "[offline-scheduler] rank_step=%s admitted_requests=%s",
                self.current_step,
                sorted(current_indices),
            )
            self.current_step += 1
            self.admitted_current_step_indices.clear()
            self.release_current_step()
            self.advance_empty_steps()

    def finish_requests(self, request_ids: str | Iterable[str] | None) -> None:
        if isinstance(request_ids, str):
            requested_ids = {request_ids}
        elif request_ids is None:
            requested_ids = {
                request.request_id for request in self.future_requests.values()
            }
        else:
            requested_ids = set(request_ids)

        for request_index, request in list(self.future_requests.items()):
            if request.request_id not in requested_ids:
                continue
            self.future_requests.pop(request_index)
            self.request_id_to_index.pop(request.request_id, None)

    def num_unfinished_future_requests(self) -> int:
        return len(self.future_requests)

    def next_request_index(self) -> int:
        arrival_index = self.next_arrival_index
        self.next_arrival_index += 1
        if self.request_indices is None:
            request_index = arrival_index
        else:
            if arrival_index >= len(self.request_indices):
                raise ValueError(
                    "offline scheduler received more requests than "
                    "AFD_OFFLINE_SCHEDULER_REQUEST_INDICES describes"
                )
            request_index = self.request_indices[arrival_index]

        if request_index not in self.schedule.index_to_step:
            raise ValueError(
                f"request index {request_index} does not belong to this DP rank's "
                "offline CSV schedule"
            )
        return request_index

    def current_step_indices(self) -> set[int]:
        if self.current_step >= len(self.schedule.rank_steps):
            return set()
        return set(self.schedule.rank_steps[self.current_step])

    def release_current_step(self) -> None:
        if self.current_step >= len(self.schedule.rank_steps):
            return

        for request_index in self.schedule.rank_steps[self.current_step]:
            request = self.future_requests.pop(request_index, None)
            if request is not None:
                self.scheduler._enqueue_waiting_request(request)

    def advance_empty_steps(self) -> None:
        while (
            self.current_step < len(self.schedule.rank_steps)
            and not self.schedule.rank_steps[self.current_step]
        ):
            self.current_step += 1
            self.release_current_step()


class OfflineCsvScheduler(Scheduler):
    """Scheduler that admits new requests according to an offline CSV schedule."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        schedule_csv = os.environ[envs.AFD_OFFLINE_SCHEDULER_CSV]
        dp_size = int(os.environ[envs.AFD_OFFLINE_SCHEDULER_DP_SIZE])
        dp_rank = int(os.environ[envs.AFD_OFFLINE_SCHEDULER_DP_RANK])
        request_indices = _parse_request_indices(
            os.environ.get(envs.AFD_OFFLINE_SCHEDULER_REQUEST_INDICES, "")
        )
        schedule = OfflineCsvSchedule.from_csv(schedule_csv, dp_size, dp_rank)
        expected_request_indices = tuple(schedule.rank_request_indices())
        if request_indices is not None and request_indices != expected_request_indices:
            missing = sorted(set(expected_request_indices) - set(request_indices))
            extra = sorted(set(request_indices) - set(expected_request_indices))
            raise ValueError(
                "AFD_OFFLINE_SCHEDULER_REQUEST_INDICES must match this DP "
                "rank's CSV-assigned requests in order; "
                f"expected={len(expected_request_indices)} "
                f"received={len(request_indices)} "
                f"missing={_format_indices(missing)} "
                f"extra={_format_indices(extra)}"
            )
        self.offline_admission_gate = OfflineAdmissionGate(
            self,
            schedule,
            request_indices,
        )
        logger.info(
            "[offline-scheduler] enabled dp_rank=%s dp_size=%s steps=%s "
            "total_csv_requests=%s request_indices=%s submitted_requests=%s",
            dp_rank,
            dp_size,
            len(schedule.rank_steps),
            schedule.total_requests,
            "custom" if request_indices is not None else "arrival-order",
            len(expected_request_indices),
        )

    def add_request(self, request: Any) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            super().add_request(request)
            return

        should_enqueue_now = self.offline_admission_gate.add_request(request)
        if should_enqueue_now:
            super().add_request(request)

    def schedule(self) -> Any:
        scheduler_output = super().schedule()
        self.offline_admission_gate.after_schedule(scheduler_output)
        return scheduler_output

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: Any,
    ) -> list[tuple[str, int]]:
        finished = super().finish_requests(request_ids, finished_status)
        self.offline_admission_gate.finish_requests(request_ids)
        return finished

    def get_num_unfinished_requests(self) -> int:
        return (
            super().get_num_unfinished_requests()
            + self.offline_admission_gate.num_unfinished_future_requests()
        )

    def get_request_counts(self) -> tuple[int, int]:
        running, waiting = super().get_request_counts()
        waiting += self.offline_admission_gate.num_unfinished_future_requests()
        return running, waiting


def _parse_request_indices(value: str) -> tuple[int, ...] | None:
    if not value:
        return None
    return tuple(int(item) for item in value.split(",") if item)


def _format_indices(indices: list[int], limit: int = 8) -> str:
    if not indices:
        return "[]"
    shown = ",".join(str(index) for index in indices[:limit])
    suffix = "" if len(indices) <= limit else f",...(+{len(indices) - limit})"
    return f"[{shown}{suffix}]"


__all__ = ["OfflineCsvScheduler"]
