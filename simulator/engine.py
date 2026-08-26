"""Discrete-event simulation for merged and CAMAsync AFD Prefill."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from simulator.config import OutputConfig, SimulationConfig
from simulator.profiles import ProfileBundle
from simulator.workload import RequestSpec, RuntimeRequest, generate_workload

AFD_DP_COUNT = 2
MERGED_DP_COUNT = 4


@dataclass(frozen=True)
class BatchSegment:
    request: RuntimeRequest
    prefix_tokens: int
    query_tokens: int


@dataclass(frozen=True)
class SchedulerBatch:
    batch_id: int
    dp: int
    segments: tuple[BatchSegment, ...]

    @property
    def query_tokens(self) -> int:
        return sum(segment.query_tokens for segment in self.segments)


@dataclass(frozen=True)
class AfdStage:
    segments: tuple[BatchSegment, ...]

    @property
    def query_tokens(self) -> int:
        return sum(segment.query_tokens for segment in self.segments)


@dataclass
class AfdDpState:
    dp: int
    batch: SchedulerBatch
    stages: tuple[AfdStage, ...]
    ops: list[tuple[str, int, int]]
    time_ms: float
    op_index: int = 0


@dataclass(frozen=True)
class FfnJob:
    end_ms: float


@dataclass(frozen=True)
class PendingFfnJob:
    key: str
    arrival_ms: float
    layer: int
    batch: int
    stage: int
    tokens: int


class RequestDispatcher:
    """Route requests to DP queues when they arrive."""

    def __init__(
        self,
        specs: tuple[RequestSpec, ...],
        dp_count: int,
        policy: str,
    ) -> None:
        self.dp_count = dp_count
        self.policy = policy
        self.requests = [RuntimeRequest(spec=spec, assigned_dp=-1) for spec in specs]
        self.pending = sorted(
            enumerate(self.requests),
            key=lambda item: (item[1].spec.arrival_ms, item[0]),
        )
        self.pending_index = 0
        self.round_robin_index = 0
        self.tie_start_index = 0
        self.queues: list[list[RuntimeRequest]] = [[] for _ in range(dp_count)]
        self.active: list[deque[RuntimeRequest]] = [deque() for _ in range(dp_count)]

    @property
    def has_pending(self) -> bool:
        return self.pending_index < len(self.pending)

    @property
    def next_arrival_ms(self) -> float:
        if not self.has_pending:
            return math.inf
        return self.pending[self.pending_index][1].spec.arrival_ms

    def dispatch_until(self, time_ms: float) -> None:
        while self.next_arrival_ms <= time_ms:
            request = self.pending[self.pending_index][1]
            self.pending_index += 1
            dp = self._select_dp(request.spec.arrival_ms)
            request.assigned_dp = dp
            self.queues[dp].append(request)
            if self.policy == "vllm_queue_aware":
                self.active[dp].append(request)

    def _select_dp(self, arrival_ms: float) -> int:
        if self.policy == "round_robin":
            dp = self.round_robin_index
            self.round_robin_index = (self.round_robin_index + 1) % self.dp_count
            return dp

        loads = []
        for dp in range(self.dp_count):
            active = self.active[dp]
            while (
                active
                and active[0].completion_ms is not None
                and active[0].completion_ms <= arrival_ms
            ):
                active.popleft()
            loads.append(len(active))
        min_load = min(loads)
        for offset in range(self.dp_count):
            dp = (self.tie_start_index + offset) % self.dp_count
            if loads[dp] == min_load:
                self.tie_start_index = (self.tie_start_index + 1) % self.dp_count
                return dp
        raise RuntimeError("DP dispatcher failed to select an engine")


def compare_architectures(
    config: SimulationConfig,
    profiles: ProfileBundle,
    requests: tuple[RequestSpec, ...] | None = None,
) -> dict[str, Any]:
    """Run one identical workload through AFD and merged architectures."""

    workload = requests if requests is not None else generate_workload(config)
    if not workload:
        raise ValueError("workload contains no requests")
    _validate_workload(config, workload)
    afd = simulate_afd(config, profiles, workload)
    merged = simulate_merged(config, profiles, workload)
    return {
        "config": config.to_mapping(),
        "profile": profiles.summary(),
        "workload": {
            "request_count": len(workload),
            "input_tokens": sum(item.input_tokens for item in workload),
            "cached_tokens": sum(item.cached_prefix_tokens for item in workload),
        },
        "afd": afd,
        "merged": merged,
        "comparison": _comparison(afd["summary"], merged["summary"]),
    }


def simulate_merged(
    config: SimulationConfig,
    profiles: ProfileBundle,
    specs: tuple[RequestSpec, ...],
) -> dict[str, Any]:
    dispatcher = RequestDispatcher(specs, MERGED_DP_COUNT, config.scheduler.policy)
    requests = dispatcher.requests
    queues = dispatcher.queues
    timeline: list[dict[str, Any]] = []
    wave_start = 0.0
    batch_id = 0
    total_barrier_ms = 0.0
    busy_ms = [0.0 for _ in range(MERGED_DP_COUNT)]
    ep_busy_ms = 0.0

    while dispatcher.has_pending or any(queues):
        dispatcher.dispatch_until(wave_start)
        next_ready = min(
            (_request_ready_ms(queue[0]) for queue in queues if queue),
            default=math.inf,
        )
        if dispatcher.next_arrival_ms < next_ready:
            wave_start = max(wave_start, dispatcher.next_arrival_ms)
            dispatcher.dispatch_until(wave_start)
            continue
        wave_start = max(wave_start, next_ready)
        dispatcher.dispatch_until(wave_start)
        batches = []
        for dp, queue in enumerate(queues):
            batch = _pack_batch(
                config,
                queue,
                dp=dp,
                batch_id=batch_id,
                start_ms=wave_start,
            )
            batches.append(batch)
            if batch is not None:
                batch_id += 1
        if not any(batches):
            wave_start = next_ready
            continue

        layer_start = wave_start
        for layer_idx in range(profiles.layer_count):
            attention_ends: list[float | None] = []
            for dp, batch in enumerate(batches):
                if batch is None:
                    attention_ends.append(None)
                    continue
                duration = _segments_duration(
                    profiles,
                    "merged",
                    layer_idx,
                    "attention_router",
                    batch.segments,
                )
                end = layer_start + duration
                attention_ends.append(end)
                busy_ms[dp] += duration
                _append_timeline(
                    timeline,
                    config,
                    architecture="merged",
                    resource=f"DP{dp} Attention",
                    phase="attention_router",
                    start_ms=layer_start,
                    end_ms=end,
                    layer=layer_idx,
                    batch=batch.batch_id,
                    tokens=batch.query_tokens,
                )
            active_ends = [value for value in attention_ends if value is not None]
            barrier_end = max(active_ends, default=layer_start)
            for dp, end in enumerate(attention_ends):
                wait_start = layer_start if end is None else end
                if wait_start < barrier_end:
                    total_barrier_ms += barrier_end - wait_start
                    _append_timeline(
                        timeline,
                        config,
                        architecture="merged",
                        resource=f"DP{dp} Attention",
                        phase="barrier",
                        start_ms=wait_start,
                        end_ms=barrier_end,
                        layer=layer_idx,
                        batch=None,
                        tokens=0,
                    )

            global_ep_tokens = sum(
                batch.query_tokens for batch in batches if batch is not None
            )
            global_profile_query = max(1.0, global_ep_tokens / MERGED_DP_COUNT)
            max_dp_tokens = max(
                batch.query_tokens for batch in batches if batch is not None
            )
            cursor = barrier_end
            for phase in (
                "merged_dispatch",
                "routed_experts",
                "merged_combine",
                "merged_combine_local",
                "shared_expert",
                "merged_sp_post",
            ):
                is_global_phase = phase in {
                    "merged_dispatch",
                    "routed_experts",
                    "merged_combine",
                }
                phase_tokens = global_ep_tokens if is_global_phase else max_dp_tokens
                profile_query_tokens = (
                    global_profile_query if is_global_phase else max_dp_tokens
                )
                duration = profiles.duration_ms(
                    "merged", layer_idx, phase, 0, profile_query_tokens
                )
                end = cursor + duration
                ep_busy_ms += duration
                _append_timeline(
                    timeline,
                    config,
                    architecture="merged",
                    resource="Global EP16",
                    phase=phase,
                    start_ms=cursor,
                    end_ms=end,
                    layer=layer_idx,
                    batch=None,
                    tokens=phase_tokens,
                    profile_query_tokens=profile_query_tokens,
                    expert_details=_expert_event_details(
                        profiles,
                        "merged",
                        layer_idx,
                        phase,
                        profile_query_tokens,
                    ),
                )
                cursor = end
            layer_start = cursor

        wave_start = layer_start
        for batch in batches:
            if batch is not None:
                _complete_batch(batch, wave_start)

    makespan = max(request.completion_ms or 0.0 for request in requests)
    utilization = {
        f"attention_dp{dp}": value / makespan if makespan else 0.0
        for dp, value in enumerate(busy_ms)
    }
    utilization["global_ep16"] = ep_busy_ms / makespan if makespan else 0.0
    result = _build_result(
        "merged",
        config,
        requests,
        timeline,
        makespan,
        utilization,
    )
    result["summary"]["barrier_wait_ms"] = total_barrier_ms
    return result


def simulate_afd(
    config: SimulationConfig,
    profiles: ProfileBundle,
    specs: tuple[RequestSpec, ...],
) -> dict[str, Any]:
    dispatcher = RequestDispatcher(specs, AFD_DP_COUNT, config.scheduler.policy)
    dispatcher.dispatch_until(0.0)
    requests = dispatcher.requests
    queues = dispatcher.queues
    timeline: list[dict[str, Any]] = []
    states: dict[int, AfdDpState] = {}
    next_batch_id = 0
    jobs: dict[str, FfnJob] = {}
    pending_jobs: list[PendingFfnJob] = []
    ffn_available_ms = 0.0
    ffn_busy_ms = 0.0
    attention_busy_ms = [0.0 for _ in range(AFD_DP_COUNT)]
    attention_wait_ms = 0.0
    dp_available_ms = [0.0 for _ in range(AFD_DP_COUNT)]

    while states or dispatcher.has_pending or any(queues):
        candidates = []
        for state in states.values():
            op_type, stage_idx, layer_idx = state.ops[state.op_index]
            ready_ms = state.time_ms
            if op_type == "combine":
                job = jobs.get(_job_key(state.batch.batch_id, stage_idx, layer_idx))
                if job is None:
                    continue
                ready_ms = max(ready_ms, job.end_ms)
            candidates.append((ready_ms, state.dp, op_type, stage_idx, layer_idx))

        next_state_op = min(candidates) if candidates else None
        next_ffn_job = min(
            pending_jobs,
            key=lambda item: (item.arrival_ms, item.batch, item.stage, item.layer),
            default=None,
        )
        ffn_ready_ms = (
            max(ffn_available_ms, next_ffn_job.arrival_ms)
            if next_ffn_job is not None
            else math.inf
        )
        idle_ready = [
            (
                max(dp_available_ms[dp], _request_ready_ms(queues[dp][0])),
                dp,
            )
            for dp in range(AFD_DP_COUNT)
            if dp not in states and queues[dp]
        ]
        next_idle = min(idle_ready, default=(math.inf, -1))
        next_state_ms = next_state_op[0] if next_state_op is not None else math.inf
        next_operation_ms = min(next_state_ms, ffn_ready_ms, next_idle[0])
        if dispatcher.next_arrival_ms <= next_operation_ms:
            dispatcher.dispatch_until(dispatcher.next_arrival_ms)
            continue
        if next_idle[0] <= min(next_state_ms, ffn_ready_ms):
            start_ms, idle_dp = next_idle
            state, next_batch_id = _load_afd_state(
                config,
                queues[idle_dp],
                idle_dp,
                current_time_ms=start_ms,
                next_batch_id=next_batch_id,
                layer_count=profiles.layer_count,
            )
            if state is None:
                raise RuntimeError("AFD scheduler failed to load a ready batch")
            states[idle_dp] = state
            continue
        if next_ffn_job is not None and (
            next_state_op is None or next_state_op[0] > ffn_ready_ms
        ):
            pending_jobs.remove(next_ffn_job)
            recv_start = ffn_ready_ms
            recv_end = recv_start + config.cam.dispatch_recv.latency_ms(
                next_ffn_job.tokens
            )
            compute_duration = sum(
                profiles.duration_ms(
                    "afd",
                    next_ffn_job.layer,
                    phase,
                    0,
                    next_ffn_job.tokens,
                )
                for phase in ("routed_experts", "shared_expert")
            )
            compute_end = recv_end + compute_duration
            combine_send_end = compute_end + config.cam.combine_send.latency_ms(
                next_ffn_job.tokens
            )
            for phase, start, end in (
                ("dispatch_recv", recv_start, recv_end),
                ("ffn_compute", recv_end, compute_end),
                ("combine_send", compute_end, combine_send_end),
            ):
                _append_timeline(
                    timeline,
                    config,
                    architecture="afd",
                    resource="FFN EP8",
                    phase=phase,
                    start_ms=start,
                    end_ms=end,
                    layer=next_ffn_job.layer,
                    batch=next_ffn_job.batch,
                    stage=next_ffn_job.stage,
                    tokens=next_ffn_job.tokens,
                    profile_query_tokens=(
                        next_ffn_job.tokens if phase == "ffn_compute" else None
                    ),
                    expert_details=_expert_event_details(
                        profiles,
                        "afd",
                        next_ffn_job.layer,
                        phase,
                        next_ffn_job.tokens,
                    ),
                )
            ffn_busy_ms += combine_send_end - recv_start
            ffn_available_ms = combine_send_end
            jobs[next_ffn_job.key] = FfnJob(end_ms=combine_send_end)
            continue

        if next_state_op is None:
            raise RuntimeError("AFD event loop has no runnable operation")
        _, dp, op_type, stage_idx, layer_idx = next_state_op
        state = states[dp]
        stage = state.stages[stage_idx]

        if op_type == "attention":
            start = state.time_ms
            duration = _segments_duration(
                profiles,
                "afd",
                layer_idx,
                "attention_router",
                stage.segments,
            )
            end = start + duration
            attention_busy_ms[dp] += duration
            _append_timeline(
                timeline,
                config,
                architecture="afd",
                resource=f"DP{dp} Attention",
                phase="attention_router",
                start_ms=start,
                end_ms=end,
                layer=layer_idx,
                batch=state.batch.batch_id,
                stage=stage_idx,
                tokens=stage.query_tokens,
            )
            state.time_ms = end
        elif op_type == "dispatch":
            send_start = state.time_ms
            send_end = send_start + config.cam.dispatch_send.latency_ms(
                stage.query_tokens
            )
            _append_timeline(
                timeline,
                config,
                architecture="afd",
                resource=f"DP{dp} CAM",
                phase="dispatch_send",
                start_ms=send_start,
                end_ms=send_end,
                layer=layer_idx,
                batch=state.batch.batch_id,
                stage=stage_idx,
                tokens=stage.query_tokens,
            )
            key = _job_key(state.batch.batch_id, stage_idx, layer_idx)
            pending_jobs.append(
                PendingFfnJob(
                    key=key,
                    arrival_ms=send_end,
                    layer=layer_idx,
                    batch=state.batch.batch_id,
                    stage=stage_idx,
                    tokens=stage.query_tokens,
                )
            )
            state.time_ms = send_end
        else:
            job = jobs.pop(_job_key(state.batch.batch_id, stage_idx, layer_idx))
            if job.end_ms > state.time_ms:
                _append_timeline(
                    timeline,
                    config,
                    architecture="afd",
                    resource=f"DP{dp} Attention",
                    phase="wait_ffn",
                    start_ms=state.time_ms,
                    end_ms=job.end_ms,
                    layer=layer_idx,
                    batch=state.batch.batch_id,
                    stage=stage_idx,
                    tokens=stage.query_tokens,
                )
                attention_wait_ms += job.end_ms - state.time_ms
                state.time_ms = job.end_ms
            recv_start = state.time_ms
            post_duration = profiles.duration_ms(
                "afd", layer_idx, "afd_post", 0, stage.query_tokens
            )
            recv_end = (
                recv_start
                + config.cam.combine_recv.latency_ms(stage.query_tokens)
                + post_duration
            )
            _append_timeline(
                timeline,
                config,
                architecture="afd",
                resource=f"DP{dp} CAM",
                phase="combine_recv_post",
                start_ms=recv_start,
                end_ms=recv_end,
                layer=layer_idx,
                batch=state.batch.batch_id,
                stage=stage_idx,
                tokens=stage.query_tokens,
            )
            state.time_ms = recv_end

        state.op_index += 1
        if state.op_index == len(state.ops):
            _complete_batch(state.batch, state.time_ms)
            dp_available_ms[dp] = state.time_ms
            del states[dp]

    makespan = max(request.completion_ms or 0.0 for request in requests)
    utilization = {
        f"attention_dp{dp}": value / makespan if makespan else 0.0
        for dp, value in enumerate(attention_busy_ms)
    }
    utilization["ffn_ep8"] = ffn_busy_ms / makespan if makespan else 0.0
    result = _build_result("afd", config, requests, timeline, makespan, utilization)
    result["summary"]["attention_wait_ms"] = attention_wait_ms
    result["summary"]["cam_calibrated"] = config.cam.calibrated
    return result


def sweep_qps(config: SimulationConfig, profiles: ProfileBundle) -> dict[str, Any]:
    """Find the highest SLO-feasible QPS with coarse search and refinement."""

    if config.mode != "continuous":
        raise ValueError("QPS sweep requires mode='continuous'")
    if config.arrival.kind == "trace":
        raise ValueError("QPS sweep is unavailable for exact timestamp trace replay")
    sweep = config.sweep
    ratio = sweep.max_qps / sweep.min_qps
    coarse = [
        sweep.min_qps * ratio ** (index / (sweep.coarse_points - 1))
        for index in range(sweep.coarse_points)
    ]
    cache: dict[float, dict[str, Any]] = {}
    compact_output = OutputConfig(
        include_timeline=False,
        timeline_max_events=config.output.timeline_max_events,
        include_requests=False,
    )

    def evaluate(qps: float) -> dict[str, Any]:
        key = round(qps, 9)
        if key not in cache:
            scenario = replace(
                config,
                arrival=replace(config.arrival, qps=qps),
                output=compact_output,
            )
            cache[key] = compare_architectures(scenario, profiles)
        return cache[key]

    for qps in coarse:
        evaluate(qps)

    capacities = {}
    for architecture in ("afd", "merged"):
        passed = [
            qps for qps in coarse if _sweep_pass(evaluate(qps), architecture, config)
        ]
        failed = [
            qps
            for qps in coarse
            if not _sweep_pass(evaluate(qps), architecture, config)
        ]
        lower = max(passed, default=0.0)
        upper_candidates = [qps for qps in failed if qps > lower]
        upper = min(upper_candidates, default=sweep.max_qps)
        if lower > 0 and upper > lower:
            for _ in range(sweep.refinement_steps):
                midpoint = (lower + upper) / 2.0
                if _sweep_pass(evaluate(midpoint), architecture, config):
                    lower = midpoint
                else:
                    upper = midpoint
        capacities[architecture] = lower

    series = {"afd": [], "merged": []}
    for qps in sorted(cache):
        result = cache[qps]
        for architecture in series:
            summary = result[architecture]["summary"]
            series[architecture].append(
                {
                    "qps": qps,
                    "throughput_rps": summary["throughput_rps"],
                    "ttft_p50_ms": summary["ttft_p50_ms"],
                    "ttft_p90_ms": summary["ttft_p90_ms"],
                    "ttft_p99_ms": summary["ttft_p99_ms"],
                    "slo_attainment": summary["slo_attainment"],
                    "passed": _sweep_pass(result, architecture, config),
                }
            )
    return {
        "capacity_qps": capacities,
        "series": series,
        "slo": {
            "ttft_limit_ms": config.slo.ttft_limit_ms,
            "target_ratio": config.slo.target_ratio,
            "throughput_tolerance_ratio": sweep.throughput_tolerance_ratio,
        },
    }


def _sweep_pass(
    result: dict[str, Any], architecture: str, config: SimulationConfig
) -> bool:
    summary = result[architecture]["summary"]
    offered = summary["offered_qps"]
    return (
        summary["slo_attainment"] >= config.slo.target_ratio
        and summary["throughput_rps"]
        >= offered * config.sweep.throughput_tolerance_ratio
    )


def _request_ready_ms(request: RuntimeRequest) -> float:
    return request.spec.arrival_ms + request.spec.cache_lookup_ms


def _pack_batch(
    config: SimulationConfig,
    queue: list[RuntimeRequest],
    *,
    dp: int,
    batch_id: int,
    start_ms: float,
) -> SchedulerBatch | None:
    budget = config.scheduler.max_num_batched_tokens
    segments = []
    while queue and len(segments) < config.scheduler.max_num_seqs and budget > 0:
        request = queue[0]
        if _request_ready_ms(request) > start_ms:
            break
        remaining = request.remaining_query_tokens
        if not config.scheduler.chunked_prefill:
            if remaining > budget:
                if not segments:
                    raise ValueError(
                        f"request {request.spec.request_id} has {remaining} "
                        "uncached tokens, "
                        "exceeding non-chunked max_num_batched_tokens"
                    )
                break
            query_tokens = remaining
        else:
            query_tokens = min(remaining, config.scheduler.chunk_size, budget)
        prefix_tokens = request.current_prefix_tokens
        request.computed_query_tokens += query_tokens
        if request.first_scheduled_ms is None:
            request.first_scheduled_ms = start_ms
        segments.append(
            BatchSegment(
                request=request,
                prefix_tokens=prefix_tokens,
                query_tokens=query_tokens,
            )
        )
        budget -= query_tokens
        if request.remaining_query_tokens == 0:
            queue.pop(0)
        else:
            break
    if not segments:
        return None
    return SchedulerBatch(
        batch_id=batch_id,
        dp=dp,
        segments=tuple(segments),
    )


def _split_afd_stages(batch: SchedulerBatch, split: str) -> tuple[AfdStage, ...]:
    segments = batch.segments
    if split == "request":
        if len(segments) < 2:
            return (AfdStage(segments),)
        cumulative = 0
        total = batch.query_tokens
        best_index = 1
        best_distance = math.inf
        for index in range(1, len(segments)):
            cumulative += segments[index - 1].query_tokens
            distance = abs(cumulative * 2 - total)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return (
            AfdStage(segments[:best_index]),
            AfdStage(segments[best_index:]),
        )

    if batch.query_tokens < 2:
        return (AfdStage(segments),)
    split_token = (batch.query_tokens + 1) // 2
    stage_segments: list[list[BatchSegment]] = [[], []]
    cursor = 0
    for segment in segments:
        segment_start = cursor
        segment_end = cursor + segment.query_tokens
        for stage_idx, (start, end) in enumerate(
            ((0, split_token), (split_token, batch.query_tokens))
        ):
            overlap_start = max(segment_start, start)
            overlap_end = min(segment_end, end)
            if overlap_end > overlap_start:
                offset = overlap_start - segment_start
                stage_segments[stage_idx].append(
                    BatchSegment(
                        request=segment.request,
                        prefix_tokens=segment.prefix_tokens + offset,
                        query_tokens=overlap_end - overlap_start,
                    )
                )
        cursor = segment_end
    return tuple(AfdStage(tuple(items)) for items in stage_segments if items)


def _build_afd_ops(stage_count: int, layer_count: int) -> list[tuple[str, int, int]]:
    if stage_count == 1:
        return [
            operation
            for layer in range(layer_count)
            for operation in (
                ("attention", 0, layer),
                ("dispatch", 0, layer),
                ("combine", 0, layer),
            )
        ]
    ops = [("attention", 0, 0), ("dispatch", 0, 0)]
    for layer in range(layer_count - 1):
        ops.extend(
            (
                ("attention", 1, layer),
                ("combine", 0, layer),
                ("dispatch", 1, layer),
                ("attention", 0, layer + 1),
                ("combine", 1, layer),
                ("dispatch", 0, layer + 1),
            )
        )
    last = layer_count - 1
    ops.extend(
        (
            ("attention", 1, last),
            ("combine", 0, last),
            ("dispatch", 1, last),
            ("combine", 1, last),
        )
    )
    return ops


def _load_afd_state(
    config: SimulationConfig,
    queue: list[RuntimeRequest],
    dp: int,
    *,
    current_time_ms: float,
    next_batch_id: int,
    layer_count: int,
) -> tuple[AfdDpState | None, int]:
    if not queue:
        return None, next_batch_id
    start_ms = max(current_time_ms, _request_ready_ms(queue[0]))
    batch = _pack_batch(
        config,
        queue,
        dp=dp,
        batch_id=next_batch_id,
        start_ms=start_ms,
    )
    if batch is None:
        return None, next_batch_id
    stages = _split_afd_stages(batch, config.afd.ubatch_split)
    return (
        AfdDpState(
            dp=dp,
            batch=batch,
            stages=stages,
            ops=_build_afd_ops(len(stages), layer_count),
            time_ms=start_ms,
        ),
        next_batch_id + 1,
    )


def _segments_duration(
    profiles: ProfileBundle,
    topology: str,
    layer_idx: int,
    phase: str,
    segments: Iterable[BatchSegment],
) -> float:
    return sum(
        profiles.duration_ms(
            topology,
            layer_idx,
            phase,
            segment.prefix_tokens,
            segment.query_tokens,
        )
        for segment in segments
    )


def _complete_batch(batch: SchedulerBatch, completion_ms: float) -> None:
    for segment in batch.segments:
        request = segment.request
        if request.remaining_query_tokens == 0:
            request.completion_ms = completion_ms


def _job_key(batch_id: int, stage_idx: int, layer_idx: int) -> str:
    return f"b{batch_id}-u{stage_idx}-l{layer_idx}"


def _expert_event_details(
    profiles: ProfileBundle,
    topology: str,
    layer_idx: int,
    phase: str,
    tokens: int | float,
) -> dict[str, Any] | None:
    if phase not in {"ffn_compute", "routed_experts", "shared_expert"}:
        return None
    model_config = profiles.metadata.get("model_config")
    topology_spec = profiles.topology_specs[topology]
    if not model_config or not topology_spec:
        return None
    expert_spec = topology_spec["ffn"] if topology == "afd" else topology_spec
    return {
        "top_k": (None if phase == "shared_expert" else int(model_config["moe_top_k"])),
        "ep_size": int(expert_spec["ep_size"]),
        "sampled_input_shapes": profiles.expert_shape_samples(
            topology, layer_idx, phase, tokens
        ),
    }


def _append_timeline(
    timeline: list[dict[str, Any]],
    config: SimulationConfig,
    *,
    architecture: str,
    resource: str,
    phase: str,
    start_ms: float,
    end_ms: float,
    layer: int,
    batch: int | None,
    tokens: int,
    stage: int | None = None,
    profile_query_tokens: int | float | None = None,
    expert_details: dict[str, Any] | None = None,
) -> None:
    if not config.output.include_timeline:
        return
    if len(timeline) > config.output.timeline_max_events:
        return
    event = {
        "architecture": architecture,
        "resource": resource,
        "phase": phase,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "layer": layer,
        "batch": batch,
        "stage": stage,
        "tokens": tokens,
    }
    if profile_query_tokens is not None:
        event["profile_query_tokens"] = profile_query_tokens
    if expert_details:
        event.update(expert_details)
    timeline.append(event)


def _build_result(
    architecture: str,
    config: SimulationConfig,
    requests: list[RuntimeRequest],
    timeline: list[dict[str, Any]],
    makespan_ms: float,
    utilization: dict[str, float],
) -> dict[str, Any]:
    measurement_start = (
        config.arrival.warmup_s * 1_000.0 if config.mode == "continuous" else 0.0
    )
    measured = [
        request for request in requests if request.spec.arrival_ms >= measurement_start
    ]
    ttfts = [
        (request.completion_ms or 0.0)
        - request.spec.arrival_ms
        + config.fixed_ttft_overhead_ms
        for request in measured
    ]
    if config.mode == "continuous":
        nominal_end = measurement_start + config.arrival.duration_s * 1_000.0
        actual_end = max(
            nominal_end,
            max((request.completion_ms or 0.0) for request in measured),
        )
        elapsed_s = max((actual_end - measurement_start) / 1_000.0, 1e-9)
        offered_qps = len(measured) / config.arrival.duration_s
    else:
        first_arrival = min(request.spec.arrival_ms for request in measured)
        elapsed_s = max((makespan_ms - first_arrival) / 1_000.0, 1e-9)
        offered_qps = len(measured) / elapsed_s
    throughput = len(measured) / elapsed_s
    input_tokens = sum(request.spec.input_tokens for request in measured)
    cached_tokens = sum(request.spec.cached_prefix_tokens for request in measured)
    compute_tokens = input_tokens - cached_tokens
    slo_hits = sum(value <= config.slo.ttft_limit_ms for value in ttfts)
    summary = {
        "architecture": architecture,
        "request_count": len(measured),
        "makespan_ms": makespan_ms,
        "offered_qps": offered_qps,
        "throughput_rps": throughput,
        "input_tokens_per_s": input_tokens / elapsed_s,
        "compute_tokens_per_s": compute_tokens / elapsed_s,
        "logical_input_tokens": input_tokens,
        "computed_query_tokens": compute_tokens,
        "cached_prefix_tokens": cached_tokens,
        "cache_token_ratio": cached_tokens / input_tokens if input_tokens else 0.0,
        "ttft_mean_ms": sum(ttfts) / len(ttfts),
        "ttft_p50_ms": _percentile(ttfts, 0.50),
        "ttft_p90_ms": _percentile(ttfts, 0.90),
        "ttft_p99_ms": _percentile(ttfts, 0.99),
        "slo_attainment": slo_hits / len(ttfts),
        "slo_goodput_rps": slo_hits / elapsed_s,
        "utilization": utilization,
        "timeline_truncated": len(timeline) > config.output.timeline_max_events,
    }
    request_payload = []
    if config.output.include_requests:
        request_payload = [
            {
                "request_id": request.spec.request_id,
                "arrival_ms": request.spec.arrival_ms,
                "input_tokens": request.spec.input_tokens,
                "cached_prefix_tokens": request.spec.cached_prefix_tokens,
                "query_tokens": request.spec.query_tokens,
                "assigned_dp": request.assigned_dp,
                "first_scheduled_ms": request.first_scheduled_ms,
                "completion_ms": request.completion_ms,
                "ttft_ms": (
                    (request.completion_ms or 0.0)
                    - request.spec.arrival_ms
                    + config.fixed_ttft_overhead_ms
                ),
            }
            for request in requests
        ]
    return {
        "summary": summary,
        "requests": request_payload,
        "timeline": timeline[: config.output.timeline_max_events],
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * ratio


def _comparison(afd: dict[str, Any], merged: dict[str, Any]) -> dict[str, float | None]:
    return {
        "throughput_speedup": _safe_ratio(
            afd["throughput_rps"], merged["throughput_rps"]
        ),
        "p99_ttft_ratio": _safe_ratio(afd["ttft_p99_ms"], merged["ttft_p99_ms"]),
        "slo_goodput_speedup": _safe_ratio(
            afd["slo_goodput_rps"], merged["slo_goodput_rps"]
        ),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _validate_workload(
    config: SimulationConfig, workload: tuple[RequestSpec, ...]
) -> None:
    if config.mode == "continuous" and not any(
        request.arrival_ms >= config.arrival.warmup_s * 1_000.0 for request in workload
    ):
        raise ValueError("measurement window contains no requests")
    if not config.scheduler.chunked_prefill:
        too_large = [
            request.request_id
            for request in workload
            if request.query_tokens > config.scheduler.max_num_batched_tokens
        ]
        if too_large:
            raise ValueError(
                "non-chunked requests exceed max_num_batched_tokens: "
                + ", ".join(too_large[:5])
            )


__all__ = [
    "compare_architectures",
    "simulate_afd",
    "simulate_merged",
    "sweep_qps",
]
