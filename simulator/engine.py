"""Discrete-event simulation for merged and CAMAsync AFD Prefill."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from simulator.config import OutputConfig, SchedulerBatchConfig, SimulationConfig
from simulator.profiles import ProfileBundle
from simulator.workload import RequestSpec, RuntimeRequest, generate_workload

AFD_WAVE_TOKEN_SUM_POLICY = "afd_wave_token_sum"
AFD_WAVE_TOKEN_SQUARE_SUM_POLICY = "afd_wave_token_square_sum"
AFD_WAVE_POLICIES = frozenset(
    {AFD_WAVE_TOKEN_SUM_POLICY, AFD_WAVE_TOKEN_SQUARE_SUM_POLICY}
)
MERGED_WAVE_TOKEN_SUM_POLICY = "merged_wave_token_sum"
MERGED_WAVE_TOKEN_SQUARE_SUM_POLICY = "merged_wave_token_square_sum"
MERGED_WAVE_POLICIES = frozenset(
    {MERGED_WAVE_TOKEN_SUM_POLICY, MERGED_WAVE_TOKEN_SQUARE_SUM_POLICY}
)
CURRENT_RUNTIME_POLICY = "current_runtime"
PREFILL_TOKEN_GREEDY_POLICY = "prefill_token_greedy"
PREFILL_TOKEN_SQUARE_GREEDY_POLICY = "prefill_token_square_greedy"
ROUND_ROBIN_POLICY = "round_robin"
VLLM_QUEUE_AWARE_POLICY = "vllm_queue_aware"
VLLM_WAITING_REQUEST_WEIGHT = 4
VLLM_DP_STATS_UPDATE_INTERVAL_MS = 100.0

DpScore = tuple[float, ...]
DpScoreProvider = Callable[[RuntimeRequest], tuple[DpScore, ...]]


@dataclass(frozen=True)
class BatchSegment:
    request: RuntimeRequest
    prefix_tokens: int
    query_tokens: int


@dataclass(frozen=True)
class FifoRequestState:
    request: RuntimeRequest
    prefix_tokens: int
    remaining_query_tokens: int


@dataclass(frozen=True)
class FifoWaveSegment:
    request: RuntimeRequest
    prefix_tokens: int
    query_tokens: int
    remaining_query_tokens: int


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
    last_op_type: str | None = None
    last_op_stage: int = 0
    last_op_start_ms: float = 0.0
    last_op_end_ms: float = 0.0


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
    """Route requests using the configured DP load-balancing policy."""

    def __init__(
        self,
        specs: tuple[RequestSpec, ...],
        dp_count: int,
        policy: str,
        score_provider: DpScoreProvider | None = None,
    ) -> None:
        self.dp_count = dp_count
        self.policy = policy
        self.score_provider = score_provider
        self.requests = [RuntimeRequest(spec=spec, assigned_dp=-1) for spec in specs]
        self.pending = sorted(
            enumerate(self.requests),
            key=lambda item: (item[1].spec.arrival_ms, item[0]),
        )
        self.pending_index = 0
        self.round_robin_index = 0
        self.tie_start_index = 0
        self.queues: list[list[RuntimeRequest]] = [[] for _ in range(dp_count)]
        self.running_request_ids: list[set[int]] = [set() for _ in range(dp_count)]
        self.outstanding_query_tokens = [0 for _ in range(dp_count)]
        self.outstanding_query_token_squares = [0 for _ in range(dp_count)]
        self.reported_counts = [[0, 0] for _ in range(dp_count)]
        self.next_stats_update_ms = VLLM_DP_STATS_UPDATE_INTERVAL_MS

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
            self.advance_time(request.spec.arrival_ms)
            dp = self._select_dp(request)
            request.assigned_dp = dp
            self.queues[dp].append(request)
            self.outstanding_query_tokens[dp] += request.remaining_query_tokens
            self.outstanding_query_token_squares[dp] += (
                request.remaining_query_tokens**2
            )
            if self.policy == VLLM_QUEUE_AWARE_POLICY:
                # vLLM increments its local waiting count immediately to balance
                # requests arriving between coordinator updates.
                self.reported_counts[dp][0] += 1
        self.advance_time(time_ms)

    def advance_time(self, time_ms: float) -> None:
        """Publish engine request counts at vLLM's 100 ms update cadence."""

        if (
            self.policy != VLLM_QUEUE_AWARE_POLICY
            or time_ms < self.next_stats_update_ms
        ):
            return
        self.reported_counts = [
            list(self.request_counts(dp)) for dp in range(self.dp_count)
        ]
        elapsed_intervals = math.floor(
            (time_ms - self.next_stats_update_ms) / VLLM_DP_STATS_UPDATE_INTERVAL_MS
        )
        self.next_stats_update_ms += (
            elapsed_intervals + 1
        ) * VLLM_DP_STATS_UPDATE_INTERVAL_MS

    def mark_batch_started(self, batch: SchedulerBatch) -> None:
        running_ids = self.running_request_ids[batch.dp]
        running_ids.update(id(segment.request) for segment in batch.segments)

    def mark_batch_finished(self, batch: SchedulerBatch) -> None:
        running_ids = self.running_request_ids[batch.dp]
        for segment in batch.segments:
            running_ids.discard(id(segment.request))
        self.outstanding_query_tokens[batch.dp] -= batch.query_tokens
        for segment in batch.segments:
            remaining_after = segment.request.remaining_query_tokens
            remaining_before = remaining_after + segment.query_tokens
            self.outstanding_query_token_squares[batch.dp] -= (
                remaining_before**2 - remaining_after**2
            )

    def request_counts(self, dp: int) -> tuple[int, int]:
        """Return the engine's current ``(waiting, running)`` request counts."""

        running_ids = self.running_request_ids[dp]
        waiting = sum(id(request) not in running_ids for request in self.queues[dp])
        return waiting, len(running_ids)

    def _select_dp(self, request: RuntimeRequest) -> int:
        if self.policy == ROUND_ROBIN_POLICY:
            dp = self.round_robin_index
            self.round_robin_index = (self.round_robin_index + 1) % self.dp_count
            return dp

        if self.policy in AFD_WAVE_POLICIES | MERGED_WAVE_POLICIES:
            if self.score_provider is None:
                raise RuntimeError(f"{self.policy} requires a wave score provider")
            scores = self.score_provider(request)
            if len(scores) != self.dp_count:
                raise RuntimeError("wave score provider returned wrong DP count")
        elif self.policy == PREFILL_TOKEN_GREEDY_POLICY:
            scores = tuple((float(value),) for value in self.outstanding_query_tokens)
        elif self.policy == PREFILL_TOKEN_SQUARE_GREEDY_POLICY:
            scores = tuple(
                (float(value),) for value in self.outstanding_query_token_squares
            )
        else:
            scores = tuple(
                (float(waiting * VLLM_WAITING_REQUEST_WEIGHT + running),)
                for waiting, running in self.reported_counts
            )
        min_score = min(scores)
        for offset in range(self.dp_count):
            dp = (self.tie_start_index + offset) % self.dp_count
            if scores[dp] == min_score:
                self.tie_start_index = (self.tie_start_index + 1) % self.dp_count
                return dp
        raise RuntimeError("DP dispatcher failed to select an engine")


def resolve_scheduler_policy(configured_policy: str, architecture: str) -> str:
    """Resolve the architecture-specific policy used by today's runtime."""

    if configured_policy in AFD_WAVE_POLICIES:
        if architecture == "afd":
            return configured_policy
        raise ValueError(f"{configured_policy} is unavailable for {architecture}")
    if configured_policy in MERGED_WAVE_POLICIES:
        if architecture == "merged":
            return configured_policy
        raise ValueError(f"{configured_policy} is unavailable for {architecture}")
    if configured_policy != CURRENT_RUNTIME_POLICY:
        return configured_policy
    if architecture == "afd":
        return ROUND_ROBIN_POLICY
    if architecture == "merged":
        return VLLM_QUEUE_AWARE_POLICY
    raise ValueError(f"unknown architecture: {architecture}")


def compare_architectures(
    config: SimulationConfig,
    profiles: ProfileBundle,
    requests: tuple[RequestSpec, ...] | None = None,
) -> dict[str, Any]:
    """Run one identical workload through AFD and merged architectures."""

    profiles.device_budget()
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
    merged_spec = profiles.topology_specs["merged"]
    merged_dp_count = int(merged_spec["dp_size"])
    merged_tp_size = int(merged_spec["tp_size"])
    merged_ep_size = int(merged_spec["ep_size"])
    if min(merged_dp_count, merged_tp_size, merged_ep_size) <= 0:
        raise ValueError("merged profile parallel sizes must be positive")
    dispatcher: RequestDispatcher

    def wave_score_provider(request: RuntimeRequest) -> tuple[DpScore, ...]:
        return _merged_wave_completion_scores(
            config,
            request,
            dispatcher.queues,
            dispatcher.policy,
        )

    dispatcher = RequestDispatcher(
        specs,
        merged_dp_count,
        resolve_scheduler_policy(config.scheduler.merged_policy, "merged"),
        score_provider=wave_score_provider,
    )
    requests = dispatcher.requests
    queues = dispatcher.queues
    timeline: list[dict[str, Any]] = []
    wave_start = 0.0
    batch_id = 0
    total_barrier_ms = 0.0
    busy_ms = [0.0 for _ in range(merged_dp_count)]
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
                config.scheduler.merged,
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
        for batch in batches:
            if batch is not None:
                dispatcher.mark_batch_started(batch)

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
            global_profile_query = max(1.0, global_ep_tokens / merged_dp_count)
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
                    resource=f"Global EP{merged_ep_size}",
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
        dispatcher.dispatch_until(wave_start)
        for batch in batches:
            if batch is not None:
                _complete_batch(batch, wave_start)
                dispatcher.mark_batch_finished(batch)

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
    result["summary"]["topology"] = {
        "dp_size": merged_dp_count,
        "tp_size": merged_tp_size,
        "ep_size": merged_ep_size,
    }
    result["summary"]["scheduler_policy"] = dispatcher.policy
    result["summary"]["barrier_wait_ms"] = total_barrier_ms
    return result


def simulate_afd(
    config: SimulationConfig,
    profiles: ProfileBundle,
    specs: tuple[RequestSpec, ...],
) -> dict[str, Any]:
    afd_spec = profiles.topology_specs["afd"]
    attention_spec = afd_spec["attention"]
    ffn_spec = afd_spec["ffn"]
    afd_dp_count = int(attention_spec["dp_size"])
    afd_tp_size = int(attention_spec["tp_size"])
    afd_ep_size = int(ffn_spec["ep_size"])
    if min(afd_dp_count, afd_tp_size, afd_ep_size) <= 0:
        raise ValueError("AFD profile parallel sizes must be positive")
    states: dict[int, AfdDpState] = {}
    dispatcher: RequestDispatcher

    def wave_score_provider(request: RuntimeRequest) -> tuple[DpScore, ...]:
        return _afd_wave_completion_scores(
            config,
            request,
            dispatcher.queues,
            states,
            profiles.layer_count,
            dispatcher.policy,
        )

    dispatcher = RequestDispatcher(
        specs,
        afd_dp_count,
        resolve_scheduler_policy(config.scheduler.afd_policy, "afd"),
        score_provider=wave_score_provider,
    )
    dispatcher.dispatch_until(0.0)
    requests = dispatcher.requests
    queues = dispatcher.queues
    timeline: list[dict[str, Any]] = []
    next_batch_id = 0
    jobs: dict[str, FfnJob] = {}
    pending_jobs: list[PendingFfnJob] = []
    ffn_available_ms = 0.0
    ffn_busy_ms = 0.0
    attention_busy_ms = [0.0 for _ in range(afd_dp_count)]
    attention_wait_ms = 0.0
    dp_available_ms = [0.0 for _ in range(afd_dp_count)]

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
            for dp in range(afd_dp_count)
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
            dispatcher.advance_time(start_ms)
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
            dispatcher.mark_batch_started(state.batch)
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
                    resource=f"FFN EP{afd_ep_size}",
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

        state.last_op_type = op_type
        state.last_op_stage = stage_idx
        state.last_op_start_ms = start if op_type == "attention" else state.time_ms
        state.last_op_end_ms = end if op_type == "attention" else state.time_ms
        state.op_index += 1
        if state.op_index == len(state.ops):
            dispatcher.advance_time(state.time_ms)
            _complete_batch(state.batch, state.time_ms)
            dispatcher.mark_batch_finished(state.batch)
            dp_available_ms[dp] = state.time_ms
            del states[dp]

    makespan = max(request.completion_ms or 0.0 for request in requests)
    utilization = {
        f"attention_dp{dp}": value / makespan if makespan else 0.0
        for dp, value in enumerate(attention_busy_ms)
    }
    utilization[f"ffn_ep{afd_ep_size}"] = ffn_busy_ms / makespan if makespan else 0.0
    result = _build_result("afd", config, requests, timeline, makespan, utilization)
    result["summary"]["topology"] = {
        "dp_size": afd_dp_count,
        "tp_size": afd_tp_size,
        "ep_size": afd_ep_size,
    }
    result["summary"]["scheduler_policy"] = dispatcher.policy
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
    batch_config: SchedulerBatchConfig,
    queue: list[RuntimeRequest],
    *,
    dp: int,
    batch_id: int,
    start_ms: float,
) -> SchedulerBatch | None:
    planned_segments, remaining_states = _plan_fifo_wave(
        config,
        batch_config,
        _fifo_request_states(queue),
        start_ms=start_ms,
    )
    if not planned_segments:
        return None
    segments = []
    for planned_segment in planned_segments:
        request = planned_segment.request
        request.computed_query_tokens += planned_segment.query_tokens
        if request.first_scheduled_ms is None:
            request.first_scheduled_ms = start_ms
        segments.append(
            BatchSegment(
                request=request,
                prefix_tokens=planned_segment.prefix_tokens,
                query_tokens=planned_segment.query_tokens,
            )
        )
    queue[:] = [state.request for state in remaining_states]
    return SchedulerBatch(
        batch_id=batch_id,
        dp=dp,
        segments=tuple(segments),
    )


def _fifo_request_states(
    requests: Iterable[RuntimeRequest],
) -> tuple[FifoRequestState, ...]:
    return tuple(
        FifoRequestState(
            request=request,
            prefix_tokens=request.current_prefix_tokens,
            remaining_query_tokens=request.remaining_query_tokens,
        )
        for request in requests
    )


def _plan_fifo_wave(
    config: SimulationConfig,
    batch_config: SchedulerBatchConfig,
    states: tuple[FifoRequestState, ...],
    *,
    start_ms: float,
) -> tuple[tuple[FifoWaveSegment, ...], tuple[FifoRequestState, ...]]:
    """Plan one FIFO wave without mutating runtime requests.

    A request contributes at most one chunk to a wave. If that chunk does not
    consume the wave budget, later FIFO requests may use the remaining budget.
    Partial requests retain their relative order for the next wave.
    """

    budget = batch_config.max_num_batched_tokens
    segments: list[FifoWaveSegment] = []
    remaining_states: list[FifoRequestState] = []
    state_index = 0
    while (
        state_index < len(states)
        and len(segments) < config.scheduler.max_num_seqs
        and budget > 0
    ):
        state = states[state_index]
        if _request_ready_ms(state.request) > start_ms:
            break
        if not config.scheduler.chunked_prefill:
            if state.remaining_query_tokens > budget:
                if not segments:
                    raise ValueError(
                        f"request {state.request.spec.request_id} has "
                        f"{state.remaining_query_tokens} uncached tokens, "
                        "exceeding non-chunked max_num_batched_tokens"
                    )
                break
            query_tokens = state.remaining_query_tokens
        else:
            query_tokens = min(
                state.remaining_query_tokens,
                batch_config.chunk_size,
                budget,
            )
        remaining_query_tokens = state.remaining_query_tokens - query_tokens
        segments.append(
            FifoWaveSegment(
                request=state.request,
                prefix_tokens=state.prefix_tokens,
                query_tokens=query_tokens,
                remaining_query_tokens=remaining_query_tokens,
            )
        )
        if remaining_query_tokens > 0:
            remaining_states.append(
                FifoRequestState(
                    request=state.request,
                    prefix_tokens=state.prefix_tokens + query_tokens,
                    remaining_query_tokens=remaining_query_tokens,
                )
            )
        budget -= query_tokens
        state_index += 1
    remaining_states.extend(states[state_index:])
    return tuple(segments), tuple(remaining_states)


def _afd_wave_completion_scores(
    config: SimulationConfig,
    request: RuntimeRequest,
    queues: list[list[RuntimeRequest]],
    states: dict[int, AfdDpState],
    layer_count: int,
    policy: str,
) -> tuple[DpScore, ...]:
    """Estimate each DP's FIFO tail-wave completion in Attention work units."""

    use_squares = policy == AFD_WAVE_TOKEN_SQUARE_SUM_POLICY
    arrival_ms = request.spec.arrival_ms
    scores = []
    for dp, queue in enumerate(queues):
        running_work = _afd_remaining_attention_work(
            states.get(dp), arrival_ms, use_squares
        )
        queued_waves = _fifo_waves(
            config,
            config.scheduler.afd,
            (*queue, request),
        )
        completion_wave_idx = _fifo_completion_wave_index(queued_waves, request)
        queued_work = sum(
            _afd_wave_attention_work(
                tuple(segment.query_tokens for segment in wave),
                config.afd.ubatch_split,
                layer_count,
                use_squares,
            )
            for wave in queued_waves[: completion_wave_idx + 1]
        )
        scores.append((running_work + queued_work,))
    return tuple(scores)


def _merged_wave_completion_scores(
    config: SimulationConfig,
    request: RuntimeRequest,
    queues: list[list[RuntimeRequest]],
    policy: str,
) -> tuple[DpScore, ...]:
    """Score FIFO placement by global wave count and synchronous drain work."""

    use_squares = policy == MERGED_WAVE_TOKEN_SQUARE_SUM_POLICY
    scores = []
    for candidate_dp in range(len(queues)):
        waves_by_dp = [
            _fifo_waves(
                config,
                config.scheduler.merged,
                (*queue, request) if dp == candidate_dp else tuple(queue),
            )
            for dp, queue in enumerate(queues)
        ]
        global_wave_count = max(map(len, waves_by_dp))
        global_wave_work = tuple(
            max(
                (
                    _query_token_work(
                        tuple(segment.query_tokens for segment in waves[wave_idx]),
                        use_squares,
                    )
                    for waves in waves_by_dp
                    if wave_idx < len(waves)
                ),
                default=0.0,
            )
            for wave_idx in range(global_wave_count)
        )
        candidate_wave_idx = _fifo_completion_wave_index(
            waves_by_dp[candidate_dp], request
        )
        scores.append(
            (
                float(global_wave_count),
                sum(global_wave_work),
                sum(global_wave_work[: candidate_wave_idx + 1]),
            )
        )
    return tuple(scores)


def _afd_remaining_attention_work(
    state: AfdDpState | None,
    arrival_ms: float,
    use_squares: bool,
) -> float:
    if state is None:
        return 0.0
    work = 0.0
    if (
        state.last_op_type == "attention"
        and state.last_op_start_ms < arrival_ms < state.last_op_end_ms
    ):
        stage = state.stages[state.last_op_stage]
        remaining_ratio = (state.last_op_end_ms - arrival_ms) / (
            state.last_op_end_ms - state.last_op_start_ms
        )
        work += _query_token_work(
            tuple(segment.query_tokens for segment in stage.segments),
            use_squares,
        ) * remaining_ratio
    for op_type, stage_idx, _ in state.ops[state.op_index :]:
        if op_type != "attention":
            continue
        work += _query_token_work(
            tuple(
                segment.query_tokens
                for segment in state.stages[stage_idx].segments
            ),
            use_squares,
        )
    return work


def _fifo_waves(
    config: SimulationConfig,
    batch_config: SchedulerBatchConfig,
    requests: tuple[RuntimeRequest, ...],
) -> tuple[tuple[FifoWaveSegment, ...], ...]:
    """Plan all future FIFO waves using the runtime batch planner."""

    waves = []
    states = _fifo_request_states(requests)
    while states:
        segments, remaining_states = _plan_fifo_wave(
            config,
            batch_config,
            states,
            start_ms=math.inf,
        )
        if not segments:
            raise RuntimeError("FIFO wave planner made no progress")
        waves.append(segments)
        states = remaining_states
    return tuple(waves)


def _fifo_completion_wave_index(
    waves: tuple[tuple[FifoWaveSegment, ...], ...],
    request: RuntimeRequest,
) -> int:
    for wave_idx, wave in enumerate(waves):
        if any(
            segment.request is request and segment.remaining_query_tokens == 0
            for segment in wave
        ):
            return wave_idx
    raise RuntimeError(f"request {request.spec.request_id} has no completion wave")


def _afd_wave_attention_work(
    query_tokens: tuple[int, ...],
    ubatch_split: str,
    layer_count: int,
    use_squares: bool,
) -> float:
    stages = _split_query_token_lengths(query_tokens, ubatch_split)
    per_layer_work = sum(
        _query_token_work(stage, use_squares) for stage in stages
    )
    return per_layer_work * layer_count


def _split_query_token_lengths(
    query_tokens: tuple[int, ...], split: str
) -> tuple[tuple[int, ...], ...]:
    total_tokens = sum(query_tokens)
    if len(query_tokens) < 2 and split == "request":
        return (query_tokens,)
    if split == "request":
        cumulative = 0
        best_index = 1
        best_distance = math.inf
        for index in range(1, len(query_tokens)):
            cumulative += query_tokens[index - 1]
            distance = abs(cumulative * 2 - total_tokens)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return (query_tokens[:best_index], query_tokens[best_index:])
    if total_tokens < 2:
        return (query_tokens,)
    split_token = (total_tokens + 1) // 2
    stages: list[list[int]] = [[], []]
    cursor = 0
    for tokens in query_tokens:
        segment_start = cursor
        segment_end = cursor + tokens
        for stage_idx, (start, end) in enumerate(
            ((0, split_token), (split_token, total_tokens))
        ):
            overlap = min(segment_end, end) - max(segment_start, start)
            if overlap > 0:
                stages[stage_idx].append(overlap)
        cursor = segment_end
    return tuple(tuple(stage) for stage in stages if stage)


def _query_token_work(query_tokens: tuple[int, ...], use_squares: bool) -> float:
    if use_squares:
        return float(sum(tokens**2 for tokens in query_tokens))
    return float(sum(query_tokens))


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
        config.scheduler.afd,
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
    timeline_max_events = config.output.timeline_max_events
    if timeline_max_events is not None and len(timeline) > timeline_max_events:
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
        "timeline_truncated": (
            config.output.timeline_max_events is not None
            and len(timeline) > config.output.timeline_max_events
        ),
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
        "timeline": (
            timeline
            if config.output.timeline_max_events is None
            else timeline[: config.output.timeline_max_events]
        ),
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
        max_num_batched_tokens = min(
            config.scheduler.afd.max_num_batched_tokens,
            config.scheduler.merged.max_num_batched_tokens,
        )
        too_large = [
            request.request_id
            for request in workload
            if request.query_tokens > max_num_batched_tokens
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
