"""Workload generation, CSV replay, and Prefix Cache sampling."""

from __future__ import annotations

import csv
import io
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from simulator.config import PrefixCacheConfig, SimulationConfig


@dataclass(frozen=True)
class CsvRequest:
    input_length: int
    request_id: str | None = None
    arrival_time_ms: float | None = None
    cached_prefix_tokens: int | None = None


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_ms: float
    input_tokens: int
    cached_prefix_tokens: int
    cache_lookup_ms: float

    @property
    def query_tokens(self) -> int:
        return self.input_tokens - self.cached_prefix_tokens


@dataclass
class RuntimeRequest:
    spec: RequestSpec
    assigned_dp: int
    computed_query_tokens: int = 0
    first_scheduled_ms: float | None = None
    completion_ms: float | None = None

    @property
    def remaining_query_tokens(self) -> int:
        return self.spec.query_tokens - self.computed_query_tokens

    @property
    def current_prefix_tokens(self) -> int:
        return self.spec.cached_prefix_tokens + self.computed_query_tokens


def read_csv_requests(
    *,
    csv_path: str | None = None,
    csv_text: str | None = None,
) -> tuple[CsvRequest, ...]:
    if bool(csv_path) == bool(csv_text):
        raise ValueError("provide exactly one of csv_path or csv_text")
    if csv_path:
        text = Path(csv_path).read_text(encoding="utf-8-sig")
    else:
        text = (csv_text or "").lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "input_length" not in reader.fieldnames:
        raise ValueError("CSV must contain an input_length header")
    requests = []
    for row_number, row in enumerate(reader, start=2):
        try:
            input_length = int(row["input_length"] or "")
            request_id = (row.get("request_id") or "").strip() or None
            arrival_raw = (row.get("arrival_time_ms") or "").strip()
            cached_raw = (row.get("cached_prefix_tokens") or "").strip()
            arrival_time_ms = float(arrival_raw) if arrival_raw else None
            cached_prefix_tokens = int(cached_raw) if cached_raw else None
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid CSV value on row {row_number}: {exc}") from exc
        if input_length <= 0:
            raise ValueError(f"input_length must be positive on row {row_number}")
        if arrival_time_ms is not None and arrival_time_ms < 0:
            raise ValueError(f"arrival_time_ms cannot be negative on row {row_number}")
        if (
            cached_prefix_tokens is not None
            and not 0 <= cached_prefix_tokens < input_length
        ):
            raise ValueError(
                f"cached_prefix_tokens must be in [0, input_length) on row {row_number}"
            )
        requests.append(
            CsvRequest(
                input_length=input_length,
                request_id=request_id,
                arrival_time_ms=arrival_time_ms,
                cached_prefix_tokens=cached_prefix_tokens,
            )
        )
    if not requests:
        raise ValueError("CSV contains no requests")
    has_arrival = [item.arrival_time_ms is not None for item in requests]
    if any(has_arrival) and not all(has_arrival):
        raise ValueError("arrival_time_ms must be present on every CSV row or none")
    if all(has_arrival):
        timestamps = [float(item.arrival_time_ms or 0.0) for item in requests]
        if timestamps != sorted(timestamps):
            raise ValueError("CSV arrival_time_ms must be non-decreasing")
    return tuple(requests)


def generate_workload(config: SimulationConfig) -> tuple[RequestSpec, ...]:
    csv_requests = None
    if config.csv_path or config.csv_text:
        csv_requests = read_csv_requests(
            csv_path=config.csv_path,
            csv_text=config.csv_text,
        )

    if config.mode == "fixed":
        raw_requests = (
            csv_requests
            if csv_requests is not None
            else tuple(CsvRequest(length) for length in config.fixed_lengths)
        )
        arrivals = [
            item.arrival_time_ms
            if item.arrival_time_ms is not None and config.arrival.kind == "trace"
            else 0.0
            for item in raw_requests
        ]
        return _materialize_requests(raw_requests, arrivals, config.prefix_cache)

    if csv_requests and all(item.arrival_time_ms is not None for item in csv_requests):
        if config.arrival.kind != "trace":
            raise ValueError(
                "CSV arrival_time_ms requires arrival.kind=trace for exact replay"
            )
        first_arrival = float(csv_requests[0].arrival_time_ms or 0.0)
        arrivals = [
            float(item.arrival_time_ms or 0.0) - first_arrival for item in csv_requests
        ]
        end_ms = (config.arrival.warmup_s + config.arrival.duration_s) * 1_000.0
        window = [
            (request, arrival)
            for request, arrival in zip(csv_requests, arrivals, strict=True)
            if arrival < end_ms
        ]
        raw_requests, arrivals = zip(*window, strict=True) if window else ((), ())
        return _materialize_requests(raw_requests, arrivals, config.prefix_cache)

    arrival_rng = random.Random(config.arrival.seed)
    length_rng = random.Random(config.arrival.seed + 1)
    end_ms = (config.arrival.warmup_s + config.arrival.duration_s) * 1_000.0
    arrivals = _generate_arrivals(
        config.arrival.kind,
        config.arrival.qps,
        end_ms,
        arrival_rng,
    )
    if csv_requests:
        raw_requests = tuple(
            _choose_csv_request(
                csv_requests,
                index,
                config.csv_sampling,
                length_rng,
            )
            for index in range(len(arrivals))
        )
    else:
        raw_requests = tuple(
            CsvRequest(_choose_weighted_length(config, length_rng)) for _ in arrivals
        )
    return _materialize_requests(raw_requests, arrivals, config.prefix_cache)


def _generate_arrivals(
    kind: str,
    qps: float,
    end_ms: float,
    rng: random.Random,
) -> list[float]:
    if kind == "trace":
        raise ValueError("arrival.kind=trace requires CSV arrival_time_ms")
    arrivals = []
    current_ms = 0.0
    while current_ms < end_ms:
        arrivals.append(current_ms)
        if kind == "constant":
            current_ms += 1_000.0 / qps
        else:
            current_ms += rng.expovariate(qps) * 1_000.0
    return arrivals


def _choose_csv_request(
    requests: tuple[CsvRequest, ...],
    index: int,
    sampling: str,
    rng: random.Random,
) -> CsvRequest:
    if sampling == "cycle":
        return requests[index % len(requests)]
    return requests[rng.randrange(len(requests))]


def _choose_weighted_length(config: SimulationConfig, rng: random.Random) -> int:
    total = sum(bucket.weight for bucket in config.length_mix)
    cursor = rng.random() * total
    for bucket in config.length_mix:
        cursor -= bucket.weight
        if cursor <= 0:
            return bucket.tokens
    return config.length_mix[-1].tokens


def _materialize_requests(
    raw_requests: Iterable[CsvRequest],
    arrivals: Iterable[float],
    prefix_cache: PrefixCacheConfig,
) -> tuple[RequestSpec, ...]:
    cache_rng = random.Random(prefix_cache.seed)
    result = []
    for index, (raw, arrival_ms) in enumerate(zip(raw_requests, arrivals, strict=True)):
        cached_tokens = _cached_tokens(raw, prefix_cache, cache_rng)
        request_id = raw.request_id or f"r{index + 1}"
        result.append(
            RequestSpec(
                request_id=request_id,
                arrival_ms=float(arrival_ms),
                input_tokens=raw.input_length,
                cached_prefix_tokens=cached_tokens,
                cache_lookup_ms=prefix_cache.lookup_latency_ms(cached_tokens),
            )
        )
    return tuple(result)


def _cached_tokens(
    raw: CsvRequest,
    config: PrefixCacheConfig,
    rng: random.Random,
) -> int:
    if not config.enabled:
        return 0
    if raw.cached_prefix_tokens is not None:
        return raw.cached_prefix_tokens
    if rng.random() >= config.request_hit_rate:
        return 0
    candidate = math.floor(raw.input_length * config.matched_prefix_ratio)
    candidate = (candidate // config.block_size) * config.block_size
    max_cached = ((raw.input_length - 1) // config.block_size) * config.block_size
    return min(candidate, max_cached)


__all__ = [
    "CsvRequest",
    "RequestSpec",
    "RuntimeRequest",
    "generate_workload",
    "read_csv_requests",
]
