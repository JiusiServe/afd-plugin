"""Typed configuration and validation for the Prefill simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_PROFILE_MIN_TOKENS = 128
DEFAULT_PROFILE_MAX_TOKENS = 131_072
DEFAULT_MAX_NUM_SEQS = 64
DEFAULT_MAX_BATCHED_TOKENS = 8_192
DEFAULT_SIMULATION_DURATION_S = 60.0
DEFAULT_WARMUP_S = 10.0
DEFAULT_RANDOM_SEED = 1_024
DEFAULT_SLO_LIMIT_MS = 1_000.0
DEFAULT_SLO_TARGET_RATIO = 0.99
DEFAULT_TIMELINE_MAX_EVENTS = 20_000


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _positive(value: float | int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class LengthBucket:
    tokens: int
    weight: float = 1.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> LengthBucket:
        item = cls(tokens=int(raw["tokens"]), weight=float(raw.get("weight", 1.0)))
        _positive(item.tokens, "length_mix.tokens")
        _positive(item.weight, "length_mix.weight")
        return item


@dataclass(frozen=True)
class ArrivalConfig:
    kind: str = "constant"
    qps: float = 1.0
    duration_s: float = DEFAULT_SIMULATION_DURATION_S
    warmup_s: float = DEFAULT_WARMUP_S
    seed: int = DEFAULT_RANDOM_SEED

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ArrivalConfig:
        item = cls(
            kind=str(raw.get("kind", "constant")),
            qps=float(raw.get("qps", 1.0)),
            duration_s=float(raw.get("duration_s", DEFAULT_SIMULATION_DURATION_S)),
            warmup_s=float(raw.get("warmup_s", DEFAULT_WARMUP_S)),
            seed=int(raw.get("seed", DEFAULT_RANDOM_SEED)),
        )
        if item.kind not in {"constant", "poisson", "trace"}:
            raise ValueError("arrival.kind must be constant, poisson, or trace")
        if item.kind != "trace":
            _positive(item.qps, "arrival.qps")
        _positive(item.duration_s, "arrival.duration_s")
        if item.warmup_s < 0:
            raise ValueError("arrival.warmup_s must be non-negative")
        return item


@dataclass(frozen=True)
class SchedulerConfig:
    policy: str = "round_robin"
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    max_num_batched_tokens: int = DEFAULT_MAX_BATCHED_TOKENS
    chunked_prefill: bool = False
    chunk_size: int = DEFAULT_MAX_BATCHED_TOKENS

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> SchedulerConfig:
        max_batched_tokens = int(
            raw.get("max_num_batched_tokens", DEFAULT_MAX_BATCHED_TOKENS)
        )
        item = cls(
            policy=str(raw.get("policy", "round_robin")),
            max_num_seqs=int(raw.get("max_num_seqs", DEFAULT_MAX_NUM_SEQS)),
            max_num_batched_tokens=max_batched_tokens,
            chunked_prefill=bool(raw.get("chunked_prefill", False)),
            chunk_size=int(raw.get("chunk_size", max_batched_tokens)),
        )
        if item.policy not in {"round_robin", "vllm_queue_aware"}:
            raise ValueError("scheduler.policy must be round_robin or vllm_queue_aware")
        _positive(item.max_num_seqs, "scheduler.max_num_seqs")
        _positive(item.max_num_batched_tokens, "scheduler.max_num_batched_tokens")
        _positive(item.chunk_size, "scheduler.chunk_size")
        if item.chunk_size > item.max_num_batched_tokens:
            raise ValueError(
                "scheduler.chunk_size cannot exceed max_num_batched_tokens"
            )
        return item


@dataclass(frozen=True)
class PrefixCacheConfig:
    enabled: bool = False
    request_hit_rate: float = 0.0
    matched_prefix_ratio: float = 0.0
    block_size: int = 32
    lookup_fixed_ms: float = 0.0
    lookup_per_block_ms: float = 0.0
    seed: int = DEFAULT_RANDOM_SEED

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> PrefixCacheConfig:
        item = cls(
            enabled=bool(raw.get("enabled", False)),
            request_hit_rate=float(raw.get("request_hit_rate", 0.0)),
            matched_prefix_ratio=float(raw.get("matched_prefix_ratio", 0.0)),
            block_size=int(raw.get("block_size", 32)),
            lookup_fixed_ms=float(raw.get("lookup_fixed_ms", 0.0)),
            lookup_per_block_ms=float(raw.get("lookup_per_block_ms", 0.0)),
            seed=int(raw.get("seed", DEFAULT_RANDOM_SEED)),
        )
        for name, value in (
            ("request_hit_rate", item.request_hit_rate),
            ("matched_prefix_ratio", item.matched_prefix_ratio),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"prefix_cache.{name} must be in [0, 1]")
        _positive(item.block_size, "prefix_cache.block_size")
        if item.lookup_fixed_ms < 0 or item.lookup_per_block_ms < 0:
            raise ValueError("prefix cache lookup latency cannot be negative")
        return item

    def lookup_latency_ms(self, cached_tokens: int) -> float:
        if not self.enabled:
            return 0.0
        blocks = cached_tokens // self.block_size
        return self.lookup_fixed_ms + blocks * self.lookup_per_block_ms


@dataclass(frozen=True)
class CamLegConfig:
    fixed_ms: float
    per_token_ms: float

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        default_fixed_ms: float,
        default_per_token_ms: float,
    ) -> CamLegConfig:
        item = cls(
            fixed_ms=float(raw.get("fixed_ms", default_fixed_ms)),
            per_token_ms=float(raw.get("per_token_ms", default_per_token_ms)),
        )
        if item.fixed_ms < 0 or item.per_token_ms < 0:
            raise ValueError("CAM latency parameters cannot be negative")
        return item

    def latency_ms(self, tokens: int) -> float:
        return self.fixed_ms + self.per_token_ms * tokens


@dataclass(frozen=True)
class CamConfig:
    calibrated: bool = False
    dispatch_send: CamLegConfig = field(
        default_factory=lambda: CamLegConfig(0.11, 1.0 / 52_000)
    )
    dispatch_recv: CamLegConfig = field(
        default_factory=lambda: CamLegConfig(0.10, 1.0 / 68_000)
    )
    combine_send: CamLegConfig = field(
        default_factory=lambda: CamLegConfig(0.10, 1.0 / 70_000)
    )
    combine_recv: CamLegConfig = field(
        default_factory=lambda: CamLegConfig(0.12, 1.0 / 58_000)
    )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> CamConfig:
        return cls(
            calibrated=bool(raw.get("calibrated", False)),
            dispatch_send=CamLegConfig.from_mapping(
                _mapping(raw.get("dispatch_send"), "cam.dispatch_send"),
                default_fixed_ms=0.11,
                default_per_token_ms=1.0 / 52_000,
            ),
            dispatch_recv=CamLegConfig.from_mapping(
                _mapping(raw.get("dispatch_recv"), "cam.dispatch_recv"),
                default_fixed_ms=0.10,
                default_per_token_ms=1.0 / 68_000,
            ),
            combine_send=CamLegConfig.from_mapping(
                _mapping(raw.get("combine_send"), "cam.combine_send"),
                default_fixed_ms=0.10,
                default_per_token_ms=1.0 / 70_000,
            ),
            combine_recv=CamLegConfig.from_mapping(
                _mapping(raw.get("combine_recv"), "cam.combine_recv"),
                default_fixed_ms=0.12,
                default_per_token_ms=1.0 / 58_000,
            ),
        )


@dataclass(frozen=True)
class AfdConfig:
    ubatch_split: str = "request"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> AfdConfig:
        item = cls(ubatch_split=str(raw.get("ubatch_split", "request")))
        if item.ubatch_split not in {"request", "token"}:
            raise ValueError("afd.ubatch_split must be request or token")
        return item


@dataclass(frozen=True)
class SloConfig:
    ttft_limit_ms: float = DEFAULT_SLO_LIMIT_MS
    target_ratio: float = DEFAULT_SLO_TARGET_RATIO

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> SloConfig:
        item = cls(
            ttft_limit_ms=float(raw.get("ttft_limit_ms", DEFAULT_SLO_LIMIT_MS)),
            target_ratio=float(raw.get("target_ratio", DEFAULT_SLO_TARGET_RATIO)),
        )
        _positive(item.ttft_limit_ms, "slo.ttft_limit_ms")
        if not 0 < item.target_ratio <= 1:
            raise ValueError("slo.target_ratio must be in (0, 1]")
        return item


@dataclass(frozen=True)
class SweepConfig:
    min_qps: float = 0.5
    max_qps: float = 64.0
    coarse_points: int = 10
    refinement_steps: int = 7
    throughput_tolerance_ratio: float = 0.99

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> SweepConfig:
        item = cls(
            min_qps=float(raw.get("min_qps", 0.5)),
            max_qps=float(raw.get("max_qps", 64.0)),
            coarse_points=int(raw.get("coarse_points", 10)),
            refinement_steps=int(raw.get("refinement_steps", 7)),
            throughput_tolerance_ratio=float(
                raw.get("throughput_tolerance_ratio", 0.99)
            ),
        )
        _positive(item.min_qps, "sweep.min_qps")
        if item.max_qps <= item.min_qps:
            raise ValueError("sweep.max_qps must exceed min_qps")
        if item.coarse_points < 2 or item.refinement_steps < 0:
            raise ValueError("invalid sweep point counts")
        if not 0 < item.throughput_tolerance_ratio <= 1:
            raise ValueError("sweep.throughput_tolerance_ratio must be in (0, 1]")
        return item


@dataclass(frozen=True)
class OutputConfig:
    include_timeline: bool = True
    timeline_max_events: int = DEFAULT_TIMELINE_MAX_EVENTS
    include_requests: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> OutputConfig:
        item = cls(
            include_timeline=bool(raw.get("include_timeline", True)),
            timeline_max_events=int(
                raw.get("timeline_max_events", DEFAULT_TIMELINE_MAX_EVENTS)
            ),
            include_requests=bool(raw.get("include_requests", True)),
        )
        _positive(item.timeline_max_events, "output.timeline_max_events")
        return item


@dataclass(frozen=True)
class SimulationConfig:
    mode: str = "fixed"
    fixed_lengths: tuple[int, ...] = (512, 8_192, 2_048, 6_144)
    length_mix: tuple[LengthBucket, ...] = (
        LengthBucket(512, 0.5),
        LengthBucket(8_192, 0.5),
    )
    csv_path: str | None = None
    csv_text: str | None = None
    csv_sampling: str = "cycle"
    arrival: ArrivalConfig = field(default_factory=ArrivalConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    prefix_cache: PrefixCacheConfig = field(default_factory=PrefixCacheConfig)
    afd: AfdConfig = field(default_factory=AfdConfig)
    cam: CamConfig = field(default_factory=CamConfig)
    slo: SloConfig = field(default_factory=SloConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    fixed_ttft_overhead_ms: float = 0.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> SimulationConfig:
        fixed_lengths = tuple(int(value) for value in raw.get("fixed_lengths", []))
        length_mix = tuple(
            LengthBucket.from_mapping(_mapping(value, "length_mix item"))
            for value in raw.get("length_mix", [])
        )
        item = cls(
            mode=str(raw.get("mode", "fixed")),
            fixed_lengths=(
                fixed_lengths or cls.__dataclass_fields__["fixed_lengths"].default
            ),
            length_mix=length_mix or cls.__dataclass_fields__["length_mix"].default,
            csv_path=raw.get("csv_path"),
            csv_text=raw.get("csv_text"),
            csv_sampling=str(raw.get("csv_sampling", "cycle")),
            arrival=ArrivalConfig.from_mapping(_mapping(raw.get("arrival"), "arrival")),
            scheduler=SchedulerConfig.from_mapping(
                _mapping(raw.get("scheduler"), "scheduler")
            ),
            prefix_cache=PrefixCacheConfig.from_mapping(
                _mapping(raw.get("prefix_cache"), "prefix_cache")
            ),
            afd=AfdConfig.from_mapping(_mapping(raw.get("afd"), "afd")),
            cam=CamConfig.from_mapping(_mapping(raw.get("cam"), "cam")),
            slo=SloConfig.from_mapping(_mapping(raw.get("slo"), "slo")),
            sweep=SweepConfig.from_mapping(_mapping(raw.get("sweep"), "sweep")),
            output=OutputConfig.from_mapping(_mapping(raw.get("output"), "output")),
            fixed_ttft_overhead_ms=float(raw.get("fixed_ttft_overhead_ms", 0.0)),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if self.mode not in {"fixed", "continuous"}:
            raise ValueError("mode must be fixed or continuous")
        if self.mode == "fixed" and self.arrival.kind == "trace":
            raise ValueError("arrival.kind=trace requires mode='continuous'")
        if self.csv_sampling not in {"cycle", "sample"}:
            raise ValueError("csv_sampling must be cycle or sample")
        if self.fixed_ttft_overhead_ms < 0:
            raise ValueError("fixed_ttft_overhead_ms cannot be negative")
        for tokens in self.fixed_lengths:
            _positive(tokens, "fixed_lengths item")
        if (
            self.mode == "fixed"
            and not self.scheduler.chunked_prefill
            and any(
                tokens > self.scheduler.max_num_batched_tokens
                for tokens in self.fixed_lengths
            )
            and not (self.csv_path or self.csv_text)
        ):
            raise ValueError(
                "non-chunked fixed request exceeds scheduler.max_num_batched_tokens"
            )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def default_config_mapping() -> dict[str, Any]:
    return SimulationConfig().to_mapping()


__all__ = [
    "AfdConfig",
    "ArrivalConfig",
    "CamConfig",
    "LengthBucket",
    "PrefixCacheConfig",
    "SchedulerConfig",
    "SimulationConfig",
    "SloConfig",
    "SweepConfig",
    "default_config_mapping",
]
