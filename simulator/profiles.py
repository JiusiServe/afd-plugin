"""Normalized msModeling profile bundle and interpolation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_SCHEMA_VERSION = 1
PROFILE_PHASES = (
    "attention_router",
    "merged_dispatch",
    "routed_experts",
    "merged_combine",
    "merged_combine_local",
    "shared_expert",
    "merged_sp_post",
    "afd_post",
)
EXPERT_SHAPE_KEYS = {
    "routed_experts": "routed_expert_input_shape",
    "ffn_compute": "routed_expert_input_shape",
    "shared_expert": "shared_expert_input_shape",
}
ROUTED_EXPERT_SAMPLE_SHAPES = "routed_expert_sample_shapes"


@dataclass(frozen=True)
class ProfilePoint:
    prefix_tokens: int
    query_tokens: int
    layers: tuple[dict[str, Any], ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], layer_count: int) -> ProfilePoint:
        layers = tuple(_normalize_layer(layer) for layer in raw["layers"])
        if len(layers) != layer_count:
            raise ValueError(
                f"profile point has {len(layers)} layers, expected {layer_count}"
            )
        return cls(
            prefix_tokens=int(raw["prefix_tokens"]),
            query_tokens=int(raw["query_tokens"]),
            layers=layers,
        )


def _normalize_layer(raw: dict[str, Any]) -> dict[str, Any]:
    layer: dict[str, Any] = {
        phase: float(raw.get(phase, 0.0)) for phase in PROFILE_PHASES
    }
    for shape_key in set(EXPERT_SHAPE_KEYS.values()):
        if shape_key in raw:
            layer[shape_key] = tuple(int(value) for value in raw[shape_key])
    if ROUTED_EXPERT_SAMPLE_SHAPES in raw:
        layer[ROUTED_EXPERT_SAMPLE_SHAPES] = tuple(
            {
                "shape": tuple(int(value) for value in sample["shape"]),
                "count": int(sample["count"]),
            }
            for sample in raw[ROUTED_EXPERT_SAMPLE_SHAPES]
        )
    return layer


class TopologyProfile:
    """One topology's two-dimensional `(prefix, query)` profile grid."""

    def __init__(
        self,
        name: str,
        points: tuple[ProfilePoint, ...],
        layer_count: int,
        max_context_tokens: int | None = None,
    ):
        if not points:
            raise ValueError(f"topology profile {name!r} has no points")
        self.name = name
        self.points = points
        self.layer_count = layer_count
        self.max_context_tokens = max_context_tokens
        self._grid = {
            (point.prefix_tokens, point.query_tokens): point for point in points
        }
        if len(self._grid) != len(points):
            raise ValueError(f"topology profile {name!r} contains duplicate points")
        self.prefix_anchors = tuple(sorted({point.prefix_tokens for point in points}))
        self.query_anchors = tuple(sorted({point.query_tokens for point in points}))

    def duration_ms(
        self,
        layer_idx: int,
        phase: str,
        prefix_tokens: int,
        query_tokens: int | float,
    ) -> float:
        if not 0 <= layer_idx < self.layer_count:
            raise ValueError(f"layer_idx {layer_idx} is outside profile")
        if phase not in PROFILE_PHASES:
            raise ValueError(f"unknown profile phase {phase!r}")
        if prefix_tokens < 0 or query_tokens <= 0:
            raise ValueError("profile lookup requires prefix>=0 and query>0")
        if (
            self.max_context_tokens is not None
            and prefix_tokens + query_tokens > self.max_context_tokens
        ):
            raise ValueError(
                f"prefix+query={prefix_tokens + query_tokens} outside profile "
                f"context limit {self.max_context_tokens}"
            )

        lower_prefix, upper_prefix = _bounds(
            self.prefix_anchors, prefix_tokens, "prefix"
        )
        lower_value = self._interpolate_query(
            lower_prefix, layer_idx, phase, query_tokens
        )
        if upper_prefix == lower_prefix:
            return lower_value
        ratio = (prefix_tokens - lower_prefix) / (upper_prefix - lower_prefix)
        if (
            self.max_context_tokens is not None
            and query_tokens > self.max_context_tokens - upper_prefix
        ):
            return self._interpolate_triangle(
                lower_prefix,
                upper_prefix,
                ratio,
                layer_idx,
                phase,
                query_tokens,
            )
        upper_value = self._interpolate_query(
            upper_prefix, layer_idx, phase, query_tokens
        )
        return lower_value + (upper_value - lower_value) * ratio

    def expert_shape_samples(
        self,
        layer_idx: int,
        phase: str,
        query_tokens: int | float,
    ) -> list[dict[str, Any]]:
        """Return the trace anchors whose measured shapes back this lookup."""

        shape_key = EXPERT_SHAPE_KEYS.get(phase)
        if shape_key is None:
            return []
        anchors = tuple(sorted(query for prefix, query in self._grid if prefix == 0))
        lower_query, upper_query = _bounds(anchors, query_tokens, "query")
        weights = [(lower_query, 1.0)]
        if upper_query != lower_query:
            upper_weight = (query_tokens - lower_query) / (upper_query - lower_query)
            weights = [
                (lower_query, 1.0 - upper_weight),
                (upper_query, upper_weight),
            ]
        samples = []
        for sampled_query, weight in weights:
            shape = self._grid[(0, sampled_query)].layers[layer_idx].get(shape_key)
            if shape is not None:
                sample = {
                    "query_tokens": sampled_query,
                    "weight": weight,
                    "shape": list(shape),
                }
                if phase != "shared_expert":
                    sample["expert_shapes"] = [
                        {
                            "shape": list(item["shape"]),
                            "count": item["count"],
                        }
                        for item in self._grid[(0, sampled_query)]
                        .layers[layer_idx]
                        .get(ROUTED_EXPERT_SAMPLE_SHAPES, ())
                    ]
                samples.append(sample)
        return samples

    def _interpolate_triangle(
        self,
        lower_prefix: int,
        upper_prefix: int,
        prefix_ratio: float,
        layer_idx: int,
        phase: str,
        query_tokens: int | float,
    ) -> float:
        """Interpolate the strip next to the prefix+query context boundary."""

        assert self.max_context_tokens is not None
        lower_boundary = self.max_context_tokens - lower_prefix
        upper_boundary = self.max_context_tokens - upper_prefix
        lower_at_boundary = self._interpolate_query(
            lower_prefix, layer_idx, phase, lower_boundary
        )
        upper_at_boundary = self._interpolate_query(
            upper_prefix, layer_idx, phase, upper_boundary
        )
        distance = lower_boundary - upper_boundary
        lower_weight = (query_tokens - upper_boundary) / distance
        upper_weight = prefix_ratio
        corner_weight = 1.0 - lower_weight - upper_weight
        lower_corner = self._interpolate_query(
            lower_prefix, layer_idx, phase, upper_boundary
        )
        return (
            corner_weight * lower_corner
            + lower_weight * lower_at_boundary
            + upper_weight * upper_at_boundary
        )

    def _interpolate_query(
        self,
        prefix_tokens: int,
        layer_idx: int,
        phase: str,
        query_tokens: int | float,
    ) -> float:
        anchors = tuple(
            sorted(query for prefix, query in self._grid if prefix == prefix_tokens)
        )
        lower_query, upper_query = _bounds(
            anchors, query_tokens, f"query at prefix={prefix_tokens}"
        )
        lower = self._grid[(prefix_tokens, lower_query)].layers[layer_idx][phase]
        if upper_query == lower_query:
            return lower
        upper = self._grid[(prefix_tokens, upper_query)].layers[layer_idx][phase]
        ratio = (query_tokens - lower_query) / (upper_query - lower_query)
        return lower + (upper - lower) * ratio


class ProfileBundle:
    """Versioned profiles consumed by the simulation runtime."""

    def __init__(
        self,
        *,
        metadata: dict[str, Any],
        layer_count: int,
        topologies: dict[str, TopologyProfile],
        topology_specs: dict[str, dict[str, Any]],
    ) -> None:
        self.metadata = metadata
        self.layer_count = layer_count
        self.topologies = topologies
        self.topology_specs = topology_specs
        for required in ("afd", "merged"):
            if required not in topologies:
                raise ValueError(f"profile bundle is missing {required!r} topology")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ProfileBundle:
        if int(raw.get("schema_version", 0)) != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported profile schema_version {raw.get('schema_version')!r}"
            )
        layer_count = int(raw.get("layer_count", 43))
        topology_profiles = {}
        topology_specs = {}
        for name, topology in raw.get("topologies", {}).items():
            points = tuple(
                ProfilePoint.from_mapping(point, layer_count)
                for point in topology.get("points", [])
            )
            max_context = topology.get("max_context_tokens")
            topology_profiles[name] = TopologyProfile(
                name,
                points,
                layer_count,
                int(max_context) if max_context is not None else None,
            )
            topology_specs[name] = dict(topology.get("spec", {}))
        return cls(
            metadata=dict(raw.get("metadata", {})),
            layer_count=layer_count,
            topologies=topology_profiles,
            topology_specs=topology_specs,
        )

    @classmethod
    def load(cls, path: str | Path) -> ProfileBundle:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def duration_ms(
        self,
        topology: str,
        layer_idx: int,
        phase: str,
        prefix_tokens: int,
        query_tokens: int | float,
    ) -> float:
        try:
            profile = self.topologies[topology]
        except KeyError as exc:
            raise ValueError(f"unknown topology profile {topology!r}") from exc
        return profile.duration_ms(layer_idx, phase, prefix_tokens, query_tokens)

    def expert_shape_samples(
        self,
        topology: str,
        layer_idx: int,
        phase: str,
        query_tokens: int | float,
    ) -> list[dict[str, Any]]:
        return self.topologies[topology].expert_shape_samples(
            layer_idx, phase, query_tokens
        )

    def device_budget(self) -> dict[str, int]:
        """Return device counts and reject non-equal architecture budgets."""

        afd_spec = self.topology_specs["afd"]
        attention_devices = _topology_device_count(
            afd_spec["attention"], "AFD Attention"
        )
        ffn_devices = _topology_device_count(afd_spec["ffn"], "AFD FFN")
        merged_devices = _topology_device_count(
            self.topology_specs["merged"], "merged"
        )
        afd_devices = attention_devices + ffn_devices
        if afd_devices != merged_devices:
            raise ValueError(
                "profile device budget mismatch: "
                f"AFD uses {attention_devices} Attention + {ffn_devices} FFN "
                f"= {afd_devices} dies, but merged uses {merged_devices} dies"
            )
        return {
            "afd_attention": attention_devices,
            "afd_ffn": ffn_devices,
            "afd_total": afd_devices,
            "merged": merged_devices,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "layer_count": self.layer_count,
            "topologies": {
                name: {
                    "point_count": len(profile.points),
                    "prefix_anchors": profile.prefix_anchors,
                    "query_anchors": profile.query_anchors,
                    "max_context_tokens": profile.max_context_tokens,
                    "spec": self.topology_specs[name],
                }
                for name, profile in self.topologies.items()
            },
        }


def _topology_device_count(spec: dict[str, Any], label: str) -> int:
    num_devices = int(spec["num_devices"])
    dp_size = int(spec["dp_size"])
    tp_size = int(spec["tp_size"])
    if min(num_devices, dp_size, tp_size) <= 0:
        raise ValueError(f"{label} parallel sizes must be positive")
    if dp_size * tp_size != num_devices:
        raise ValueError(
            f"{label} DP{dp_size} x TP{tp_size} requires "
            f"{dp_size * tp_size} dies, not {num_devices}"
        )
    return num_devices


def _bounds(
    anchors: tuple[int, ...], value: int | float, label: str
) -> tuple[int, int]:
    if not anchors:
        raise ValueError(f"profile contains no {label} anchors")
    if value < anchors[0] or value > anchors[-1]:
        raise ValueError(
            f"{label}={value} outside profile domain [{anchors[0]}, {anchors[-1]}]"
        )
    for anchor in anchors:
        if value == anchor:
            return anchor, anchor
        if value < anchor:
            index = anchors.index(anchor)
            return anchors[index - 1], anchor
    return anchors[-1], anchors[-1]


__all__ = [
    "PROFILE_PHASES",
    "PROFILE_SCHEMA_VERSION",
    "ProfileBundle",
    "ProfilePoint",
    "TopologyProfile",
]
