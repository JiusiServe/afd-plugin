"""Generate normalized DSV4 profiles from msModeling analytic traces."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simulator.profiles import PROFILE_PHASES, PROFILE_SCHEMA_VERSION

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_DEVICE = "ATLAS_800_A3_752T_128G_DIE"
DEFAULT_HIDDEN_SIZE = 4_096
DEFAULT_MOE_TOP_K = 6
EXPECTED_LAYER_COUNT = 43
GATE_OPS = {
    "tensor_cast.moe_gating_top_k.default",
    "tensor_cast.moe_gating_top_k_hash.default",
}
ALL_TO_ALL_OP = "tensor_cast.all_to_all.default"
ALL_GATHER_OP = "tensor_cast.all_gather.default"
HC_PRE_OP = "tensor_cast.hc_pre_inv_rms.default"
HC_POST_OP = "tensor_cast.hc_post.default"
SHARED_EXPERT_START_OP = "tensor_cast.static_quant_linear.default"
SPLIT_WITH_SIZES_OP = "aten.split_with_sizes.default"
ROUTED_EXPERT_SHAPE = "routed_expert_input_shape"
ROUTED_EXPERT_SAMPLE_SHAPES = "routed_expert_sample_shapes"
SHARED_EXPERT_SHAPE = "shared_expert_input_shape"


@dataclass(frozen=True)
class TopologySpec:
    name: str
    num_devices: int
    dp_size: int
    tp_size: int
    ep_size: int
    sequence_parallel: bool = True


DEFAULT_TOPOLOGIES = (
    TopologySpec("afd", num_devices=8, dp_size=2, tp_size=4, ep_size=8),
    TopologySpec("merged", num_devices=16, dp_size=4, tp_size=4, ep_size=16),
)
DEFAULT_AFD_FFN_TOPOLOGY = TopologySpec(
    "afd_ffn",
    num_devices=8,
    dp_size=8,
    tp_size=1,
    ep_size=8,
    sequence_parallel=False,
)


def build_profile_bundle(
    *,
    msmodeling_root: str | Path,
    python_executable: str,
    output_path: str | Path,
    query_anchors: Iterable[int],
    prefix_anchors: Iterable[int],
    model_id: str = DEFAULT_MODEL_ID,
    device: str = DEFAULT_DEVICE,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    moe_top_k: int = DEFAULT_MOE_TOP_K,
    topologies: tuple[TopologySpec, ...] = DEFAULT_TOPOLOGIES,
    afd_ffn_topology: TopologySpec = DEFAULT_AFD_FFN_TOPOLOGY,
    keep_traces: str | Path | None = None,
) -> dict[str, Any]:
    """Run msModeling for a triangular `(prefix, query)` grid."""

    if hidden_size <= 0 or moe_top_k <= 0:
        raise ValueError("hidden_size and moe_top_k must be positive")

    root = Path(msmodeling_root).resolve()
    if not (root / "cli" / "inference" / "text_generate.py").is_file():
        raise ValueError(f"not an msModeling checkout: {root}")
    query_values = tuple(sorted({int(value) for value in query_anchors}))
    prefix_values = tuple(sorted({int(value) for value in prefix_anchors}))
    max_context_tokens, anchor_grid = _anchor_grid(query_values, prefix_values)

    trace_root_context = (
        tempfile.TemporaryDirectory(prefix="dsv4-profile-")
        if keep_traces is None
        else None
    )
    trace_root = Path(
        trace_root_context.name if trace_root_context else keep_traces
    ).resolve()
    trace_root.mkdir(parents=True, exist_ok=True)
    command_log: list[list[str]] = []
    topology_payload: dict[str, Any] = {}
    try:
        for topology in topologies:
            points = []
            ffn_cache: dict[int, list[dict[str, Any]]] = {}
            for prefix_tokens, row_queries in anchor_grid.items():
                for query_tokens in row_queries:
                    trace_path = trace_root / (
                        f"{topology.name}-p{prefix_tokens}-q{query_tokens}.json"
                    )
                    command = _msmodeling_command(
                        python_executable=python_executable,
                        model_id=model_id,
                        device=device,
                        topology=topology,
                        prefix_tokens=prefix_tokens,
                        query_tokens=query_tokens,
                        trace_path=trace_path,
                    )
                    command_log.append(command)
                    subprocess.run(command, cwd=root, check=True)
                    layers = aggregate_trace(trace_path)
                    if topology.name == "afd":
                        local_query = max(
                            1,
                            (query_tokens + afd_ffn_topology.dp_size - 1)
                            // afd_ffn_topology.dp_size,
                        )
                        if local_query not in ffn_cache:
                            ffn_trace_path = trace_root / (
                                f"afd-ffn-q{local_query}.json"
                            )
                            ffn_command = _msmodeling_command(
                                python_executable=python_executable,
                                model_id=model_id,
                                device=device,
                                topology=afd_ffn_topology,
                                prefix_tokens=0,
                                query_tokens=local_query,
                                trace_path=ffn_trace_path,
                            )
                            command_log.append(ffn_command)
                            subprocess.run(ffn_command, cwd=root, check=True)
                            ffn_cache[local_query] = aggregate_trace(ffn_trace_path)
                        layers = _compose_afd_layers(
                            attention_layers=layers,
                            ffn_layers=ffn_cache[local_query],
                        )
                    if prefix_tokens != 0:
                        for layer in layers:
                            layer.pop(ROUTED_EXPERT_SHAPE, None)
                            layer.pop(ROUTED_EXPERT_SAMPLE_SHAPES, None)
                            layer.pop(SHARED_EXPERT_SHAPE, None)
                    points.append(
                        {
                            "prefix_tokens": prefix_tokens,
                            "query_tokens": query_tokens,
                            "layers": layers,
                        }
                    )
            spec: dict[str, Any] = topology.__dict__
            if topology.name == "afd":
                spec = {
                    "attention": topology.__dict__,
                    "ffn": afd_ffn_topology.__dict__,
                    "ffn_job_mapping": ("local_query=ceil(job_query/ffn.dp_size)"),
                }
            topology_payload[topology.name] = {
                "spec": spec,
                "max_context_tokens": max_context_tokens,
                "points": points,
            }
    finally:
        if trace_root_context is not None:
            trace_root_context.cleanup()

    payload = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "layer_count": EXPECTED_LAYER_COUNT,
        "metadata": {
            "model": model_id,
            "device": device,
            "model_config": {
                "hidden_size": hidden_size,
                "moe_top_k": moe_top_k,
            },
            "performance_model": "analytic",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "msmodeling_root": str(root),
            "commands": command_log,
            "phase_sources": {
                "afd.attention_router,afd.afd_post": "afd attention trace",
                "afd.routed_experts,afd.shared_expert": "afd_ffn job trace",
                "merged.*": "merged trace",
            },
            "notes": "CAM communication is not included in these profiles.",
        },
        "topologies": topology_payload,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _anchor_grid(
    query_values: tuple[int, ...], prefix_values: tuple[int, ...]
) -> tuple[int, dict[int, tuple[int, ...]]]:
    if not query_values or not prefix_values:
        raise ValueError("profile anchors cannot be empty")
    max_context_tokens = max(query_values)
    if max_context_tokens < 2:
        raise ValueError("largest query anchor must be at least 2")
    if any(prefix < 0 or prefix >= max_context_tokens for prefix in prefix_values):
        raise ValueError("prefix anchors must be in [0, max context)")
    prefixes = sorted({0, *prefix_values, max_context_tokens - 1})
    queries_with_endpoints = {1, *query_values}
    grid = {}
    for prefix in prefixes:
        queries = {
            query
            for query in queries_with_endpoints
            if query > 0 and prefix + query <= max_context_tokens
        }
        queries.add(max_context_tokens - prefix)
        grid[prefix] = tuple(sorted(queries))
    return max_context_tokens, grid


def _compose_afd_layers(
    *,
    attention_layers: list[dict[str, Any]],
    ffn_layers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **attention,
            "routed_experts": ffn["routed_experts"],
            "shared_expert": ffn["shared_expert"],
            ROUTED_EXPERT_SHAPE: ffn[ROUTED_EXPERT_SHAPE],
            ROUTED_EXPERT_SAMPLE_SHAPES: ffn[ROUTED_EXPERT_SAMPLE_SHAPES],
            SHARED_EXPERT_SHAPE: ffn[SHARED_EXPERT_SHAPE],
        }
        for attention, ffn in zip(attention_layers, ffn_layers, strict=True)
    ]


def aggregate_trace(path: str | Path) -> list[dict[str, Any]]:
    """Aggregate one msModeling trace into stable per-layer phases."""

    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    events = [
        event
        for event in raw.get("traceEvents", [])
        if event.get("ph") == "X" and event.get("cat") == "analytic"
    ]
    gates = [
        index for index, event in enumerate(events) if event.get("name") in GATE_OPS
    ]
    if len(gates) != EXPECTED_LAYER_COUNT:
        raise ValueError(
            f"trace {path} contains {len(gates)} MoE gates, "
            f"expected {EXPECTED_LAYER_COUNT}"
        )

    layers = []
    for layer_idx, gate_index in enumerate(gates):
        previous_gate = gates[layer_idx - 1] if layer_idx else 0
        next_gate = gates[layer_idx + 1] if layer_idx + 1 < len(gates) else len(events)
        hcpres = [
            index
            for index in range(previous_gate, gate_index)
            if events[index].get("name") == HC_PRE_OP
        ]
        layer_start = hcpres[-2] if len(hcpres) >= 2 else previous_gate
        all_to_alls = [
            index
            for index in range(gate_index, next_gate)
            if events[index].get("name") == ALL_TO_ALL_OP
        ]
        if len(all_to_alls) != 2:
            raise ValueError(
                f"layer {layer_idx} contains {len(all_to_alls)} "
                "all_to_all ops, expected 2"
            )
        dispatch_index, combine_index = all_to_alls
        gather_indices = [
            index
            for index in range(combine_index, next_gate)
            if events[index].get("name") == ALL_GATHER_OP
        ]
        hcpost_indices = [
            index
            for index in range(combine_index, next_gate)
            if events[index].get("name") == HC_POST_OP
        ]
        if not hcpost_indices:
            raise ValueError(f"layer {layer_idx} is missing hc_post")
        hcpost_index = hcpost_indices[0]
        gather_index = gather_indices[0] if gather_indices else None
        shared_end = gather_index if gather_index is not None else hcpost_index
        layer_end = hcpost_index + 1
        shared_starts = [
            index
            for index in range(combine_index + 1, shared_end)
            if events[index].get("name") == SHARED_EXPERT_START_OP
        ]
        shared_start = shared_starts[0] if shared_starts else shared_end

        layer = {phase: 0.0 for phase in PROFILE_PHASES}
        layer["attention_router"] = _duration_ms(events[layer_start:dispatch_index])
        layer["merged_dispatch"] = _duration_ms(
            events[dispatch_index : dispatch_index + 1]
        )
        layer["routed_experts"] = _duration_ms(
            events[dispatch_index + 1 : combine_index]
        )
        layer["merged_combine"] = _duration_ms(
            events[combine_index : combine_index + 1]
        )
        layer["merged_combine_local"] = _duration_ms(
            events[combine_index + 1 : shared_start]
        )
        layer["shared_expert"] = _duration_ms(events[shared_start:shared_end])
        layer[ROUTED_EXPERT_SHAPE] = _output_shape(events[dispatch_index], path)
        routed_splits = [
            event
            for event in events[dispatch_index + 1 : combine_index]
            if event.get("name") == SPLIT_WITH_SIZES_OP
        ]
        if not routed_splits:
            raise ValueError(
                f"trace {path} layer {layer_idx} has no routed expert split"
            )
        layer[ROUTED_EXPERT_SAMPLE_SHAPES] = _output_shape_counts(
            routed_splits[0], path
        )
        layer[SHARED_EXPERT_SHAPE] = _input_shape(events[shared_start], path)
        if gather_index is not None:
            layer["merged_sp_post"] = _duration_ms(events[gather_index:layer_end])
        layer["afd_post"] = _duration_ms([events[hcpost_index]])
        layers.append(layer)
    return layers


def _output_shape(event: dict[str, Any], path: str | Path) -> list[int]:
    output = str(event.get("args", {}).get("Output", ""))
    match = re.search(r"size=\((\d+(?:\s*,\s*\d+)*)\)", output)
    if match is None:
        raise ValueError(f"trace {path} event {event.get('name')} has no output shape")
    return [int(value.strip()) for value in match.group(1).split(",")]


def _input_shape(event: dict[str, Any], path: str | Path) -> list[int]:
    raw_shapes = event.get("args", {}).get("simulation_shapes")
    shapes = json.loads(raw_shapes) if isinstance(raw_shapes, str) else raw_shapes
    if not shapes or not shapes[0]:
        raise ValueError(f"trace {path} event {event.get('name')} has no input shape")
    shape = [int(value) for value in shapes[0]]
    return [math.prod(shape[:-1]), shape[-1]]


def _output_shape_counts(
    event: dict[str, Any], path: str | Path
) -> list[dict[str, Any]]:
    output = str(event.get("args", {}).get("Output", ""))
    shapes = re.findall(r"size=\((\d+(?:\s*,\s*\d+)*)\)", output)
    if not shapes:
        raise ValueError(f"trace {path} event {event.get('name')} has no output shapes")
    counts = Counter(
        tuple(int(value.strip()) for value in shape.split(",")) for shape in shapes
    )
    return [{"shape": list(shape), "count": count} for shape, count in counts.items()]


def _duration_ms(events: Iterable[dict[str, Any]]) -> float:
    return sum(float(event.get("dur", 0.0)) for event in events) / 1_000.0


def _msmodeling_command(
    *,
    python_executable: str,
    model_id: str,
    device: str,
    topology: TopologySpec,
    prefix_tokens: int,
    query_tokens: int,
    trace_path: Path,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "cli.inference.text_generate",
        model_id,
        "--device",
        device,
        "--num-devices",
        str(topology.num_devices),
        "--tp-size",
        str(topology.tp_size),
        "--dp-size",
        str(topology.dp_size),
        "--ep-size",
        str(topology.ep_size),
    ]
    if topology.sequence_parallel:
        command.append("--enable-sequence-parallel")
    command.extend(
        [
            "--compile",
            "--num-queries",
            str(topology.dp_size),
            "--query-length",
            str(query_tokens),
            "--context-length",
            str(prefix_tokens),
            "--performance-model",
            "analytic",
            "--chrome-trace",
            str(trace_path),
        ]
    )
    return command


__all__ = [
    "DEFAULT_DEVICE",
    "DEFAULT_AFD_FFN_TOPOLOGY",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MOE_TOP_K",
    "DEFAULT_TOPOLOGIES",
    "TopologySpec",
    "aggregate_trace",
    "build_profile_bundle",
]
