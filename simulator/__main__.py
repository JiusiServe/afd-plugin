"""Command-line entry point for the DSV4 Prefill simulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simulator.config import SimulationConfig
from simulator.engine import compare_architectures, sweep_qps
from simulator.profile_builder import (
    DEFAULT_AFD_TOPOLOGY,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MERGED_TOPOLOGY,
    DEFAULT_MOE_TOP_K,
    TopologySpec,
    build_profile_bundle,
    compose_retargeted_afd_profile,
)
from simulator.profiles import ProfileBundle
from simulator.server import serve

DEFAULT_QUERY_ANCHORS = "1,128,512,2048,4096,8192,16384,32768,65536,131072"
DEFAULT_PREFIX_ANCHORS = "0,8192,32768,65536,98304,122880"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve_parser = commands.add_parser("serve", help="start the local web UI")
    serve_parser.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help="one or more normalized profile JSON files",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)

    simulate_parser = commands.add_parser("simulate", help="run one simulation")
    _add_profile_argument(simulate_parser)
    _add_config_arguments(simulate_parser)

    sweep_parser = commands.add_parser("sweep", help="run an SLO QPS sweep")
    _add_profile_argument(sweep_parser)
    _add_config_arguments(sweep_parser)

    profiles_parser = commands.add_parser("profiles", help="profile operations")
    profile_commands = profiles_parser.add_subparsers(
        dest="profile_command", required=True
    )
    build_parser = profile_commands.add_parser(
        "build", help="generate an analytic profile grid with msModeling"
    )
    build_parser.add_argument("--msmodeling-root", required=True)
    build_parser.add_argument("--python", dest="python_executable", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--model-id", default=None)
    build_parser.add_argument("--device", default=None)
    build_parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    build_parser.add_argument("--moe-top-k", type=int, default=DEFAULT_MOE_TOP_K)
    build_parser.add_argument("--query-anchors", default=DEFAULT_QUERY_ANCHORS)
    build_parser.add_argument("--prefix-anchors", default=DEFAULT_PREFIX_ANCHORS)
    build_parser.add_argument(
        "--topology",
        choices=("all", "merged"),
        default="all",
        help="build both default topologies or only the configured merged topology",
    )
    build_parser.add_argument(
        "--merged-num-devices",
        type=int,
        default=DEFAULT_MERGED_TOPOLOGY.num_devices,
    )
    build_parser.add_argument(
        "--merged-dp-size",
        type=int,
        default=DEFAULT_MERGED_TOPOLOGY.dp_size,
    )
    build_parser.add_argument(
        "--merged-tp-size",
        type=int,
        default=DEFAULT_MERGED_TOPOLOGY.tp_size,
    )
    build_parser.add_argument(
        "--merged-ep-size",
        type=int,
        default=DEFAULT_MERGED_TOPOLOGY.ep_size,
    )
    build_parser.add_argument("--keep-traces")

    compose_parser = profile_commands.add_parser(
        "compose",
        help="reuse AFD per-DP points with a new DP count and merged profile",
    )
    compose_parser.add_argument("--afd-profile", required=True)
    compose_parser.add_argument("--merged-profile", required=True)
    compose_parser.add_argument("--afd-dp-size", type=int, required=True)
    compose_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "serve":
        loaded_profiles = {}
        for path in args.profiles:
            profile_id = Path(path).stem
            if profile_id in loaded_profiles:
                parser.error(f"duplicate profile id: {profile_id}")
            loaded_profiles[profile_id] = ProfileBundle.load(path)
        serve(loaded_profiles, host=args.host, port=args.port)
        return 0
    if args.command in {"simulate", "sweep"}:
        config = _load_config(args.config)
        profiles = ProfileBundle.load(args.profiles)
        result = (
            compare_architectures(config, profiles)
            if args.command == "simulate"
            else sweep_qps(config, profiles)
        )
        _write_result(result, args.output)
        return 0
    if args.command == "profiles" and args.profile_command == "build":
        if min(
            args.merged_num_devices,
            args.merged_dp_size,
            args.merged_tp_size,
            args.merged_ep_size,
        ) <= 0:
            parser.error("merged parallel sizes must be positive")
        if (
            args.merged_dp_size * args.merged_tp_size
            != args.merged_num_devices
        ):
            parser.error(
                "merged DP size multiplied by TP size must equal "
                "merged num devices"
            )
        merged_topology = TopologySpec(
            "merged",
            num_devices=args.merged_num_devices,
            dp_size=args.merged_dp_size,
            tp_size=args.merged_tp_size,
            ep_size=args.merged_ep_size,
        )
        topologies = (
            (DEFAULT_AFD_TOPOLOGY, merged_topology)
            if args.topology == "all"
            else (merged_topology,)
        )
        kwargs = {
            "msmodeling_root": args.msmodeling_root,
            "python_executable": args.python_executable,
            "output_path": args.output,
            "query_anchors": _parse_anchors(args.query_anchors),
            "prefix_anchors": _parse_anchors(args.prefix_anchors),
            "hidden_size": args.hidden_size,
            "moe_top_k": args.moe_top_k,
            "keep_traces": args.keep_traces,
            "topologies": topologies,
        }
        if args.model_id:
            kwargs["model_id"] = args.model_id
        if args.device:
            kwargs["device"] = args.device
        build_profile_bundle(**kwargs)
        print(args.output)
        return 0
    if args.command == "profiles" and args.profile_command == "compose":
        compose_retargeted_afd_profile(
            afd_profile_path=args.afd_profile,
            merged_profile_path=args.merged_profile,
            output_path=args.output,
            afd_dp_size=args.afd_dp_size,
        )
        print(args.output)
        return 0
    parser.error("unsupported command")
    return 2


def _add_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profiles", required=True, help="normalized profile JSON")


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="simulation config JSON")
    parser.add_argument("--output", help="result JSON; stdout when omitted")


def _load_config(path: str) -> SimulationConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config JSON must be an object")
    return SimulationConfig.from_mapping(raw)


def _write_result(result: dict, output: str | None) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
        print(output)
    else:
        sys.stdout.write(text + "\n")


def _parse_anchors(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value.strip()) for value in raw.split(",")}))
    if not values or any(value < 0 for value in values):
        raise ValueError("anchors must be comma-separated non-negative integers")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
