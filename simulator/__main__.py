"""Command-line entry point for the DSV4 Prefill simulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simulator.config import SimulationConfig
from simulator.engine import compare_architectures, sweep_qps
from simulator.profile_builder import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MOE_TOP_K,
    build_profile_bundle,
)
from simulator.profiles import ProfileBundle
from simulator.server import serve

DEFAULT_QUERY_ANCHORS = "1,128,512,2048,4096,8192,16384,32768,65536,131072"
DEFAULT_PREFIX_ANCHORS = "0,8192,32768,65536,98304,122880"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve_parser = commands.add_parser("serve", help="start the local web UI")
    _add_profile_argument(serve_parser)
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
    build_parser.add_argument("--keep-traces")

    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(ProfileBundle.load(args.profiles), host=args.host, port=args.port)
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
        kwargs = {
            "msmodeling_root": args.msmodeling_root,
            "python_executable": args.python_executable,
            "output_path": args.output,
            "query_anchors": _parse_anchors(args.query_anchors),
            "prefix_anchors": _parse_anchors(args.prefix_anchors),
            "hidden_size": args.hidden_size,
            "moe_top_k": args.moe_top_k,
            "keep_traces": args.keep_traces,
        }
        if args.model_id:
            kwargs["model_id"] = args.model_id
        if args.device:
            kwargs["device"] = args.device
        build_profile_bundle(**kwargs)
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
