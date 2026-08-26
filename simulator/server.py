"""Dependency-free local HTTP server for the simulator UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from simulator.config import SimulationConfig, default_config_mapping
from simulator.engine import compare_architectures, sweep_qps
from simulator.length_datasets import LENGTH_DATASETS, length_dataset_catalog
from simulator.profiles import ProfileBundle

WEB_ROOT = Path(__file__).with_name("web")


@dataclass(frozen=True)
class TopologyChoice:
    topology_id: str
    source_profile_id: str
    source_profile: ProfileBundle
    num_devices: int


class SimulatorServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        profiles: dict[str, ProfileBundle],
    ) -> None:
        if not profiles:
            raise ValueError("at least one profile is required")
        if any(not profile_id for profile_id in profiles):
            raise ValueError("profile ids must be non-empty")
        self.afd_topologies: dict[str, TopologyChoice] = {}
        self.merged_topologies: dict[str, TopologyChoice] = {}
        self.default_merged_by_afd: dict[str, str] = {}
        self.length_datasets = length_dataset_catalog()
        for profile_id, profile in profiles.items():
            budget = profile.device_budget()
            afd_id = _topology_id("afd", profile.topology_specs["afd"])
            merged_id = _topology_id("merged", profile.topology_specs["merged"])
            self.afd_topologies.setdefault(
                afd_id,
                TopologyChoice(
                    topology_id=afd_id,
                    source_profile_id=profile_id,
                    source_profile=profile,
                    num_devices=budget["afd_total"],
                ),
            )
            self.merged_topologies.setdefault(
                merged_id,
                TopologyChoice(
                    topology_id=merged_id,
                    source_profile_id=profile_id,
                    source_profile=profile,
                    num_devices=budget["merged"],
                ),
            )
            self.default_merged_by_afd.setdefault(afd_id, merged_id)
        self.default_afd_topology_id = next(iter(self.afd_topologies))
        super().__init__(address, SimulatorRequestHandler)

    def comparison_profile(
        self,
        afd_topology_id: str,
        merged_topology_id: str,
    ) -> ProfileBundle:
        try:
            afd = self.afd_topologies[afd_topology_id]
        except KeyError as exc:
            raise ValueError(f"unknown afd_topology_id: {afd_topology_id}") from exc
        try:
            merged = self.merged_topologies[merged_topology_id]
        except KeyError as exc:
            raise ValueError(
                f"unknown merged_topology_id: {merged_topology_id}"
            ) from exc
        if afd.num_devices != merged.num_devices:
            raise ValueError(
                "topology device budget mismatch: "
                f"AFD uses {afd.num_devices} dies, but merged uses "
                f"{merged.num_devices} dies"
            )
        if afd.source_profile.layer_count != merged.source_profile.layer_count:
            raise ValueError("selected topologies have different layer counts")
        metadata = dict(afd.source_profile.metadata)
        metadata["topology_sources"] = {
            "afd": afd.source_profile_id,
            "merged": merged.source_profile_id,
        }
        comparison = ProfileBundle(
            metadata=metadata,
            layer_count=afd.source_profile.layer_count,
            topologies={
                "afd": afd.source_profile.topologies["afd"],
                "merged": merged.source_profile.topologies["merged"],
            },
            topology_specs={
                "afd": afd.source_profile.topology_specs["afd"],
                "merged": merged.source_profile.topology_specs["merged"],
            },
        )
        comparison.device_budget()
        return comparison


class SimulatorRequestHandler(BaseHTTPRequestHandler):
    server: SimulatorServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send_bytes(
                (WEB_ROOT / "index.html").read_bytes(),
                content_type="text/html; charset=utf-8",
            )
            return
        if self.path == "/api/defaults":
            self._send_json(
                {
                    "config": default_config_mapping(),
                    "length_datasets": [
                        dataset.summary() for dataset in LENGTH_DATASETS
                    ],
                    "default_afd_topology_id": (
                        self.server.default_afd_topology_id
                    ),
                    "afd_topologies": [
                        _topology_catalog_summary(
                            choice,
                            "afd",
                            default_merged_topology_id=(
                                self.server.default_merged_by_afd[choice.topology_id]
                            ),
                        )
                        for choice in sorted(
                            self.server.afd_topologies.values(),
                            key=lambda item: (item.num_devices, item.topology_id),
                        )
                    ],
                    "merged_topologies": [
                        _topology_catalog_summary(choice, "merged")
                        for choice in sorted(
                            self.server.merged_topologies.values(),
                            key=lambda item: (item.num_devices, item.topology_id),
                        )
                    ],
                }
            )
            return
        length_dataset_prefix = "/api/length-datasets/"
        if self.path.startswith(length_dataset_prefix):
            dataset_id = self.path.removeprefix(length_dataset_prefix)
            try:
                dataset = self.server.length_datasets[dataset_id]
            except KeyError:
                self._send_json(
                    {"error": f"unknown length dataset: {dataset_id}"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_bytes(
                dataset.csv_bytes(),
                content_type="text/csv; charset=utf-8",
            )
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            raw = self._read_json()
            afd_topology_id = raw.pop("afd_topology_id", None)
            merged_topology_id = raw.pop("merged_topology_id", None)
            if not isinstance(afd_topology_id, str) or not afd_topology_id:
                raise ValueError("afd_topology_id is required")
            if not isinstance(merged_topology_id, str) or not merged_topology_id:
                raise ValueError("merged_topology_id is required")
            profiles = self.server.comparison_profile(
                afd_topology_id,
                merged_topology_id,
            )
            config = SimulationConfig.from_mapping(raw)
            if self.path == "/api/simulate":
                result = compare_architectures(config, profiles)
                result["topology_selection"] = {
                    "afd": afd_topology_id,
                    "merged": merged_topology_id,
                }
                self._send_json(result)
                return
            if self.path == "/api/sweep":
                result = sweep_qps(config, profiles)
                result["topology_selection"] = {
                    "afd": afd_topology_id,
                    "merged": merged_topology_id,
                }
                self._send_json(result)
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.UNPROCESSABLE_ENTITY)
        except Exception as exc:  # pragma: no cover - server safety boundary
            self._send_json(
                {"error": f"internal error: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[simulator] {self.address_string()} {format % args}")

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("request body is empty")
        if content_length > 10 * 1024 * 1024:
            raise ValueError("request body exceeds 10 MiB")
        body = self.rfile.read(content_length)
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def _send_bytes(
        self,
        payload: bytes,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def _topology_id(topology: str, spec: dict[str, Any]) -> str:
    if topology == "afd":
        attention = spec["attention"]
        ffn = spec["ffn"]
        return (
            f"afd-attn-dp{attention['dp_size']}tp{attention['tp_size']}"
            f"ep{attention['ep_size']}-ffn-dp{ffn['dp_size']}"
            f"tp{ffn['tp_size']}ep{ffn['ep_size']}"
        )
    if topology == "merged":
        return (
            f"merged-dp{spec['dp_size']}tp{spec['tp_size']}ep{spec['ep_size']}"
        )
    raise ValueError(f"unknown topology: {topology}")


def _topology_catalog_summary(
    choice: TopologyChoice,
    topology: str,
    *,
    default_merged_topology_id: str | None = None,
) -> dict[str, Any]:
    profile_summary = choice.source_profile.summary()
    topology_summary = profile_summary["topologies"][topology]
    metadata = profile_summary["metadata"]
    return {
        "id": choice.topology_id,
        "source_profile_id": choice.source_profile_id,
        "num_devices": choice.num_devices,
        "default_merged_topology_id": default_merged_topology_id,
        "metadata": {
            key: metadata[key]
            for key in (
                "model",
                "device",
                "performance_model",
                "generated_at",
                "notes",
            )
            if key in metadata
        },
        "layer_count": profile_summary["layer_count"],
        "spec": topology_summary["spec"],
        "max_context_tokens": topology_summary["max_context_tokens"],
    }


def serve(
    profiles: dict[str, ProfileBundle],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = SimulatorServer((host, port), profiles)
    print(f"DSV4 Prefill simulator: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["SimulatorServer", "serve"]
