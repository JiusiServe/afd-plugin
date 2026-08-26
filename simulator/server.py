"""Dependency-free local HTTP server for the simulator UI."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from simulator.config import SimulationConfig, default_config_mapping
from simulator.engine import compare_architectures, sweep_qps
from simulator.profiles import ProfileBundle

WEB_ROOT = Path(__file__).with_name("web")


class SimulatorServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        profiles: ProfileBundle,
    ) -> None:
        self.profiles = profiles
        super().__init__(address, SimulatorRequestHandler)


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
                    "profile": self.server.profiles.summary(),
                }
            )
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            raw = self._read_json()
            config = SimulationConfig.from_mapping(raw)
            if self.path == "/api/simulate":
                self._send_json(compare_architectures(config, self.server.profiles))
                return
            if self.path == "/api/sweep":
                self._send_json(sweep_qps(config, self.server.profiles))
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


def serve(
    profiles: ProfileBundle,
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
