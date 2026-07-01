# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""vllm-omni API server compatibility patch for AFD FFN daemon mode.

When the vllm-omni API server (`omni_run_server_worker`) is the entry point
(i.e. ``--omni`` / ``--deploy-config`` flags are used), it starts an HTTP
service independently of the engine-core subprocess.  For the FFN role this
is wrong: the FFN node should run its connector loop and never expose an HTTP
endpoint.

This patch intercepts ``omni_run_server_worker``.  If the launch args carry
``additional_config.afd.role == "ffn"``, the HTTP layer is replaced by a
simple asyncio wait-for-signal loop so that:

1. ``build_async_omni`` is still entered → the ``StageEngineCoreProc``
   subprocess starts and the patched ``run_busy_loop`` executes the FFN
   connector loop as usual.
2. The parent (API-server) process blocks on SIGTERM / SIGINT instead of
   serving HTTP.
3. On signal the context manager exits and ``AsyncOmni.shutdown()`` cleans up
   the subprocess.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_afd_plugin_omni_server_patch_state"


def apply_omni_server_patch() -> None:
    """Patch ``omni_run_server_worker`` to skip HTTP for the FFN role.

    The patch is idempotent: a second call is a no-op.  If vllm-omni is not
    installed the patch is silently skipped.
    """
    try:
        import vllm_omni.entrypoints.openai.api_server as _server_mod
    except ImportError:
        logger.debug("AFD omni-server patch skipped: vllm-omni is unavailable")
        return

    if hasattr(_server_mod, _PATCH_ATTR):
        return

    original_run_server_worker = _server_mod.omni_run_server_worker

    async def patched_omni_run_server_worker(
        listen_address: Any,
        sock: Any,
        args: Any,
        **kwargs: Any,
    ) -> None:
        if _is_ffn_args(args):
            await _run_ffn_daemon(args, _server_mod)
            return
        await original_run_server_worker(listen_address, sock, args, **kwargs)

    _server_mod.omni_run_server_worker = patched_omni_run_server_worker
    setattr(_server_mod, _PATCH_ATTR, original_run_server_worker)
    logger.debug("AFD omni-server patch applied")


def _is_ffn_args(args: Any) -> bool:
    """Return True when the launch carries ``afd.role == "ffn"``.

    Two entry shapes are handled:

    1. Top-level CLI args (``vllm serve ... --additional-config '{...}'``),
       where ``args.additional_config`` holds the afd block directly.
    2. Deploy-config launches (``vllm serve ... --omni --deploy-config x.yaml``),
       where the afd block lives inside a stage entry of the YAML pointed to by
       ``args.deploy_config`` / ``args.stage_configs_path``. The top-level args
       carry no ``additional_config`` in this case, so we must inspect the YAML.
    """
    if _afd_dict_is_ffn(getattr(args, "additional_config", None)):
        return True
    for attr in ("deploy_config", "stage_configs_path"):
        path = getattr(args, attr, None)
        if path and _deploy_config_is_ffn(path):
            return True
    return False


def _afd_dict_is_ffn(additional: Any) -> bool:
    """Return True when an ``additional_config`` value selects the FFN role."""
    if isinstance(additional, str):
        try:
            import json

            additional = json.loads(additional)
        except Exception:
            return False
    if not isinstance(additional, dict):
        return False
    afd = additional.get("afd", {})
    if not isinstance(afd, dict):
        return False
    return bool(afd.get("enabled", False)) and afd.get("role") == "ffn"


def _deploy_config_is_ffn(path: Any) -> bool:
    """Return True when any stage in the deploy-config YAML selects the FFN role.

    The deploy config may be a filesystem path or a registered/built-in name.
    We resolve it with vllm-omni's own loader when possible (so ``base_config``
    inheritance and the built-in search path are honoured), falling back to a
    plain YAML read of an existing file.
    """
    raw: Any = None
    try:
        from vllm_omni.config.stage_config import resolve_deploy_yaml

        raw = resolve_deploy_yaml(path)
    except Exception:
        try:
            import os

            import yaml

            if not (isinstance(path, str) and os.path.isfile(path)):
                return False
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except Exception:
            return False

    if not isinstance(raw, dict):
        return False
    for stage in raw.get("stages", []) or []:
        if isinstance(stage, dict) and _afd_dict_is_ffn(stage.get("additional_config")):
            return True
    return False


async def _run_ffn_daemon(args: Any, server_mod: Any) -> None:
    """Enter the engine context and wait for a shutdown signal.

    The HTTP server is intentionally NOT started.  The ``StageEngineCoreProc``
    subprocess started by ``build_async_omni`` will execute the AFD FFN
    connector loop via the patched ``EngineCoreProc.run_busy_loop``.
    """
    build_async_omni = getattr(server_mod, "build_async_omni", None)
    if build_async_omni is None:
        logger.error(
            "AFD FFN daemon: build_async_omni not found in server module; "
            "falling back to normal HTTP server startup"
        )
        return

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _set_shutdown() -> None:
        if not shutdown_event.is_set():
            shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _set_shutdown)
        except (NotImplementedError, RuntimeError):
            # Signal handlers are not supported on all platforms / event loops.
            pass

    try:
        async with build_async_omni(args) as _engine:
            logger.info(
                "AFD FFN omni server: engine started. "
                "HTTP API server is intentionally skipped for the FFN role. "
                "Waiting for shutdown signal."
            )
            await shutdown_event.wait()
    finally:
        logger.info("AFD FFN omni server: shutting down.")


__all__ = ["apply_omni_server_patch"]
