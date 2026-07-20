# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Loader for plugin-owned Ascend custom operators."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

AFD_ASCEND_OPS_NAMESPACE = "afd_ascend"
AFD_ASCEND_VENDOR_NAME = "afd-plugin"
AFD_CUST_OPAPI_ENV = "AFD_CUST_OPAPI_LIB_PATH"
CAM_OP_NAMESPACE = "umdk_cam_op_lib"
CAM_DISPATCH_SEND = "async_dispatch_send"
CAM_DISPATCH_RECV = "async_dispatch_recv"
CAM_COMBINE_SEND = "async_combine_send"
CAM_COMBINE_RECV = "async_combine_recv"


def get_afd_cann_vendor_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "_cann_ops_custom"
        / "vendors"
        / AFD_ASCEND_VENDOR_NAME
    )


def get_afd_cust_opapi_path() -> Path:
    return get_afd_cann_vendor_path() / "op_api" / "lib" / "libcust_opapi.so"


def _prepend_env_path(name: str, path: Path) -> None:
    path_str = str(path)
    current = os.environ.get(name, "")
    entries = [entry for entry in current.split(":") if entry]
    if path_str in entries:
        return
    os.environ[name] = ":".join([path_str, *entries])


def _ensure_afd_custom_opp_env() -> None:
    vendor_dir = get_afd_cann_vendor_path()
    if not vendor_dir.exists():
        return

    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
    op_api_lib = vendor_dir / "op_api" / "lib"
    if op_api_lib.exists():
        _prepend_env_path("LD_LIBRARY_PATH", op_api_lib)
    cust_opapi = get_afd_cust_opapi_path()
    if cust_opapi.exists():
        os.environ[AFD_CUST_OPAPI_ENV] = str(cust_opapi)


def _assert_afd_namespace_registered(torch: object) -> None:
    _ = torch.ops.afd_ascend.a2e
    _ = torch.ops.afd_ascend.e2a


def _assert_cam_namespace_registered(torch: object) -> None:
    _ = (
        torch.ops.umdk_cam_op_lib.async_dispatch_send,
        torch.ops.umdk_cam_op_lib.async_dispatch_recv,
        torch.ops.umdk_cam_op_lib.async_combine_send,
        torch.ops.umdk_cam_op_lib.async_combine_recv,
    )


@lru_cache(maxsize=1)
def ensure_cam_p2p_ops_available() -> None:
    """Import the custom operators used by ``CAMP2pAFDConnector``.

    The extension is optional at package import time.  It is built by default
    in an Ascend environment unless ``AFD_BUILD_ASCEND_OPS=0`` is set.
    """

    _ensure_afd_custom_opp_env()
    try:
        import torch

        import afd_plugin._C_ascend  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "CAMP2P Ascend custom ops are not available. Build the package with "
            "Ascend ops enabled in a torch-npu/CANN environment.",
        ) from exc
    _assert_afd_namespace_registered(torch)


def ensure_afd_ascend_ops_loaded() -> None:
    """Backward-compatible alias for the CAMP2P custom-op loader."""

    ensure_cam_p2p_ops_available()


def has_afd_ascend_ops() -> bool:
    try:
        ensure_cam_p2p_ops_available()
    except RuntimeError:
        return False
    return True


def ensure_cam_async_ops_available() -> None:
    """Ensure the runtime exposes the real CAM async operator namespace."""

    try:
        import torch
        import torch_npu  # noqa: F401
        import umdk_cam_op_lib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "CAMAsyncAFDConnector requires torch, torch_npu, umdk_cam_op_lib, "
            "and preloaded real torch.ops.umdk_cam_op_lib CAM ops.",
        ) from exc

    try:
        _assert_cam_namespace_registered(torch)
    except AttributeError as exc:
        raise RuntimeError(
            "CAMAsyncAFDConnector requires real torch.ops.umdk_cam_op_lib CAM ops "
            "(async_dispatch_send, async_dispatch_recv, async_combine_send, "
            "async_combine_recv). Install or preload the CAM operator binaries "
            "before initializing the connector.",
        ) from exc


__all__ = [
    "AFD_ASCEND_OPS_NAMESPACE",
    "AFD_ASCEND_VENDOR_NAME",
    "AFD_CUST_OPAPI_ENV",
    "CAM_COMBINE_RECV",
    "CAM_COMBINE_SEND",
    "CAM_DISPATCH_RECV",
    "CAM_DISPATCH_SEND",
    "CAM_OP_NAMESPACE",
    "ensure_afd_ascend_ops_loaded",
    "ensure_cam_async_ops_available",
    "ensure_cam_p2p_ops_available",
    "get_afd_cann_vendor_path",
    "get_afd_cust_opapi_path",
    "has_afd_ascend_ops",
]
