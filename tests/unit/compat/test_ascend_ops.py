from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from afd_plugin.compat.ascend import ops
from afd_plugin.compat.ascend.ops import (
    ensure_afd_ascend_ops_loaded,
    ensure_cam_ops_available,
    get_afd_cann_vendor_path,
    get_afd_cust_opapi_path,
    has_afd_ascend_ops,
)


def test_ascend_ops_loader_fails_clearly_without_extension():
    if has_afd_ascend_ops():
        pytest.skip("AFD Ascend extension is installed in this environment")

    with pytest.raises(RuntimeError, match="AFD Ascend custom ops"):
        ensure_afd_ascend_ops_loaded()


def test_ascend_ops_paths_are_plugin_owned():
    vendor_path = get_afd_cann_vendor_path()
    cust_opapi_path = get_afd_cust_opapi_path()

    assert vendor_path.parts[-3:] == ("_cann_ops_custom", "vendors", "afd-plugin")
    assert "vllm-ascend" not in vendor_path.parts
    assert cust_opapi_path.name == "libcust_opapi.so"
    assert cust_opapi_path.parts[-6:] == (
        "_cann_ops_custom",
        "vendors",
        "afd-plugin",
        "op_api",
        "lib",
        "libcust_opapi.so",
    )


def test_ascend_ops_loader_sets_explicit_cust_opapi_path(tmp_path, monkeypatch):
    vendor_path = (
        tmp_path / "afd_plugin" / "_cann_ops_custom" / "vendors" / "afd-plugin"
    )
    op_api_lib = vendor_path / "op_api" / "lib"
    op_api_lib.mkdir(parents=True)
    cust_opapi_path = op_api_lib / "libcust_opapi.so"
    cust_opapi_path.touch()
    monkeypatch.setattr(ops, "get_afd_cann_vendor_path", lambda: vendor_path)
    monkeypatch.setattr(ops, "get_afd_cust_opapi_path", lambda: cust_opapi_path)
    monkeypatch.delenv("ASCEND_CUSTOM_OPP_PATH", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("AFD_CUST_OPAPI_LIB_PATH", raising=False)

    ops._ensure_custom_opp_env()

    assert str(vendor_path) in ops.os.environ["ASCEND_CUSTOM_OPP_PATH"]
    assert str(op_api_lib) in ops.os.environ["LD_LIBRARY_PATH"]
    assert ops.os.environ["AFD_CUST_OPAPI_LIB_PATH"] == str(cust_opapi_path)


def test_ascend_ops_namespace_check_allows_vllm_ascend_coexistence():
    class _Namespace:
        a2e = object()
        e2a = object()

    class _Ops:
        _C_ascend = _Namespace()
        afd_ascend = _Namespace()

    class _Torch:
        ops = _Ops()

    ops._assert_afd_namespace_registered(_Torch)


def test_cam_ops_namespace_check_requires_real_async_ops(monkeypatch):
    fake_torch = SimpleNamespace(ops=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    with pytest.raises(RuntimeError, match="umdk_cam_op_lib"):
        ensure_cam_ops_available()


def test_cam_ops_namespace_check_imports_runtime_modules_in_order(monkeypatch):
    cam_namespace = SimpleNamespace(
        async_dispatch_send=object(),
        async_dispatch_recv=object(),
        async_combine_send=object(),
        async_combine_recv=object(),
    )
    fake_torch = SimpleNamespace(
        ops=SimpleNamespace(umdk_cam_op_lib=cam_namespace),
    )
    imports = []
    original_import = builtins.__import__

    def record_import(name, *args, **kwargs):
        if name in {"torch", "torch_npu", "umdk_cam_op_lib"}:
            imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", record_import)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "umdk_cam_op_lib", SimpleNamespace())

    ensure_cam_ops_available()

    assert imports[:3] == ["torch", "torch_npu", "umdk_cam_op_lib"]
