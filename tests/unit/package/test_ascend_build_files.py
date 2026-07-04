from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import pytest
import setuptools

_ASCEND_ENV_VARS = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_TOOLKIT_HOME",
    "TORCH_NPU_PATH",
)


def _run_setup_py(
    monkeypatch: pytest.MonkeyPatch,
    *,
    afd_build_ascend_ops: str | None = None,
    has_torch_npu: bool = False,
    has_ascend_toolkit: bool = False,
    ascend_env_var: str | None = None,
) -> list[str]:
    root = Path(__file__).resolve().parents[3]
    captured: dict[str, object] = {}

    def fake_setup(**kwargs: object) -> None:
        captured.update(kwargs)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object | None:
        if name == "torch_npu":
            return object() if has_torch_npu else None
        return real_find_spec(name, *args, **kwargs)

    real_path_exists = Path.exists

    def fake_path_exists(path: Path) -> bool:
        if str(path) == "/usr/local/Ascend/ascend-toolkit/latest":
            return has_ascend_toolkit
        return real_path_exists(path)

    monkeypatch.setattr(setuptools, "setup", fake_setup)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(Path, "exists", fake_path_exists)
    monkeypatch.delenv("AFD_BUILD_ASCEND_OPS", raising=False)
    for name in _ASCEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    if afd_build_ascend_ops is not None:
        monkeypatch.setenv("AFD_BUILD_ASCEND_OPS", afd_build_ascend_ops)
    if ascend_env_var is not None:
        monkeypatch.setenv(ascend_env_var, "/opt/ascend")

    runpy.run_path(str(root / "setup.py"))

    ext_modules = captured["ext_modules"]
    return [ext.name for ext in ext_modules]  # type: ignore[attr-defined]


def test_ascend_a2e_e2a_sources_are_vendored():
    root = Path(__file__).resolve().parents[3]
    required = [
        "csrc/npu/a2e/op_host/aclnn_a2e.cpp",
        "csrc/npu/a2e/op_kernel/a2e.cpp",
        "csrc/npu/a2e/op_kernel/comm_args.h",
        "csrc/npu/a2e/op_kernel/moe_distribute_base.h",
        "csrc/npu/e2a/op_host/aclnn_e2a.cpp",
        "csrc/npu/e2a/op_kernel/e2a.cpp",
        "csrc/npu/e2a/op_kernel/comm_args.h",
        "csrc/npu/e2a/op_kernel/moe_distribute_base.h",
        "csrc/npu/build_aclnn.sh",
        "csrc/npu/torch_extension/CMakeLists.txt",
        "csrc/npu/torch_extension/torch_binding.cpp",
        "csrc/npu/torch_extension/torch_binding_meta.cpp",
    ]

    for relpath in required:
        assert (root / relpath).is_file(), relpath


def test_ascend_ops_build_is_disabled_by_default_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
):
    assert _run_setup_py(monkeypatch) == []


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"has_torch_npu": True}, ["afd_plugin._C_ascend"]),
        ({"has_ascend_toolkit": True}, ["afd_plugin._C_ascend"]),
        ({"ascend_env_var": "ASCEND_HOME_PATH"}, ["afd_plugin._C_ascend"]),
    ],
)
def test_ascend_ops_build_is_enabled_by_default_on_npu(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    expected: list[str],
):
    assert _run_setup_py(monkeypatch, **kwargs) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", ["afd_plugin._C_ascend"]),
        ("true", ["afd_plugin._C_ascend"]),
        ("yes", ["afd_plugin._C_ascend"]),
        ("on", ["afd_plugin._C_ascend"]),
        ("0", []),
        ("false", []),
        ("no", []),
        ("off", []),
    ],
)
def test_ascend_ops_build_env_overrides_platform_default(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: list[str],
):
    assert _run_setup_py(
        monkeypatch,
        afd_build_ascend_ops=value,
        has_torch_npu=value in {"0", "false", "no", "off"},
    ) == expected


def test_ascend_ops_build_env_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
):
    with pytest.raises(RuntimeError, match="AFD_BUILD_ASCEND_OPS"):
        _run_setup_py(monkeypatch, afd_build_ascend_ops="maybe")


def test_ascend_ops_use_isolated_namespace_and_vendor_path():
    root = Path(__file__).resolve().parents[3]
    torch_binding = (root / "csrc/npu/torch_extension/torch_binding.cpp").read_text()
    torch_binding_meta = (
        root / "csrc/npu/torch_extension/torch_binding_meta.cpp"
    ).read_text()
    torch_cmake = (root / "csrc/npu/torch_extension/CMakeLists.txt").read_text()
    cann_cmake = (root / "csrc/npu/CMakeLists.txt").read_text()
    op_api_common = (root / "csrc/npu/aclnn_torch_adapter/op_api_common.h").read_text()

    assert "TORCH_LIBRARY(afd_ascend" in torch_binding
    assert "TORCH_LIBRARY(_C_ascend" not in torch_binding
    assert "TORCH_LIBRARY_IMPL(afd_ascend, Meta" in torch_binding_meta
    assert "TORCH_LIBRARY_IMPL(_C_ascend, Meta" not in torch_binding_meta
    assert "vendors/afd-plugin/op_api/lib" in torch_cmake
    assert "vendors/vllm-ascend/op_api/lib" not in torch_cmake
    assert '"afd-plugin"' in cann_cmake
    assert '"vllm-ascend"' not in cann_cmake
    assert "AFD_CUST_OPAPI_LIB_PATH" in op_api_common
    assert 'return "libcust_opapi.so"' not in op_api_common
