# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe validation helpers for AFD runtime wiring."""

from __future__ import annotations

import importlib
from typing import Any, Final

from afd_plugin.config import AFDConfig, parse_afd_config

ATTENTION_WORKER_FQCN: Final[str] = "afd_plugin.v1.worker.AFDAttentionWorker"
FFN_WORKER_FQCN: Final[str] = "afd_plugin.v1.worker.AFDFFNWorker"
ATTENTION_MODEL_RUNNER_FQCN: Final[str] = "afd_plugin.v1.worker.AFDAttentionModelRunner"
FFN_MODEL_RUNNER_FQCN: Final[str] = "afd_plugin.v1.worker.GPUFFNModelRunner"
UBATCH_WRAPPER_FQCN: Final[str] = "afd_plugin.v1.worker.AFDUBatchWrapper"
NPU_ATTENTION_WORKER_FQCN: Final[str] = (
    "afd_plugin.v1.worker.ascend.AFDNPUAttentionWorker"
)
NPU_FFN_WORKER_FQCN: Final[str] = "afd_plugin.v1.worker.ascend.AFDNPUFFNWorker"
NPU_ATTENTION_MODEL_RUNNER_FQCN: Final[str] = (
    "afd_plugin.v1.worker.ascend.AFDNPUAttentionModelRunner"
)
NPU_FFN_MODEL_RUNNER_FQCN: Final[str] = (
    "afd_plugin.v1.worker.ascend.AFDNPUFFNModelRunner"
)


def normalize_qualname(value: str) -> str:
    return value.replace(":", ".")


def resolve_class_from_qualname(qualname: str, *, role: str = "class") -> type[Any]:
    """Resolve a dotted or colon-separated class path."""

    normalized = normalize_qualname(qualname.strip())
    if not normalized or "." not in normalized:
        raise ValueError(
            f"{role} must be a dotted qualname, got {qualname!r}",
        )
    module_name, obj_name = normalized.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, obj_name)
    if not isinstance(obj, type):
        raise TypeError(
            f"{role} resolved to {type(obj).__name__}, expected a class",
        )
    return obj


def expected_worker_qualname(role: str) -> str:
    if role == "attention":
        return ATTENTION_WORKER_FQCN
    if role == "ffn":
        return FFN_WORKER_FQCN
    raise ValueError(f"unknown AFD role {role!r}")


def expected_npu_worker_qualname(role: str) -> str:
    if role == "attention":
        return NPU_ATTENTION_WORKER_FQCN
    if role == "ffn":
        return NPU_FFN_WORKER_FQCN
    raise ValueError(f"unknown AFD role {role!r}")


def assert_compatible_afd_stack(
    vllm_config: object,
    *,
    caller: str,
    expected_role: str | None = None,
    expected_worker_qualname_override: str | None = None,
    require_enabled: bool = True,
) -> AFDConfig:
    """Validate AFD config and worker class wiring.

    This helper intentionally uses duck typing so unit tests and local CPU
    development do not need to construct a real vLLM ``VllmConfig``.
    """

    def _ctx() -> str:
        return f" (context: {caller!r})"

    config = parse_afd_config(vllm_config, expected_role=expected_role)
    if require_enabled and not config.enabled:
        raise ValueError(f"AFD is not enabled in additional_config['afd']{_ctx()}")

    parallel_config = vllm_config.parallel_config
    async_expected_worker = (
        expected_npu_worker_qualname(config.role)
        if config.connector == "afdasyncconnector"
        else None
    )

    worker_cls_raw = parallel_config.worker_cls
    if not isinstance(worker_cls_raw, str):
        raise ValueError(
            "parallel_config.worker_cls must be a qualname string "
            f"(got type {type(worker_cls_raw).__name__}){_ctx()}",
        )
    if worker_cls_raw.strip() == "auto":
        expected_worker = (
            async_expected_worker
            or expected_worker_qualname_override
            or expected_worker_qualname(config.role)
        )
        raise ValueError(
            "parallel_config.worker_cls is still 'auto'; pass --worker-cls "
            f"{expected_worker}{_ctx()}",
        )

    expected_qualname = (
        async_expected_worker
        or expected_worker_qualname_override
        or expected_worker_qualname(config.role)
    )
    worker_fqcn = normalize_qualname(worker_cls_raw.strip())
    expected_fqcn = normalize_qualname(expected_qualname)
    if worker_fqcn != expected_fqcn:
        prefix = (
            "AFDAsyncConnector requires Ascend NPU worker class: "
            if async_expected_worker is not None
            else "invalid worker class for AFD runtime stack: "
        )
        raise ValueError(
            prefix +
            f"got={worker_fqcn!r} expected={expected_qualname!r}; "
            f"pass --worker-cls {expected_qualname}{_ctx()}",
        )

    return config


__all__ = [
    "ATTENTION_MODEL_RUNNER_FQCN",
    "ATTENTION_WORKER_FQCN",
    "FFN_MODEL_RUNNER_FQCN",
    "FFN_WORKER_FQCN",
    "NPU_ATTENTION_MODEL_RUNNER_FQCN",
    "NPU_ATTENTION_WORKER_FQCN",
    "NPU_FFN_MODEL_RUNNER_FQCN",
    "NPU_FFN_WORKER_FQCN",
    "UBATCH_WRAPPER_FQCN",
    "assert_compatible_afd_stack",
    "expected_npu_worker_qualname",
    "expected_worker_qualname",
    "normalize_qualname",
    "resolve_class_from_qualname",
]
