from __future__ import annotations

import pytest

from afd_plugin.validation import (
    ATTENTION_MODEL_RUNNER_FQCN,
    ATTENTION_WORKER_FQCN,
    FFN_MODEL_RUNNER_FQCN,
    FFN_WORKER_FQCN,
    NPU_ATTENTION_MODEL_RUNNER_FQCN,
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_MODEL_RUNNER_FQCN,
    NPU_FFN_WORKER_FQCN,
    UBATCH_WRAPPER_FQCN,
    resolve_class_from_qualname,
)

GPU_RUNTIME_CLASS_PATHS = [
    ATTENTION_WORKER_FQCN,
    ATTENTION_MODEL_RUNNER_FQCN,
    FFN_WORKER_FQCN,
    FFN_MODEL_RUNNER_FQCN,
    UBATCH_WRAPPER_FQCN,
    "afd_plugin.v1.worker:AFDAttentionWorker",
]

NPU_RUNTIME_CLASS_PATHS = [
    NPU_ATTENTION_WORKER_FQCN,
    NPU_ATTENTION_MODEL_RUNNER_FQCN,
    NPU_FFN_WORKER_FQCN,
    NPU_FFN_MODEL_RUNNER_FQCN,
]


@pytest.mark.parametrize(
    "qualname",
    GPU_RUNTIME_CLASS_PATHS + NPU_RUNTIME_CLASS_PATHS,
)
def test_runtime_class_paths_are_plugin_paths(qualname):
    assert qualname.startswith("afd_plugin.v1.worker")


@pytest.mark.vllm_runtime
@pytest.mark.parametrize("qualname", GPU_RUNTIME_CLASS_PATHS)
def test_gpu_runtime_class_paths_resolve_when_vllm_is_available(qualname):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    cls = resolve_class_from_qualname(qualname)

    assert isinstance(cls, type)
    assert cls.__module__.startswith("afd_plugin.v1.worker")


@pytest.mark.vllm_runtime
@pytest.mark.parametrize("qualname", NPU_RUNTIME_CLASS_PATHS)
def test_npu_runtime_class_paths_resolve_when_vllm_ascend_is_available(qualname):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_ascend")

    cls = resolve_class_from_qualname(qualname)

    assert isinstance(cls, type)
    assert cls.__module__.startswith("afd_plugin.v1.worker.npu")
