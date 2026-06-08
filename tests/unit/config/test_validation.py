from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.validation import (
    ATTENTION_WORKER_FQCN,
    FFN_WORKER_FQCN,
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_WORKER_FQCN,
    assert_compatible_afd_stack,
)


def _vllm_like_config(*, afd, worker_cls):
    return SimpleNamespace(
        additional_config={"afd": afd},
        parallel_config=SimpleNamespace(worker_cls=worker_cls),
    )


def test_attention_stack_validation_accepts_matching_worker():
    vllm_config = _vllm_like_config(
        afd={"enabled": True, "role": "attention"},
        worker_cls=ATTENTION_WORKER_FQCN,
    )

    config = assert_compatible_afd_stack(
        vllm_config,
        caller="test",
        expected_role="attention",
    )

    assert config.role == "attention"


def test_ffn_stack_validation_accepts_matching_worker():
    vllm_config = _vllm_like_config(
        afd={"enabled": True, "role": "ffn"},
        worker_cls=FFN_WORKER_FQCN,
    )

    config = assert_compatible_afd_stack(
        vllm_config,
        caller="test",
        expected_role="ffn",
    )

    assert config.role == "ffn"


def test_stack_validation_rejects_disabled_config():
    vllm_config = _vllm_like_config(
        afd={"enabled": False, "role": "attention"},
        worker_cls=ATTENTION_WORKER_FQCN,
    )

    with pytest.raises(ValueError, match="AFD is not enabled"):
        assert_compatible_afd_stack(vllm_config, caller="test")


def test_stack_validation_rejects_wrong_worker():
    vllm_config = _vllm_like_config(
        afd={"enabled": True, "role": "ffn"},
        worker_cls=ATTENTION_WORKER_FQCN,
    )

    with pytest.raises(ValueError, match="invalid worker class"):
        assert_compatible_afd_stack(vllm_config, caller="test")


def test_stack_validation_rejects_auto_worker():
    vllm_config = _vllm_like_config(
        afd={"enabled": True, "role": "attention"},
        worker_cls="auto",
    )

    with pytest.raises(ValueError, match="worker_cls is still 'auto'"):
        assert_compatible_afd_stack(vllm_config, caller="test")


def test_stack_validation_accepts_npu_worker_override():
    vllm_config = _vllm_like_config(
        afd={
            "enabled": True,
            "role": "attention",
            "connector": "camp2pconnector",
        },
        worker_cls=NPU_ATTENTION_WORKER_FQCN,
    )

    config = assert_compatible_afd_stack(
        vllm_config,
        caller="test",
        expected_role="attention",
        expected_worker_qualname_override=NPU_ATTENTION_WORKER_FQCN,
    )

    assert config.connector == "camp2pconnector"


def test_async_connector_requires_npu_attention_worker():
    vllm_config = _vllm_like_config(
        afd={
            "enabled": True,
            "role": "attention",
            "connector": "afdasyncconnector",
        },
        worker_cls=ATTENTION_WORKER_FQCN,
    )

    with pytest.raises(ValueError, match="requires Ascend NPU worker"):
        assert_compatible_afd_stack(
            vllm_config,
            caller="test",
            expected_role="attention",
        )

    vllm_config.parallel_config.worker_cls = NPU_ATTENTION_WORKER_FQCN
    config = assert_compatible_afd_stack(
        vllm_config,
        caller="test",
        expected_role="attention",
    )
    assert config.connector == "afdasyncconnector"


def test_async_connector_requires_npu_ffn_worker():
    vllm_config = _vllm_like_config(
        afd={
            "enabled": True,
            "role": "ffn",
            "connector": "afdasyncconnector",
        },
        worker_cls=NPU_FFN_WORKER_FQCN,
    )

    config = assert_compatible_afd_stack(
        vllm_config,
        caller="test",
        expected_role="ffn",
    )

    assert config.connector == "afdasyncconnector"
