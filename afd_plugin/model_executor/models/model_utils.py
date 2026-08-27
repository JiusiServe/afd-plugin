# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Utilities for AFD model configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from afd_plugin import _MODEL_REGISTRATIONS, _QWEN3_5_MODEL_REGISTRATIONS

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def get_afd_model_config(
    model_config: ModelConfig,
    *,
    device_type: Literal["cuda", "npu"],
) -> ModelConfig:
    """Return a model config that resolves to an AFD model implementation."""

    # ``get_afd_model_config`` also runs in every spawned vLLM worker.  Those
    # workers import the explicit AFD worker class directly and do not invoke
    # the ``vllm.general_plugins`` entry point, so register the lazy AFD model
    # names before replacing a native architecture with its ``AFD...`` name.
    # Registration is idempotent in ``register_afd``.
    from afd_plugin import register_afd

    register_afd()

    for model_arch in model_config.hf_config.architectures:
        if model_arch in _MODEL_REGISTRATIONS:
            if model_arch in _QWEN3_5_MODEL_REGISTRATIONS and device_type != "cuda":
                raise ValueError(
                    "AFD Qwen3.5/3.6 supports CUDA execution only; "
                    f"got device_type={device_type!r}",
                )
            # deepcopy preserves aliasing within the copied object graph, so
            # the pure-text identity hf_text_config is hf_config is retained
            # automatically. vLLM Ascend uses that identity to distinguish
            # text models from multimodal models.
            afd_model_config = deepcopy(model_config)
            afd_model_config.hf_config.architectures = [f"AFD{model_arch}"]
            return afd_model_config
    return model_config
