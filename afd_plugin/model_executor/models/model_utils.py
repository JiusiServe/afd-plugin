# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Utilities for AFD model configuration."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

from afd_plugin import _DEEPSEEK_MODEL_REGISTRATIONS

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def get_afd_model_config(model_config: ModelConfig) -> ModelConfig:
    """Return a model config that resolves to an AFD model implementation."""

    for model_arch in model_config.hf_config.architectures:
        if model_arch in _DEEPSEEK_MODEL_REGISTRATIONS:
            afd_model_config = copy(model_config)
            afd_model_config.hf_config = copy(model_config.hf_config)
            afd_model_config.hf_config.architectures = [f"AFD{model_arch}"]
            return afd_model_config
    return model_config
