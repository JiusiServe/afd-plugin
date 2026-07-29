# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side worker for AFD GPU execution."""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker

from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.attention_model_runner import (
    AFDAttentionModelRunner,
    fail_if_unsupported_ubatching,
)
from afd_plugin.validation import assert_compatible_afd_stack


class AFDAttentionWorker(Worker):
    """Attention worker that injects :class:`AFDAttentionModelRunner`."""

    afd_expected_role = "attention"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker,
        )

    def init_device(self):
        """Initialize the native GPU worker and swap in the AFD runner."""

        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDAttentionWorker.init_device",
            expected_role="attention",
        )
        if self.use_v2_model_runner:
            raise RuntimeError(
                "AFD Attention runtime currently supports only the vLLM v1 "
                "GPUModelRunner; set VLLM_USE_V2_MODEL_RUNNER=0",
            )

        fail_if_unsupported_ubatching(self.vllm_config)

        super().init_device()
        self.vllm_config.model_config = get_afd_model_config(
            self.vllm_config.model_config,
        )
        self.model_runner = AFDAttentionModelRunner(self.vllm_config, self.device)

        torch.accelerator.empty_cache()


__all__ = ["AFDAttentionWorker"]
