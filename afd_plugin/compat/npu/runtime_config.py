# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Config compatibility adjustments for AFD Ascend workers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

FLASHINFER_ALL2ALLV_BACKEND = "flashinfer_all2allv"


def npu_afd_num_ubatches(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    if parallel_config.use_ubatching:
        return int(parallel_config.num_ubatches)
    return 1


def fix_all2all_backend_for_afd(vllm_config: VllmConfig) -> None:
    """Apply vLLM-Ascend's default-worker all2all fix to AFD workers.

    vLLM-Ascend normally rewrites ``all2all_backend`` to
    ``"flashinfer_all2allv"`` when sequence parallelism is disabled, but that
    compatibility rewrite is gated on the default ``worker_cls == "auto"``.
    AFD Attention/FFN workers use custom worker classes, so they miss the
    rewrite and keep the default ``"allgather_reducescatter"`` backend.

    Leaving that backend in place can make the Ascend MoE path think sequence
    parallel MoE is enabled and split tokens through ``sequence_parallel_chunk``,
    which is not the layout AFD's NPU connector path sends. Mirror the upstream
    rewrite here before AFD creates the NPU model runner.
    """
    parallel_config = vllm_config.parallel_config
    if (
        not vllm_config.compilation_config.pass_config.enable_sp
        and parallel_config.all2all_backend != FLASHINFER_ALL2ALLV_BACKEND
    ):
        parallel_config.all2all_backend = FLASHINFER_ALL2ALLV_BACKEND


__all__ = ["fix_all2all_backend_for_afd", "npu_afd_num_ubatches"]
