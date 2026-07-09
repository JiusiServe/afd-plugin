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
    """Mirror vllm-ascend's platform.py all2all_backend override.

    vllm-ascend sets ``all2all_backend = "flashinfer_all2allv"`` when
    ``enable_sp`` is False, but only when ``worker_cls == "auto"``.
    AFD workers use a custom ``worker_cls``, so this override never fires
    and ``all2all_backend`` keeps its default ``"allgather_reducescatter"``.
    That value triggers ``use_sequence_parallel_moe = True`` (because
    ``enable_expert_parallel=True``, ``tp_size > 1``, ``dp_size > 1``),
    which incorrectly splits MoE tokens via ``sequence_parallel_chunk``,
    producing wrong output.

    This function applies the same fix for AFD workers.
    """
    parallel_config = vllm_config.parallel_config
    if (
        not vllm_config.compilation_config.pass_config.enable_sp
        and parallel_config.all2all_backend != FLASHINFER_ALL2ALLV_BACKEND
    ):
        parallel_config.all2all_backend = FLASHINFER_ALL2ALLV_BACKEND


__all__ = ["fix_all2all_backend_for_afd", "npu_afd_num_ubatches"]
