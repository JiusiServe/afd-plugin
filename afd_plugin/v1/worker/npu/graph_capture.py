# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD control-plane coordination around upstream NPU graph capture."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        AFDNPUAttentionModelRunner,
    )


class AFDNPUGraphCapture:
    """Capture separate single-stage/uBatch keys outside the model graph."""

    @staticmethod
    def warmup_and_capture(
        runner: AFDNPUAttentionModelRunner,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None = None,
        allow_microbatching: bool = False,
        num_warmups: int | None = None,
    ) -> None:
        if num_warmups is None:
            num_warmups = runner.compilation_config.cudagraph_num_of_warmups
        if allow_microbatching:
            AFDNPUGraphCapture._capture_once(
                runner,
                desc,
                cudagraph_runtime_mode,
                profile_seq_lens,
                allow_microbatching=False,
                num_warmups=num_warmups,
            )
        AFDNPUGraphCapture._capture_once(
            runner,
            desc,
            cudagraph_runtime_mode,
            profile_seq_lens,
            allow_microbatching=allow_microbatching,
            num_warmups=num_warmups,
        )

    @staticmethod
    def _capture_once(
        runner: AFDNPUAttentionModelRunner,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None,
        *,
        allow_microbatching: bool,
        num_warmups: int,
    ) -> None:
        force_attention = cudagraph_runtime_mode is CUDAGraphMode.FULL
        previous_is_warmup = runner._is_warmup
        try:
            runner._is_warmup = True
            for _ in range(num_warmups):
                runner._dummy_run(
                    desc.num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    force_attention=force_attention,
                    uniform_decode=desc.uniform,
                    allow_microbatching=allow_microbatching,
                    skip_eplb=True,
                    remove_lora=False,
                    num_active_loras=desc.num_active_loras,
                )
        finally:
            runner._is_warmup = previous_is_warmup

        previous_metadata = runner._afd_pending_metadata
        previous_suppress_send = runner._afd_suppress_metadata_send
        previous_is_graph_capturing = runner._afd_is_graph_capturing
        try:
            runner._afd_is_graph_capturing = True
            if allow_microbatching:
                runner._afd_pending_metadata = None
                runner._afd_suppress_metadata_send = False
            else:
                runner._afd_pending_metadata = runner._build_afd_metadata(
                    None,
                    int(desc.num_tokens),
                )
                if runner.connector.control_plane is not None:
                    runner._send_dp_metadata(
                        runner._build_capture_dp_metadata(int(desc.num_tokens)),
                        None,
                    )
                runner._afd_suppress_metadata_send = True

            runner._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                uniform_decode=desc.uniform,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
                is_graph_capturing=True,
                profile_seq_lens=profile_seq_lens,
            )
        finally:
            runner._afd_is_graph_capturing = previous_is_graph_capturing
            runner._afd_suppress_metadata_send = previous_suppress_send
            runner._afd_pending_metadata = previous_metadata


__all__ = ["AFDNPUGraphCapture"]
