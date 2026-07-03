# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-side worker for AFD GPU execution."""

from __future__ import annotations

import logging
import threading
from typing import Any

import torch
from vllm.v1.worker.gpu_worker import Worker

from afd_plugin.v1.worker.attention_model_runner import fail_if_unsupported_ubatching
from afd_plugin.v1.worker.ffn_model_runner import GPUFFNModelRunner
from afd_plugin.validation import assert_compatible_afd_stack

logger = logging.getLogger(__name__)


class AFDFFNWorker(Worker):
    """FFN worker that owns the AFD daemon loop.

    The FFN side enters through native ``vllm serve --worker-cls``. The native
    scheduler may still be present, but FFN work is driven by connector
    metadata.
    """

    afd_expected_role = "ffn"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ffn_thread: threading.Thread | None = None
        self._ffn_shutdown_event: threading.Event | None = None
        self._ffn_loop_error: BaseException | None = None

    def init_device(self) -> None:
        """Initialize the native GPU worker and swap in the FFN runner."""

        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDFFNWorker.init_device",
            expected_role="ffn",
        )
        if self.use_v2_model_runner:
            raise RuntimeError(
                "AFD FFN runtime currently supports only the vLLM v1 "
                "GPUModelRunner interface; unset VLLM_USE_V2_MODEL_RUNNER",
            )

        fail_if_unsupported_ubatching(self.vllm_config)

        super().init_device()
        self.model_runner = GPUFFNModelRunner(self.vllm_config, self.device)

        torch.accelerator.empty_cache()

    def get_kv_cache_spec(self) -> dict[str, Any]:
        """FFN workers do not allocate KV cache."""

        return {}

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        """Skip KV cache allocation and start the FFN connector loop."""

        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.model_runner.initialize_kv_cache(kv_cache_config)
        self.model_runner.initialize_afd_connector()
        self.start_ffn_server_loop()

    def compile_or_warm_up_model(self) -> float:
        """FFN workers perform no warmup/capture; model execution is driven
        entirely by connector metadata.
        """

        return 0.0

    def execute_model(self, scheduler_output: Any) -> None:
        """Fail fast if the default scheduler tries to execute FFN work."""

        raise RuntimeError(
            "AFD FFN workers are connector-driven; scheduler-driven "
            "execute_model() is not supported.",
        )

    def start_ffn_server_loop(self) -> None:
        if self._ffn_thread is not None and self._ffn_thread.is_alive():
            self.raise_ffn_loop_error_if_any()
            return

        self.raise_ffn_loop_error_if_any()
        connector = self.model_runner.connector
        if not connector.is_initialized:
            self.model_runner.initialize_afd_connector()

        self._ffn_shutdown_event = threading.Event()
        self._ffn_loop_error = None

        def ffn_worker_loop() -> None:
            try:
                self._run_ffn_server_loop()
            except Exception as exc:
                self._ffn_loop_error = exc
                logger.exception("AFD FFN worker loop failed")

        self._ffn_thread = threading.Thread(
            target=ffn_worker_loop,
            name="afd-ffn-worker-loop",
            daemon=True,
        )
        self._ffn_thread.start()

    def _run_ffn_server_loop(self) -> None:
        event = self._ffn_shutdown_event
        if event is None:
            return

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        while not event.is_set():
            try:
                (
                    dp_metadata_list,
                    is_attn_graph_capturing,
                    is_warmup,
                ) = self.model_runner.connector.recv_dp_metadata_list(timeout_ms=100)
            except TimeoutError:
                continue

            if self.model_runner.use_cuda_graph and (
                is_warmup or is_attn_graph_capturing
            ):
                self.model_runner.capture_model(
                    dp_metadata_list=dp_metadata_list,
                    is_warmup=is_warmup,
                    is_attn_graph_capturing=is_attn_graph_capturing,
                )
            else:
                self.model_runner.execute_model(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=is_attn_graph_capturing,
                    is_warmup=is_warmup,
                )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

    def raise_ffn_loop_error_if_any(self) -> None:
        error = self._ffn_loop_error
        if error is not None:
            self._ffn_loop_error = None
            raise RuntimeError("AFD FFN worker loop failed") from error

    def stop_ffn_server_loop(self) -> None:
        event = self._ffn_shutdown_event
        if event is not None:
            event.set()
        try:
            self.model_runner.shutdown()
        finally:
            thread = self._ffn_thread
            if thread is not None:
                thread.join(timeout=5)
            self._ffn_thread = None
            self._ffn_shutdown_event = None
        self.raise_ffn_loop_error_if_any()

    def shutdown(self) -> None:
        self.stop_ffn_server_loop()
        super().shutdown()


__all__ = ["AFDFFNWorker"]
