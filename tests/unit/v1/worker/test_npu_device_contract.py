# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from pathlib import Path


def test_npu_connectors_use_model_device_index_for_dp_workers():
    for source_path in (
        Path("afd_plugin/v1/worker/npu/attention_model_runner.py"),
        Path("afd_plugin/v1/worker/npu/ffn_model_runner.py"),
    ):
        source = source_path.read_text()

        assert "rank, _ = _resolve_world_ranks()" in source
        assert "local_rank = int(device.index)" in source
        assert "rank, local_rank = _resolve_world_ranks()" not in source


def test_cam_memory_headroom_check_does_not_adjust_utilization():
    for source_path in (
        Path("afd_plugin/v1/worker/npu/attention_worker.py"),
        Path("afd_plugin/v1/worker/npu/ffn_worker.py"),
    ):
        source = source_path.read_text()

        assert "warn_if_cam_memory_headroom_is_low(" in source
        assert "self.init_snapshot.total_memory" in source
        assert "self.requested_memory = (" not in source


def test_cam_workers_do_not_modify_global_hccl_buffer_configuration():
    for source_path in (
        Path("afd_plugin/v1/worker/npu/attention_worker.py"),
        Path("afd_plugin/v1/worker/npu/ffn_worker.py"),
    ):
        source = source_path.read_text()

        assert "HCCL_BUFFSIZE" not in source
