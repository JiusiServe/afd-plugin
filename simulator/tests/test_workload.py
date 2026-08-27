from __future__ import annotations

import unittest

from simulator.config import SimulationConfig
from simulator.workload import generate_workload, read_csv_requests


class WorkloadTests(unittest.TestCase):
    def test_csv_length_list_preserves_online_distribution(self) -> None:
        rows = read_csv_requests(csv_text="\ufeffinput_length\n512\n8192\n8192\n")

        self.assertEqual([row.input_length for row in rows], [512, 8192, 8192])

    def test_csv_timestamp_and_cached_prefix_are_replayed(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "continuous",
                "csv_text": (
                    "request_id,arrival_time_ms,input_length,cached_prefix_tokens\n"
                    "a,100,1024,512\n"
                    "b,125,2048,0\n"
                ),
                "arrival": {"kind": "trace", "duration_s": 1, "warmup_s": 0},
                "prefix_cache": {"enabled": True},
            }
        )

        workload = generate_workload(config)

        self.assertEqual([item.arrival_ms for item in workload], [0.0, 25.0])
        self.assertEqual(workload[0].cached_prefix_tokens, 512)
        self.assertEqual(workload[0].query_tokens, 512)

    def test_csv_trace_is_clipped_to_the_configured_window(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "continuous",
                "csv_text": (
                    "arrival_time_ms,input_length\n100,128\n125,128\n160,128\n"
                ),
                "arrival": {
                    "kind": "trace",
                    "duration_s": 0.04,
                    "warmup_s": 0.01,
                },
            }
        )

        workload = generate_workload(config)

        self.assertEqual([item.arrival_ms for item in workload], [0.0, 25.0])

    def test_scaled_trace_preserves_bursts_and_repeats_at_target_qps(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "continuous",
                "csv_text": (
                    "arrival_time_ms,input_length\n"
                    "100,128\n"
                    "200,256\n"
                    "500,512\n"
                ),
                "arrival": {
                    "kind": "scaled_trace",
                    "qps": 2,
                    "duration_s": 3,
                    "warmup_s": 0,
                },
            }
        )

        workload = generate_workload(config)

        self.assertEqual(
            [request.arrival_ms for request in workload],
            [0.0, 250.0, 1_000.0, 1_500.0, 1_750.0, 2_500.0],
        )
        self.assertEqual(
            [request.input_tokens for request in workload],
            [128, 256, 512, 128, 256, 512],
        )

    def test_scaled_trace_keeps_zero_gap_arrivals_simultaneous(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "continuous",
                "csv_text": (
                    "arrival_time_ms,input_length\n"
                    "0,128\n"
                    "0,256\n"
                    "1000,512\n"
                ),
                "arrival": {
                    "kind": "scaled_trace",
                    "qps": 2,
                    "duration_s": 1,
                    "warmup_s": 0,
                },
            }
        )

        workload = generate_workload(config)

        self.assertEqual([request.arrival_ms for request in workload], [0.0, 0.0])

    def test_synthetic_arrival_uses_timestamped_csv_as_length_pool(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "continuous",
                "csv_text": (
                    "arrival_time_ms,input_length\n"
                    "100,128\n"
                    "125,256\n"
                    "160,512\n"
                ),
                "arrival": {
                    "kind": "constant",
                    "qps": 2,
                    "duration_s": 1,
                    "warmup_s": 0,
                },
            }
        )

        workload = generate_workload(config)

        self.assertEqual([request.arrival_ms for request in workload], [0.0, 500.0])
        self.assertEqual(
            [request.input_tokens for request in workload],
            [128, 256],
        )

    def test_prefix_cache_sampling_is_deterministic_and_block_aligned(self) -> None:
        raw = {
            "mode": "fixed",
            "fixed_lengths": [1024, 1024, 1024],
            "prefix_cache": {
                "enabled": True,
                "request_hit_rate": 1.0,
                "matched_prefix_ratio": 0.73,
                "block_size": 32,
                "seed": 7,
            },
        }
        first = generate_workload(SimulationConfig.from_mapping(raw))
        second = generate_workload(SimulationConfig.from_mapping(raw))

        self.assertEqual(first, second)
        self.assertEqual({item.cached_prefix_tokens for item in first}, {736})

    def test_full_cache_ratio_keeps_one_aligned_query_block(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [1024],
                "prefix_cache": {
                    "enabled": True,
                    "request_hit_rate": 1,
                    "matched_prefix_ratio": 1,
                    "block_size": 32,
                },
            }
        )

        request = generate_workload(config)[0]

        self.assertEqual(request.cached_prefix_tokens, 992)
        self.assertEqual(request.query_tokens, 32)
