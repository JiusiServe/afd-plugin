from __future__ import annotations

import unittest
from collections import deque

from simulator.config import SimulationConfig
from simulator.engine import RequestDispatcher, compare_architectures, sweep_qps
from simulator.tests.helpers import make_profile
from simulator.workload import RequestSpec


class EngineTests(unittest.TestCase):
    def test_scheduler_policy_validation(self) -> None:
        config = SimulationConfig.from_mapping(
            {"scheduler": {"policy": "vllm_queue_aware"}}
        )
        self.assertEqual(config.scheduler.policy, "vllm_queue_aware")

        with self.assertRaisesRegex(ValueError, "scheduler.policy"):
            SimulationConfig.from_mapping({"scheduler": {"policy": "least_tokens"}})

    def test_vllm_queue_aware_routes_to_the_less_loaded_dp(self) -> None:
        specs = (
            RequestSpec("long", 0.0, 8_192, 0, 0.0),
            RequestSpec("short", 0.0, 1, 0, 0.0),
            RequestSpec("later", 5.0, 1, 0, 0.0),
        )
        base = {
            "mode": "fixed",
            "scheduler": {"max_num_batched_tokens": 8_192},
            "cam": {
                "dispatch_send": {"fixed_ms": 0, "per_token_ms": 0.001},
                "dispatch_recv": {"fixed_ms": 0, "per_token_ms": 0},
                "combine_send": {"fixed_ms": 0, "per_token_ms": 0},
                "combine_recv": {"fixed_ms": 0, "per_token_ms": 0},
            },
        }
        round_robin = compare_architectures(
            SimulationConfig.from_mapping(base),
            make_profile(layer_count=1),
            specs,
        )
        base["scheduler"]["policy"] = "vllm_queue_aware"
        queue_aware = compare_architectures(
            SimulationConfig.from_mapping(base),
            make_profile(layer_count=1),
            specs,
        )

        self.assertEqual(round_robin["afd"]["requests"][2]["assigned_dp"], 0)
        self.assertEqual(queue_aware["afd"]["requests"][2]["assigned_dp"], 1)
        self.assertLess(
            queue_aware["afd"]["requests"][2]["completion_ms"],
            round_robin["afd"]["requests"][2]["completion_ms"],
        )

    def test_queue_aware_active_sets_are_pruned_in_place(self) -> None:
        specs = tuple(
            RequestSpec(f"request-{index}", 0.0, 1, 0, 0.0) for index in range(10_000)
        )
        dispatcher = RequestDispatcher(specs, 4, "vllm_queue_aware")
        active_ids = [id(active) for active in dispatcher.active]

        dispatcher.dispatch_until(0.0)

        self.assertTrue(all(isinstance(active, deque) for active in dispatcher.active))
        self.assertEqual([id(active) for active in dispatcher.active], active_ids)
        self.assertEqual([len(active) for active in dispatcher.active], [2_500] * 4)

        round_robin = RequestDispatcher(specs, 4, "round_robin")
        round_robin.dispatch_until(0.0)
        self.assertTrue(all(not active for active in round_robin.active))

    def test_fixed_workload_runs_both_architectures_and_exposes_barrier(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [128, 512, 2048, 8192],
                "scheduler": {"max_num_batched_tokens": 16384},
            }
        )

        result = compare_architectures(config, make_profile())

        self.assertEqual(result["afd"]["summary"]["request_count"], 4)
        self.assertEqual(result["merged"]["summary"]["request_count"], 4)
        self.assertGreater(result["merged"]["summary"]["barrier_wait_ms"], 0)
        self.assertTrue(
            all(item["completion_ms"] is not None for item in result["afd"]["requests"])
        )

    def test_expert_timeline_events_expose_shape_topk_and_ep(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [128, 128, 128, 128],
                "scheduler": {"max_num_batched_tokens": 128},
            }
        )

        result = compare_architectures(config, make_profile(layer_count=1))
        afd_ffn = next(
            event
            for event in result["afd"]["timeline"]
            if event["phase"] == "ffn_compute"
        )
        routed = next(
            event
            for event in result["merged"]["timeline"]
            if event["phase"] == "routed_experts"
        )
        shared = next(
            event
            for event in result["merged"]["timeline"]
            if event["phase"] == "shared_expert"
        )

        self.assertEqual(
            afd_ffn["sampled_input_shapes"],
            [
                {
                    "query_tokens": 128,
                    "weight": 1.0,
                    "shape": [96, 4_096],
                    "expert_shapes": [{"shape": [96, 4_096], "count": 1}],
                }
            ],
        )
        self.assertEqual(afd_ffn["top_k"], 6)
        self.assertEqual(afd_ffn["ep_size"], 8)
        self.assertEqual(
            routed["sampled_input_shapes"],
            [
                {
                    "query_tokens": 128,
                    "weight": 1.0,
                    "shape": [48, 4_096],
                    "expert_shapes": [{"shape": [48, 4_096], "count": 1}],
                }
            ],
        )
        self.assertEqual(routed["top_k"], 6)
        self.assertEqual(routed["ep_size"], 16)
        self.assertEqual(routed["tokens"], 512)
        self.assertEqual(routed["profile_query_tokens"], 128)
        self.assertIsNone(shared["top_k"])

    def test_merged_ep_uses_global_tokens_with_dp4_equivalent_query(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [512, 8_192, 2_048, 6_144],
                "scheduler": {"max_num_batched_tokens": 8_192},
            }
        )
        profiles = make_profile(layer_count=1)

        result = compare_architectures(config, profiles)
        routed = next(
            event
            for event in result["merged"]["timeline"]
            if event["phase"] == "routed_experts"
        )

        self.assertEqual(routed["tokens"], 16_896)
        self.assertEqual(routed["profile_query_tokens"], 4_224)
        global_ep_events = [
            event
            for event in result["merged"]["timeline"]
            if event["phase"] in {"merged_dispatch", "routed_experts", "merged_combine"}
        ]
        self.assertTrue(all(event["tokens"] == 16_896 for event in global_ep_events))
        self.assertTrue(
            all(event["profile_query_tokens"] == 4_224 for event in global_ep_events)
        )
        local_tail_events = [
            event
            for event in result["merged"]["timeline"]
            if event["phase"]
            in {
                "merged_combine_local",
                "shared_expert",
                "merged_sp_post",
            }
        ]
        self.assertTrue(all(event["tokens"] == 8_192 for event in local_tail_events))
        self.assertTrue(
            all(event["profile_query_tokens"] == 8_192 for event in local_tail_events)
        )
        self.assertAlmostEqual(
            routed["end_ms"] - routed["start_ms"],
            profiles.duration_ms("merged", 0, "routed_experts", 0, 4_224),
        )

    def test_merged_ep_preserves_fractional_dp4_equivalent_query(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [128, 129, 130, 132],
                "scheduler": {"max_num_batched_tokens": 132},
            }
        )

        result = compare_architectures(config, make_profile(layer_count=1))
        routed = next(
            event
            for event in result["merged"]["timeline"]
            if event["phase"] == "routed_experts"
        )

        self.assertEqual(routed["tokens"], 519)
        self.assertEqual(routed["profile_query_tokens"], 129.75)

    def test_chunked_prefill_and_token_ubatch_preserve_request_completion(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [2048],
                "scheduler": {
                    "max_num_batched_tokens": 1024,
                    "chunked_prefill": True,
                    "chunk_size": 1024,
                },
                "afd": {"ubatch_split": "token"},
            }
        )

        result = compare_architectures(config, make_profile())

        request = result["afd"]["requests"][0]
        self.assertEqual(request["query_tokens"], 2048)
        self.assertGreater(request["completion_ms"], request["first_scheduled_ms"])
        self.assertTrue(any(event["stage"] == 1 for event in result["afd"]["timeline"]))

    def test_prefix_cache_reduces_computed_tokens_but_not_logical_tokens(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "csv_text": "input_length,cached_prefix_tokens\n1024,512\n",
                "scheduler": {"max_num_batched_tokens": 2048},
                "prefix_cache": {"enabled": True},
            }
        )

        result = compare_architectures(config, make_profile())
        summary = result["afd"]["summary"]

        self.assertEqual(summary["logical_input_tokens"], 1024)
        self.assertEqual(summary["computed_query_tokens"], 512)
        self.assertEqual(summary["cache_token_ratio"], 0.5)

    def test_continuous_poisson_and_sweep_are_reproducible(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "continuous",
                "length_mix": [{"tokens": 128, "weight": 1}],
                "arrival": {
                    "kind": "poisson",
                    "qps": 1,
                    "duration_s": 2,
                    "warmup_s": 0,
                    "seed": 11,
                },
                "scheduler": {"max_num_batched_tokens": 1024},
                "sweep": {
                    "min_qps": 0.5,
                    "max_qps": 2,
                    "coarse_points": 3,
                    "refinement_steps": 1,
                },
            }
        )

        first = sweep_qps(config, make_profile(layer_count=1))
        second = sweep_qps(config, make_profile(layer_count=1))

        self.assertEqual(first, second)
        self.assertTrue(first["series"]["afd"])

    def test_afd_ffn_is_fcfs_by_dispatch_arrival(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [100, 1],
                "scheduler": {"max_num_batched_tokens": 100},
                "cam": {
                    "dispatch_send": {"fixed_ms": 0, "per_token_ms": 1},
                    "dispatch_recv": {"fixed_ms": 0, "per_token_ms": 0},
                    "combine_send": {"fixed_ms": 0, "per_token_ms": 0},
                    "combine_recv": {"fixed_ms": 0, "per_token_ms": 0},
                },
            }
        )

        result = compare_architectures(config, make_profile(layer_count=1))
        receives = sorted(
            (
                event
                for event in result["afd"]["timeline"]
                if event["phase"] == "dispatch_recv"
            ),
            key=lambda event: event["start_ms"],
        )

        self.assertEqual([event["tokens"] for event in receives], [1, 100])

    def test_sweep_rejects_fixed_workload(self) -> None:
        config = SimulationConfig.from_mapping(
            {"mode": "fixed", "fixed_lengths": [128]}
        )

        with self.assertRaisesRegex(ValueError, "continuous"):
            sweep_qps(config, make_profile(layer_count=1))

    def test_timeline_truncated_only_when_an_event_is_dropped(self) -> None:
        exact = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [128],
                "output": {"timeline_max_events": 7},
            }
        )
        truncated = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [128],
                "output": {"timeline_max_events": 6},
            }
        )

        exact_result = compare_architectures(exact, make_profile(layer_count=1))
        truncated_result = compare_architectures(truncated, make_profile(layer_count=1))

        self.assertFalse(exact_result["afd"]["summary"]["timeline_truncated"])
        self.assertTrue(truncated_result["afd"]["summary"]["timeline_truncated"])
        self.assertEqual(len(truncated_result["afd"]["timeline"]), 6)

    def test_continuous_window_requires_a_measured_request(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "continuous",
                "csv_text": "arrival_time_ms,input_length\n0,128\n",
                "arrival": {
                    "kind": "trace",
                    "warmup_s": 1,
                    "duration_s": 1,
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "measurement window"):
            compare_architectures(config, make_profile(layer_count=1))

    def test_trace_metrics_use_only_the_measurement_window(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "continuous",
                "csv_text": (
                    "arrival_time_ms,input_length\n100,128\n125,128\n160,128\n"
                ),
                "arrival": {
                    "kind": "trace",
                    "warmup_s": 0.01,
                    "duration_s": 0.04,
                },
            }
        )

        result = compare_architectures(config, make_profile(layer_count=1))

        self.assertEqual(result["afd"]["summary"]["request_count"], 1)
        self.assertEqual(result["afd"]["summary"]["offered_qps"], 25)
