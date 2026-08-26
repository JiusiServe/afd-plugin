from __future__ import annotations

import unittest

from simulator.config import SimulationConfig
from simulator.engine import (
    BatchSegment,
    RequestDispatcher,
    SchedulerBatch,
    _afd_wave_completion_scores,
    _merged_wave_completion_scores,
    _pack_batch,
    compare_architectures,
    resolve_scheduler_policy,
    sweep_qps,
)
from simulator.tests.helpers import make_profile
from simulator.workload import RequestSpec, RuntimeRequest


def _symmetric_batch_configs(
    max_num_batched_tokens: int,
    chunk_size: int | None = None,
) -> dict[str, dict[str, int]]:
    batch_config = {
        "max_num_batched_tokens": max_num_batched_tokens,
        "chunk_size": (
            max_num_batched_tokens if chunk_size is None else chunk_size
        ),
    }
    return {"afd": batch_config, "merged": dict(batch_config)}


class EngineTests(unittest.TestCase):
    def test_scheduler_policy_validation(self) -> None:
        default_config = SimulationConfig.from_mapping({})
        config = SimulationConfig.from_mapping(
            {
                "scheduler": {
                    "afd_policy": "vllm_queue_aware",
                    "merged_policy": "vllm_queue_aware",
                }
            }
        )
        token_config = SimulationConfig.from_mapping(
            {
                "scheduler": {
                    "afd_policy": "prefill_token_greedy",
                    "merged_policy": "prefill_token_greedy",
                }
            }
        )
        square_config = SimulationConfig.from_mapping(
            {"scheduler": {"afd_policy": "prefill_token_square_greedy"}}
        )
        wave_config = SimulationConfig.from_mapping(
            {"scheduler": {"afd_policy": "afd_wave_token_sum"}}
        )
        wave_square_config = SimulationConfig.from_mapping(
            {"scheduler": {"afd_policy": "afd_wave_token_square_sum"}}
        )
        separate_config = SimulationConfig.from_mapping(
            {
                "scheduler": {
                    "afd_policy": "afd_wave_token_sum",
                    "merged_policy": "merged_wave_token_square_sum",
                    "afd": {
                        "max_num_batched_tokens": 32_768,
                        "chunk_size": 16_384,
                    },
                    "merged": {
                        "max_num_batched_tokens": 65_536,
                        "chunk_size": 8_192,
                    },
                }
            }
        )

        self.assertEqual(default_config.scheduler.afd_policy, "current_runtime")
        self.assertEqual(default_config.scheduler.merged_policy, "current_runtime")
        self.assertEqual(config.scheduler.afd_policy, "vllm_queue_aware")
        self.assertEqual(config.scheduler.merged_policy, "vllm_queue_aware")
        self.assertEqual(token_config.scheduler.afd_policy, "prefill_token_greedy")
        self.assertEqual(token_config.scheduler.merged_policy, "prefill_token_greedy")
        self.assertEqual(
            square_config.scheduler.afd_policy,
            "prefill_token_square_greedy",
        )
        self.assertEqual(wave_config.scheduler.afd_policy, "afd_wave_token_sum")
        self.assertEqual(wave_config.scheduler.merged_policy, "current_runtime")
        self.assertEqual(
            wave_square_config.scheduler.afd_policy,
            "afd_wave_token_square_sum",
        )
        self.assertEqual(
            separate_config.scheduler.afd_policy,
            "afd_wave_token_sum",
        )
        self.assertEqual(
            separate_config.scheduler.merged_policy,
            "merged_wave_token_square_sum",
        )
        self.assertEqual(
            separate_config.scheduler.afd.max_num_batched_tokens,
            32_768,
        )
        self.assertEqual(separate_config.scheduler.afd.chunk_size, 16_384)
        self.assertEqual(
            separate_config.scheduler.merged.max_num_batched_tokens,
            65_536,
        )
        self.assertEqual(separate_config.scheduler.merged.chunk_size, 8_192)

        with self.assertRaisesRegex(ValueError, "scheduler.policy was removed"):
            SimulationConfig.from_mapping({"scheduler": {"policy": "round_robin"}})
        with self.assertRaisesRegex(ValueError, "scheduler.afd_policy"):
            SimulationConfig.from_mapping(
                {"scheduler": {"afd_policy": "merged_wave_token_sum"}}
            )
        with self.assertRaisesRegex(ValueError, "scheduler.merged_policy"):
            SimulationConfig.from_mapping(
                {"scheduler": {"merged_policy": "afd_wave_token_sum"}}
            )
        with self.assertRaisesRegex(ValueError, "were removed"):
            SimulationConfig.from_mapping(
                {"scheduler": {"max_num_batched_tokens": 8_192}}
            )
        with self.assertRaisesRegex(ValueError, "scheduler.afd.chunk_size"):
            SimulationConfig.from_mapping(
                {
                    "scheduler": {
                        "afd": {
                            "max_num_batched_tokens": 8_192,
                            "chunk_size": 16_384,
                        }
                    }
                }
            )

    def test_current_runtime_resolves_policy_by_architecture(self) -> None:
        self.assertEqual(
            resolve_scheduler_policy("current_runtime", "afd"), "round_robin"
        )
        self.assertEqual(
            resolve_scheduler_policy("current_runtime", "merged"),
            "vllm_queue_aware",
        )
        self.assertEqual(
            resolve_scheduler_policy("vllm_queue_aware", "afd"),
            "vllm_queue_aware",
        )
        self.assertEqual(
            resolve_scheduler_policy("afd_wave_token_sum", "afd"),
            "afd_wave_token_sum",
        )
        with self.assertRaisesRegex(ValueError, "unavailable for merged"):
            resolve_scheduler_policy("afd_wave_token_sum", "merged")
        self.assertEqual(
            resolve_scheduler_policy("merged_wave_token_sum", "merged"),
            "merged_wave_token_sum",
        )
        with self.assertRaisesRegex(ValueError, "unavailable for afd"):
            resolve_scheduler_policy("merged_wave_token_sum", "afd")

    def test_vllm_queue_aware_scores_waiting_four_times_running(self) -> None:
        specs = (
            RequestSpec("initial-0", 0.0, 1, 0, 0.0),
            RequestSpec("initial-1", 0.0, 1, 0, 0.0),
            RequestSpec("initial-2", 0.0, 1, 0, 0.0),
            RequestSpec("initial-3", 0.0, 1, 0, 0.0),
            RequestSpec("later", 100.0, 1, 0, 0.0),
        )
        dispatcher = RequestDispatcher(specs, 2, "vllm_queue_aware")
        dispatcher.dispatch_until(0.0)
        running = [
            dispatcher.queues[0].pop(0),
            dispatcher.queues[0].pop(0),
            dispatcher.queues[1].pop(0),
        ]
        batch = SchedulerBatch(
            batch_id=0,
            dp=0,
            segments=tuple(
                BatchSegment(request=request, prefix_tokens=0, query_tokens=1)
                for request in running
            ),
        )
        dispatcher.mark_batch_started(batch)

        self.assertEqual(dispatcher.request_counts(0), (0, 3))
        self.assertEqual(dispatcher.request_counts(1), (1, 0))
        dispatcher.dispatch_until(100.0)

        self.assertEqual(dispatcher.requests[-1].assigned_dp, 0)

    def test_vllm_queue_aware_balances_burst_via_optimistic_counts(self) -> None:
        specs = tuple(
            RequestSpec(f"request-{index}", 0.0, 1, 0, 0.0) for index in range(10_000)
        )
        dispatcher = RequestDispatcher(specs, 4, "vllm_queue_aware")

        dispatcher.dispatch_until(0.0)

        self.assertEqual([len(queue) for queue in dispatcher.queues], [2_500] * 4)
        self.assertEqual(dispatcher.reported_counts, [[2_500, 0]] * 4)

    def test_prefill_token_greedy_routes_by_outstanding_query_tokens(self) -> None:
        specs = (
            RequestSpec("long", 0.0, 8_192, 0, 0.0),
            RequestSpec("cached-short", 0.0, 8_193, 8_192, 0.0),
            RequestSpec("medium", 0.0, 4_096, 0, 0.0),
        )
        dispatcher = RequestDispatcher(specs, 2, "prefill_token_greedy")

        dispatcher.dispatch_until(0.0)

        self.assertEqual(
            [request.assigned_dp for request in dispatcher.requests],
            [0, 1, 1],
        )
        self.assertEqual(dispatcher.outstanding_query_tokens, [8_192, 4_097])

    def test_prefill_token_square_greedy_routes_by_request_token_squares(self) -> None:
        specs = tuple(
            RequestSpec(str(index), 0.0, tokens, 0, 0.0)
            for index, tokens in enumerate((8, 5, 5, 1))
        )
        linear_dispatcher = RequestDispatcher(specs, 2, "prefill_token_greedy")
        square_dispatcher = RequestDispatcher(
            specs,
            2,
            "prefill_token_square_greedy",
        )

        linear_dispatcher.dispatch_until(0.0)
        square_dispatcher.dispatch_until(0.0)

        self.assertEqual(
            [request.assigned_dp for request in linear_dispatcher.requests],
            [0, 1, 1, 0],
        )
        self.assertEqual(
            [request.assigned_dp for request in square_dispatcher.requests],
            [0, 1, 1, 1],
        )
        self.assertEqual(square_dispatcher.outstanding_query_token_squares, [64, 51])

    def test_afd_wave_estimators_route_by_fifo_attention_work(self) -> None:
        specs = tuple(
            RequestSpec(str(index), 0.0, tokens, 0, 0.0)
            for index, tokens in enumerate((8, 5, 5, 1))
        )
        base = {
            "mode": "fixed",
            "fixed_lengths": [8],
            "scheduler": _symmetric_batch_configs(32),
            "output": {"include_timeline": False},
        }
        linear_config = SimulationConfig.from_mapping(
            {
                **base,
                "scheduler": {
                    **base["scheduler"],
                    "afd_policy": "afd_wave_token_sum",
                },
            }
        )
        square_config = SimulationConfig.from_mapping(
            {
                **base,
                "scheduler": {
                    **base["scheduler"],
                    "afd_policy": "afd_wave_token_square_sum",
                },
            }
        )

        linear = compare_architectures(linear_config, make_profile(), specs)
        square = compare_architectures(square_config, make_profile(), specs)

        self.assertEqual(
            [request["assigned_dp"] for request in linear["afd"]["requests"]],
            [0, 1, 1, 0],
        )
        self.assertEqual(
            [request["assigned_dp"] for request in square["afd"]["requests"]],
            [0, 1, 1, 1],
        )
        self.assertEqual(
            linear["merged"]["summary"]["scheduler_policy"],
            "vllm_queue_aware",
        )

    def test_merged_wave_estimators_are_selected_independently(self) -> None:
        specs = tuple(
            RequestSpec(str(index), 0.0, tokens, 0, 0.0)
            for index, tokens in enumerate((8, 5, 5, 1))
        )
        base = {
            "mode": "fixed",
            "fixed_lengths": [8],
            "scheduler": {
                "afd_policy": "round_robin",
                **_symmetric_batch_configs(32),
            },
            "output": {"include_timeline": False},
        }
        linear_config = SimulationConfig.from_mapping(
            {
                **base,
                "scheduler": {
                    **base["scheduler"],
                    "merged_policy": "merged_wave_token_sum",
                },
            }
        )
        square_config = SimulationConfig.from_mapping(
            {
                **base,
                "scheduler": {
                    **base["scheduler"],
                    "merged_policy": "merged_wave_token_square_sum",
                },
            }
        )

        linear = compare_architectures(
            linear_config,
            make_profile(merged_dp_size=2, merged_tp_size=8),
            specs,
        )
        square = compare_architectures(
            square_config,
            make_profile(merged_dp_size=2, merged_tp_size=8),
            specs,
        )

        self.assertEqual(
            [request["assigned_dp"] for request in linear["merged"]["requests"]],
            [0, 1, 1, 0],
        )
        self.assertEqual(
            [request["assigned_dp"] for request in square["merged"]["requests"]],
            [0, 1, 1, 1],
        )
        self.assertEqual(linear["afd"]["summary"]["scheduler_policy"], "round_robin")
        self.assertEqual(
            linear["merged"]["summary"]["scheduler_policy"],
            "merged_wave_token_sum",
        )

    def test_merged_wave_prefers_existing_global_wave(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [10],
                "scheduler": {
                    "merged_policy": "merged_wave_token_sum",
                    **_symmetric_batch_configs(10),
                },
            }
        )

        def runtime_request(request_id: str, tokens: int) -> RuntimeRequest:
            return RuntimeRequest(
                RequestSpec(request_id, 0.0, tokens, 0, 0.0),
                assigned_dp=-1,
            )

        queues = [
            [runtime_request("dp0-wave0", 10), runtime_request("dp0-wave1", 10)],
            [runtime_request("dp1-wave0", 10)],
        ]
        candidate = runtime_request("candidate", 1)

        scores = _merged_wave_completion_scores(
            config,
            candidate,
            queues,
            "merged_wave_token_sum",
        )

        self.assertEqual(scores[0], (3.0, 21.0, 21.0))
        self.assertEqual(scores[1], (2.0, 20.0, 20.0))

    def test_chunked_prefill_fills_wave_with_later_fifo_requests(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "scheduler": {
                    "chunked_prefill": True,
                    **_symmetric_batch_configs(32, 16),
                }
            }
        )
        long_request = RuntimeRequest(
            RequestSpec("long", 0.0, 40, 0, 0.0), assigned_dp=0
        )
        short_request = RuntimeRequest(
            RequestSpec("short", 0.0, 16, 0, 0.0), assigned_dp=0
        )
        queue = [long_request, short_request]

        first_batch = _pack_batch(
            config,
            config.scheduler.afd,
            queue,
            dp=0,
            batch_id=0,
            start_ms=0.0,
        )

        self.assertIsNotNone(first_batch)
        if first_batch is None:
            self.fail("expected the first FIFO batch to be planned")
        self.assertEqual(
            [segment.query_tokens for segment in first_batch.segments],
            [16, 16],
        )
        self.assertEqual(
            [segment.request.spec.request_id for segment in first_batch.segments],
            ["long", "short"],
        )
        self.assertEqual(queue, [long_request])
        self.assertEqual(long_request.remaining_query_tokens, 24)
        self.assertEqual(short_request.remaining_query_tokens, 0)

        second_batch = _pack_batch(
            config,
            config.scheduler.afd,
            queue,
            dp=0,
            batch_id=1,
            start_ms=1.0,
        )
        third_batch = _pack_batch(
            config,
            config.scheduler.afd,
            queue,
            dp=0,
            batch_id=2,
            start_ms=2.0,
        )

        self.assertIsNotNone(second_batch)
        self.assertIsNotNone(third_batch)
        if second_batch is None or third_batch is None:
            self.fail("expected the long request to require three batches")
        self.assertEqual(second_batch.query_tokens, 16)
        self.assertEqual(third_batch.query_tokens, 8)
        self.assertEqual(queue, [])

    def test_architectures_use_independent_batch_and_chunk_sizes(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [48],
                "scheduler": {
                    "chunked_prefill": True,
                    "afd": {
                        "max_num_batched_tokens": 32,
                        "chunk_size": 16,
                    },
                    "merged": {
                        "max_num_batched_tokens": 64,
                        "chunk_size": 32,
                    },
                },
            }
        )

        result = compare_architectures(
            config,
            make_profile(
                layer_count=1,
                afd_dp_size=1,
                afd_tp_size=8,
                merged_dp_size=1,
                merged_tp_size=16,
            ),
        )
        afd_attention = next(
            event
            for event in result["afd"]["timeline"]
            if event["phase"] == "attention_router"
        )
        merged_attention = next(
            event
            for event in result["merged"]["timeline"]
            if event["phase"] == "attention_router"
        )

        self.assertEqual(afd_attention["tokens"], 16)
        self.assertEqual(merged_attention["tokens"], 32)

    def test_chunked_wave_scores_use_candidate_completion_wave(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "scheduler": {
                    "chunked_prefill": True,
                    "afd": {
                        "max_num_batched_tokens": 32,
                        "chunk_size": 16,
                    },
                    "merged": {
                        "max_num_batched_tokens": 16,
                        "chunk_size": 8,
                    },
                }
            }
        )

        def runtime_request(request_id: str, tokens: int) -> RuntimeRequest:
            return RuntimeRequest(
                RequestSpec(request_id, 0.0, tokens, 0, 0.0),
                assigned_dp=-1,
            )

        long_request = runtime_request("long", 40)
        candidate = runtime_request("candidate", 16)
        queues = [[long_request], []]

        afd_scores = _afd_wave_completion_scores(
            config,
            candidate,
            queues,
            {},
            1,
            "afd_wave_token_sum",
        )
        merged_scores = _merged_wave_completion_scores(
            config,
            candidate,
            queues,
            "merged_wave_token_sum",
        )

        self.assertEqual(afd_scores, ((32.0,), (16.0,)))
        self.assertEqual(merged_scores[0], (5.0, 56.0, 32.0))
        self.assertEqual(merged_scores[1], (5.0, 40.0, 16.0))

    def test_chunked_running_request_is_not_also_counted_as_waiting(self) -> None:
        specs = (RequestSpec("chunked", 0.0, 2, 0, 0.0),)
        dispatcher = RequestDispatcher(specs, 1, "vllm_queue_aware")
        dispatcher.dispatch_until(0.0)
        dispatcher.requests[0].computed_query_tokens = 1
        batch = SchedulerBatch(
            batch_id=0,
            dp=0,
            segments=(
                BatchSegment(
                    request=dispatcher.requests[0],
                    prefix_tokens=0,
                    query_tokens=1,
                ),
            ),
        )

        dispatcher.mark_batch_started(batch)
        self.assertEqual(dispatcher.request_counts(0), (0, 1))
        self.assertEqual(dispatcher.outstanding_query_tokens, [2])
        self.assertEqual(dispatcher.outstanding_query_token_squares, [4])
        dispatcher.mark_batch_finished(batch)
        self.assertEqual(dispatcher.request_counts(0), (1, 0))
        self.assertEqual(dispatcher.outstanding_query_tokens, [1])
        self.assertEqual(dispatcher.outstanding_query_token_squares, [1])

    def test_fixed_workload_runs_both_architectures_and_exposes_barrier(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [128, 512, 2048, 8192],
                "scheduler": _symmetric_batch_configs(16_384),
            }
        )

        result = compare_architectures(config, make_profile())

        self.assertEqual(result["afd"]["summary"]["request_count"], 4)
        self.assertEqual(result["merged"]["summary"]["request_count"], 4)
        self.assertEqual(result["afd"]["summary"]["scheduler_policy"], "round_robin")
        self.assertEqual(
            result["merged"]["summary"]["scheduler_policy"],
            "vllm_queue_aware",
        )
        self.assertGreater(result["merged"]["summary"]["barrier_wait_ms"], 0)
        self.assertTrue(
            all(item["completion_ms"] is not None for item in result["afd"]["requests"])
        )

    def test_expert_timeline_events_expose_shape_topk_and_ep(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [128, 128, 128, 128],
                "scheduler": _symmetric_batch_configs(128),
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
                "scheduler": _symmetric_batch_configs(8_192),
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
                "scheduler": _symmetric_batch_configs(132),
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

    def test_merged_topology_uses_profile_dp_size(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [512, 8_192],
                "scheduler": {
                    "afd_policy": "round_robin",
                    "merged_policy": "round_robin",
                    **_symmetric_batch_configs(8_192),
                },
            }
        )

        result = compare_architectures(
            config,
            make_profile(layer_count=1, merged_dp_size=2, merged_tp_size=8),
        )
        merged = result["merged"]
        routed = next(
            event for event in merged["timeline"] if event["phase"] == "routed_experts"
        )

        self.assertEqual(merged["summary"]["topology"]["dp_size"], 2)
        self.assertEqual(merged["summary"]["topology"]["tp_size"], 8)
        self.assertEqual(
            {request["assigned_dp"] for request in merged["requests"]},
            {0, 1},
        )
        self.assertEqual(routed["tokens"], 8_704)
        self.assertEqual(routed["profile_query_tokens"], 4_352)

    def test_afd_topology_uses_profile_dp_size(self) -> None:
        specs = tuple(
            RequestSpec(str(index), 0.0, 128, 0, 0.0) for index in range(6)
        )
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [128],
                "scheduler": _symmetric_batch_configs(8_192),
                "output": {"include_timeline": False},
            }
        )

        result = compare_architectures(
            config,
            make_profile(
                afd_dp_size=3,
                afd_tp_size=8,
                merged_dp_size=4,
                merged_tp_size=8,
            ),
            specs,
        )

        self.assertEqual(
            result["afd"]["summary"]["topology"],
            {"dp_size": 3, "tp_size": 8, "ep_size": 8},
        )
        self.assertEqual(
            {request["assigned_dp"] for request in result["afd"]["requests"]},
            {0, 1, 2},
        )

    def test_afd_topology_requires_explicit_parallel_sizes(self) -> None:
        profile = make_profile(layer_count=1)
        del profile.topology_specs["afd"]["attention"]["dp_size"]
        config = SimulationConfig.from_mapping(
            {"mode": "fixed", "fixed_lengths": [128]}
        )

        with self.assertRaises(KeyError):
            compare_architectures(config, profile)

    def test_chunked_prefill_and_token_ubatch_preserve_request_completion(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [2048],
                "scheduler": {
                    "chunked_prefill": True,
                    **_symmetric_batch_configs(1_024),
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
                "scheduler": _symmetric_batch_configs(2_048),
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
                "scheduler": _symmetric_batch_configs(1_024),
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
                "scheduler": _symmetric_batch_configs(100),
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
        unlimited = SimulationConfig.from_mapping(
            {
                "mode": "fixed",
                "fixed_lengths": [128],
                "output": {"timeline_max_events": None},
            }
        )

        exact_result = compare_architectures(exact, make_profile(layer_count=1))
        truncated_result = compare_architectures(truncated, make_profile(layer_count=1))
        unlimited_result = compare_architectures(unlimited, make_profile(layer_count=1))

        self.assertFalse(exact_result["afd"]["summary"]["timeline_truncated"])
        self.assertTrue(truncated_result["afd"]["summary"]["timeline_truncated"])
        self.assertEqual(len(truncated_result["afd"]["timeline"]), 6)
        self.assertIsNone(unlimited.output.timeline_max_events)
        self.assertFalse(unlimited_result["afd"]["summary"]["timeline_truncated"])
        self.assertEqual(
            unlimited_result["afd"]["timeline"], exact_result["afd"]["timeline"]
        )

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
