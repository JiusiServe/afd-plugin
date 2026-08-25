from __future__ import annotations

import unittest

from simulator.profile_builder import (
    DEFAULT_AFD_FFN_TOPOLOGY,
    ROUTED_EXPERT_SAMPLE_SHAPES,
    ROUTED_EXPERT_SHAPE,
    SHARED_EXPERT_SHAPE,
    _anchor_grid,
    _compose_afd_layers,
)
from simulator.profiles import PROFILE_PHASES, ProfileBundle
from simulator.tests.helpers import make_profile


class ProfileTests(unittest.TestCase):
    def test_profile_returns_anchor_and_interpolates_both_dimensions(self) -> None:
        profile = make_profile(layer_count=1)
        exact = profile.duration_ms("afd", 0, "attention_router", 0, 128)
        middle = profile.duration_ms("afd", 0, "attention_router", 256, 256)

        self.assertGreater(exact, 0)
        self.assertGreater(middle, exact)

    def test_profile_rejects_extrapolation(self) -> None:
        profile = make_profile(layer_count=1)

        with self.assertRaisesRegex(ValueError, "outside profile domain"):
            profile.duration_ms("afd", 0, "attention_router", 0, 65_536)

    def test_expert_shapes_report_the_actual_interpolation_anchors(self) -> None:
        profile = make_profile(layer_count=1)

        samples = profile.expert_shape_samples("afd", 0, "routed_experts", 96)

        self.assertEqual(
            samples,
            [
                {
                    "query_tokens": 64,
                    "weight": 0.5,
                    "shape": [48, 4_096],
                    "expert_shapes": [{"shape": [48, 4_096], "count": 1}],
                },
                {
                    "query_tokens": 128,
                    "weight": 0.5,
                    "shape": [96, 4_096],
                    "expert_shapes": [{"shape": [96, 4_096], "count": 1}],
                },
            ],
        )

    def test_triangular_grid_covers_full_context_boundary(self) -> None:
        max_context, grid = _anchor_grid(
            (128, 512, 65_536, 131_072),
            (8_192, 32_768, 122_880),
        )

        self.assertEqual(max_context, 131_072)
        self.assertIn(0, grid)
        self.assertIn(1, grid[8_192])
        self.assertIn(122_880, grid[8_192])
        self.assertEqual(grid[131_071], (1,))

    def test_triangular_profile_interpolates_near_context_boundary(self) -> None:
        points = []
        for prefix, queries in {0: (1, 8, 16), 8: (1, 8), 15: (1,)}.items():
            for query in queries:
                layer = {phase: 0.0 for phase in PROFILE_PHASES}
                layer["attention_router"] = prefix + 2 * query
                points.append(
                    {
                        "prefix_tokens": prefix,
                        "query_tokens": query,
                        "layers": [layer],
                    }
                )
        profile = ProfileBundle.from_mapping(
            {
                "schema_version": 1,
                "layer_count": 1,
                "topologies": {
                    name: {"max_context_tokens": 16, "points": points}
                    for name in ("afd", "merged")
                },
            }
        )

        value = profile.duration_ms("afd", 0, "attention_router", 4, 12)

        self.assertEqual(value, 28)

    def test_afd_profile_composes_attention_and_single_job_ffn_phases(self) -> None:
        attention = [{phase: 1.0 for phase in PROFILE_PHASES}]
        ffn = [
            {
                **{phase: 2.0 for phase in PROFILE_PHASES},
                ROUTED_EXPERT_SHAPE: [192, 4_096],
                ROUTED_EXPERT_SAMPLE_SHAPES: [{"shape": [6, 4_096], "count": 32}],
                SHARED_EXPERT_SHAPE: [32, 4_096],
            }
        ]

        layer = _compose_afd_layers(
            attention_layers=attention,
            ffn_layers=ffn,
        )[0]

        self.assertEqual(layer["attention_router"], 1.0)
        self.assertEqual(layer["afd_post"], 1.0)
        self.assertEqual(layer["routed_experts"], 2.0)
        self.assertEqual(layer["shared_expert"], 2.0)
        self.assertEqual(layer[ROUTED_EXPERT_SHAPE], [192, 4_096])
        self.assertEqual(
            layer[ROUTED_EXPERT_SAMPLE_SHAPES],
            [{"shape": [6, 4_096], "count": 32}],
        )
        self.assertEqual(layer[SHARED_EXPERT_SHAPE], [32, 4_096])
        self.assertEqual(DEFAULT_AFD_FFN_TOPOLOGY.dp_size, 8)
        self.assertEqual(DEFAULT_AFD_FFN_TOPOLOGY.tp_size, 1)
        self.assertFalse(DEFAULT_AFD_FFN_TOPOLOGY.sequence_parallel)
