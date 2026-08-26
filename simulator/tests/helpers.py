"""Small deterministic profile fixtures."""

from __future__ import annotations

from simulator.profiles import PROFILE_PHASES, ProfileBundle

QUERY_ANCHORS = (1, 64, 128, 256, 512, 1_024, 2_048, 8_192, 32_768)
PREFIX_ANCHORS = (0, 256, 512, 4_096)
AFD_FFN_DEVICE_COUNT = 8


def make_profile(
    layer_count: int = 3,
    *,
    afd_dp_size: int = 2,
    afd_tp_size: int = 4,
    merged_dp_size: int = 4,
    merged_tp_size: int = 4,
) -> ProfileBundle:
    afd_attention_devices = afd_dp_size * afd_tp_size
    merged_devices = merged_dp_size * merged_tp_size
    topologies = {}
    for topology, factor in (("afd", 0.9), ("merged", 1.0)):
        expert_parallel_size = (
            AFD_FFN_DEVICE_COUNT if topology == "afd" else merged_devices
        )
        points = []
        for prefix in PREFIX_ANCHORS:
            for query in QUERY_ANCHORS:
                layers = []
                for layer in range(layer_count):
                    scale = factor * (1.0 + layer * 0.05)
                    token_cost = query / 4_096.0
                    prefix_cost = prefix * query / (4_096.0 * 4_096.0)
                    values = {phase: 0.0 for phase in PROFILE_PHASES}
                    values.update(
                        {
                            "attention_router": scale
                            * (0.6 + 0.8 * token_cost + 0.2 * prefix_cost),
                            "merged_dispatch": scale * (0.12 + 0.05 * token_cost),
                            "routed_experts": scale * (0.3 + 0.4 * token_cost),
                            "merged_combine": scale * (0.08 + 0.04 * token_cost),
                            "merged_combine_local": scale * 0.05,
                            "shared_expert": scale * (0.12 + 0.2 * token_cost),
                            "merged_sp_post": scale * 0.11,
                            "afd_post": scale * 0.04,
                        }
                    )
                    layers.append(values)
                points.append(
                    {
                        "prefix_tokens": prefix,
                        "query_tokens": query,
                        "layers": [
                            {
                                **layer,
                                "routed_expert_input_shape": [
                                    (query * 6 + expert_parallel_size - 1)
                                    // expert_parallel_size,
                                    4_096,
                                ],
                                "routed_expert_sample_shapes": [
                                    {
                                        "shape": [
                                            (
                                                query * 6
                                                + expert_parallel_size
                                                - 1
                                            )
                                            // expert_parallel_size,
                                            4_096,
                                        ],
                                        "count": 1,
                                    }
                                ],
                                "shared_expert_input_shape": [query, 4_096],
                            }
                            for layer in layers
                        ],
                    }
                )
        topologies[topology] = {"points": points}
    return ProfileBundle.from_mapping(
        {
            "schema_version": 1,
            "layer_count": layer_count,
            "metadata": {
                "model": "test-dsv4",
                "performance_model": "analytic",
                "model_config": {"hidden_size": 4_096, "moe_top_k": 6},
            },
            "topologies": {
                "afd": {
                    **topologies["afd"],
                    "spec": {
                        "attention": {
                            "num_devices": afd_attention_devices,
                            "dp_size": afd_dp_size,
                            "tp_size": afd_tp_size,
                            "ep_size": AFD_FFN_DEVICE_COUNT,
                        },
                        "ffn": {
                            "num_devices": AFD_FFN_DEVICE_COUNT,
                            "dp_size": AFD_FFN_DEVICE_COUNT,
                            "tp_size": 1,
                            "ep_size": AFD_FFN_DEVICE_COUNT,
                        },
                    },
                },
                "merged": {
                    **topologies["merged"],
                    "spec": {
                        "num_devices": merged_devices,
                        "dp_size": merged_dp_size,
                        "tp_size": merged_tp_size,
                        "ep_size": merged_devices,
                    },
                },
            },
        }
    )
