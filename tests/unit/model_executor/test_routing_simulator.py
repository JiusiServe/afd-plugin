# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.model_executor.layers.fused_moe.router.router_factory import (  # noqa: E402
    create_fused_moe_router,
)
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (  # noqa: E402
    RoutingSimulatorRouter,
)

from afd_plugin.model_executor import routing_simulator as routing  # noqa: E402

pytestmark = pytest.mark.vllm_runtime


def _config(**overrides: int) -> routing._RoutingConfig:
    values = {
        "num_experts": 16,
        "ep_size": 4,
        "ep_rank": 0,
        "top_k": 8,
        "topn_per_rank": 2,
        **overrides,
    }
    return routing._RoutingConfig(**values)


def test_routing_cycles_are_deterministic_and_balanced() -> None:
    full_config = _config(topn_per_rank=0)
    first = routing._build_expert_cycle(full_config, torch.device("cpu"))
    second = routing._build_expert_cycle(full_config, torch.device("cpu"))
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(full_config.num_experts))

    target_ranks = torch.cat(
        [
            routing._build_routing_buffers(
                _config(num_experts=32, ep_rank=rank, topn_per_rank=4),
                max_tokens=1,
                device=torch.device("cpu"),
                indices_dtype=torch.int32,
            ).expert_ids.div(8, rounding_mode="floor")
            for rank in range(4)
        ]
    )
    assert torch.bincount(target_ranks.flatten(), minlength=4).tolist() == [8] * 4


def test_invalid_routing_configs_are_rejected() -> None:
    invalid_configs = (
        _config(num_experts=0),
        _config(ep_size=0),
        _config(ep_rank=4),
        _config(top_k=0),
        _config(num_experts=15),
        _config(topn_per_rank=5),
        _config(ep_size=1, topn_per_rank=4),
    )
    for config in invalid_configs:
        with pytest.raises(ValueError):
            routing._build_expert_cycle(config, torch.device("cpu"))


def test_strategy_returns_normalized_weights_and_reuses_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routing,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    strategy = routing.AFDBalancedRoutingStrategy(topn_per_rank=2)

    weights, expert_ids = strategy.route_tokens(
        torch.empty((2, 8)),
        torch.empty((2, 8)),
        top_k=4,
    )
    reused_weights, reused_ids = strategy.route_tokens(
        torch.empty((1, 8)),
        torch.empty((1, 8)),
        top_k=4,
    )

    assert weights.tolist() == [[0.25] * 4] * 2
    assert expert_ids.tolist() == [[0, 1, 4, 5]] * 2
    assert expert_ids.dtype == torch.int32
    assert weights.data_ptr() == reused_weights.data_ptr()
    assert expert_ids.data_ptr() == reused_ids.data_ptr()


def test_strategy_registers_with_vllm_and_preserves_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
        routing.AFD_BALANCED_ROUTING_STRATEGY,
    )
    monkeypatch.setenv("AFD_BENCHMARK_FORCE_LB_TOPN_PER_RANK", "2")
    monkeypatch.setattr(
        routing,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    monkeypatch.setattr(routing.RoutingSimulator, "_routing_strategies", {})
    routing.register_afd_balanced_routing_strategy()

    router = create_fused_moe_router(top_k=4, global_num_experts=8)
    captured_ids: list[torch.Tensor] = []
    router.set_capture_fn(captured_ids.append)
    _, expert_ids = router.select_experts(
        torch.empty((1, 8)),
        torch.empty((1, 8)),
    )

    assert isinstance(router, RoutingSimulatorRouter)
    assert expert_ids.tolist() == [[0, 1, 4, 5]]
    assert len(captured_ids) == 1
    assert torch.equal(captured_ids[0], expert_ids)
