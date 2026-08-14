# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Deterministic routing strategy for controlled CUDA MoE benchmarks.

Select ``afd_balanced`` with vLLM's
``VLLM_MOE_ROUTING_SIMULATION_STRATEGY`` environment variable. The optional
``AFD_BENCHMARK_FORCE_LB_TOPN_PER_RANK`` variable limits the local expert pool
on every EP rank; ``0`` selects all local experts.

The strategy returns normalized uniform weights and deterministic expert IDs.
It changes model outputs and is intended only for benchmark and profiling runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulator,
    RoutingStrategy,
)

AFD_BALANCED_ROUTING_STRATEGY = "afd_balanced"
_TOPN_PER_RANK_ENV = "AFD_BENCHMARK_FORCE_LB_TOPN_PER_RANK"
_DETERMINISTIC_SEED = 1024


@dataclass(frozen=True)
class _RoutingConfig:
    num_experts: int
    ep_size: int
    ep_rank: int
    top_k: int
    topn_per_rank: int


@dataclass(frozen=True)
class _RoutingBuffers:
    weights: torch.Tensor
    expert_ids: torch.Tensor


def _validate_config(config: _RoutingConfig) -> None:
    if config.num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if config.ep_size <= 0:
        raise ValueError("ep_size must be positive")
    if not 0 <= config.ep_rank < config.ep_size:
        raise ValueError("ep_rank must be within the expert-parallel group")
    if config.top_k <= 0:
        raise ValueError("top_k must be positive")
    if config.num_experts % config.ep_size != 0:
        raise ValueError("num_experts must be divisible by ep_size")

    local_experts = config.num_experts // config.ep_size
    selected_per_rank = config.topn_per_rank or local_experts
    if not 0 < selected_per_rank <= local_experts:
        raise ValueError(f"{_TOPN_PER_RANK_ENV} must be between 0 and {local_experts}")
    if config.top_k > selected_per_rank * config.ep_size:
        raise ValueError("selected expert pool must contain at least top_k experts")


def _build_expert_cycle(
    config: _RoutingConfig,
    device: torch.device,
) -> torch.Tensor:
    _validate_config(config)
    local_experts = config.num_experts // config.ep_size

    if config.topn_per_rank:
        expert_cycle = torch.cat(
            [
                torch.arange(
                    rank * local_experts,
                    rank * local_experts + config.topn_per_rank,
                    device=device,
                    dtype=torch.int32,
                )
                for rank in range(config.ep_size)
            ]
        )
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_DETERMINISTIC_SEED)
        expert_cycle = torch.randperm(
            config.num_experts,
            generator=generator,
            device="cpu",
            dtype=torch.int32,
        ).to(device=device, non_blocking=True)

    source_rank_offset = config.ep_rank * local_experts
    return (expert_cycle + source_rank_offset) % config.num_experts


def _build_routing_buffers(
    config: _RoutingConfig,
    max_tokens: int,
    device: torch.device,
    indices_dtype: torch.dtype,
) -> _RoutingBuffers:
    expert_cycle = _build_expert_cycle(config, device)
    total_ids = max_tokens * config.top_k
    repeat_count = (total_ids + expert_cycle.numel() - 1) // expert_cycle.numel()
    expert_ids = expert_cycle.repeat(repeat_count)[:total_ids].reshape(
        max_tokens,
        config.top_k,
    )
    expert_ids = expert_ids.to(dtype=indices_dtype)
    weights = torch.full(
        (max_tokens, config.top_k),
        1.0 / config.top_k,
        device=device,
        dtype=torch.float32,
    )
    return _RoutingBuffers(weights=weights, expert_ids=expert_ids)


class AFDBalancedRoutingStrategy(RoutingStrategy):
    """Generate deterministic, EP-balanced routing for performance tests."""

    def __init__(self, topn_per_rank: int | None = None) -> None:
        if topn_per_rank is None:
            topn_per_rank = int(os.environ.get(_TOPN_PER_RANK_ENV, "0"))
        self.topn_per_rank = topn_per_rank
        self._buffers: dict[
            tuple[_RoutingConfig, torch.device, torch.dtype],
            _RoutingBuffers,
        ] = {}

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ep_group = get_ep_group()
        config = _RoutingConfig(
            num_experts=router_logits.shape[-1],
            ep_size=ep_group.world_size,
            ep_rank=ep_group.rank_in_group,
            top_k=top_k,
            topn_per_rank=self.topn_per_rank,
        )
        indices_dtype = indices_type or torch.int32
        key = (config, hidden_states.device, indices_dtype)
        buffers = self._buffers.get(key)
        max_tokens = hidden_states.shape[0]
        if buffers is None or buffers.expert_ids.shape[0] < max_tokens:
            buffers = _build_routing_buffers(
                config,
                max_tokens,
                hidden_states.device,
                indices_dtype,
            )
            self._buffers[key] = buffers

        num_tokens = hidden_states.shape[0]
        return buffers.weights[:num_tokens], buffers.expert_ids[:num_tokens]


def register_afd_balanced_routing_strategy() -> None:
    """Register the opt-in AFD strategy with vLLM's routing simulator."""
    RoutingSimulator.register_strategy(
        AFD_BALANCED_ROUTING_STRATEGY,
        AFDBalancedRoutingStrategy(),
    )


__all__ = [
    "AFD_BALANCED_ROUTING_STRATEGY",
    "AFDBalancedRoutingStrategy",
    "register_afd_balanced_routing_strategy",
]
