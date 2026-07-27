from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")


class _QuantType:
    NONE = 0
    W8A8 = 1


def _install_fake_modules(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    class AscendFusedMoE:
        """Stand-in for vllm_ascend AscendFusedMoE."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    def build_fused_experts_input(*args: object, **kwargs: object) -> torch.Tensor:
        """Fake builder: returns the possibly swapped topk_ids."""

        del args
        return kwargs["topk_ids"]

    class AscendW8A8DynamicFusedMoEMethod:
        def __init__(self):
            self.multistream_overlap_gate = False
            self.in_dtype = torch.float32
            self.dynamic_eplb = False
            self.quant_type = _QuantType.W8A8

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            router_logits: torch.Tensor,
            top_k: int,
            renormalize: bool,
            use_grouped_topk: bool = False,
            num_experts: int = -1,
            expert_map: torch.Tensor | None = None,
            topk_group: int | None = None,
            num_expert_group: int | None = None,
            custom_routing_function: Callable | None = None,
            scoring_func: str = "softmax",
            routed_scaling_factor: float = 1.0,
            e_score_correction_bias: torch.Tensor | None = None,
            is_prefill: bool = True,
            enable_force_load_balance: bool = False,
            log2phy: torch.Tensor | None = None,
            global_redundant_expert_num: int = 0,
            pertoken_scale: Any | None = None,
            activation: str = "silu",
            apply_router_weight_on_input: bool = False,
            mc2_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            import vllm_ascend.quantization.methods.w8a8_dynamic as mod

            del (
                layer,
                top_k,
                renormalize,
                use_grouped_topk,
                num_experts,
                expert_map,
                topk_group,
                num_expert_group,
                custom_routing_function,
                scoring_func,
                routed_scaling_factor,
                e_score_correction_bias,
                is_prefill,
                enable_force_load_balance,
                log2phy,
                global_redundant_expert_num,
                pertoken_scale,
                activation,
                apply_router_weight_on_input,
                mc2_mask,
            )
            return mod.build_fused_experts_input(
                hidden_states=x,
                topk_weights=torch.ones_like(router_logits, dtype=torch.float32),
                topk_ids=router_logits,
                w1=torch.empty(0),
                w2=torch.empty(0),
                quant_type=_QuantType.W8A8,
                dynamic_eplb=False,
            )

    vllm = types.ModuleType("vllm")
    vllm_config = types.ModuleType("vllm.config")
    vllm_config.VllmConfig = object

    root = types.ModuleType("vllm_ascend")
    envs_mod = types.ModuleType("vllm_ascend.envs")
    envs_mod.VLLM_ASCEND_ENABLE_FUSED_MC2 = 0
    ascend_forward_context_mod = types.ModuleType("vllm_ascend.ascend_forward_context")
    ascend_forward_context_mod.MoECommType = SimpleNamespace(FUSED_MC2="fused_mc2")
    ascend_forward_context_mod._EXTRA_CTX = SimpleNamespace(
        moe_comm_method=SimpleNamespace(
            fused_experts=lambda fused_experts_input: fused_experts_input
        ),
        moe_comm_type=None,
    )
    flash_common3_context_mod = types.ModuleType("vllm_ascend.flash_common3_context")
    flash_common3_context_mod.get_flash_common3_context = lambda: None
    ops = types.ModuleType("vllm_ascend.ops")
    fused_moe_pkg = types.ModuleType("vllm_ascend.ops.fused_moe")
    experts_selector_mod = types.ModuleType(
        "vllm_ascend.ops.fused_moe.experts_selector"
    )

    def select_experts(
        hidden_states,
        router_logits,
        top_k,
        use_grouped_topk,
        renormalize,
        topk_group,
        num_expert_group,
        custom_routing_function,
        scoring_func,
        routed_scaling_factor,
        e_score_correction_bias,
        mix_placement,
        num_logical_experts,
        num_shared_experts,
        num_experts,
    ):
        return (
            torch.ones_like(router_logits, dtype=torch.float32),
            router_logits,
        )

    experts_selector_mod.select_experts = select_experts
    experts_selector_mod.zero_experts_compute = None
    fused_moe_mod = types.ModuleType("vllm_ascend.ops.fused_moe.fused_moe")
    fused_moe_mod.AscendFusedMoE = AscendFusedMoE
    fused_moe_mod.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        info_once=lambda *args, **kwargs: None,
    )

    quant = types.ModuleType("vllm_ascend.quantization")
    methods = types.ModuleType("vllm_ascend.quantization.methods")
    w8a8_mod = types.ModuleType("vllm_ascend.quantization.methods.w8a8_dynamic")
    w8a8_mod.AscendW8A8DynamicFusedMoEMethod = AscendW8A8DynamicFusedMoEMethod
    w8a8_mod.build_fused_experts_input = build_fused_experts_input

    quant_type_mod = types.ModuleType("vllm_ascend.quantization.quant_type")
    quant_type_mod.QuantType = _QuantType

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config)
    monkeypatch.setitem(sys.modules, "vllm_ascend", root)
    monkeypatch.setitem(sys.modules, "vllm_ascend.envs", envs_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        ascend_forward_context_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.flash_common3_context",
        flash_common3_context_mod,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", ops)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe", fused_moe_pkg)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.experts_selector",
        experts_selector_mod,
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ops.fused_moe.fused_moe", fused_moe_mod
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization", quant)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.methods", methods)
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.quantization.methods.w8a8_dynamic", w8a8_mod
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.quantization.quant_type", quant_type_mod
    )
    return fused_moe_mod


@pytest.fixture
def force_lb_mod(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    _install_fake_modules(monkeypatch)
    module_name = "afd_plugin.compat.patches.npu.force_load_balance"
    sys.modules.pop(module_name, None)
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod


def _new_layer(force_lb_mod: types.ModuleType) -> object:
    return force_lb_mod.AscendFusedMoE.__new__(force_lb_mod.AscendFusedMoE)


def _aggregate_target_rank_counts(
    force_lb_mod: types.ModuleType,
    *,
    n_routed_experts: int,
    ep_size: int,
    top_k: int,
    topn_per_rank: int,
    batch_tokens: int,
) -> torch.Tensor:
    local_routed_experts = n_routed_experts // ep_size
    expert_ids: list[torch.Tensor] = []
    for ep_rank in range(ep_size):
        config = force_lb_mod.ForceLoadBalanceConfig(
            n_routed_experts=n_routed_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
            top_k=top_k,
            topn_per_rank=topn_per_rank,
        )
        expert_ids.append(
            force_lb_mod._build_topk_buffer(
                config,
                max_tokens=batch_tokens,
                device=torch.device("cpu"),
            ).flatten()
        )

    target_ranks = torch.cat(expert_ids) // local_routed_experts
    return torch.bincount(target_ranks.to(torch.int64), minlength=ep_size)


def test_force_load_balance_buffer_topn_per_rank(force_lb_mod: types.ModuleType):
    layer = _new_layer(force_lb_mod)
    layer.ep_size = 4
    layer.ep_rank = 0
    layer.n_routed_experts = 8
    layer.top_k = 2
    layer.force_load_balance_topn_per_rank = 1

    force_lb_mod._init_force_lb_buffer(
        layer,
        max_tokens=4,
        device=torch.device("cpu"),
    )

    expected = torch.tensor([[0, 2], [4, 6], [0, 2], [4, 6]], dtype=torch.int32)
    assert torch.equal(layer.force_lb_fake_topk_buffer, expected)


def test_force_load_balance_buffer_uses_max_num_batched_tokens(
    force_lb_mod: types.ModuleType,
):
    max_tokens = force_lb_mod._get_force_lb_max_tokens(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=6))
    )
    assert max_tokens == 6

    layer = _new_layer(force_lb_mod)
    layer.ep_size = 2
    layer.ep_rank = 0
    layer.n_routed_experts = 4
    layer.top_k = 2
    layer.force_load_balance_topn_per_rank = 0

    force_lb_mod._init_force_lb_buffer(
        layer,
        max_tokens=max_tokens,
        device=torch.device("cpu"),
    )

    assert layer.force_lb_fake_topk_buffer.shape == (6, 2)


def test_force_load_balance_max_tokens_falls_back_when_not_int(
    force_lb_mod: types.ModuleType,
):
    max_tokens = force_lb_mod._get_force_lb_max_tokens(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=None))
    )
    assert max_tokens == 128


def test_force_load_balance_buffer_ids_within_routed_experts(
    force_lb_mod: types.ModuleType,
):
    layer = _new_layer(force_lb_mod)
    layer.ep_size = 2
    layer.ep_rank = 0
    layer.n_routed_experts = 4
    layer.global_num_experts = 6
    layer.top_k = 2
    layer.force_load_balance_topn_per_rank = 2

    force_lb_mod._init_force_lb_buffer(
        layer,
        max_tokens=2,
        device=torch.device("cpu"),
    )

    assert int(layer.force_lb_fake_topk_buffer.max()) < layer.n_routed_experts


def test_force_load_balance_full_expert_cycle_is_deterministic(
    force_lb_mod: types.ModuleType,
):
    config = force_lb_mod.ForceLoadBalanceConfig(
        n_routed_experts=8,
        ep_size=4,
        ep_rank=0,
        top_k=2,
        topn_per_rank=0,
    )

    first = force_lb_mod._build_expert_cycle(config, torch.device("cpu"))
    second = force_lb_mod._build_expert_cycle(config, torch.device("cpu"))

    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(8))


@pytest.mark.parametrize(
    ("ep_size", "batch_tokens"),
    [
        pytest.param(64, 16, id="ep64-bs16"),
        pytest.param(16, 42, id="ep16-bs42"),
        pytest.param(16, 56, id="ep16-bs56"),
    ],
)
def test_force_load_balance_aggregates_evenly_for_published_batches(
    force_lb_mod: types.ModuleType,
    ep_size: int,
    batch_tokens: int,
):
    top_k = 8
    counts = _aggregate_target_rank_counts(
        force_lb_mod,
        n_routed_experts=256,
        ep_size=ep_size,
        top_k=top_k,
        topn_per_rank=4,
        batch_tokens=batch_tokens,
    )

    assert torch.equal(
        counts,
        torch.full((ep_size,), batch_tokens * top_k, dtype=torch.int64),
    )


def test_force_load_balance_all_experts_aggregates_partial_cycle_evenly(
    force_lb_mod: types.ModuleType,
):
    ep_size = 4
    top_k = 2
    batch_tokens = 1
    counts = _aggregate_target_rank_counts(
        force_lb_mod,
        n_routed_experts=8,
        ep_size=ep_size,
        top_k=top_k,
        topn_per_rank=0,
        batch_tokens=batch_tokens,
    )

    assert torch.equal(
        counts,
        torch.full((ep_size,), batch_tokens * top_k, dtype=torch.int64),
    )


def test_force_load_balance_buffer_grows_for_large_batch(
    force_lb_mod: types.ModuleType,
):
    layer = _new_layer(force_lb_mod)
    layer.ep_size = 2
    layer.ep_rank = 0
    layer.n_routed_experts = 4
    layer.top_k = 2
    layer.force_load_balance_topn_per_rank = 2

    force_lb_mod._init_force_lb_buffer(
        layer,
        max_tokens=2,
        device=torch.device("cpu"),
    )
    topk_ids = force_lb_mod._get_force_lb_topk_ids(
        layer,
        batch_tokens=5,
        device=torch.device("cpu"),
    )

    assert topk_ids.shape == (5, 2)
    assert layer.force_lb_fake_topk_buffer.shape[0] >= 5


def test_w8a8_apply_swaps_topk_ids_with_buffer(force_lb_mod: types.ModuleType):
    method = force_lb_mod.AscendW8A8DynamicFusedMoEMethod()

    layer = _new_layer(force_lb_mod)
    layer.enable_force_load_balance = True
    layer.mix_placement = False
    layer.top_k = 2
    layer.ep_size = 4
    layer.n_routed_experts = 8
    layer.force_load_balance_topn_per_rank = 1
    layer.force_lb_fake_topk_buffer = torch.tensor(
        [[0, 2], [4, 6], [0, 2], [4, 6]], dtype=torch.int32
    )
    layer.w13_weight = torch.empty(0)
    layer.w13_weight_scale_fp32 = torch.empty(0)
    layer.w2_weight = torch.empty(0)
    layer.w2_weight_scale = torch.empty(0)

    real_topk_ids = torch.zeros((4, 2), dtype=torch.int64)
    out = method.apply(
        layer=layer,
        x=torch.empty((4, 1)),
        router_logits=real_topk_ids,
        top_k=2,
        renormalize=True,
        num_experts=2,
    )

    expected = layer.force_lb_fake_topk_buffer.to(torch.int64)
    assert torch.equal(out, expected)


def test_w8a8_apply_passthrough_when_buffer_absent(force_lb_mod: types.ModuleType):
    method = force_lb_mod.AscendW8A8DynamicFusedMoEMethod()

    layer = _new_layer(force_lb_mod)
    layer.enable_force_load_balance = False
    layer.force_lb_fake_topk_buffer = None
    layer.mix_placement = False
    layer.top_k = 2
    layer.w13_weight = torch.empty(0)
    layer.w13_weight_scale_fp32 = torch.empty(0)
    layer.w2_weight = torch.empty(0)
    layer.w2_weight_scale = torch.empty(0)

    real_topk_ids = torch.zeros((4, 2), dtype=torch.int64)
    out = method.apply(
        layer=layer,
        x=torch.empty((4, 1)),
        router_logits=real_topk_ids,
        top_k=2,
        renormalize=True,
        num_experts=2,
    )

    assert torch.equal(out, real_topk_ids)
