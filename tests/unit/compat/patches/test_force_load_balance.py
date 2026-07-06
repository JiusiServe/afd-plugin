from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

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
        def apply(
            self,
            layer: object,
            x: object,
            router_logits: object,
            top_k: int,
            renormalize: bool,
            **kwargs: object,
        ) -> torch.Tensor:
            import vllm_ascend.quantization.methods.w8a8_dynamic as mod

            del layer, x, router_logits, top_k, renormalize
            return mod.build_fused_experts_input(topk_ids=kwargs["topk_ids"])

    vllm = types.ModuleType("vllm")
    vllm_config = types.ModuleType("vllm.config")
    vllm_config.VllmConfig = object

    root = types.ModuleType("vllm_ascend")
    ops = types.ModuleType("vllm_ascend.ops")
    fused_moe_pkg = types.ModuleType("vllm_ascend.ops.fused_moe")
    fused_moe_mod = types.ModuleType("vllm_ascend.ops.fused_moe.fused_moe")
    fused_moe_mod.AscendFusedMoE = AscendFusedMoE

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
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", ops)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe", fused_moe_pkg)
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
    mod.apply_force_load_balance_patch()
    mod.apply_force_load_balance_patch()
    return mod


def _new_layer(force_lb_mod: types.ModuleType) -> object:
    return force_lb_mod.AscendFusedMoE.__new__(force_lb_mod.AscendFusedMoE)


def test_force_load_balance_buffer_topn_per_rank(force_lb_mod: types.ModuleType):
    layer = _new_layer(force_lb_mod)
    layer.ep_size = 4
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
        top_k=2,
        topn_per_rank=0,
    )

    first = force_lb_mod._build_expert_cycle(config, torch.device("cpu"))
    second = force_lb_mod._build_expert_cycle(config, torch.device("cpu"))

    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(8))


def test_force_load_balance_buffer_grows_for_large_batch(
    force_lb_mod: types.ModuleType,
):
    layer = _new_layer(force_lb_mod)
    layer.ep_size = 2
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

    real_topk_ids = torch.zeros((4, 2), dtype=torch.int64)
    out = method.apply(
        layer=layer,
        x=None,
        router_logits=None,
        top_k=2,
        renormalize=True,
        topk_ids=real_topk_ids,
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

    real_topk_ids = torch.zeros((4, 2), dtype=torch.int64)
    out = method.apply(
        layer=layer,
        x=None,
        router_logits=None,
        top_k=2,
        renormalize=True,
        topk_ids=real_topk_ids,
    )

    assert torch.equal(out, real_topk_ids)
