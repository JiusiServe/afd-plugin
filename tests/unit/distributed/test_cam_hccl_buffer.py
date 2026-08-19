from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from afd_plugin.config import AFDConfig
from afd_plugin.distributed.cam_hccl_buffer import (
    derive_cam_hccl_buffer_plan,
    derive_cam_hccl_buffer_plan_from_config,
    warn_if_cam_memory_headroom_is_low,
)


def _vllm_config(
    *,
    connector: str = "CAMAsyncAFDConnector",
    num_npus_per_dp_group: int = 2,
    dynamic_quant: int = 1,
    gpu_memory_utilization: float = 0.9,
):
    extra_config = (
        {
            "attn_ranks_per_dp": num_npus_per_dp_group,
            "dynamicQuant": dynamic_quant,
        }
        if connector == "CAMAsyncAFDConnector"
        else {}
    )
    return SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": extra_config}},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=num_npus_per_dp_group,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8000),
        cache_config=SimpleNamespace(
            gpu_memory_utilization=gpu_memory_utilization,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=2048,
                num_experts_per_tok=6,
                n_routed_experts=64,
            ),
        ),
    )


def _afd_config(*, connector: str, role: str) -> AFDConfig:
    return AFDConfig(
        connector=connector,
        role=role,
        num_attention_ranks=2,
        num_ffn_ranks=2,
        compute_gate_on_attention=connector == "CAMAsyncAFDConnector",
    )


def test_deepseek_v2_lite_attention_gate_slot_uses_full_cam_capacity():
    plan = derive_cam_hccl_buffer_plan(
        hidden_size=2048,
        max_batch_tokens=8000,
        num_npus_per_dp_group=2,
        topk=6,
        num_routed_experts=64,
        attention_rank_size=2,
        ffn_rank_size=2,
        compute_gate_on_attention=True,
        dynamic_quant=0,
    )

    assert plan.attention_required_bytes == 229_376_000
    assert plan.ffn_required_bytes == 1_048_576_000
    assert plan.attention_buffer_size_mb == 241
    assert plan.ffn_buffer_size_mb == 1100
    assert plan.buffer_size_mb_for_role("attention") == 241
    assert plan.buffer_size_mb_for_role("ffn") == 1100


def test_attention_gate_dynamic_quant_uses_int8_dispatch_and_bf16_combine():
    plan = derive_cam_hccl_buffer_plan(
        hidden_size=2048,
        max_batch_tokens=8000,
        num_npus_per_dp_group=2,
        topk=6,
        num_routed_experts=64,
        attention_rank_size=2,
        ffn_rank_size=2,
        compute_gate_on_attention=True,
        dynamic_quant=1,
    )

    assert plan.attention_required_bytes == 172_144_000
    assert plan.ffn_required_bytes == 786_944_000
    assert plan.attention_buffer_size_mb == 181
    assert plan.ffn_buffer_size_mb == 826


def test_deepseek_v32_dynamic_quantized_slot_fits_in_four_gibibytes():
    plan = derive_cam_hccl_buffer_plan(
        hidden_size=7168,
        max_batch_tokens=65536,
        num_npus_per_dp_group=8,
        topk=8,
        num_routed_experts=256,
        attention_rank_size=16,
        ffn_rank_size=16,
        compute_gate_on_attention=True,
        dynamic_quant=1,
    )

    assert plan.attention_required_bytes == 1_585_741_824
    assert plan.ffn_required_bytes == 2_819_096_576
    assert plan.attention_buffer_size_mb == 1664
    assert plan.ffn_buffer_size_mb == 2958


def test_ffn_gate_slot_carries_one_unexpanded_row_in_each_direction():
    plan = derive_cam_hccl_buffer_plan(
        hidden_size=3,
        max_batch_tokens=5,
        num_npus_per_dp_group=2,
        topk=2,
        num_routed_experts=8,
        attention_rank_size=3,
        ffn_rank_size=2,
        compute_gate_on_attention=False,
        dynamic_quant=1,
    )

    assert plan.attention_required_bytes == 36
    assert plan.ffn_required_bytes == 144
    assert plan.attention_buffer_size_mb == 1
    assert plan.ffn_buffer_size_mb == 1


def test_buffer_plan_from_config_supports_async_and_camp2p():
    async_plan = derive_cam_hccl_buffer_plan_from_config(
        _vllm_config(),
        _afd_config(connector="CAMAsyncAFDConnector", role="attention"),
    )
    camp2p_plan = derive_cam_hccl_buffer_plan_from_config(
        _vllm_config(
            connector="CAMP2pAFDConnector",
            dynamic_quant=0,
        ),
        _afd_config(connector="CAMP2pAFDConnector", role="ffn"),
    )

    assert async_plan.attention_buffer_size_mb == 181
    assert async_plan.ffn_buffer_size_mb == 826
    assert camp2p_plan.attention_buffer_size_mb == 35
    assert camp2p_plan.ffn_buffer_size_mb == 1100


@pytest.mark.parametrize(
    "connector",
    ["CAMAsyncAFDConnector", "CAMP2pAFDConnector"],
)
def test_cam_memory_headroom_warns_without_adjusting_utilization(
    connector,
    caplog,
):
    vllm_config = _vllm_config(
        connector=connector,
        gpu_memory_utilization=0.999,
    )

    with caplog.at_level(logging.WARNING):
        warn_if_cam_memory_headroom_is_low(
            vllm_config,
            _afd_config(connector=connector, role="attention"),
            64 * 1024**3,
        )

    assert vllm_config.cache_config.gpu_memory_utilization == 0.999
    assert "consider setting gpu_memory_utilization" in caplog.text
    assert "configured value 0.999000 is unchanged" in caplog.text


def test_cam_memory_headroom_does_not_warn_when_already_safe(caplog):
    vllm_config = _vllm_config(gpu_memory_utilization=0.75)

    with caplog.at_level(logging.WARNING):
        warn_if_cam_memory_headroom_is_low(
            vllm_config,
            _afd_config(connector="CAMAsyncAFDConnector", role="attention"),
            64 * 1024**3,
        )

    assert caplog.text == ""


def test_cam_buffer_rejects_unsupported_role_and_dynamic_quant():
    plan = derive_cam_hccl_buffer_plan(
        hidden_size=16,
        max_batch_tokens=8,
        num_npus_per_dp_group=1,
        topk=2,
        num_routed_experts=8,
        attention_rank_size=4,
        ffn_rank_size=2,
        compute_gate_on_attention=True,
        dynamic_quant=1,
    )

    with pytest.raises(ValueError, match="unsupported AFD role"):
        plan.buffer_size_mb_for_role("decode")
    with pytest.raises(ValueError, match="dynamic_quant must be 0 or 1"):
        derive_cam_hccl_buffer_plan(
            hidden_size=16,
            max_batch_tokens=8,
            num_npus_per_dp_group=1,
            topk=2,
            num_routed_experts=8,
            attention_rank_size=4,
            ffn_rank_size=2,
            compute_gate_on_attention=True,
            dynamic_quant=2,
        )
    with pytest.raises(ValueError, match="ffn_rank_size must be positive"):
        derive_cam_hccl_buffer_plan(
            hidden_size=16,
            max_batch_tokens=8,
            num_npus_per_dp_group=1,
            topk=2,
            num_routed_experts=8,
            attention_rank_size=4,
            ffn_rank_size=0,
            compute_gate_on_attention=True,
            dynamic_quant=1,
        )
