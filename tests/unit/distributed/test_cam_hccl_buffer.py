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
    num_npus_per_dp_group: int = 8,
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
        scheduler_config=SimpleNamespace(max_num_batched_tokens=140000),
        cache_config=SimpleNamespace(
            gpu_memory_utilization=gpu_memory_utilization,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=7168,
                num_experts_per_tok=8,
            ),
        ),
    )


def _afd_config(*, connector: str, role: str) -> AFDConfig:
    return AFDConfig(
        connector=connector,
        role=role,
        num_attention_ranks=24,
        num_ffn_ranks=8,
    )


def test_deepseek_v32_cam_buffers_are_derived_independently_by_role():
    plan = derive_cam_hccl_buffer_plan(
        hidden_size=7168,
        max_batch_tokens=140000,
        num_npus_per_dp_group=8,
        topk=8,
        attention_rank_size=24,
        dynamic_quant=1,
    )

    assert plan.attention_required_bytes == 2_257_920_000
    assert plan.moe_required_bytes == 2_593_920_000
    assert plan.attention_buffer_size_mb == 2369
    assert plan.ffn_buffer_size_mb == 2722
    assert plan.buffer_size_mb_for_role("attention") == 2369
    assert plan.buffer_size_mb_for_role("ffn") == 2722


def test_moe_buffer_has_no_topk_multiplier_and_uses_non_quantized_width():
    low_topk = derive_cam_hccl_buffer_plan(
        hidden_size=7168,
        max_batch_tokens=140000,
        num_npus_per_dp_group=8,
        topk=1,
        attention_rank_size=24,
        dynamic_quant=0,
    )
    high_topk = derive_cam_hccl_buffer_plan(
        hidden_size=7168,
        max_batch_tokens=140000,
        num_npus_per_dp_group=8,
        topk=8,
        attention_rank_size=24,
        dynamic_quant=0,
    )

    assert low_topk.moe_required_bytes == 5_160_960_000
    assert high_topk.moe_required_bytes == low_topk.moe_required_bytes
    assert high_topk.ffn_buffer_size_mb == 5415
    assert high_topk.attention_required_bytes > low_topk.attention_required_bytes


def test_cam_buffer_rounds_partial_per_npu_token_and_mb_up():
    plan = derive_cam_hccl_buffer_plan(
        hidden_size=3,
        max_batch_tokens=5,
        num_npus_per_dp_group=2,
        topk=1,
        attention_rank_size=1,
        dynamic_quant=1,
    )

    assert plan.attention_required_bytes == 36
    assert plan.moe_required_bytes == 18_528
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

    assert async_plan.attention_buffer_size_mb == 2369
    assert async_plan.ffn_buffer_size_mb == 2722
    assert camp2p_plan.attention_buffer_size_mb == 2369
    assert camp2p_plan.ffn_buffer_size_mb == 5415


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
        gpu_memory_utilization=0.95,
    )

    with caplog.at_level(logging.WARNING):
        warn_if_cam_memory_headroom_is_low(
            vllm_config,
            _afd_config(connector=connector, role="attention"),
            64 * 1024**3,
        )

    assert vllm_config.cache_config.gpu_memory_utilization == 0.95
    assert "consider setting gpu_memory_utilization" in caplog.text
    assert "configured value 0.950000 is unchanged" in caplog.text


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
        attention_rank_size=4,
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
            attention_rank_size=4,
            dynamic_quant=2,
        )
    with pytest.raises(ValueError, match="num_npus_per_dp_group must be positive"):
        derive_cam_hccl_buffer_plan(
            hidden_size=16,
            max_batch_tokens=8,
            num_npus_per_dp_group=0,
            topk=2,
            attention_rank_size=4,
            dynamic_quant=1,
        )
