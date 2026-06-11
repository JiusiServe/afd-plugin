from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context


def test_get_afd_metadata_from_additional_kwargs():
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": {"stage": 0}},
    )

    assert get_afd_metadata_from_forward_context(forward_context) == {"stage": 0}


def test_deepseek_afd_wrapper_keeps_full_model_compile_enabled():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert "@native.support_torch_compile\nclass AFDDeepseekV2Model" in source
    assert "from __future__ import annotations" not in source
    assert "self.do_not_compile = True" not in source


def test_deepseek_afd_wrapper_treats_index_topk_as_optional():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'self.is_v32 = hasattr(config, "index_topk")' in source
    assert "self.is_v32 = config.index_topk is not None" not in source
    assert "topk_tokens = config.index_topk" in source


def test_deepseek_afd_wrapper_treats_llama_4_scaling_as_optional():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'getattr(self.config, "llama_4_scaling", None)' in source
    assert "self.config.llama_4_scaling" not in source


def test_deepseek_afd_attention_path_can_compute_gate_before_send():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    forward_with_afd = source.split("    def forward_with_afd(", 1)[1].split(
        "    def forward_with_afd_v2(",
        1,
    )[0]
    forward_with_afd_v2 = source.split("    def forward_with_afd_v2(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]

    assert 'if self.afd_role == "attention":' in source
    assert "def _forward_attention(" not in source
    assert "return self.forward_with_afd_v2(" in forward_with_afd
    assert "layer.compute_attn_output(" not in forward_with_afd
    assert "layer.compute_attn_output(" in forward_with_afd_v2
    assert "pending_ffn_recv" in forward_with_afd_v2
    assert "topk_weights" in forward_with_afd_v2
    assert "topk_ids" in forward_with_afd_v2


def test_deepseek_afd_gate_on_attention_keeps_dense_layers_local():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert "self.is_moe_layer = _is_moe_layer(config, layer_idx)" in source
    assert "self.compute_gate_on_attention and not self.is_moe_layer" in source
    assert "if not layer.is_moe_layer:" in source
    assert "self.is_dense_mlp_weight(name)" in source


def test_deepseek_afd_ffn_path_reuses_ascend_moe_mlp_after_attention_gate():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    compute_ffn_output = source.split(
        "    def compute_ffn_output(",
        1,
    )[1].split("    def _compute_moe_with_attention_gate(", 1)[0]
    compute_moe = source.split(
        "    def _compute_moe_with_attention_gate(",
        1,
    )[1].split("\n\n@native.support_torch_compile", 1)[0]

    assert "_compute_moe_with_attention_gate(" in compute_ffn_output
    assert "AFDFFNOutput(" in compute_moe
    assert "MoEMlpComputeInput(" in compute_moe
    assert "unified_apply_mlp(" in compute_moe
    assert "quant_type == QuantType.W8A8" in compute_moe
    assert "w13_weight_scale_fp32" in compute_moe
    assert "w13_weight_scale_fp32_list" in compute_moe
    assert "w2_weight_scale_list" in compute_moe
    assert "MoEQuantParams(quant_type=quant_type)" in compute_moe
