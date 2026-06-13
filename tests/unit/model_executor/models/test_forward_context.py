from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from afd_plugin.model_executor.models import (
    ASYNC_MOE_UBATCH_METADATA_KEY,
    get_afd_metadata_from_forward_context,
    get_async_moe_ubatch_metadata_from_forward_context,
)


def test_get_afd_metadata_from_additional_kwargs():
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": {"stage": 0}},
    )

    assert get_afd_metadata_from_forward_context(forward_context) == {"stage": 0}


def test_get_async_moe_ubatch_metadata_from_additional_kwargs():
    sidecar = {"ubatch_slices": ["stage0", "stage1"]}
    forward_context = SimpleNamespace(
        additional_kwargs={ASYNC_MOE_UBATCH_METADATA_KEY: sidecar},
    )

    assert (
        get_async_moe_ubatch_metadata_from_forward_context(forward_context) is sidecar
    )


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
        "    def forward_with_afd_v3(",
        1,
    )[0]

    assert 'if self.afd_role == "attention":' in source
    assert "def _forward_attention(" not in source
    assert "return self.forward_with_afd_v3(" in forward_with_afd
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


def test_deepseek_async_moe_ubatching_runs_attention_inside_stage_context():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    forward_with_afd_v3 = source.split("    def forward_with_afd_v3(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]

    assert "from afd_plugin.envs import AFD_CAMP2P_STUB_IO" in source
    assert "_AFD_ASYNC_MOE_FORWARD_LOG_ENV = AFD_CAMP2P_STUB_IO" in source
    assert "async_moe_ubatch_metadata" in forward_with_afd_v3
    assert "_log_async_moe_forward_step(" in forward_with_afd_v3
    assert '"enter"' in forward_with_afd_v3
    assert '"attention_begin"' in forward_with_afd_v3
    assert '"send_end"' in forward_with_afd_v3
    assert "first_moe_layer = int(self.config.first_k_dense_replace)" in (
        forward_with_afd_v3
    )
    assert "dense_end_layer = min(self.end_layer, first_moe_layer)" in (
        forward_with_afd_v3
    )
    assert "stage_hidden_states = [" in forward_with_afd_v3
    assert "moe_layers = list(islice(self.layers, moe_start_layer, self.end_layer))" in (
        forward_with_afd_v3
    )
    assert "def compute_stage_attention(" in forward_with_afd_v3
    assert "def send_stage_attention(" in forward_with_afd_v3
    assert "def recv_stage_ffn(" in forward_with_afd_v3
    assert "for moe_layer_offset in range(last_moe_layer_offset):" in (
        forward_with_afd_v3
    )
    assert "def flush_pending_ffn_outputs()" not in forward_with_afd_v3
    assert "torch.cat(stage_hidden_states, dim=0)" in forward_with_afd_v3
    assert "_run_async_moe_ubatch_layer(" not in source
    assert "_recv_async_moe_ubatch_outputs(" not in source
    assert "forward_context.attn_metadata = attn_metadata[stage_idx]" in source
    assert forward_with_afd_v3.index("with _use_async_moe_ubatch_forward_context(") < (
        forward_with_afd_v3.index("layer.compute_attn_output(")
    )
    assert forward_with_afd_v3.index(") = layer.compute_attn_output(") < (
        forward_with_afd_v3.index("def send_stage_attention(")
    )
    assert forward_with_afd_v3.index("first_layer = moe_layers[0]") < (
        forward_with_afd_v3.index("for moe_layer_offset in range(last_moe_layer_offset):")
    )
    assert forward_with_afd_v3.index("recv_stage_ffn(current_layer, 0") < (
        forward_with_afd_v3.index("send_stage_attention(\n                current_layer,\n                1")
    )
    assert forward_with_afd_v3.index("recv_stage_ffn(current_layer, 1") < (
        forward_with_afd_v3.index("send_stage_attention(\n                next_layer,\n                0")
    )
    assert forward_with_afd_v3.index("send_stage_attention(\n            last_layer,\n            1") < (
        forward_with_afd_v3.index("recv_stage_ffn(last_layer, 1, \"final\")")
    )


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


def test_deepseek_afd_ffn_compute_has_stub_io_diagnostics():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    compute_ffn_output = source.split(
        "    def compute_ffn_output(",
        1,
    )[1].split("    def _compute_moe_with_attention_gate(", 1)[0]
    compute_moe = source.split(
        "    def _compute_moe_with_attention_gate(",
        1,
    )[1].split("\n\n@native.support_torch_compile", 1)[0]

    assert "camp2p_stub_io_enabled()" in source
    assert "_log_ffn_compute_step(" in compute_ffn_output
    assert '"decoder_layer_enter"' in compute_ffn_output
    assert '"attention_gate_moe_begin"' in compute_ffn_output
    assert '"dense_mlp_begin"' not in compute_ffn_output
    assert '"dense_scaling_begin"' not in compute_ffn_output
    assert '"moe_weights_begin"' in compute_moe
    assert '"shared_experts_begin"' in compute_moe
    assert '"routed_unified_apply_mlp_begin"' in compute_moe
    assert '"routed_scaling_begin"' not in compute_moe
    assert '"shared_scaling_begin"' not in compute_moe
    assert '"moe_attention_gate_exit"' in compute_moe
