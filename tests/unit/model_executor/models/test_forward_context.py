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
    executor_source = Path(
        "afd_plugin/model_executor/models/deepseek_v2_async_cam_forward.py",
    ).read_text()
    forward_with_afd = source.split("    def forward_with_afd(", 1)[1].split(
        "    def forward_with_afd_v2(",
        1,
    )[0]
    forward_with_afd_v2 = source.split("    def forward_with_afd_v2(", 1)[1].split(
        "    def forward_with_afd_v3(",
        1,
    )[0]
    attention_gate_forward = executor_source.split(
        "def run_attention_gate_afd_forward(",
        1,
    )[1].split("def run_async_moe_ubatch_afd_forward(", 1)[0]

    assert 'if self.afd_role == "attention":' in source
    assert "def _forward_attention(" not in source
    assert "return self.forward_with_afd_v3(" in forward_with_afd
    assert "return self.forward_with_afd_v2(" in forward_with_afd
    assert "return run_attention_gate_afd_forward(" in forward_with_afd_v2
    assert "layer.compute_attn_output(" not in forward_with_afd
    assert "layer.compute_attn_output(" in attention_gate_forward
    assert "pending_ffn_recv" in attention_gate_forward
    assert "topk_weights" in attention_gate_forward
    assert "topk_ids" in attention_gate_forward


def test_deepseek_afd_attention_gate_can_force_balanced_topk_ids():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    compute_attn_output = source.split("    def compute_attn_output(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]

    assert "force_balanced_topk_ids_enabled" in source
    assert "def _force_balanced_topk_ids(" in source
    assert "topk_ids.copy_(balanced_topk_ids)" in source
    assert "topk_weights, topk_ids = afd_connector.select_experts(" in (
        compute_attn_output
    )
    assert "if force_balanced_topk_ids_enabled():" in compute_attn_output
    assert (
        compute_attn_output.index(
            "topk_weights, topk_ids = afd_connector.select_experts(",
        )
        < compute_attn_output.index("if force_balanced_topk_ids_enabled():")
        < compute_attn_output.index("topk_weights = topk_weights.to(torch.float32)")
    )


def test_deepseek_afd_gate_on_attention_keeps_dense_layers_local():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    executor_source = Path(
        "afd_plugin/model_executor/models/deepseek_v2_async_cam_forward.py",
    ).read_text()

    assert "self.is_moe_layer = _is_moe_layer(config, layer_idx)" in source
    assert "self.compute_gate_on_attention and not self.is_moe_layer" in source
    assert "if not layer.is_moe_layer:" in executor_source
    assert "self.is_dense_mlp_weight(name)" in source


def test_deepseek_async_moe_ubatching_runs_attention_inside_stage_context():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    executor_source = Path(
        "afd_plugin/model_executor/models/deepseek_v2_async_cam_forward.py",
    ).read_text()
    forward_with_afd_v3 = source.split("    def forward_with_afd_v3(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]
    async_ubatch_forward = executor_source.split(
        "def run_async_moe_ubatch_afd_forward(",
        1,
    )[1].split(
        "_MISSING_FORWARD_CONTEXT_ATTR = object()",
        1,
    )[0]

    assert "async_moe_ubatch_metadata" in forward_with_afd_v3
    assert "return run_async_moe_ubatch_afd_forward(" in forward_with_afd_v3
    assert "_log_async_moe_forward_step(" not in async_ubatch_forward
    assert "first_moe_layer = int(self.config.first_k_dense_replace)" in (
        async_ubatch_forward
    )
    assert "dense_end_layer = min(model.end_layer, first_moe_layer)" in (
        async_ubatch_forward
    )
    assert "stage_hidden_states = [" in async_ubatch_forward
    assert (
        "moe_layers = list(islice(model.layers, moe_start_layer, model.end_layer))"
        in async_ubatch_forward
    )
    assert "def compute_stage_attention(" in async_ubatch_forward
    assert "def send_stage_attention(" in async_ubatch_forward
    assert "def recv_stage_ffn(" in async_ubatch_forward
    assert "for moe_layer_offset in range(last_moe_layer_offset):" in (
        async_ubatch_forward
    )
    assert "def flush_pending_ffn_outputs()" not in async_ubatch_forward
    assert "torch.cat(stage_hidden_states, dim=0)" in async_ubatch_forward
    assert "_run_async_moe_ubatch_layer(" not in executor_source
    assert "_recv_async_moe_ubatch_outputs(" not in executor_source
    assert "forward_context.attn_metadata = attn_metadata[stage_idx]" in executor_source
    assert async_ubatch_forward.index(
        "with _use_async_moe_ubatch_forward_context(",
    ) < (
        async_ubatch_forward.index("layer.compute_attn_output(")
    )
    assert async_ubatch_forward.index(") = layer.compute_attn_output(") < (
        async_ubatch_forward.index("def send_stage_attention(")
    )
    assert async_ubatch_forward.index(
        "first_layer = moe_layers[0]",
    ) < async_ubatch_forward.index(
        "for moe_layer_offset in range(last_moe_layer_offset):",
    )
    assert async_ubatch_forward.index("recv_stage_ffn(0)") < (
        async_ubatch_forward.index(
            "send_stage_attention(\n                current_layer,\n                1",
        )
    )
    assert async_ubatch_forward.index("recv_stage_ffn(1)") < (
        async_ubatch_forward.index(
            "send_stage_attention(\n                next_layer,\n                0",
        )
    )
    assert async_ubatch_forward.index(
        "send_stage_attention(\n            last_layer,\n            1",
    ) < (async_ubatch_forward.rindex("recv_stage_ffn(1)"))


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
    assert "_gmmswigluquant_fusion_enabled()" in compute_moe
    assert "fusion=use_gmmswigluquant_fusion" in compute_moe
    assert "_compute_w8a8_shared_experts_from_int8(" in compute_moe
    assert "shared_input.dtype == torch.int8" in compute_moe
    assert "fusion=False" not in compute_moe


def test_deepseek_afd_ffn_compute_omits_stub_io_diagnostics():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    compute_ffn_output = source.split(
        "    def compute_ffn_output(",
        1,
    )[1].split("    def _compute_moe_with_attention_gate(", 1)[0]
    compute_moe = source.split(
        "    def _compute_moe_with_attention_gate(",
        1,
    )[1].split("\n\n@native.support_torch_compile", 1)[0]

    assert "camp2p_stub_io_enabled()" not in source
    assert "_log_ffn_compute_step(" not in compute_ffn_output
    assert '"dense_mlp_begin"' not in compute_ffn_output
    assert '"dense_scaling_begin"' not in compute_ffn_output
    assert "_log_ffn_compute_step(" not in compute_moe
    assert '"routed_scaling_begin"' not in compute_moe
    assert '"shared_scaling_begin"' not in compute_moe
