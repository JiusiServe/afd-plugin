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


def test_deepseek_afd_attention_path_uses_decoder_layer_forward():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    forward_with_afd = source.split("    def forward_with_afd(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]

    assert 'if self.afd_role == "attention":' in source
    assert "def _forward_attention(" not in source
    assert "hidden_states, residual = layer(\n" in forward_with_afd
    assert "layer.compute_attn_output(" not in forward_with_afd
