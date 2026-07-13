from __future__ import annotations

import importlib.metadata
from pathlib import Path

import afd_plugin
from afd_plugin.compat import is_vllm_version_supported


def test_package_import_is_cpu_safe():
    assert afd_plugin.__version__
    assert afd_plugin.AFDConfig().connector == "P2pNcclAFDConnector"


def test_register_afd_is_idempotent():
    afd_plugin.register_afd()
    afd_plugin.register_afd()


def test_deepseek_afd_model_registration_paths_are_lazy_strings():
    registrations = afd_plugin._DEEPSEEK_MODEL_REGISTRATIONS

    assert registrations["DeepseekV2ForCausalLM"] == (
        "afd_plugin.model_executor.models.deepseek_v2:AFDDeepseekV2ForCausalLM"
    )
    assert registrations["DeepseekV3ForCausalLM"] == (
        "afd_plugin.model_executor.models.deepseek_v2:AFDDeepseekV3ForCausalLM"
    )
    assert registrations["DeepseekV32ForCausalLM"] == (
        "afd_plugin.model_executor.models.deepseek_v2:AFDDeepseekV3ForCausalLM"
    )


def test_entry_point_is_registered():
    entry_points = importlib.metadata.entry_points(group="vllm.general_plugins")
    matches = [ep for ep in entry_points if ep.name == "afd"]
    assert matches
    assert matches[0].value == "afd_plugin:register_afd"


def test_connectors_export_attn_output_without_recv_alias():
    root = Path(__file__).resolve().parents[3]
    metadata_source = (root / "afd_plugin/connectors/metadata.py").read_text()
    namespace_source = (root / "afd_plugin/connectors/__init__.py").read_text()

    assert "class AFDA2FTransferPayload:" in metadata_source
    assert '"AFDA2FTransferPayload"' in metadata_source
    assert "AFDRecvOutput" not in metadata_source
    assert "AFDA2FTransferPayload," in namespace_source
    assert '"AFDA2FTransferPayload"' in namespace_source
    assert "AFDRecvOutput" not in namespace_source


def test_vllm_version_support_is_exact_target():
    assert is_vllm_version_supported("0.19.1")
    assert not is_vllm_version_supported("0.19.0")
    assert not is_vllm_version_supported("0.19.2")
