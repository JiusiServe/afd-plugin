from __future__ import annotations

import importlib.metadata

import afd_plugin
from afd_plugin.compat import is_vllm_version_supported


def test_package_import_is_cpu_safe():
    assert afd_plugin.__version__
    assert afd_plugin.AFDConfig().connector == "p2pconnector"


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


def test_vllm_version_support_is_exact_target():
    assert is_vllm_version_supported("0.23.0")
    assert not is_vllm_version_supported("0.22.0")
    assert not is_vllm_version_supported("0.23.1")
