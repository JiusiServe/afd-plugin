from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from afd_plugin.compat.patches.omni_server import _is_ffn_args


def _install_fake_resolve_deploy_yaml(monkeypatch, mapping):
    """Inject a fake ``vllm_omni.config.stage_config.resolve_deploy_yaml``.

    ``mapping`` maps a deploy-config path to the raw dict it should resolve to,
    letting us exercise the YAML-inspection branch without pyyaml or a real
    vllm-omni install.
    """
    omni_pkg = types.ModuleType("vllm_omni")
    config_pkg = types.ModuleType("vllm_omni.config")
    stage_mod = types.ModuleType("vllm_omni.config.stage_config")

    def resolve_deploy_yaml(path):
        return mapping[path]

    stage_mod.resolve_deploy_yaml = resolve_deploy_yaml
    monkeypatch.setitem(sys.modules, "vllm_omni", omni_pkg)
    monkeypatch.setitem(sys.modules, "vllm_omni.config", config_pkg)
    monkeypatch.setitem(sys.modules, "vllm_omni.config.stage_config", stage_mod)


def test_top_level_additional_config_dict_ffn():
    args = SimpleNamespace(additional_config={"afd": {"enabled": True, "role": "ffn"}})
    assert _is_ffn_args(args) is True


def test_top_level_additional_config_json_string_ffn():
    args = SimpleNamespace(additional_config='{"afd": {"enabled": true, "role": "ffn"}}')
    assert _is_ffn_args(args) is True


def test_top_level_attention_role_is_not_ffn():
    args = SimpleNamespace(additional_config={"afd": {"enabled": True, "role": "attention"}})
    assert _is_ffn_args(args) is False


def test_deploy_config_stage_ffn(monkeypatch):
    _install_fake_resolve_deploy_yaml(
        monkeypatch,
        {"ffn.yaml": {"stages": [{"stage_id": 0, "additional_config": {"afd": {"enabled": True, "role": "ffn"}}}]}},
    )
    args = SimpleNamespace(deploy_config="ffn.yaml", stage_configs_path=None)
    assert _is_ffn_args(args) is True


def test_deploy_config_stage_attention_is_not_ffn(monkeypatch):
    _install_fake_resolve_deploy_yaml(
        monkeypatch,
        {"attn.yaml": {"stages": [{"stage_id": 0, "additional_config": {"afd": {"enabled": True, "role": "attention"}}}]}},
    )
    args = SimpleNamespace(deploy_config="attn.yaml", stage_configs_path=None)
    assert _is_ffn_args(args) is False


def test_stage_configs_path_ffn(monkeypatch):
    _install_fake_resolve_deploy_yaml(
        monkeypatch,
        {"legacy.yaml": {"stages": [{"stage_id": 0, "additional_config": {"afd": {"enabled": True, "role": "ffn"}}}]}},
    )
    args = SimpleNamespace(deploy_config=None, stage_configs_path="legacy.yaml")
    assert _is_ffn_args(args) is True


def test_no_afd_config_is_not_ffn():
    args = SimpleNamespace(deploy_config=None, stage_configs_path=None, additional_config=None)
    assert _is_ffn_args(args) is False
