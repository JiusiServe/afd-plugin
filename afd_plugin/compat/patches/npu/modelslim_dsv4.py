# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compatibility mapping for DSV4 ModelSlim quantization.

DSV4 checkpoints use names such as ``attn`` and ``ffn`` while the runtime
uses the corresponding Hugging Face names. Resolve a runtime prefix to the
matching checkpoint key without changing the quantization description.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_PATCH_ATTR = "_afd_dsv4_modelslim_patch"

_DSV4_RUNTIME_TO_CHECKPOINT = (
    (".self_attn.", ".attn."),
    (".post_attention_layernorm.", ".ffn_norm."),
    (".input_layernorm.", ".attn_norm."),
    (".mlp.", ".ffn."),
    (".gate_proj.", ".w1."),
    (".up_proj.", ".w3."),
    (".down_proj.", ".w2."),
)


def _replace_once(value: str, old: str, new: str) -> str:
    return value.replace(old, new, 1)


def _runtime_name_variants(prefix: str) -> list[str]:
    """Return DSV4 runtime/checkpoint spelling variants for ``prefix``."""

    variants = [prefix]
    for old, new in _DSV4_RUNTIME_TO_CHECKPOINT:
        variants.append(_replace_once(prefix, old, new))
    return variants


def _has_quant_weight(
    quant_description: Mapping[str, Any],
    prefix: str,
    packed_modules_mapping: Mapping[str, list[str]],
) -> bool:
    """Check a linear/MoE prefix without indexing a missing description key."""

    projection = prefix.rsplit(".", 1)[-1]
    if projection in packed_modules_mapping:
        return all(
            f"{prefix.replace(projection, shard)}.weight" in quant_description
            for shard in packed_modules_mapping[projection]
        )
    return f"{prefix}.weight" in quant_description


def _resolve_dsv4_quant_prefix(
    prefix: str,
    quant_description: Mapping[str, Any],
    packed_modules_mapping: Mapping[str, list[str]],
) -> str:
    """Resolve a DSV4 runtime prefix against ModelSlim description keys."""

    seen: set[str] = set()
    for candidate in _runtime_name_variants(prefix):
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_quant_weight(quant_description, candidate, packed_modules_mapping):
            return candidate
    return prefix


def _patch_dsv4_packed_mapping(modelslim_module: Any) -> None:
    """Use DSV4 checkpoint shard names for fused runtime modules."""

    mapping = modelslim_module.packed_modules_model_mapping.setdefault(
        "deepseek_v4", {}
    )
    mapping.update(
        {
            "gate_up_proj": ["w1", "w3"],
            "fused_wqa_wkv": ["wq_a", "wkv"],
            "experts": ["experts.0.w1", "experts.0.w3", "experts.0.w2"],
        }
    )


def apply_dsv4_modelslim_patch() -> bool:
    """Install the idempotent vLLM-Ascend ModelSlim prefix resolver."""

    try:
        from vllm_ascend.quantization import modelslim_config
    except ImportError:
        return False

    if getattr(modelslim_config, _PATCH_ATTR, None) is not None:
        return True

    original_mapper = modelslim_config.AscendModelSlimConfig.quant_prefix_mapper
    _patch_dsv4_packed_mapping(modelslim_config)

    def quant_prefix_mapper(self, model_type: str, prefix: str) -> str:
        mapped_prefix = original_mapper(self, model_type, prefix)
        if model_type != "deepseek_v4":
            return mapped_prefix

        return _resolve_dsv4_quant_prefix(
            mapped_prefix,
            getattr(self, "quant_description", {}),
            getattr(self, "packed_modules_mapping", {}),
        )

    modelslim_config.AscendModelSlimConfig.quant_prefix_mapper = quant_prefix_mapper
    setattr(modelslim_config, _PATCH_ATTR, original_mapper)
    return True


__all__ = ["apply_dsv4_modelslim_patch", "_resolve_dsv4_quant_prefix"]
