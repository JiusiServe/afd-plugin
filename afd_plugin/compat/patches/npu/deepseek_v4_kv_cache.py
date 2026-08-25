# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek-V4 KV-cache compatibility for the Ascend v0.26 runtime.

The Ascend hybrid-cache grouping helper needs every DSV4 MLA/SWA page size
when it builds a packed cache layout.  Without that normalization its output
can expose multiple descriptors for one packed backing allocation to the
Ascend allocator, which then allocates each descriptor independently.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from typing import Any

logger = logging.getLogger(__name__)

_GROUPING_PATCHED = False
_ALLOCATOR_PATCHED = False


def _is_uniform_mla_group(group: Any) -> bool:
    specs = getattr(group, "kv_cache_specs", None)
    return bool(specs and hasattr(group, "get_page_sizes")) and all(
        hasattr(spec, "compress_ratio") and hasattr(spec, "page_size_bytes")
        for spec in specs.values()
    )


def apply_deepseek_v4_hybrid_kv_cache_group_patch() -> bool:
    """Normalize DSV4 hybrid KV page-size buckets before planning.

    This is independent of speculative decoding: target-only DSV4 has the
    same full-MLA and sliding-window cache-spec topology.
    """

    global _GROUPING_PATCHED
    try:
        import vllm.v1.core.kv_cache_utils as kv_cache_utils
    except (ImportError, ModuleNotFoundError):
        return True

    with suppress(ImportError, ModuleNotFoundError):
        importlib.import_module("vllm_ascend.patch.platform.patch_kv_cache_utils")

    original = getattr(kv_cache_utils, "_get_kv_cache_groups_uniform_groups", None)
    if original is None:
        return True
    if getattr(original, "_afd_dsv4_page_size_patch", False):
        _GROUPING_PATCHED = True
        return True

    def get_kv_cache_groups_uniform_groups(
        grouped_specs: list[Any],
        _original: Callable[..., Any] = original,
    ) -> Any:
        if len(grouped_specs) <= 2 or not all(
            _is_uniform_mla_group(grouped_specs[index]) for index in (0, 1)
        ):
            return _original(grouped_specs)

        full_mla_spec = grouped_specs[0]
        page_sizes = set(full_mla_spec.get_page_sizes())
        for group in grouped_specs[1:]:
            page_sizes.update(group.get_page_sizes())
        if page_sizes.issubset(set(full_mla_spec.get_page_sizes())):
            return _original(grouped_specs)

        spec_type = type(full_mla_spec)

        class FullMLAWithAllPageSizes(spec_type):
            def get_page_sizes(self) -> list[int]:
                return sorted(page_sizes)

        patched_specs = list(grouped_specs)
        patched_specs[0] = FullMLAWithAllPageSizes(
            block_size=full_mla_spec.block_size,
            kv_cache_specs=full_mla_spec.kv_cache_specs,
        )
        logger.info(
            "AFD DSV4 normalized Ascend KV page buckets: %s -> %s",
            full_mla_spec.get_page_sizes(),
            sorted(page_sizes),
        )
        return _original(patched_specs)

    get_kv_cache_groups_uniform_groups._afd_dsv4_page_size_patch = True  # type: ignore[attr-defined]
    kv_cache_utils._get_kv_cache_groups_uniform_groups = (
        get_kv_cache_groups_uniform_groups
    )
    _GROUPING_PATCHED = True
    return True


def apply_deepseek_v4_ascend_allocator_patch() -> bool:
    """Let Ascend allocate DSV4 state-only cache entries.

    Ascend's allocator recognizes compressed cache entries by an ``attn``
    substring.  DSV4 compressor/SWA state entries do not have it, so alias
    them internally only while the upstream allocator constructs the tensors.
    """

    global _ALLOCATOR_PATCHED
    try:
        import vllm_ascend.worker.model_runner_v1 as model_runner_module
    except (ImportError, ModuleNotFoundError):
        return True

    runner_cls = getattr(model_runner_module, "NPUModelRunner", None)
    original = getattr(runner_cls, "_allocate_kv_cache_tensors", None)
    if original is None:
        return True
    if getattr(original, "_afd_dsv4_allocator_patch", False):
        _ALLOCATOR_PATCHED = True
        return True

    def allocate_kv_cache_tensors(
        self: Any,
        kv_cache_config: Any,
        _original: Callable[..., Any] = original,
    ) -> Any:
        state_names: set[str] = set()
        for group in kv_cache_config.kv_cache_groups:
            specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
            if specs is None:
                continue
            for name, spec in specs.items():
                if (
                    type(spec).__name__ == "AscendSlidingWindowMLASpec"
                    and "attn" not in name
                ):
                    state_names.add(name)
        if not state_names:
            return _original(self, kv_cache_config)

        aliases = {name: f"{name}.afd_attn_cache" for name in state_names}
        def alias(name: str) -> str:
            return aliases.get(name, name)

        patched_groups = []
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            specs = getattr(spec, "kv_cache_specs", None)
            if specs is not None:
                spec = replace(
                    spec,
                    kv_cache_specs={
                        alias(name): value for name, value in specs.items()
                    },
                )
            patched_groups.append(
                replace(
                    group,
                    layer_names=[alias(name) for name in group.layer_names],
                    kv_cache_spec=spec,
                )
            )
        patched_tensors = [
            replace(tensor, shared_by=[alias(name) for name in tensor.shared_by])
            for tensor in kv_cache_config.kv_cache_tensors
        ]
        patched_config = replace(
            kv_cache_config,
            kv_cache_groups=patched_groups,
            kv_cache_tensors=patched_tensors,
        )
        raw_tensors = _original(self, patched_config)
        return {aliases.get(name, name): tensor for name, tensor in raw_tensors.items()}

    allocate_kv_cache_tensors._afd_dsv4_allocator_patch = True  # type: ignore[attr-defined]
    runner_cls._allocate_kv_cache_tensors = allocate_kv_cache_tensors
    _ALLOCATOR_PATCHED = True
    return True


def apply_deepseek_v4_kv_cache_patches() -> None:
    """Install the DSV4-only Ascend KV compatibility patches."""

    apply_deepseek_v4_hybrid_kv_cache_group_patch()
    apply_deepseek_v4_ascend_allocator_patch()
