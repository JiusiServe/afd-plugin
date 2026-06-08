# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small Ascend runtime shims kept out of worker/model-runner modules."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from afd_plugin.config import AFDConfig, parse_afd_config

_PATCHES_APPLIED = False
_ASCEND_PLATFORM_PATCH_ATTR = "_afd_plugin_ascend_platform_patch_state"


def ensure_ascend_runtime_available() -> None:
    try:
        import vllm_ascend  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "AFD NPU runtime requires an importable vllm-ascend runtime",
        ) from exc


def apply_afd_ascend_patches_if_needed() -> None:
    """Apply plugin-owned, AFD-scoped Ascend patches."""

    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    _apply_afd_ascend_dbo_config_patch()
    _PATCHES_APPLIED = True


def init_ascend_workspace_for_afd(device: object, *, num_ubatches: int = 1) -> None:
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(device, int(num_ubatches))


def npu_afd_num_ubatches(vllm_config: object) -> int:
    parallel_config = vllm_config.parallel_config
    if parallel_config.use_ubatching:
        return int(parallel_config.num_ubatches)
    return 1


def fail_if_unsupported_npu_afd_features(vllm_config: object) -> None:
    """Fail fast for NPU AFD settings that are not currently supported."""

    afd_config = parse_afd_config(vllm_config)
    extra = afd_config.extra_config or {}
    if afd_config.connector == "afdasyncconnector":
        _fail_if_unsupported_npu_afd_async_features(vllm_config, afd_config)
        return

    if afd_config.compute_gate_on_attention or _truthy(
        extra.get("compute_gate_on_attention"),
    ):
        raise RuntimeError(
            "AFD NPU runtime does not support compute_gate_on_attention=true yet",
        )

    quant_mode = extra.get("quant_mode", 0)
    if quant_mode not in (None, "", 0, "0"):
        raise RuntimeError("AFD NPU runtime currently supports only quant_mode=0")

    if _truthy(extra.get("is_multistream")):
        raise RuntimeError("AFD NPU runtime does not support multistream yet")
    if _truthy(extra.get("is_attn_multistream")):
        raise RuntimeError(
            "AFD NPU runtime does not support attention multistream yet",
        )
    if _truthy(extra.get("is_ffn_multistream")):
        raise RuntimeError("AFD NPU runtime does not support FFN multistream yet")

    multistream_info = extra.get("multistream_info")
    if isinstance(multistream_info, Mapping):
        for key in ("enable", "enabled", "attn_enable", "ffn_enable"):
            if _truthy(multistream_info.get(key)):
                raise RuntimeError(
                    "AFD NPU runtime does not support multistream_info enabled",
                )

    if bool(vllm_config.parallel_config.use_ubatching) and (
        int(vllm_config.parallel_config.num_ubatches) != 2
    ):
        raise RuntimeError(
            "AFD NPU runtime supports exactly two ubatches when DBO is enabled",
        )

    if not bool(vllm_config.model_config.enforce_eager):
        _npu_aclgraph_mode_name(vllm_config)


def _fail_if_unsupported_npu_afd_async_features(
    vllm_config: object,
    afd_config: AFDConfig,
) -> None:
    extra = afd_config.extra_config or {}
    parallel_config = vllm_config.parallel_config
    if not bool(parallel_config.async_dp):
        raise RuntimeError("AFDAsyncConnector requires async_dp")
    if not bool(vllm_config.model_config.enforce_eager):
        raise RuntimeError(
            "AFDAsyncConnector supports only eager Attention/FFN execution",
        )
    if bool(parallel_config.use_ubatching):
        raise RuntimeError("AFDAsyncConnector does not support ubatching/DBO")
    if _truthy(extra.get("is_multistream")):
        raise RuntimeError("AFDAsyncConnector does not support multistream")
    if _truthy(extra.get("is_attn_multistream")):
        raise RuntimeError("AFDAsyncConnector does not support attention multistream")
    if _truthy(extra.get("is_ffn_multistream")):
        raise RuntimeError("AFDAsyncConnector does not support FFN multistream")

    multistream_info = extra.get("multistream_info")
    if isinstance(multistream_info, Mapping):
        for key in ("enable", "enabled", "attn_enable", "ffn_enable"):
            if _truthy(multistream_info.get(key)):
                raise RuntimeError(
                    "AFDAsyncConnector does not support multistream_info enabled",
                )

    quant_mode = extra.get("dynamicQuant", extra.get("quant_mode", 0))
    if quant_mode not in (None, "", 0, "0", 1, "1"):
        raise RuntimeError(
            "AFDAsyncConnector currently supports only quant_mode/dynamicQuant 0 or 1",
        )


def _npu_aclgraph_mode_name(vllm_config: object) -> str:
    return vllm_config.compilation_config.cudagraph_mode.name


def mirror_afd_metadata_on_forward_context(
    forward_context: object,
    afd_metadata: object,
) -> None:
    """Store AFD metadata in canonical kwargs and Ascend's mirrored attribute."""

    if forward_context.additional_kwargs is None:
        forward_context.additional_kwargs = {}
    forward_context.additional_kwargs["afd_metadata"] = afd_metadata
    forward_context.afd_metadata = afd_metadata


@contextmanager
def ascend_forward_context(
    *,
    vllm_config: object,
    afd_metadata: object | None = None,
    model_instance: object | None = None,
    num_tokens: int = 0,
    num_tokens_across_dp: object | None = None,
    aclgraph_runtime_mode: object | None = None,
) -> Iterator[object | None]:
    """Create the minimal forward context needed by connector-driven FFN steps."""

    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    except Exception:
        yield None
        return

    if aclgraph_runtime_mode is None:
        aclgraph_runtime_mode = CUDAGraphMode.NONE

    context_kwargs = {
        "attn_metadata": None,
        "vllm_config": vllm_config,
        "batch_descriptor": None,
        "aclgraph_runtime_mode": aclgraph_runtime_mode,
        "model_instance": model_instance,
        "afd_metadata": afd_metadata,
        "num_tokens": int(num_tokens),
        "num_tokens_across_dp": num_tokens_across_dp,
    }
    signature = inspect.signature(set_ascend_forward_context)
    if not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        context_kwargs = {
            key: value
            for key, value in context_kwargs.items()
            if key in signature.parameters
        }

    with set_ascend_forward_context(**context_kwargs):
        forward_context = get_forward_context()
        if afd_metadata is not None:
            mirror_afd_metadata_on_forward_context(forward_context, afd_metadata)
        yield forward_context


def ensure_vllm_config_has_afd_proxy(
    vllm_config: object,
    afd_config: AFDConfig | None = None,
) -> object | None:
    """Install an instance-local AFD proxy for vLLM-Ascend builds that read it.

    The plugin's public config channel remains ``additional_config["afd"]``.
    This shim only gives vLLM-Ascend code that still does ``vllm_config.afd_config``
    a read-only compatibility view.
    """

    config = afd_config or parse_afd_config(vllm_config, validate=False)
    if not config.enabled:
        return None
    proxy = _AscendAFDConfigProxy(config)
    vllm_config.afd_config = proxy
    return proxy


def _apply_afd_ascend_dbo_config_patch() -> None:
    """Preserve AFD DBO settings after vLLM-Ascend's compatibility reset.

    The reverted vLLM-Ascend baseline resets ``--enable-dbo`` and
    ``--ubatch-size`` for every Ascend run.  AFD owns the Ascend DBO path now,
    so this patch restores those fields only for AFD-enabled configs.
    """

    try:
        from vllm_ascend.platform import NPUPlatform
    except Exception:
        return

    if hasattr(NPUPlatform, _ASCEND_PLATFORM_PATCH_ATTR):
        return

    original_fix = NPUPlatform._fix_incompatible_config

    def patched_fix_incompatible_config(vllm_config: object) -> Any:
        saved = _snapshot_afd_dbo_config(vllm_config)
        result = original_fix(vllm_config)
        if saved is not None:
            _restore_afd_dbo_config(vllm_config, saved)
        return result

    NPUPlatform._fix_incompatible_config = staticmethod(patched_fix_incompatible_config)
    setattr(NPUPlatform, _ASCEND_PLATFORM_PATCH_ATTR, original_fix)


def _snapshot_afd_dbo_config(vllm_config: object) -> dict[str, Any] | None:
    if not _is_afd_config_enabled(vllm_config):
        return None
    parallel_config = vllm_config.parallel_config
    return {
        "enable_dbo": parallel_config.enable_dbo,
        "use_ubatching": parallel_config.use_ubatching,
        "num_ubatches": parallel_config.num_ubatches,
        "ubatch_size": parallel_config.ubatch_size,
    }


def _restore_afd_dbo_config(vllm_config: object, saved: dict[str, Any]) -> None:
    parallel_config = vllm_config.parallel_config
    if not (
        saved["enable_dbo"]
        or saved["use_ubatching"]
        or int(saved["ubatch_size"] or 0) != 0
    ):
        return
    parallel_config.enable_dbo = saved["enable_dbo"]
    parallel_config.ubatch_size = saved["ubatch_size"]


def _is_afd_config_enabled(vllm_config: object) -> bool:
    try:
        return parse_afd_config(vllm_config, validate=False).enabled
    except Exception:
        return False


@dataclass(frozen=True)
class _AscendAFDConfigProxy:
    _config: AFDConfig

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def afd_extra_config(self) -> dict[str, Any]:
        return self._config.extra_config

    @property
    def afd_connector(self) -> str:
        return self._config.connector

    @property
    def afd_role(self) -> str:
        return self._config.role

    @property
    def afd_port(self) -> int:
        return self._config.port

    @property
    def afd_host(self) -> str:
        return self._config.host

    @property
    def is_attention_server(self) -> bool:
        return self._config.role == "attention"

    @property
    def is_ffn_server(self) -> bool:
        return self._config.role == "ffn"

    @property
    def compute_gate_on_attention(self) -> bool:
        return bool(
            self._config.compute_gate_on_attention
            or self._config.extra_config.get("compute_gate_on_attention", False),
        )

    @property
    def quant_mode(self) -> int:
        return int(self._config.extra_config.get("quant_mode", 0) or 0)

    @property
    def is_multistream(self) -> bool:
        return bool(self._config.extra_config.get("is_multistream", False))

    @property
    def is_attn_multistream(self) -> bool:
        return bool(self._config.extra_config.get("is_attn_multistream", False))

    @property
    def is_ffn_multistream(self) -> bool:
        return bool(self._config.extra_config.get("is_ffn_multistream", False))

    @property
    def multistream_info(self) -> dict[str, Any]:
        value = self._config.extra_config.get("multistream_info", {})
        return value if isinstance(value, dict) else {}

    def compute_hash(self) -> str:
        return self._config.compute_hash()


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def fix_all2all_backend_for_afd(vllm_config: Any) -> None:
    """Mirror vllm-ascend's platform.py all2all_backend override.

    vllm-ascend sets ``all2all_backend = "flashinfer_all2allv"`` when
    ``enable_sp`` is False, but only when ``worker_cls == "auto"``.
    AFD workers use a custom ``worker_cls``, so this override never fires
    and ``all2all_backend`` keeps its default ``"allgather_reducescatter"``.
    That value triggers ``use_sequence_parallel_moe = True`` (because
    ``enable_expert_parallel=True``, ``tp_size > 1``, ``dp_size > 1``),
    which incorrectly splits MoE tokens via ``sequence_parallel_chunk``,
    producing wrong output.

    This function applies the same fix for AFD workers.
    """
    parallel_config = vllm_config.parallel_config
    if not vllm_config.compilation_config.pass_config.enable_sp:
        current = getattr(parallel_config, "all2all_backend", None)
        if current != "flashinfer_all2allv":
            parallel_config.all2all_backend = "flashinfer_all2allv"


__all__ = [
    "apply_afd_ascend_patches_if_needed",
    "ascend_forward_context",
    "ensure_ascend_runtime_available",
    "ensure_vllm_config_has_afd_proxy",
    "fail_if_unsupported_npu_afd_features",
    "fix_all2all_backend_for_afd",
    "init_ascend_workspace_for_afd",
    "mirror_afd_metadata_on_forward_context",
    "npu_afd_num_ubatches",
]
