# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Debug helpers for Ascend PCP metadata handling."""

from __future__ import annotations

import copy
import os
from typing import Any


def debug_pcp_metadata_enabled() -> bool:
    return os.getenv("AFD_DEBUG_PCP_METADATA", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def debug_slice_summary(value: Any) -> tuple[Any, Any, Any]:
    return (
        getattr(value, "start", None),
        getattr(value, "stop", None),
        getattr(value, "step", None),
    )


def debug_scalar(value: Any) -> Any:
    try:
        if hasattr(value, "item"):
            return value.item()
        return int(value)
    except Exception:
        return repr(value)


def debug_value_summary(value: Any, *, limit: int = 8) -> Any:
    if value is None:
        return None
    summary: dict[str, Any] = {"type": type(value).__name__}
    if hasattr(value, "shape"):
        try:
            summary["shape"] = tuple(int(dim) for dim in value.shape)
        except Exception:
            summary["shape"] = repr(value.shape)
    if hasattr(value, "dtype"):
        summary["dtype"] = str(value.dtype)
    if hasattr(value, "device"):
        summary["device"] = str(value.device)

    try:
        flat = value.detach().flatten().to("cpu") if hasattr(value, "detach") else None
        if flat is not None:
            total = int(flat.numel())
            summary["numel"] = total
            head_count = min(limit, total)
            summary["head"] = [debug_scalar(item) for item in flat[:head_count]]
            if total > limit:
                tail_count = min(limit, total)
                summary["tail"] = [debug_scalar(item) for item in flat[-tail_count:]]
            return summary
    except Exception as exc:
        summary["values_error"] = repr(exc)
        return summary

    try:
        if hasattr(value, "reshape") and hasattr(value, "size"):
            flat = value.reshape(-1)
            total = int(flat.size)
            summary["size"] = total
            head_count = min(limit, total)
            summary["head"] = [debug_scalar(item) for item in flat[:head_count]]
            if total > limit:
                summary["tail"] = [debug_scalar(item) for item in flat[-limit:]]
            return summary
    except Exception as exc:
        summary["values_error"] = repr(exc)
        return summary

    if isinstance(value, list | tuple):
        total = len(value)
        summary["len"] = total
        summary["head"] = [debug_scalar(item) for item in value[:limit]]
        if total > limit:
            summary["tail"] = [debug_scalar(item) for item in value[-limit:]]
        return summary

    if isinstance(value, dict):
        summary["len"] = len(value)
        summary["keys"] = list(value.keys())[:limit]
        return summary

    return value


def debug_pcp_metadata_summary(pcp_metadata: Any) -> Any:
    if pcp_metadata is None:
        return None
    fields = (
        "query_start_loc",
        "query_start_loc_cpu",
        "seq_lens",
        "seq_lens_cpu",
        "num_computed_tokens_cpu",
        "q_head_idx_tensor",
        "q_tail_idx_tensor",
        "q_full_idx",
        "pcp_allgather_restore_idx",
        "pcp_unpad_mask",
        "pcp_fa_query_idx",
        "pcp_enter_fa_restore_idx",
        "pcp_exit_fa_scatter_idx",
        "num_computed_tokens_of_pcp_dcp",
        "query_lens_pcp_full_cpu",
        "num_actual_tokens_pcp_padded",
        "actual_seq_lengths_q",
        "actual_seq_lengths_kv",
        "actual_seq_lengths_query",
        "actual_seq_lengths_key",
        "slot_mapping_cp",
    )
    summary: dict[str, Any] = {"type": type(pcp_metadata).__name__}
    for name in fields:
        if hasattr(pcp_metadata, name):
            summary[name] = debug_value_summary(getattr(pcp_metadata, name))
    return summary


def debug_pcp_common_metadata_summary(common_attn_metadata: Any) -> dict[str, Any]:
    fields = (
        "num_reqs",
        "num_actual_tokens",
        "num_input_tokens",
        "max_query_len",
        "max_seq_len",
        "query_start_loc",
        "query_start_loc_cpu",
        "seq_lens",
        "seq_lens_cpu",
        "num_computed_tokens_cpu",
        "block_table_tensor",
        "slot_mapping",
    )
    summary: dict[str, Any] = {"type": type(common_attn_metadata).__name__}
    for name in fields:
        if hasattr(common_attn_metadata, name):
            summary[name] = debug_value_summary(getattr(common_attn_metadata, name))
    if hasattr(common_attn_metadata, "prefill_context_parallel_metadata"):
        summary["prefill_context_parallel_metadata"] = debug_pcp_metadata_summary(
            common_attn_metadata.prefill_context_parallel_metadata,
        )
    return summary


def debug_pcp_manager_summary(pcp_manager: Any) -> dict[str, Any]:
    fields = (
        "num_reqs",
        "num_decode_reqs",
        "num_prefill_reqs",
        "num_decode_tokens",
        "num_scheduled_tokens_padded",
        "pcp_padded_tokens_length",
        "pcp_padded_tokens_fla",
        "num_actual_tokens_pcp_padded",
        "total_num_sampled_tokens_pcp",
        "pcp_tokens",
        "pcp_tokens_padded",
        "num_pcp_pads_cpu",
        "pcp_unpad_mask_cpu",
        "max_num_tokens_across_pcp",
        "total_num_scheduled_tokens",
        "total_pcp_padding_tokens_fla",
        "q_head_idx_tensor",
        "q_tail_idx_tensor",
        "q_full_idx",
    )
    summary: dict[str, Any] = {"type": type(pcp_manager).__name__}
    for name in fields:
        if hasattr(pcp_manager, name):
            summary[name] = debug_value_summary(getattr(pcp_manager, name))
    if hasattr(pcp_manager, "query_lens_pcp_full"):
        query_lens = pcp_manager.query_lens_pcp_full
        summary["query_lens_pcp_full.cpu"] = debug_value_summary(
            getattr(query_lens, "cpu", None),
        )
        summary["query_lens_pcp_full.gpu"] = debug_value_summary(
            getattr(query_lens, "gpu", None),
        )
    if hasattr(pcp_manager, "pcp_allgather_restore_idx"):
        restore_idx = pcp_manager.pcp_allgather_restore_idx
        summary["pcp_allgather_restore_idx.np"] = debug_value_summary(
            getattr(restore_idx, "np", None),
        )
        summary["pcp_allgather_restore_idx.gpu"] = debug_value_summary(
            getattr(restore_idx, "gpu", None),
        )
    return summary


def snapshot_pcp_manager_state(pcp_manager: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in (
        "num_reqs",
        "num_decode_reqs",
        "num_prefill_reqs",
        "num_decode_tokens",
        "num_scheduled_tokens_padded",
        "pcp_padded_tokens_length",
        "pcp_padded_tokens_fla",
        "num_actual_tokens_pcp_padded",
        "total_num_sampled_tokens_pcp",
        "pcp_tokens_padded",
        "max_num_tokens_across_pcp",
        "total_num_scheduled_tokens",
        "total_pcp_padding_tokens_fla",
        "q_head_idx_tensor",
        "q_tail_idx_tensor",
        "q_full_idx",
        "kv_idx_names",
        "extra_long_seq_kwargs",
        "long_seq_metadata",
    ):
        if hasattr(pcp_manager, name):
            state[name] = copy.copy(getattr(pcp_manager, name))
    for name in ("pcp_tokens", "num_pcp_pads_cpu", "pcp_unpad_mask_cpu"):
        if hasattr(pcp_manager, name):
            state[name] = getattr(pcp_manager, name).copy()
    if hasattr(pcp_manager, "query_lens_pcp_full"):
        state["query_lens_pcp_full_cpu"] = pcp_manager.query_lens_pcp_full.cpu.clone()
    if hasattr(pcp_manager, "pcp_allgather_restore_idx"):
        restore_idx = pcp_manager.pcp_allgather_restore_idx
        state["pcp_allgather_restore_idx_np"] = restore_idx.np.copy()
        state["pcp_allgather_restore_idx_gpu"] = restore_idx.gpu.clone()
    return state


def restore_pcp_manager_state(pcp_manager: Any, state: dict[str, Any]) -> None:
    for name, value in state.items():
        if name == "query_lens_pcp_full_cpu":
            pcp_manager.query_lens_pcp_full.cpu.copy_(value)
            pcp_manager.query_lens_pcp_full.copy_to_gpu()
            continue
        if name == "pcp_allgather_restore_idx_np":
            pcp_manager.pcp_allgather_restore_idx.np[...] = value
            continue
        if name == "pcp_allgather_restore_idx_gpu":
            pcp_manager.pcp_allgather_restore_idx.gpu.copy_(value)
            continue
        if name in ("pcp_tokens", "num_pcp_pads_cpu", "pcp_unpad_mask_cpu"):
            getattr(pcp_manager, name)[...] = value
            continue
        setattr(pcp_manager, name, value)


def clone_pcp_metadata(pcp_metadata: Any) -> Any:
    if pcp_metadata is None:
        return None
    cloned = copy.copy(pcp_metadata)
    for name, value in vars(pcp_metadata).items():
        if hasattr(value, "clone"):
            setattr(cloned, name, value.clone())
        elif isinstance(value, list):
            setattr(cloned, name, list(value))
        else:
            setattr(cloned, name, copy.copy(value))
    return cloned
