# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Forward-context helpers for plugin-owned Ascend ubatching."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, DPMetadata, ForwardContext
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

from afd_plugin.compat.patches.npu.mla_graph import (
    AFD_MLA_GRAPH_PARAMS_KEY,
)
from afd_plugin.v1.worker.ubatch_wrapper import (
    build_ubatch_additional_kwargs,
    build_ubatch_afd_metadata,
)

if TYPE_CHECKING:
    from vllm_ascend.compilation.acl_graph import GraphParams


def _get_ascend_extra(
    forward_context: ForwardContext,
    vllm_config: VllmConfig,
    name: str,
):
    if vllm_config.use_v2_model_runner:
        return forward_context.additional_kwargs.get(name)
    return getattr(forward_context, name)


def _set_ascend_extra(
    forward_context: ForwardContext,
    vllm_config: VllmConfig,
    name: str,
    value,
) -> None:
    if vllm_config.use_v2_model_runner:
        forward_context.additional_kwargs[name] = value
    else:
        setattr(forward_context, name, value)


def create_ascend_forward_context(
    cur_forward_context: ForwardContext,
    attn_metadata,
    vllm_config: VllmConfig,
    ubatch_slices: UBatchSlices,
    ubatch_num: int = 0,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode | None = None,
    batch_descriptor: BatchDescriptor | None = None,
    skip_compiled: bool = False,
    mla_graph_params: GraphParams | None = None,
) -> ForwardContext:
    if cudagraph_runtime_mode is None:
        cudagraph_runtime_mode = CUDAGraphMode.NONE

    parent_kwargs = dict(cur_forward_context.additional_kwargs or {})
    afd_metadata = parent_kwargs.get("afd_metadata")
    if afd_metadata is not None:
        parent_kwargs = build_ubatch_additional_kwargs(
            parent_kwargs,
            build_ubatch_afd_metadata(afd_metadata, ubatch_slices, ubatch_num),
        )
    if mla_graph_params is not None:
        parent_kwargs[AFD_MLA_GRAPH_PARAMS_KEY] = mla_graph_params

    ubatch_slice = ubatch_slices[ubatch_num]
    is_padding = (
        cur_forward_context.is_padding[ubatch_slice.token_slice]
        if cur_forward_context.is_padding is not None
        else None
    )
    new_forward_context = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        all_moe_layers=cur_forward_context.all_moe_layers,
        attn_metadata=attn_metadata,
        slot_mapping={},
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
        skip_compiled=skip_compiled,
        additional_kwargs=parent_kwargs,
        is_padding=is_padding,
    )

    num_tokens = ubatch_slice.num_tokens
    tp_world_size = get_tensor_model_parallel_world_size()
    dp_world_size = get_dp_group().world_size

    moe_comm_type = _get_ascend_extra(
        cur_forward_context,
        vllm_config,
        "moe_comm_type",
    )
    _set_ascend_extra(
        new_forward_context,
        vllm_config,
        "moe_comm_type",
        moe_comm_type,
    )
    _set_ascend_extra(
        new_forward_context,
        vllm_config,
        "moe_comm_method",
        get_moe_comm_method(moe_comm_type),
    )
    for name in (
        "in_profile_run",
        "mmrs_fusion",
        "is_first_layer",
        "layer_idx",
        "prefetch_mlp_gate_up_proj",
        "prefetch_mlp_down_proj",
        "model_instance",
        "is_draft_model",
        "is_draft_model_prefill",
        "draft_attn_metadatas",
        "max_tokens_across_pcp",
        "sinks",
        "input_ids",
        "eplb_heat_collection_status",
    ):
        _set_ascend_extra(
            new_forward_context,
            vllm_config,
            name,
            _get_ascend_extra(cur_forward_context, vllm_config, name),
        )
    _set_ascend_extra(
        new_forward_context,
        vllm_config,
        "capturing",
        mla_graph_params is not None
        or _get_ascend_extra(cur_forward_context, vllm_config, "capturing"),
    )
    _set_ascend_extra(
        new_forward_context,
        vllm_config,
        "num_tokens",
        num_tokens,
    )
    new_forward_context.ubatch_idx = int(ubatch_num)
    new_forward_context.num_ubatches = len(ubatch_slices)
    flash_comm_v1_enabled = bool(
        _get_ascend_extra(
            cur_forward_context,
            vllm_config,
            "flash_comm_v1_enabled",
        )
    )
    _set_ascend_extra(
        new_forward_context,
        vllm_config,
        "flash_comm_v1_enabled",
        flash_comm_v1_enabled,
    )
    _set_ascend_extra(
        new_forward_context,
        vllm_config,
        "pad_size",
        0,
    )

    if flash_comm_v1_enabled:
        _set_ascend_extra(
            new_forward_context,
            vllm_config,
            "pad_size",
            (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size,
        )

    if dp_world_size > 1 and dp_metadata is not None:
        max_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu.max().item()
        if flash_comm_v1_enabled:
            padded_length = (
                (max_tokens_across_dp + tp_world_size - 1)
                // tp_world_size
                * tp_world_size
            )
            _set_ascend_extra(
                new_forward_context,
                vllm_config,
                "padded_length",
                padded_length,
            )
            _set_ascend_extra(
                new_forward_context,
                vllm_config,
                "pad_size",
                padded_length - num_tokens,
            )
    else:
        max_tokens_across_dp = num_tokens
    _set_ascend_extra(
        new_forward_context,
        vllm_config,
        "max_tokens_across_dp",
        max_tokens_across_dp,
    )

    padded_num_tokens = math.ceil(max_tokens_across_dp / tp_world_size) * tp_world_size
    _set_ascend_extra(
        new_forward_context,
        vllm_config,
        "padded_num_tokens",
        padded_num_tokens,
    )
    cur_mc2_mask = _get_ascend_extra(
        cur_forward_context,
        vllm_config,
        "mc2_mask",
    )
    if cur_mc2_mask is not None:
        mc2_mask = torch.zeros(
            (padded_num_tokens,),
            dtype=cur_mc2_mask.dtype,
            device=cur_mc2_mask.device,
        )
        mc2_mask[:num_tokens] = True
        mc2_mask[num_tokens:] = False
        _set_ascend_extra(
            new_forward_context,
            vllm_config,
            "mc2_mask",
            mc2_mask,
        )

    new_forward_context.dbo_enabled = True
    return new_forward_context


__all__ = ["create_ascend_forward_context"]
