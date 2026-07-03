# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend ubatch helpers owned by the AFD plugin.

Copied from vLLM-Ascend commit cdd212830271249a1cafcb850c210133f21771c5;
kept plugin-owned so AFD retains DBO support independent of upstream changes.
"""

import logging

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    UBatchSlices,
    check_ubatch_thresholds,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

logger = logging.getLogger(__name__)


def is_last_ubatch_empty(
    orig_num_tokens: int,
    padded_num_tokens: int,
    num_ubatches: int,
) -> bool:
    return (padded_num_tokens // num_ubatches) * (num_ubatches - 1) >= orig_num_tokens


def _cp_enabled(vllm_config: VllmConfig) -> bool:
    parallel_config = vllm_config.parallel_config
    return (
        getattr(parallel_config, "prefill_context_parallel_size", 1) > 1
        or getattr(parallel_config, "decode_context_parallel_size", 1) > 1
    )


def check_enable_ubatch(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    uniform_decode: bool,
    vllm_config: VllmConfig,
    moe_comm_type: MoECommType | None,
) -> bool:
    parallel_config = vllm_config.parallel_config
    num_ubatches = getattr(parallel_config, "num_ubatches", 2)
    if num_ubatches != 2:
        _log_ubatch_check(
            False,
            "num_ubatches_not_2",
            num_tokens_unpadded,
            num_tokens_padded,
            uniform_decode,
            parallel_config,
            moe_comm_type,
            num_ubatches,
        )
        return False
    if num_tokens_padded < num_ubatches:
        _log_ubatch_check(
            False,
            "padded_tokens_less_than_num_ubatches",
            num_tokens_unpadded,
            num_tokens_padded,
            uniform_decode,
            parallel_config,
            moe_comm_type,
            num_ubatches,
        )
        return False

    if _cp_enabled(vllm_config):
        _log_ubatch_check(
            False,
            "context_parallel_enabled",
            num_tokens_unpadded,
            num_tokens_padded,
            uniform_decode,
            parallel_config,
            moe_comm_type,
            num_ubatches,
        )
        return False

    should_attempt_ubatching = check_ubatch_thresholds(
        parallel_config,
        num_tokens_unpadded,
        uniform_decode=uniform_decode,
    )
    if not getattr(parallel_config, "enable_dbo", False):
        _log_ubatch_check(
            False,
            "enable_dbo_false",
            num_tokens_unpadded,
            num_tokens_padded,
            uniform_decode,
            parallel_config,
            moe_comm_type,
            num_ubatches,
        )
        return False
    if not should_attempt_ubatching:
        _log_ubatch_check(
            False,
            "threshold_not_met",
            num_tokens_unpadded,
            num_tokens_padded,
            uniform_decode,
            parallel_config,
            moe_comm_type,
            num_ubatches,
        )
        return False

    should_ubatch = not is_last_ubatch_empty(
        num_tokens_unpadded,
        num_tokens_padded,
        num_ubatches,
    )
    _log_ubatch_check(
        should_ubatch,
        "ok" if should_ubatch else "last_ubatch_empty",
        num_tokens_unpadded,
        num_tokens_padded,
        uniform_decode,
        parallel_config,
        moe_comm_type,
        num_ubatches,
    )
    return should_ubatch


def _log_ubatch_check(
    enabled: bool,
    reason: str,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    uniform_decode: bool,
    parallel_config: object,
    moe_comm_type: MoECommType | None,
    num_ubatches: int,
) -> None:
    logger.warning(
        "AFD NPU ubatch check; enabled=%s reason=%s "
        "num_tokens_unpadded=%s num_tokens_padded=%s uniform_decode=%s "
        "enable_dbo=%s use_ubatching=%s num_ubatches=%s ubatch_size=%s "
        "decode_threshold=%s prefill_threshold=%s moe_comm_type=%s",
        enabled,
        reason,
        num_tokens_unpadded,
        num_tokens_padded,
        uniform_decode,
        getattr(parallel_config, "enable_dbo", None),
        getattr(parallel_config, "use_ubatching", None),
        num_ubatches,
        getattr(parallel_config, "ubatch_size", None),
        getattr(parallel_config, "dbo_decode_token_threshold", None),
        getattr(parallel_config, "dbo_prefill_token_threshold", None),
        getattr(moe_comm_type, "name", str(moe_comm_type)),
    )


def pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices,
    num_total_tokens: int,
    num_reqs_padded: int,
) -> UBatchSlices:
    last_slice = ubatch_slices[-1]
    padded_last_request_slice = slice(last_slice.request_slice.start, num_reqs_padded)
    padded_last_token_slice = slice(last_slice.token_slice.start, num_total_tokens)
    return ubatch_slices[:-1] + [
        UBatchSlice(padded_last_request_slice, padded_last_token_slice)
    ]


def create_ubatch_slices(
    num_scheduled_tokens: np.ndarray,
    token_split_points: list[int],
) -> UBatchSlices:
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    ubatch_slices: UBatchSlices = []
    start_token = 0
    for end_token in token_split_points + [int(cu_num_tokens[-1])]:
        token_slice = slice(start_token, end_token)
        req_start = int(np.searchsorted(cu_num_tokens, start_token, side="right") - 1)
        req_stop = int(np.searchsorted(cu_num_tokens, end_token, side="left"))
        ubatch_slices.append(UBatchSlice(slice(req_start, req_stop), token_slice))
        start_token = end_token
    return ubatch_slices


def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    vllm_config: VllmConfig,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
    if not should_ubatch:
        return None, None

    num_ubatches = getattr(vllm_config.parallel_config, "num_ubatches", 2)
    assert num_ubatches == 2, "Ascend ubatching currently supports exactly 2 ubatches."

    split_point = int(num_tokens_padded) // num_ubatches
    token_split_points = [split_point * i for i in range(1, num_ubatches)]
    ubatch_slices = create_ubatch_slices(
        num_scheduled_tokens_per_request,
        token_split_points,
    )
    ubatch_slices_padded = pad_out_ubatch_slices(
        ubatch_slices,
        num_tokens_padded,
        num_reqs_padded,
    )
    assert sum(ubatch_slice.num_tokens for ubatch_slice in ubatch_slices_padded) == (
        num_tokens_padded
    )
    return ubatch_slices, ubatch_slices_padded


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice,
    attn_metadata: AscendCommonAttentionMetadata,
    max_num_tokens: int = 0,
) -> AscendCommonAttentionMetadata:
    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice
    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token slice start outside of first request"
    )

    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(
        attn_metadata.query_start_loc,
        request_slice,
    )

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped

    seq_lens = attn_metadata.seq_lens[request_slice]
    seq_lens_cpu = (
        attn_metadata.seq_lens_cpu[request_slice]
        if attn_metadata.seq_lens_cpu is not None
        else None
    )

    if splits_last_request:
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped
        seq_lens = seq_lens.clone()
        seq_lens[-1] -= tokens_skipped
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu.clone()
            seq_lens_cpu[-1] -= tokens_skipped

    seq_lens_cpu_for_max = (
        seq_lens_cpu if seq_lens_cpu is not None else seq_lens.to("cpu")
    )
    max_seq_len = int(seq_lens_cpu_for_max.max())
    num_computed_tokens_cpu = (
        attn_metadata.num_computed_tokens_cpu[request_slice]
        if attn_metadata.num_computed_tokens_cpu is not None
        else None
    )

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    if len(attn_metadata.actual_seq_lengths_q) > 0:
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q[token_slice]
        if max_num_tokens and len(actual_seq_lengths_q) == 0:
            actual_seq_lengths_q = list(
                range(
                    attn_metadata.decode_token_per_req,
                    max_num_tokens + 1,
                    attn_metadata.decode_token_per_req,
                )
            )
    else:
        actual_seq_lengths_q = []

    metadata = AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu_for_max,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=attn_metadata.block_table_tensor[request_slice],
        slot_mapping=attn_metadata.slot_mapping[token_slice],
        causal=attn_metadata.causal,
        num_input_tokens=num_actual_tokens,
        actual_seq_lengths_q=actual_seq_lengths_q,
        positions=attn_metadata.positions[token_slice],
        attn_state=attn_metadata.attn_state,
        graph_pad_size=attn_metadata.graph_pad_size,
        decode_token_per_req=attn_metadata.decode_token_per_req,
        kvcomp_metadata=attn_metadata.kvcomp_metadata,
    )
    metadata.encoder_seq_lens = (
        attn_metadata.encoder_seq_lens[request_slice]
        if attn_metadata.encoder_seq_lens is not None
        else None
    )
    metadata.encoder_seq_lens_cpu = (
        attn_metadata.encoder_seq_lens_cpu[request_slice]
        if attn_metadata.encoder_seq_lens_cpu is not None
        else None
    )
    metadata.logits_indices_padded = attn_metadata.logits_indices_padded
    metadata.num_logits_indices = attn_metadata.num_logits_indices
    return metadata


def split_attn_metadata(
    ubatch_slices: UBatchSlices,
    common_attn_metadata: AscendCommonAttentionMetadata,
    max_num_tokens: int = 0,
) -> list[AscendCommonAttentionMetadata]:
    return [
        _make_metadata_with_slice(ubatch_slice, common_attn_metadata, max_num_tokens)
        for ubatch_slice in ubatch_slices
    ]


__all__ = [
    "UBatchSlice",
    "UBatchSlices",
    "check_enable_ubatch",
    "create_ubatch_slices",
    "is_last_ubatch_empty",
    "maybe_create_ubatch_slices",
    "pad_out_ubatch_slices",
    "slice_query_start_locs",
    "split_attn_metadata",
]
