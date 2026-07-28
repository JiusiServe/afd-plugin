# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend ubatch helpers owned by the AFD plugin.

Ubatch token-space planning (split policies, padding, attention-metadata
rebuilds) plus the TP/SP-aware local-token mapping consumed by model-side
stage slicing. Parts of this module mirror the Ascend DBO logic from
vLLM-Ascend commit cdd212830271249a1cafcb850c210133f21771c5, kept
plugin-owned so AFD retains DBO support independent of upstream changes.
"""

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
        return False
    if num_tokens_padded < num_ubatches:
        return False

    if _cp_enabled(vllm_config):
        return False

    should_attempt_ubatching = check_ubatch_thresholds(
        parallel_config,
        num_tokens_unpadded,
        uniform_decode=uniform_decode,
    )
    if not getattr(parallel_config, "enable_dbo", False):
        return False
    if not should_attempt_ubatching:
        return False

    return not is_last_ubatch_empty(
        num_tokens_unpadded,
        num_tokens_padded,
        num_ubatches,
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


def create_request_boundary_ubatch_slices(
    num_scheduled_tokens: np.ndarray,
    *,
    num_ubatches: int = 2,
) -> UBatchSlices | None:
    """Split scheduled tokens on request boundaries.

    Async MoE ubatching keeps dense layers on the full batch and only slices
    connector payloads.  Splitting on request boundaries avoids partial request
    metadata and keeps each stage's common attention metadata rebuildable by
    the existing Ascend builder stack.  Among those boundaries, pick the split
    whose two stages have the closest token counts.
    """

    assert num_ubatches == 2, "Async MoE ubatching currently supports 2 stages."
    num_reqs = len(num_scheduled_tokens)
    if num_reqs < num_ubatches:
        return None

    cu_num_tokens = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])
    total_tokens = int(cu_num_tokens[-1])
    if total_tokens < num_ubatches:
        return None

    split_req = min(
        range(1, num_reqs),
        key=lambda req_idx: (
            abs(int(cu_num_tokens[req_idx]) * 2 - total_tokens),
            abs(req_idx * num_ubatches - num_reqs),
        ),
    )
    split_token = int(cu_num_tokens[split_req])
    if split_token <= 0 or split_token >= total_tokens:
        return None

    return [
        UBatchSlice(slice(0, split_req), slice(0, split_token)),
        UBatchSlice(slice(split_req, num_reqs), slice(split_token, total_tokens)),
    ]


ASYNC_MOE_SPLIT_REQUEST_BOUNDARY = "request_boundary"
ASYNC_MOE_SPLIT_TOKEN_BALANCED_TP = "token_balanced_tp"


def enable_token_balanced_async_moe_split(vllm_config: VllmConfig) -> bool:
    """Whether async MoE ubatches split on TP-aligned token counts.

    Non-PCP DP+TP/SP topologies can schedule requests with heavily skewed
    lengths, where a request-boundary split produces imbalanced stages; a
    token-balanced split keeps both pipeline stages close to half the token
    workload. PCP topologies keep the request-boundary split because their
    metadata is rebuilt per request.
    """
    parallel_config = vllm_config.parallel_config
    if int(parallel_config.tensor_parallel_size) <= 1:
        return False
    if int(parallel_config.prefill_context_parallel_size) != 1:
        return False
    return int(parallel_config.decode_context_parallel_size) == 1


def token_balanced_split_points(
    num_tokens: int,
    num_ubatches: int,
    tp_size: int,
) -> list[int] | None:
    """Even token split points aligned down to multiples of ``tp_size``.

    Alignment keeps every stage divisible by the TP size so that, under
    sequence parallelism, each rank's local stage shards have equal length.
    Returns None when an aligned split is impossible (too few tokens, or
    alignment collapses a stage boundary).
    """
    if num_ubatches <= 1 or num_tokens < num_ubatches:
        return None
    split_points: list[int] = []
    for idx in range(1, num_ubatches):
        split_point = (num_tokens * idx) // num_ubatches
        if tp_size > 1:
            split_point = (split_point // tp_size) * tp_size
        if split_point <= 0 or split_point >= num_tokens:
            return None
        if split_points and split_point <= split_points[-1]:
            return None
        split_points.append(split_point)
    return split_points


def create_async_moe_ubatch_slices(
    vllm_config: VllmConfig,
    num_scheduled_tokens_np: np.ndarray,
    *,
    num_tokens: int,
    num_tokens_padded: int | None,
    num_reqs_padded: int | None,
    num_ubatches: int,
    split: str,
) -> tuple[UBatchSlices | None, str]:
    """Split async MoE work into ubatches and report the split policy used.

    ``split`` is the connector's ``async_moe_split`` value. ``"token"``
    selects the TP-aligned token-balanced split on non-PCP DP+TP/SP
    topologies: the padded token count is split evenly with split points
    aligned to the TP size, so both stages — and, under SP, every rank's
    local stage shard — carry close to half the workload. When DP padding
    makes the balanced split land past this rank's real tokens, the split
    point is moved left to the largest TP-aligned boundary that still leaves
    real tokens in both stages. If no such boundary exists, the step runs
    unbatched; falling back to request boundaries would create stage lengths
    that cannot be represented by equal SP shards.

    Any non-token policy selects the request-boundary split. Unsupported token
    topologies also retain the request-boundary result because startup
    validation rejects that configuration before serving.

    The returned policy string (``"token_balanced_tp"`` or
    ``"request_boundary"``) describes the split actually applied to this
    batch, so callers can adapt downstream handling without re-deriving the
    decision; it is a string (not a boolean) so additional split policies
    can be introduced without changing the contract.

    Returns ``(None, policy)`` when the batch is too small to split; callers
    should run such steps unbatched. Raises AssertionError if
    ``num_ubatches`` is not 2.
    """
    assert num_ubatches == 2, "Async MoE ubatching currently supports 2 stages."
    if split == "token" and enable_token_balanced_async_moe_split(vllm_config):
        split_total = int(num_tokens_padded or num_tokens)
        tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        token_split_points = token_balanced_split_points(
            split_total,
            num_ubatches,
            tp_size,
        )
        if token_split_points is not None:
            largest_real_split = ((int(num_tokens) - 1) // tp_size) * tp_size
            token_split_points[-1] = min(
                token_split_points[-1],
                largest_real_split,
            )
        if token_split_points is not None and token_split_points[-1] > 0:
            ubatch_slices = create_ubatch_slices(
                num_scheduled_tokens_np,
                token_split_points,
            )
            if num_tokens_padded is not None and num_reqs_padded is not None:
                ubatch_slices = pad_out_ubatch_slices(
                    ubatch_slices,
                    int(num_tokens_padded),
                    int(num_reqs_padded),
                )
            return ubatch_slices, ASYNC_MOE_SPLIT_TOKEN_BALANCED_TP
        return None, ASYNC_MOE_SPLIT_TOKEN_BALANCED_TP

    return (
        create_request_boundary_ubatch_slices(
            num_scheduled_tokens_np,
            num_ubatches=num_ubatches,
        ),
        ASYNC_MOE_SPLIT_REQUEST_BOUNDARY,
    )


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
    num_parent_actual_tokens = int(attn_metadata.num_actual_tokens)
    actual_token_stop = min(int(token_slice.stop), num_parent_actual_tokens)
    num_actual_tokens = actual_token_stop - int(token_slice.start)
    if num_actual_tokens <= 0:
        raise ValueError(
            "Async MoE stage must contain at least one real token; "
            f"got token_slice={token_slice} and "
            f"num_actual_tokens={num_parent_actual_tokens}",
        )

    # DP/SP padding extends the final stage's input token range without
    # creating real requests. Exclude repeated padded request entries from
    # query metadata while retaining the full input range for positions and
    # slot_mapping.
    actual_request_stop = int(
        np.searchsorted(
            start_locs.numpy(),
            actual_token_stop,
            side="left",
        ),
    )
    request_slice = slice(
        int(request_slice.start),
        min(int(request_slice.stop), actual_request_stop),
    )
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = actual_token_stop - 1

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
        tokens_skipped = start_locs[last_req + 1] - actual_token_stop
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
    num_input_tokens = int(token_slice.stop) - int(token_slice.start)
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    if len(attn_metadata.actual_seq_lengths_q) > 0:
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q[
            int(token_slice.start) : actual_token_stop
        ]
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
        num_input_tokens=num_input_tokens,
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
    "ASYNC_MOE_SPLIT_REQUEST_BOUNDARY",
    "ASYNC_MOE_SPLIT_TOKEN_BALANCED_TP",
    "UBatchSlice",
    "UBatchSlices",
    "check_enable_ubatch",
    "create_async_moe_ubatch_slices",
    "create_request_boundary_ubatch_slices",
    "create_ubatch_slices",
    "enable_token_balanced_async_moe_split",
    "is_last_ubatch_empty",
    "maybe_create_ubatch_slices",
    "pad_out_ubatch_slices",
    "slice_query_start_locs",
    "split_attn_metadata",
    "token_balanced_split_points",
]
