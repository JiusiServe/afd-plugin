# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend/vLLM-Ascend compatibility helpers for AFD runtime classes."""

from afd_plugin.compat.npu.ops import (
    AFD_ASCEND_OPS_NAMESPACE,
    AFD_ASCEND_VENDOR_NAME,
    AFD_CUST_OPAPI_ENV,
    CAM_COMBINE_RECV,
    CAM_COMBINE_SEND,
    CAM_DISPATCH_RECV,
    CAM_DISPATCH_SEND,
    CAM_OP_NAMESPACE,
    ensure_afd_ascend_ops_loaded,
    ensure_cam_async_ops_available,
    ensure_cam_p2p_ops_available,
    get_afd_cann_vendor_path,
    get_afd_cust_opapi_path,
    has_afd_ascend_ops,
)
from afd_plugin.compat.npu.runtime import (
    apply_afd_ascend_patches_if_needed,
    ascend_forward_context,
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.compat.npu.version import (
    TARGET_VLLM_ASCEND_VERSION,
    assert_vllm_ascend_version_supported,
    get_installed_vllm_ascend_version,
    is_vllm_ascend_version_supported,
)

__all__ = [
    "apply_afd_ascend_patches_if_needed",
    "assert_vllm_ascend_version_supported",
    "ascend_forward_context",
    "AFD_ASCEND_OPS_NAMESPACE",
    "AFD_ASCEND_VENDOR_NAME",
    "AFD_CUST_OPAPI_ENV",
    "CAM_COMBINE_RECV",
    "CAM_COMBINE_SEND",
    "CAM_DISPATCH_RECV",
    "CAM_DISPATCH_SEND",
    "CAM_OP_NAMESPACE",
    "ensure_afd_ascend_ops_loaded",
    "ensure_cam_async_ops_available",
    "ensure_cam_p2p_ops_available",
    "fail_if_unsupported_npu_afd_features",
    "fix_all2all_backend_for_afd",
    "get_afd_cann_vendor_path",
    "get_afd_cust_opapi_path",
    "get_installed_vllm_ascend_version",
    "has_afd_ascend_ops",
    "is_vllm_ascend_version_supported",
    "npu_afd_num_ubatches",
    "TARGET_VLLM_ASCEND_VERSION",
]
