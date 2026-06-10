# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend/vLLM-Ascend compatibility helpers for AFD runtime classes."""

from afd_plugin.compat.ascend.ops import (
    AFD_ASCEND_OPS_NAMESPACE,
    AFD_ASCEND_VENDOR_NAME,
    AFD_CUST_OPAPI_ENV,
    CAM_COMBINE_RECV,
    CAM_COMBINE_SEND,
    CAM_DISPATCH_RECV,
    CAM_DISPATCH_SEND,
    CAM_OP_NAMESPACE,
    ensure_afd_ascend_ops_loaded,
    ensure_cam_ops_available,
    get_afd_cann_vendor_path,
    get_afd_cust_opapi_path,
    has_afd_ascend_ops,
)
from afd_plugin.compat.ascend.runtime import (
    apply_afd_ascend_patches_if_needed,
    ascend_forward_context,
    ensure_ascend_runtime_available,
    ensure_vllm_config_has_afd_proxy,
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
    init_ascend_workspace_for_afd,
    mirror_afd_metadata_on_forward_context,
    npu_afd_num_ubatches,
)

__all__ = [
    "apply_afd_ascend_patches_if_needed",
    "ascend_forward_context",
    "AFD_ASCEND_OPS_NAMESPACE",
    "AFD_ASCEND_VENDOR_NAME",
    "AFD_CUST_OPAPI_ENV",
    "CAM_COMBINE_RECV",
    "CAM_COMBINE_SEND",
    "CAM_DISPATCH_RECV",
    "CAM_DISPATCH_SEND",
    "CAM_OP_NAMESPACE",
    "ensure_ascend_runtime_available",
    "ensure_afd_ascend_ops_loaded",
    "ensure_cam_ops_available",
    "ensure_vllm_config_has_afd_proxy",
    "fail_if_unsupported_npu_afd_features",
    "fix_all2all_backend_for_afd",
    "get_afd_cann_vendor_path",
    "get_afd_cust_opapi_path",
    "has_afd_ascend_ops",
    "init_ascend_workspace_for_afd",
    "mirror_afd_metadata_on_forward_context",
    "npu_afd_num_ubatches",
]
