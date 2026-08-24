# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Platform-selected AFD wrapper for DeepSeek-V4."""

from vllm.platforms import current_platform

if current_platform.device_type == "npu":
    from afd_plugin.model_executor.models.deepseek_v4_npu import (
        AFDDeepseekV4ForCausalLM,
    )
else:
    from afd_plugin.model_executor.models.deepseek_v4_cuda import (
        AFDDeepseekV4ForCausalLM,
    )

__all__ = ["AFDDeepseekV4ForCausalLM"]
