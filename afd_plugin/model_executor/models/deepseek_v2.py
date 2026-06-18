# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 AFD model wrapper.

The wrapper does not prune weights by role. Both Attention and FFN sides load
the full model; only the forward path is split so hidden states can travel
through the AFD connector.
"""

import typing
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from itertools import islice
from typing import Any

import torch
import torch.nn as nn
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.shared_fused_moe import (
    SharedFusedMoE,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models import deepseek_v2 as native
from vllm.model_executor.models.deepseek_v2 import (
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.utils import is_pp_missing_parameter

try:
    from vllm_ascend.ascend_config import get_ascend_config
except ImportError:
    get_ascend_config = None

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors import AFDConnectorMetadata, AFDFFNOutput
from afd_plugin.envs import (
    AFD_CAMP2P_STUB_IO,
    camp2p_stub_io_enabled,
    force_balanced_topk_ids_enabled,
)
from afd_plugin.model_executor.models import (
    get_afd_metadata_from_forward_context,
    get_async_moe_ubatch_metadata_from_forward_context,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

logger = init_logger(__name__)
_AFD_ASYNC_MOE_FORWARD_LOG_ENV = AFD_CAMP2P_STUB_IO


def _log_async_moe_forward_step(event: str, **kwargs: object) -> None:
    if not camp2p_stub_io_enabled():
        return
    fields = " ".join(f"{key}={value}" for key, value in kwargs.items())
    logger.warning("AFD async MoE ubatch forward %s; %s", event, fields)


def _log_ffn_compute_step(event: str, **kwargs: object) -> None:
    del event, kwargs


def _tensor_debug_info(tensor: torch.Tensor | None) -> object:
    if tensor is None:
        return None
    return {
        "shape": tuple(tensor.shape),
        "dtype": tensor.dtype,
        "device": tensor.device,
    }


def _is_moe_layer(config: object, layer_idx: int) -> bool:
    moe_layer_freq = getattr(config, "moe_layer_freq", 1)
    return (
        config.n_routed_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % moe_layer_freq == 0
    )


def _dequantize_int8_activation(
    hidden_states: torch.Tensor,
    dynamic_scales: torch.Tensor | None,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if hidden_states.dtype != torch.int8:
        return hidden_states
    if dynamic_scales is None:
        raise RuntimeError("INT8 AFD shared experts input requires dynamic_scales")

    scales = dynamic_scales.to(torch.float32)
    while scales.dim() < hidden_states.dim():
        scales = scales.unsqueeze(-1)
    return (hidden_states.to(torch.float32) * scales).to(dtype=output_dtype)


def _gmmswigluquant_fusion_enabled() -> bool:
    if get_ascend_config is None:
        return False
    ascend_config = get_ascend_config()
    fusion_config = getattr(ascend_config, "ascend_fusion_config", None)
    return bool(getattr(fusion_config, "fusion_ops_gmmswigluquant", False))


def _force_balanced_topk_ids(
    topk_ids: torch.Tensor,
    *,
    num_logical_experts: int,
) -> torch.Tensor:
    balanced_topk_ids = torch.arange(
        topk_ids.numel(),
        device=topk_ids.device,
        dtype=torch.int64,
    ).reshape(topk_ids.shape)
    balanced_topk_ids = balanced_topk_ids.remainder(num_logical_experts).to(
        dtype=topk_ids.dtype,
    )
    topk_ids.copy_(balanced_topk_ids)
    return topk_ids


class AFDDeepseekV2DecoderLayer(native.DeepseekV2DecoderLayer):
    """DeepSeek decoder layer with separable Attention and FFN execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        vllm_config = args[0] if args else kwargs.get("vllm_config")
        afd_config = parse_afd_config(vllm_config, validate=False)
        afd_role = afd_config.role if afd_config.enabled else None

        if afd_role is None:
            super().__init__(*args, **kwargs)
            self.afd_role = None
            return

        torch.nn.Module.__init__(self)

        config = args[2] if len(args) > 2 else kwargs.get("config")
        if config is None:
            config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.vllm_config = vllm_config
        self.config = config
        self.afd_config = afd_config
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        prefix = args[1] if len(args) > 1 else kwargs.get("prefix", "")
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.is_moe_layer = _is_moe_layer(config, layer_idx)
        self.compute_gate_on_attention = bool(afd_config.compute_gate_on_attention)
        self.top_k = int(config.num_experts_per_tok)

        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )
        self.use_mha = use_mha
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.afd_role = afd_role

        # Create only the modules needed for this role.
        if afd_role == "attention":
            attn_cls = (
                native.DeepseekAttention
                if use_mha
                else (
                    native.DeepseekV2MLAAttention
                    if model_config.use_mla
                    else native.DeepseekV2Attention
                )
            )
            self.self_attn = attn_cls(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                q_lora_rank=getattr(config, "q_lora_rank", None),
                kv_lora_rank=kv_lora_rank,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=kwargs.get("topk_indices_buffer"),
            )

            if self.compute_gate_on_attention and self.is_moe_layer:
                self.gate = ReplicatedLinear(
                    config.hidden_size,
                    config.n_routed_experts,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.gate",
                )
                if getattr(config, "topk_method", None) == "noaux_tc":
                    self.gate.e_score_correction_bias = nn.Parameter(
                        torch.empty(config.n_routed_experts, dtype=torch.float32)
                    )
                else:
                    self.gate.e_score_correction_bias = None

            if self.compute_gate_on_attention and not self.is_moe_layer:
                self.mlp = native.DeepseekV2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                )

        elif afd_role == "ffn":
            if self.compute_gate_on_attention and not self.is_moe_layer:
                pass
            elif self.is_moe_layer:
                self.mlp = native.DeepseekV2MoE(
                    config=config,
                    parallel_config=parallel_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                )
            else:
                self.mlp = native.DeepseekV2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                )

        self.input_layernorm = native.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = native.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_kwargs: dict[str, Any] = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)

        if (
            not isinstance(self.self_attn, native.DeepseekAttention)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                residual *= 1.0 / self.routed_scaling_factor

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        if self.afd_role == "attention" and not (
            self.compute_gate_on_attention and not self.is_moe_layer
        ):
            return hidden_states, residual

        hidden_states = self.mlp(hidden_states)
        if (
            isinstance(self.mlp, native.DeepseekV2MLP)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
        return hidden_states, residual

    def compute_attn_output(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_kwargs: dict[str, Any] = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)

        if (
            not isinstance(self.self_attn, native.DeepseekAttention)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                residual *= 1.0 / self.routed_scaling_factor

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        topk_weights = None
        topk_ids = None
        router_logits = None
        if self.compute_gate_on_attention and self.is_moe_layer:
            router_logits, _ = self.gate(hidden_states)
            afd_metadata = get_afd_metadata_from_forward_context()
            if afd_metadata is None:
                raise RuntimeError(
                    "AFD connector required for compute_gate_on_attention "
                    "but not found in forward context",
                )
            afd_connector = afd_metadata.afd_connector
            mix_placement = bool(
                getattr(self.vllm_config, "additional_config", {}).get(
                    "mix_placement",
                    False,
                ),
            )
            num_redundant_experts = (
                self.vllm_config.parallel_config.eplb_config.num_redundant_experts
            )
            if mix_placement:
                global_num_experts = (
                    self.config.n_shared_experts
                    + self.config.n_routed_experts
                    + num_redundant_experts
                )
            else:
                global_num_experts = (
                    self.config.n_routed_experts + num_redundant_experts
                )
            routed_scaling_factor = getattr(self.config, "routed_scaling_factor", 1.0)
            topk_weights, topk_ids = afd_connector.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=True,
                renormalize=getattr(self.config, "norm_topk_prob", True),
                scoring_func=getattr(self.config, "scoring_func", "softmax"),
                num_expert_group=getattr(self.config, "n_group", 1),
                topk_group=getattr(self.config, "topk_group", 1),
                routed_scaling_factor=(routed_scaling_factor if mix_placement else 1.0),
                e_score_correction_bias=self.gate.e_score_correction_bias,
                mix_placement=mix_placement,
                num_logical_experts=router_logits.shape[1],
                num_shared_experts=self.config.n_shared_experts,
                global_num_experts=global_num_experts,
            )
            if force_balanced_topk_ids_enabled():
                topk_ids = _force_balanced_topk_ids(
                    topk_ids,
                    num_logical_experts=router_logits.shape[1],
                )
            topk_weights = topk_weights.to(torch.float32)
        return hidden_states, residual, topk_weights, topk_ids, router_logits

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        *,
        group_list: torch.Tensor | None = None,
        dynamic_scales: torch.Tensor | None = None,
        expand_x_shared: torch.Tensor | None = None,
        dynamic_scales_shared: torch.Tensor | None = None,
        topk_scales: torch.Tensor | None = None,
        group_list_type: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor | AFDFFNOutput:
        del kwargs
        _log_ffn_compute_step(
            "decoder_layer_enter",
            compute_gate_on_attention=self.compute_gate_on_attention,
            is_moe_layer=self.is_moe_layer,
            hidden_states=_tensor_debug_info(hidden_states),
            group_list=_tensor_debug_info(group_list),
            dynamic_scales=_tensor_debug_info(dynamic_scales),
            expand_x_shared=_tensor_debug_info(expand_x_shared),
            dynamic_scales_shared=_tensor_debug_info(dynamic_scales_shared),
        )
        if self.compute_gate_on_attention and not self.is_moe_layer:
            raise RuntimeError(
                "Dense DeepSeek layers are computed on the Attention side "
                "when compute_gate_on_attention=true",
            )
        if self.compute_gate_on_attention:
            if group_list is None:
                raise RuntimeError(
                    "compute_gate_on_attention FFN MoE compute requires group_list",
                )
            _log_ffn_compute_step("attention_gate_moe_begin")
            output = self._compute_moe_with_attention_gate(
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                expand_x_shared=expand_x_shared,
                dynamic_scales_shared=dynamic_scales_shared,
                topk_scales=topk_scales,
                group_list_type=group_list_type,
            )
            _log_ffn_compute_step(
                "attention_gate_moe_end",
                routed_output=_tensor_debug_info(output.routed_output),
                shared_output=_tensor_debug_info(output.shared_output),
            )
            _log_ffn_compute_step("decoder_layer_exit")
            return output
        hidden_states = self.mlp(hidden_states)
        if (
            isinstance(self.mlp, native.DeepseekV2MLP)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
        _log_ffn_compute_step(
            "decoder_layer_exit",
            hidden_states=_tensor_debug_info(hidden_states),
        )
        return hidden_states

    def _compute_moe_with_attention_gate(
        self,
        *,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        dynamic_scales: torch.Tensor | None,
        expand_x_shared: torch.Tensor | None,
        dynamic_scales_shared: torch.Tensor | None,
        topk_scales: torch.Tensor | None,
        group_list_type: int,
    ) -> AFDFFNOutput:
        from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
        from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
            MoEMlpComputeInput,
            MoEWeights,
        )
        from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams
        from vllm_ascend.quantization.quant_type import QuantType

        _log_ffn_compute_step(
            "moe_attention_gate_enter",
            hidden_states=_tensor_debug_info(hidden_states),
            group_list=_tensor_debug_info(group_list),
            dynamic_scales=_tensor_debug_info(dynamic_scales),
            expand_x_shared=_tensor_debug_info(expand_x_shared),
            dynamic_scales_shared=_tensor_debug_info(dynamic_scales_shared),
            topk_scales=_tensor_debug_info(topk_scales),
            group_list_type=group_list_type,
        )
        experts = self.mlp.experts
        quant_type = experts.quant_type
        _log_ffn_compute_step("moe_weights_begin", quant_type=quant_type)
        if quant_type == QuantType.NONE:
            moe_weights = MoEWeights(
                w1=experts.w13_weight,
                w2=experts.w2_weight,
                w1_bias=experts.w13_bias if experts.moe_config.has_bias else None,
                w2_bias=experts.w2_bias if experts.moe_config.has_bias else None,
            )
        elif quant_type == QuantType.W8A8:
            if experts.dynamic_eplb:
                moe_weights = MoEWeights(
                    w1=experts.w13_weight_list,
                    w2=experts.w2_weight_list,
                    w1_scale=experts.w13_weight_scale_fp32_list,
                    w2_scale=experts.w2_weight_scale_list,
                )
            else:
                moe_weights = MoEWeights(
                    w1=[experts.w13_weight],
                    w2=[experts.w2_weight],
                    w1_scale=[experts.w13_weight_scale_fp32],
                    w2_scale=[experts.w2_weight_scale],
                )
        else:
            raise RuntimeError(
                "compute_gate_on_attention currently supports only unquantized "
                f"or W8A8 Ascend MoE experts, got {quant_type}",
            )
        _log_ffn_compute_step("moe_weights_end", quant_type=quant_type)
        use_gmmswigluquant_fusion = (
            quant_type in (QuantType.W8A8, getattr(QuantType, "MXFP8", None))
            and _gmmswigluquant_fusion_enabled()
        )

        shared_output = None
        if experts._shared_experts is not None:
            _log_ffn_compute_step("shared_experts_begin")
            shared_input = expand_x_shared
            shared_scales = dynamic_scales_shared
            if shared_input is None:
                shared_input = hidden_states
                shared_scales = dynamic_scales
            shared_input = _dequantize_int8_activation(
                shared_input,
                shared_scales,
                output_dtype=torch.bfloat16,
            )
            shared_output = experts._shared_experts(shared_input)
            _log_ffn_compute_step(
                "shared_experts_end",
                shared_output=_tensor_debug_info(shared_output),
            )

        _log_ffn_compute_step("routed_unified_apply_mlp_begin")
        routed_output = unified_apply_mlp(
            mlp_compute_input=MoEMlpComputeInput(
                hidden_states=hidden_states,
                group_list=group_list,
                group_list_type=int(group_list_type),
                dynamic_scale=dynamic_scales,
                topk_scales=topk_scales,
                weights=moe_weights,
                quant=MoEQuantParams(quant_type=quant_type),
                fusion=use_gmmswigluquant_fusion,
                activation=experts.activation,
                need_trans=False,
                dynamic_eplb=experts.dynamic_eplb,
            ),
        )
        _log_ffn_compute_step(
            "routed_unified_apply_mlp_end",
            routed_output=_tensor_debug_info(routed_output),
        )

        if hidden_states.dtype != torch.float16:
            routed_output *= self.mlp.routed_scaling_factor
        elif shared_output is not None:
            shared_output *= 1.0 / self.mlp.routed_scaling_factor

        _log_ffn_compute_step(
            "moe_attention_gate_exit",
            routed_output=_tensor_debug_info(routed_output),
            shared_output=_tensor_debug_info(shared_output),
        )
        return AFDFFNOutput(
            routed_output=routed_output,
            shared_output=shared_output,
        )


@native.support_torch_compile
class AFDDeepseekV2Model(torch.nn.Module):
    """DeepSeek model wrapper that routes Attention outputs through AFD."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: object, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.config = config
        self.device = native.current_platform.device_type

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if native.get_pp_group().is_first_rank:
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: AFDDeepseekV2DecoderLayer(
                vllm_config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            native.make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"],
                config.hidden_size,
            )
        )
        self.aux_hidden_state_layers = tuple[int, ...]()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: native.IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | native.IntermediateTensors:
        if native.get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided "
                        "to AFDDeepseekV2Model.forward",
                    )
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        llama_4_scaling = self._get_llama_4_scaling(positions)
        afd_metadata = get_afd_metadata_from_forward_context()

        aux_hidden_states = []
        if afd_metadata is not None:
            if self.aux_hidden_state_layers:
                raise RuntimeError(
                    "AFD DeepSeekV2 E2E wrapper does not support aux hidden "
                    "state capture yet",
                )
            hidden_states, residual = self.forward_with_afd(
                hidden_states,
                residual,
                positions,
                afd_metadata,
                llama_4_scaling,
            )
        else:
            for idx, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer),
                start=self.start_layer,
            ):
                if idx in self.aux_hidden_state_layers:
                    aux_hidden_states.append(hidden_states + residual)
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                    llama_4_scaling,
                )

        if not native.get_pp_group().is_last_rank:
            return native.IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual},
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def forward_with_afd(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        afd_metadata: object,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.afd_config.compute_gate_on_attention:
            forward_context = get_forward_context()
            if (
                get_async_moe_ubatch_metadata_from_forward_context(forward_context)
                is not None
            ):
                return self.forward_with_afd_v3(
                    hidden_states,
                    residual,
                    positions,
                    afd_metadata,
                    llama_4_scaling,
                )
            return self.forward_with_afd_v2(
                hidden_states,
                residual,
                positions,
                afd_metadata,
                llama_4_scaling,
            )

        afd_connector = afd_metadata.afd_connector
        forward_context = get_forward_context()
        stage_idx = int(
            getattr(forward_context, "ubatch_idx", afd_metadata.afd_stage_idx),
        )

        for layer_offset, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
        ):
            stage_idx = int(
                getattr(forward_context, "ubatch_idx", afd_metadata.afd_stage_idx),
            )
            afd_metadata.ubatch_idx = stage_idx
            afd_metadata.afd_stage_idx = stage_idx
            if layer_offset > 0:
                hidden_states = afd_connector.recv_ffn_output(
                    ref_tensor=hidden_states,
                    ubatch_idx=stage_idx,
                )

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
            )
            metadata = AFDConnectorMetadata.create_attention_metadata(
                layer_idx=layer.layer_idx,
                stage_idx=stage_idx,
                seq_len=int(hidden_states.shape[0]),
            )
            afd_connector.send_attn_output(hidden_states, metadata)
            hidden_states = maybe_apply_dbo_yield(
                hidden_states,
                role="attention",
            )

        hidden_states = afd_connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=stage_idx,
        )
        return hidden_states, residual

    def forward_with_afd_v2(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        afd_metadata: object,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        afd_connector = afd_metadata.afd_connector
        forward_context = get_forward_context()
        stage_idx = int(
            getattr(forward_context, "ubatch_idx", afd_metadata.afd_stage_idx),
        )
        pending_ffn_recv = False

        for layer_offset, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
        ):
            stage_idx = int(
                getattr(forward_context, "ubatch_idx", afd_metadata.afd_stage_idx),
            )
            afd_metadata.ubatch_idx = stage_idx
            afd_metadata.afd_stage_idx = stage_idx
            if layer_offset > 0 and pending_ffn_recv:
                hidden_states = afd_connector.recv_ffn_output(
                    ref_tensor=hidden_states,
                    ubatch_idx=stage_idx,
                )
                pending_ffn_recv = False

            if not layer.is_moe_layer:
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                    llama_4_scaling,
                )
                continue

            (
                hidden_states,
                residual,
                topk_weights,
                topk_ids,
                router_logits,
            ) = layer.compute_attn_output(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
            )

            metadata = AFDConnectorMetadata.create_attention_metadata(
                layer_idx=layer.layer_idx,
                stage_idx=stage_idx,
                seq_len=int(hidden_states.shape[0]),
            )
            afd_connector.send_attn_output(
                hidden_states,
                metadata,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
            )
            pending_ffn_recv = True
            hidden_states = maybe_apply_dbo_yield(
                hidden_states,
                role="attention",
            )

        if pending_ffn_recv:
            hidden_states = afd_connector.recv_ffn_output(
                ref_tensor=hidden_states,
                ubatch_idx=stage_idx,
            )
        return hidden_states, residual

    def forward_with_afd_v3(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        afd_metadata: object,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        forward_context = get_forward_context()
        async_moe_ubatch_metadata = get_async_moe_ubatch_metadata_from_forward_context(
            forward_context
        )
        if async_moe_ubatch_metadata is None:
            return self.forward_with_afd_v2(
                hidden_states,
                residual,
                positions,
                afd_metadata,
                llama_4_scaling,
            )
        ubatch_slices = async_moe_ubatch_metadata["ubatch_slices"]
        afd_connector = afd_metadata.afd_connector
        first_moe_layer = int(self.config.first_k_dense_replace)
        dense_end_layer = min(self.end_layer, first_moe_layer)
        for layer in islice(self.layers, self.start_layer, dense_end_layer):
            _log_async_moe_forward_step(
                "dense_begin",
                layer_idx=layer.layer_idx,
                hidden_shape=tuple(hidden_states.shape),
            )
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
            )
            _log_async_moe_forward_step(
                "dense_end",
                layer_idx=layer.layer_idx,
                hidden_shape=tuple(hidden_states.shape),
            )
        if dense_end_layer == self.end_layer:
            _log_async_moe_forward_step(
                "exit_dense_only",
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                hidden_shape=tuple(hidden_states.shape),
            )
            return hidden_states, residual

        stage_hidden_states = [
            hidden_states[ubatch_slice.token_slice] for ubatch_slice in ubatch_slices
        ]
        stage_residual = [
            _slice_optional_first_dim(residual, ubatch_slice.token_slice)
            for ubatch_slice in ubatch_slices
        ]
        stage_positions = [
            _slice_positions(positions, ubatch_slice.token_slice)
            for ubatch_slice in ubatch_slices
        ]
        stage_llama_4_scaling = [
            _slice_llama_4_scaling(
                llama_4_scaling,
                ubatch_slice.token_slice,
                num_tokens=int(hidden_states.shape[0]),
            )
            for ubatch_slice in ubatch_slices
        ]

        moe_start_layer = max(self.start_layer, first_moe_layer)
        _log_async_moe_forward_step(
            "enter",
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            dense_end_layer=dense_end_layer,
            moe_start_layer=moe_start_layer,
            num_stages=len(ubatch_slices),
            request_slices=[
                (ubatch_slice.request_slice.start, ubatch_slice.request_slice.stop)
                for ubatch_slice in ubatch_slices
            ],
            token_slices=[
                (ubatch_slice.token_slice.start, ubatch_slice.token_slice.stop)
                for ubatch_slice in ubatch_slices
            ],
            stage_shapes=[
                tuple(stage_hidden_state.shape)
                for stage_hidden_state in stage_hidden_states
            ],
        )
        moe_layers = list(islice(self.layers, moe_start_layer, self.end_layer))

        def compute_stage_attention(
            layer: Any,
            stage_idx: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            ubatch_slice = ubatch_slices[stage_idx]
            with _use_async_moe_ubatch_forward_context(
                forward_context=forward_context,
                parent_afd_metadata=afd_metadata,
                async_moe_ubatch_metadata=async_moe_ubatch_metadata,
                stage_idx=stage_idx,
            ):
                _log_async_moe_forward_step(
                    "attention_begin",
                    layer_idx=layer.layer_idx,
                    stage_idx=stage_idx,
                    hidden_shape=tuple(stage_hidden_states[stage_idx].shape),
                    token_slice=(
                        ubatch_slice.token_slice.start,
                        ubatch_slice.token_slice.stop,
                    ),
                )
                (
                    stage_hidden_states[stage_idx],
                    stage_residual[stage_idx],
                    topk_weights,
                    topk_ids,
                    router_logits,
                ) = layer.compute_attn_output(
                    stage_positions[stage_idx],
                    stage_hidden_states[stage_idx],
                    stage_residual[stage_idx],
                    stage_llama_4_scaling[stage_idx],
                )
                _log_async_moe_forward_step(
                    "attention_end",
                    layer_idx=layer.layer_idx,
                    stage_idx=stage_idx,
                    hidden_shape=tuple(stage_hidden_states[stage_idx].shape),
                    has_topk_weights=topk_weights is not None,
                    has_topk_ids=topk_ids is not None,
                    has_router_logits=router_logits is not None,
                )
            if topk_weights is None or topk_ids is None:
                raise RuntimeError(
                    "async_moe_ubatching requires Attention-side topk payloads",
                )
            expected_tokens = int(ubatch_slice.num_tokens)
            if int(stage_hidden_states[stage_idx].shape[0]) != expected_tokens:
                raise RuntimeError(
                    "async_moe_ubatching stage output token count mismatch: "
                    f"expected {expected_tokens}, got "
                    f"{int(stage_hidden_states[stage_idx].shape[0])}",
                )
            return topk_weights, topk_ids, router_logits

        def send_stage_attention(
            layer: Any,
            stage_idx: int,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            router_logits: torch.Tensor | None,
        ) -> None:
            expected_tokens = int(ubatch_slices[stage_idx].num_tokens)
            stage_metadata = AFDConnectorMetadata.create_attention_metadata(
                layer_idx=layer.layer_idx,
                stage_idx=stage_idx,
                seq_len=expected_tokens,
            )
            _log_async_moe_forward_step(
                "send_begin",
                layer_idx=layer.layer_idx,
                stage_idx=stage_idx,
                hidden_shape=tuple(stage_hidden_states[stage_idx].shape),
            )
            afd_connector.send_attn_output(
                stage_hidden_states[stage_idx],
                stage_metadata,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
            )
            _log_async_moe_forward_step(
                "send_end",
                layer_idx=layer.layer_idx,
                stage_idx=stage_idx,
            )

        def recv_stage_ffn(layer: Any, stage_idx: int, event_prefix: str) -> None:
            _log_async_moe_forward_step(
                f"{event_prefix}_recv_begin",
                layer_idx=layer.layer_idx,
                stage_idx=stage_idx,
                ref_shape=tuple(stage_hidden_states[stage_idx].shape),
            )
            stage_hidden_states[stage_idx] = afd_connector.recv_ffn_output(
                ref_tensor=stage_hidden_states[stage_idx],
                ubatch_idx=stage_idx,
            )
            _log_async_moe_forward_step(
                f"{event_prefix}_recv_end",
                layer_idx=layer.layer_idx,
                stage_idx=stage_idx,
                hidden_shape=tuple(stage_hidden_states[stage_idx].shape),
            )

        last_moe_layer_offset = len(moe_layers) - 1
        first_layer = moe_layers[0]
        topk_weights, topk_ids, router_logits = compute_stage_attention(
            first_layer,
            0,
        )
        send_stage_attention(
            first_layer,
            0,
            topk_weights,
            topk_ids,
            router_logits,
        )

        for moe_layer_offset in range(last_moe_layer_offset):
            current_layer = moe_layers[moe_layer_offset]
            next_layer = moe_layers[moe_layer_offset + 1]

            topk_weights, topk_ids, router_logits = compute_stage_attention(
                current_layer,
                1,
            )
            recv_stage_ffn(current_layer, 0, "wavefront")
            send_stage_attention(
                current_layer,
                1,
                topk_weights,
                topk_ids,
                router_logits,
            )

            topk_weights, topk_ids, router_logits = compute_stage_attention(
                next_layer,
                0,
            )
            recv_stage_ffn(current_layer, 1, "wavefront")
            send_stage_attention(
                next_layer,
                0,
                topk_weights,
                topk_ids,
                router_logits,
            )

        last_layer = moe_layers[last_moe_layer_offset]
        topk_weights, topk_ids, router_logits = compute_stage_attention(
            last_layer,
            1,
        )
        recv_stage_ffn(last_layer, 0, "final")
        send_stage_attention(
            last_layer,
            1,
            topk_weights,
            topk_ids,
            router_logits,
        )
        recv_stage_ffn(last_layer, 1, "final")
        output_hidden_states = torch.cat(stage_hidden_states, dim=0)
        _log_async_moe_forward_step(
            "exit",
            hidden_shape=tuple(output_hidden_states.shape),
        )
        return (
            output_hidden_states,
            _cat_optional_async_moe_stage_outputs(
                stage_residual,
                stage_hidden_states,
            ),
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor | AFDFFNOutput:
        _log_ffn_compute_step(
            "model_layer_dispatch_begin",
            layer_idx=layer_idx,
            hidden_states=_tensor_debug_info(hidden_states),
        )
        output = self.layers[layer_idx].compute_ffn_output(
            hidden_states,
            **kwargs,
        )
        _log_ffn_compute_step(
            "model_layer_dispatch_end",
            layer_idx=layer_idx,
        )
        return output

    def _get_llama_4_scaling(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor | None:
        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        if llama_4_scaling_config is None:
            return None
        return native._get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"
            ],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )


_MISSING_FORWARD_CONTEXT_ATTR = object()


@contextmanager
def _use_async_moe_ubatch_forward_context(
    *,
    forward_context: object,
    parent_afd_metadata: object,
    async_moe_ubatch_metadata: dict[str, Any],
    stage_idx: int,
) -> Iterator[None]:
    ubatch_slices = async_moe_ubatch_metadata["ubatch_slices"]
    attn_metadata = async_moe_ubatch_metadata["attn_metadata"]
    stage_afd_metadata = _build_async_moe_stage_afd_metadata(
        parent_afd_metadata,
        ubatch_slices,
        stage_idx,
    )

    saved_attrs = {
        "attn_metadata": _read_forward_context_attr(
            forward_context,
            "attn_metadata",
        ),
        "additional_kwargs": _read_forward_context_attr(
            forward_context,
            "additional_kwargs",
        ),
        "afd_metadata": _read_forward_context_attr(
            forward_context,
            "afd_metadata",
        ),
        "ubatch_idx": _read_forward_context_attr(forward_context, "ubatch_idx"),
        "num_ubatches": _read_forward_context_attr(
            forward_context,
            "num_ubatches",
        ),
        "num_tokens": _read_forward_context_attr(forward_context, "num_tokens"),
    }

    original_kwargs = (
        forward_context.additional_kwargs
        if saved_attrs["additional_kwargs"] is not _MISSING_FORWARD_CONTEXT_ATTR
        else None
    )
    stage_kwargs = dict(original_kwargs or {})
    stage_kwargs["afd_metadata"] = stage_afd_metadata

    try:
        forward_context.attn_metadata = attn_metadata[stage_idx]
        forward_context.additional_kwargs = stage_kwargs
        forward_context.afd_metadata = stage_afd_metadata
        forward_context.ubatch_idx = stage_idx
        forward_context.num_ubatches = len(ubatch_slices)
        forward_context.num_tokens = int(ubatch_slices[stage_idx].num_tokens)
        yield
    finally:
        for name, value in saved_attrs.items():
            _restore_forward_context_attr(forward_context, name, value)


def _build_async_moe_stage_afd_metadata(
    parent_afd_metadata: object,
    ubatch_slices: object,
    stage_idx: int,
) -> object:
    ubatch_slice = ubatch_slices[stage_idx]
    stage_metadata = parent_afd_metadata.clone()
    stage_metadata.afd_stage_idx = stage_idx
    stage_metadata.ubatch_idx = stage_idx
    stage_metadata.num_of_stages = len(ubatch_slices)
    stage_metadata.afd_tokens_start_loc = [ubatch_slice.token_slice.start]
    stage_metadata.afd_reqs_start_loc = [ubatch_slice.request_slice.start]
    stage_metadata.afd_tokens_lens = [ubatch_slice.num_tokens]
    if len(parent_afd_metadata.afd_tokens_unpadded_lens) > stage_idx:
        unpadded_len = parent_afd_metadata.afd_tokens_unpadded_lens[stage_idx]
    else:
        unpadded_len = ubatch_slice.num_tokens
    stage_metadata.afd_tokens_unpadded_lens = [int(unpadded_len)]
    return stage_metadata


def _cat_optional_async_moe_stage_outputs(
    stage_outputs: list[torch.Tensor | None],
    fallback_outputs: list[torch.Tensor],
) -> torch.Tensor | None:
    if all(stage_output is None for stage_output in stage_outputs):
        return None
    return torch.cat(
        [
            stage_output if stage_output is not None else fallback_output
            for stage_output, fallback_output in zip(
                stage_outputs,
                fallback_outputs,
                strict=True,
            )
        ],
        dim=0,
    )


def _slice_optional_first_dim(
    tensor: torch.Tensor | None,
    token_slice: slice,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[token_slice]


def _slice_positions(positions: torch.Tensor, token_slice: slice) -> torch.Tensor:
    if positions.dim() <= 1:
        return positions[token_slice]
    return positions[..., token_slice]


def _slice_llama_4_scaling(
    llama_4_scaling: torch.Tensor | None,
    token_slice: slice,
    *,
    num_tokens: int,
) -> torch.Tensor | None:
    if llama_4_scaling is None:
        return None
    if llama_4_scaling.shape[0] == num_tokens:
        return llama_4_scaling[token_slice]
    if llama_4_scaling.dim() > 1 and llama_4_scaling.shape[1] == num_tokens:
        return llama_4_scaling[:, token_slice]
    return llama_4_scaling


def _read_forward_context_attr(forward_context: object, name: str) -> object:
    try:
        return getattr(forward_context, name)
    except AttributeError:
        return _MISSING_FORWARD_CONTEXT_ATTR


def _restore_forward_context_attr(
    forward_context: object,
    name: str,
    value: object,
) -> None:
    if value is _MISSING_FORWARD_CONTEXT_ATTR:
        with suppress(AttributeError):
            delattr(forward_context, name)
        return
    setattr(forward_context, name, value)


class AFDDeepseekV2ForCausalLM(native.DeepseekV2ForCausalLM):
    """DeepSeekV2 causal LM wrapper for AFD execution."""

    model_cls = AFDDeepseekV2Model

    def __init__(self, *, vllm_config: object, prefix: str = "") -> None:
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role if self.afd_config.enabled else None
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def set_moe_parameters(self) -> None:
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, native.PPMissingLayer):
                continue
            if not isinstance(layer, native.DeepseekV2DecoderLayer):
                continue
            mlp = layer._modules.get("mlp")
            if (self.afd_role is None or self.afd_role == "ffn") and isinstance(
                mlp, native.DeepseekV2MoE
            ):
                example_moe = mlp
                self.moe_mlp_layers.append(mlp)
                self.moe_layers.append(mlp.experts)
        if self.afd_role == "attention":
            return
        self.extract_moe_parameters(example_moe)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor | AFDFFNOutput:
        _log_ffn_compute_step(
            "causal_lm_ffn_dispatch_begin",
            layer_idx=layer_idx,
            hidden_states=_tensor_debug_info(hidden_states),
        )
        output = self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)
        _log_ffn_compute_step(
            "causal_lm_ffn_dispatch_end",
            layer_idx=layer_idx,
        )
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ascend_config = get_ascend_config() if get_ascend_config is not None else None
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]

        mix_placement = (
            getattr(ascend_config, "mix_placement", False) if ascend_config else False
        )

        if self.afd_role == "attention":
            vllm_config = get_current_vllm_config()
            num_redundant_experts = (
                vllm_config.parallel_config.eplb_config.num_redundant_experts
            )
        else:
            num_redundant_experts = self.num_redundant_experts

        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts if mix_placement else 0),
            num_redundant_experts=num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if (
                self.afd_role == "attention"
                and self.afd_config is not None
                and self.afd_config.compute_gate_on_attention
                and (
                    "mlp.gate.weight" in name
                    or "mlp.gate.e_score_correction_bias" in name
                )
            ):
                mapped_name = name.replace(".mlp.gate", ".gate")
                if mapped_name in params_dict:
                    param = params_dict[mapped_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(mapped_name)
                    continue

            if (
                self.afd_role == "attention"
                and self.is_moe_weight(name)
                and (
                    not self.afd_config.compute_gate_on_attention
                    or self.is_moe_layer_weight(name)
                )
            ):
                continue

            if (
                self.afd_role == "ffn"
                and self.afd_config.compute_gate_on_attention
                and self.is_dense_mlp_weight(name)
            ):
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue

            is_fuse_shared_experts_layer = mix_placement and (
                "mlp.shared_experts" in name
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fuse_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                num_chunks = 1
                if is_fuse_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    split_dim = 1 if "down_proj.weight" in name else 0
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} "
                        f"not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fuse_shared_experts_layer:
                        if split_dim == 0:
                            weight_to_load = loaded_weight[
                                j * chunk_size : (j + 1) * chunk_size, :
                            ]
                        else:
                            weight_to_load = loaded_weight[
                                :, j * chunk_size : (j + 1) * chunk_size
                            ]
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        is_expert_weight = True
                        if self.afd_role is not None and self.afd_role == "attention":
                            continue
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        if name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fuse_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if (
                            self.afd_role == "ffn"
                            and not self.is_moe_weight(name)
                            and not self.is_common_weight(name)
                        ):
                            continue
                        if is_expert_weight:
                            continue
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue
                        if is_pp_missing_parameter(name, self):
                            continue
                        if name not in params_dict:
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            if not is_fuse_shared_experts_layer:
                loaded_params.add(name)
        return loaded_params

    def is_moe_weight(self, name):
        return (
            "shared_experts" in name
            or "experts" in name
            or "gate" in name
            or "up" in name
            or "down" in name
        )

    def is_moe_layer_weight(self, name: str) -> bool:
        layer_idx = self.weight_layer_idx(name)
        return layer_idx is not None and _is_moe_layer(self.config, layer_idx)

    def is_dense_mlp_weight(self, name: str) -> bool:
        layer_idx = self.weight_layer_idx(name)
        return (
            ".mlp." in name
            and layer_idx is not None
            and not _is_moe_layer(self.config, layer_idx)
        )

    @staticmethod
    def weight_layer_idx(name: str) -> int | None:
        parts = name.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part != "layers":
                continue
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
        return None

    def is_common_weight(self, name):
        return (
            "lm_head" in name
            or "model.norm.weight" in name
            or "embed_tokens" in name
            or "input_layernorm" in name
            or "post_attention_layernorm" in name
        )


class AFDDeepseekForCausalLM(AFDDeepseekV2ForCausalLM):
    pass


class AFDDeepseekV3ForCausalLM(AFDDeepseekV2ForCausalLM):
    pass


class AFDGlmMoeDsaForCausalLM(AFDDeepseekV2ForCausalLM):
    pass


__all__ = [
    "AFDDeepseekForCausalLM",
    "AFDDeepseekV2ForCausalLM",
    "AFDDeepseekV3ForCausalLM",
    "AFDGlmMoeDsaForCausalLM",
]
