# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V2 AFD E2E model wrapper.

This is a deliberately small Phase 4 wrapper for 1A1F end-to-end validation.
It does not prune weights by role.
Both Attention and FFN sides load the full model; only the forward path is
split so hidden states can travel through the AFD connector.
"""

import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any

import torch
import torch.nn as nn
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
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
from afd_plugin.connectors import AFDConnectorMetadata
from afd_plugin.model_executor.models import (
    get_afd_metadata_from_forward_context,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield


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

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        prefix = args[1] if len(args) > 1 else kwargs.get("prefix", "")
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

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

        # Create only the modules needed for this role
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

            if (
                getattr(afd_config, "compute_gate_on_attention", False)
                and layer_idx >= config.first_k_dense_replace
            ):
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

        elif afd_role == "ffn":
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % moe_layer_freq == 0
            ):
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
        if self.afd_role == "attention":
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
        return hidden_states, residual

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.mlp(hidden_states)
        if (
            isinstance(self.mlp, native.DeepseekV2MLP)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
        return hidden_states


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

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        output = self.layers[layer_idx].compute_ffn_output(hidden_states)
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


class AFDDeepseekV2ForCausalLM(native.DeepseekV2ForCausalLM):
    """DeepSeekV2 causal LM wrapper for AFD Phase 4 E2E smoke tests."""

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
            if (self.afd_role is None or self.afd_role == "ffn") and isinstance(
                layer.mlp, native.DeepseekV2MoE
            ):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        if self.afd_role == "attention":
            return
        self.extract_moe_parameters(example_moe)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)

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

            if self.afd_role == "attention" and self.is_moe_weight(name):
                continue

            if (
                self.afd_role == "ffn"
                and self.afd_config.compute_gate_on_attention
                and (
                    "mlp.gate.weight" in name
                    or "mlp.gate.e_score_correction_bias" in name
                )
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
