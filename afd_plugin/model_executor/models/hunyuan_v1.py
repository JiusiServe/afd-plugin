# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""HunyuanImage-3 AFD E2E model wrapper.

A deliberately small v1 wrapper for Attention-FFN Disaggregation, mirroring
``deepseek_v2.py``. It does NOT prune weights by role: both the Attention and
FFN sides build the full model and load the full checkpoint. Only the forward
path is split so hidden states travel through the AFD connector:

* Attention side: ``AFDHunYuanModel.forward`` reads AFD metadata from the
  forward context and runs ``forward_with_afd`` (per layer: compute attention,
  ``send_attn_output``, ``recv_ffn_output``).
* FFN side: the connector-driven FFN runner calls ``compute_ffn_output``.
"""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.models import hunyuan_v1 as native
from vllm_omni.model_executor.models.hunyuan_image3 import hunyuan_image3 as himg

from afd_plugin.connectors import AFDConnectorMetadata
from afd_plugin.model_executor.models import (
    get_afd_metadata_from_forward_context,
)
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

ImageHunyuanModel = himg.HunyuanModel
ImageForConditionalGeneration = himg.HunyuanImage3ForConditionalGeneration


class AFDHunYuanDecoderLayer(native.HunYuanDecoderLayer):
    """HunYuan decoder layer with separable Attention and FFN execution.

    The module is built in full (inherits the upstream ``__init__``); only the
    forward path is split into ``compute_attn_output`` / ``compute_ffn_output``.
    """

    def compute_attn_output(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        kv_states: tuple[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states, ori_kv_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_states=kv_states,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        return hidden_states, residual, ori_kv_states

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class AFDHunYuanModel(ImageHunyuanModel):
    """HunyuanImage-3 inner model that routes Attention outputs through AFD.

    Reuses the upstream ``__init__``/``load_weights``/expert mapping verbatim;
    only swaps the decoder layer class and overrides ``forward``.
    """

    def __init__(self, *, vllm_config: object, prefix: str = "") -> None:
        # Build the upstream model but with the AFD-aware decoder layer so each
        # layer exposes compute_attn_output / compute_ffn_output.
        original_layer = native.HunYuanDecoderLayer
        native.HunYuanDecoderLayer = AFDHunYuanDecoderLayer
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            native.HunYuanDecoderLayer = original_layer

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
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is not None:
            hidden_states, residual = self.forward_with_afd(
                hidden_states,
                residual,
                positions,
                afd_metadata,
            )
        else:
            cla_factor = native._get_cla_factor(self.config)
            prev_kv_states = None
            for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer),
            ):
                hidden_states, residual, kv_states = layer(
                    positions,
                    hidden_states,
                    residual,
                    prev_kv_states,
                )
                if getattr(self.config, "use_cla", False) and i % cla_factor == 0:
                    prev_kv_states = kv_states
                else:
                    prev_kv_states = None

        if not native.get_pp_group().is_last_rank:
            return native.IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual},
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_with_afd(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        afd_metadata: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        afd_connector = afd_metadata.afd_connector
        forward_context = get_forward_context()
        stage_idx = int(
            getattr(forward_context, "ubatch_idx", afd_metadata.afd_stage_idx),
        )
        cla_factor = native._get_cla_factor(self.config)
        prev_kv_states = None

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

            hidden_states, residual, kv_states = layer.compute_attn_output(
                positions,
                hidden_states,
                residual,
                prev_kv_states,
            )
            metadata = AFDConnectorMetadata.create_attention_metadata(
                layer_idx=layer.layer_id,
                stage_idx=stage_idx,
                seq_len=int(hidden_states.shape[0]),
            )
            afd_connector.send_attn_output(hidden_states, metadata)
            hidden_states = maybe_apply_dbo_yield(
                hidden_states,
                role="attention",
            )

            if (
                getattr(self.config, "use_cla", False)
                and layer_offset % cla_factor == 0
            ):
                prev_kv_states = kv_states
            else:
                prev_kv_states = None

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
        return self.layers[layer_idx].compute_ffn_output(hidden_states)


class AFDHunyuanImage3ForConditionalGeneration(ImageForConditionalGeneration):
    """HunyuanImage-3 causal-MM wrapper for AFD E2E.

    Reuses the upstream multimodal/__init__/load_weights machinery; only swaps
    the inner model for :class:`AFDHunYuanModel` and exposes ``compute_ffn_output``.
    """

    def __init__(self, *, vllm_config: object, prefix: str = "") -> None:
        # The upstream __init__ hardcodes ``self.model = HunyuanModel(...)``
        # referencing the module global. Swap it so the AFD inner model is built,
        # then everything else (VAE/ViT/lm_head, _patch_moe_blocks,
        # _replace_rotary_embeddings) runs unchanged.
        original_model = himg.HunyuanModel
        himg.HunyuanModel = AFDHunYuanModel
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            himg.HunyuanModel = original_model

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(weights)


# Registration alias: config.json may report either architecture name.
AFDHunyuanImage3ForCausalMM = AFDHunyuanImage3ForConditionalGeneration


__all__ = [
    "AFDHunyuanImage3ForCausalMM",
    "AFDHunyuanImage3ForConditionalGeneration",
]
