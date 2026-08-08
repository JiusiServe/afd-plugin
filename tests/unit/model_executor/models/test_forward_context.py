from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.model_executor.models import (  # noqa: E402
    ASYNC_MOE_UBATCH_METADATA_KEY,
    get_afd_metadata_from_forward_context,
    get_async_moe_ubatch_metadata_from_forward_context,
)


def test_get_afd_metadata_from_additional_kwargs():
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": {"stage": 0}},
        afd_metadata={"stage": 1},
    )

    assert get_afd_metadata_from_forward_context(forward_context) == {"stage": 0}


def test_get_afd_metadata_ignores_forward_context_attribute():
    forward_context = SimpleNamespace(
        additional_kwargs={},
        afd_metadata={"stage": 0},
    )

    assert get_afd_metadata_from_forward_context(forward_context) is None


def test_get_async_moe_ubatch_metadata_from_additional_kwargs():
    sidecar = {"ubatch_slices": ["stage0", "stage1"]}
    forward_context = SimpleNamespace(
        additional_kwargs={ASYNC_MOE_UBATCH_METADATA_KEY: sidecar},
    )

    assert (
        get_async_moe_ubatch_metadata_from_forward_context(forward_context) is sidecar
    )


@pytest.mark.parametrize("raise_inside", [False, True])
def test_async_moe_ubatch_forward_context_restores_state(raise_inside):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as async_forward,
    )

    class CloneableMetadata(SimpleNamespace):
        def clone(self):
            return CloneableMetadata(**vars(self))

    parent_metadata = CloneableMetadata(
        stage_idx=7,
        num_stages=1,
        tokens_unpadded_lens=[2, 3],
    )
    ubatch_slices = [
        SimpleNamespace(
            token_slice=slice(0, 2),
            request_slice=slice(0, 1),
            num_tokens=2,
        ),
        SimpleNamespace(
            token_slice=slice(2, 5),
            request_slice=slice(1, 2),
            num_tokens=3,
        ),
    ]
    async_metadata = {
        "ubatch_slices": ubatch_slices,
        "attn_metadata": ["attention-0", "attention-1"],
    }
    original_kwargs = {
        "afd_metadata": parent_metadata,
        "preserved": object(),
    }
    forward_context = SimpleNamespace(
        attn_metadata="parent-attention",
        additional_kwargs=original_kwargs,
        ubatch_idx=7,
        num_tokens=5,
    )

    def run_context():
        with async_forward._use_async_moe_ubatch_forward_context(
            forward_context=forward_context,
            parent_afd_metadata=parent_metadata,
            async_moe_ubatch_metadata=async_metadata,
            stage_idx=1,
        ):
            assert forward_context.attn_metadata == "attention-1"
            assert forward_context.ubatch_idx == 1
            assert forward_context.num_ubatches == 2
            assert forward_context.num_tokens == 3
            assert forward_context.additional_kwargs is not original_kwargs
            assert (
                forward_context.additional_kwargs["preserved"]
                is original_kwargs["preserved"]
            )
            stage_metadata = forward_context.additional_kwargs["afd_metadata"]
            assert stage_metadata is not parent_metadata
            assert stage_metadata.stage_idx == 1
            assert stage_metadata.tokens_start_loc == [2]
            assert stage_metadata.requests_start_loc == [1]
            assert stage_metadata.tokens_lens == [3]
            assert stage_metadata.tokens_unpadded_lens == [3]
            if raise_inside:
                raise RuntimeError("test failure")

    if raise_inside:
        with pytest.raises(RuntimeError, match="test failure"):
            run_context()
    else:
        run_context()

    assert forward_context.attn_metadata == "parent-attention"
    assert forward_context.additional_kwargs is original_kwargs
    assert forward_context.ubatch_idx == 7
    assert forward_context.num_tokens == 5
    assert not hasattr(forward_context, "num_ubatches")


@pytest.mark.parametrize(
    ("is_first_rank", "is_last_rank"),
    [(True, False), (False, True)],
)
def test_async_model_forward_preserves_pp_boundaries(
    monkeypatch,
    is_first_rank,
    is_last_rank,
):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as async_forward,
    )

    class FakeIntermediateTensors(dict):
        pass

    monkeypatch.setattr(async_forward, "IntermediateTensors", FakeIntermediateTensors)
    monkeypatch.setattr(
        async_forward,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_first_rank=is_first_rank,
            is_last_rank=is_last_rank,
        ),
    )
    forward_context = SimpleNamespace()
    afd_metadata = object()
    monkeypatch.setattr(async_forward, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        async_forward,
        "get_afd_metadata_from_forward_context",
        lambda context: afd_metadata if context is forward_context else None,
    )
    monkeypatch.setattr(
        async_forward,
        "get_async_moe_ubatch_metadata_from_forward_context",
        lambda context: None,
    )

    schedule_calls = []

    def run_schedule(
        model,
        hidden_states,
        residual,
        positions,
        received_metadata,
        llama_4_scaling,
    ):
        schedule_calls.append(
            (
                model,
                hidden_states,
                residual,
                positions,
                received_metadata,
                llama_4_scaling,
            )
        )
        next_residual = (
            torch.zeros_like(hidden_states) if residual is None else residual + 2
        )
        return hidden_states + 1, next_residual

    monkeypatch.setattr(
        async_forward,
        "run_attention_gate_afd_forward",
        run_schedule,
    )
    norm_calls = []

    def run_norm(hidden_states, residual):
        norm_calls.append((hidden_states, residual))
        return hidden_states + residual, None

    model = SimpleNamespace(
        aux_hidden_state_layers=(),
        embed_input_ids=lambda input_ids: input_ids.to(torch.float32).unsqueeze(-1),
        _get_llama_4_scaling=lambda positions: None,
        norm=run_norm,
    )
    positions = torch.arange(2)
    if is_first_rank:
        input_ids = torch.tensor([3, 4])
        intermediate_tensors = None
        expected_hidden_states = model.embed_input_ids(input_ids)
        expected_residual = None
    else:
        input_ids = None
        expected_hidden_states = torch.full((2, 1), 5.0)
        expected_residual = torch.full((2, 1), 7.0)
        intermediate_tensors = FakeIntermediateTensors(
            {
                "hidden_states": expected_hidden_states,
                "residual": expected_residual,
            }
        )

    output = async_forward.run_model_forward(
        model,
        input_ids,
        positions,
        intermediate_tensors,
    )

    assert len(schedule_calls) == 1
    assert torch.equal(schedule_calls[0][1], expected_hidden_states)
    assert schedule_calls[0][2] is expected_residual
    assert schedule_calls[0][4] is afd_metadata
    scheduled_hidden_states = expected_hidden_states + 1
    scheduled_residual = (
        torch.zeros_like(expected_hidden_states)
        if expected_residual is None
        else expected_residual + 2
    )
    if is_last_rank:
        assert len(norm_calls) == 1
        assert torch.equal(output, scheduled_hidden_states + scheduled_residual)
    else:
        assert isinstance(output, FakeIntermediateTensors)
        assert torch.equal(output["hidden_states"], scheduled_hidden_states)
        assert torch.equal(output["residual"], scheduled_residual)


def test_async_cam_profile_forward_skips_real_connector_io(monkeypatch):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as async_forward,
    )

    forward_context = SimpleNamespace(
        in_profile_run=True,
        ubatch_idx=0,
    )
    monkeypatch.setattr(async_forward, "get_forward_context", lambda: forward_context)

    connector_calls = []
    connector = SimpleNamespace(
        send_attn_output=lambda *args, **kwargs: connector_calls.append("send"),
        recv_ffn_output=lambda *args, **kwargs: connector_calls.append("recv"),
    )
    afd_metadata = SimpleNamespace(connector=connector, stage_idx=0)

    class _ProfileMoELayer:
        is_moe_layer = True
        layer_idx = 0

        def compute_attn_output(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        ):
            return (
                hidden_states + 1,
                residual,
                torch.ones((hidden_states.shape[0], 1)),
                torch.zeros((hidden_states.shape[0], 1), dtype=torch.int32),
                torch.ones((hidden_states.shape[0], 1)),
            )

    model = SimpleNamespace(
        layers=[_ProfileMoELayer(), _ProfileMoELayer()],
        start_layer=0,
        end_layer=2,
    )
    hidden_states = torch.zeros((2, 4))

    output, residual = async_forward.run_attention_gate_afd_forward(
        model,
        hidden_states,
        None,
        torch.arange(2),
        afd_metadata,
    )

    assert torch.equal(output, hidden_states + 2)
    assert residual is None
    assert connector_calls == []


def test_deepseek_afd_wrapper_keeps_full_model_compile_enabled():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert "@native.support_torch_compile\nclass AFDDeepseekV2Model" in source
    assert "from __future__ import annotations" not in source
    assert "self.do_not_compile = True" not in source


def test_deepseek_afd_wrapper_treats_index_topk_as_optional():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'self.is_v32 = hasattr(config, "index_topk")' in source
    assert "self.is_v32 = config.index_topk is not None" not in source
    assert "topk_tokens = config.index_topk" in source


def test_deepseek_afd_wrapper_treats_llama_4_scaling_as_optional():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'getattr(self.config, "llama_4_scaling", None)' in source
    assert "self.config.llama_4_scaling" not in source


def test_deepseek_afd_attention_path_can_compute_gate_before_send():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    executor_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py",
    ).read_text()
    module_imports = source.split("logger = init_logger(__name__)", 1)[0]
    model_source = source.split("class AFDDeepseekV2Model", 1)[1].split(
        "class AFDDeepseekV2ForCausalLM",
        1,
    )[0]
    model_forward = model_source.split("    def forward(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]
    gate_proxy = source.split("class GateOnlyRemoteMoE", 1)[1].split(
        "class AFDDeepseekV2RemoteExpertsMoE",
        1,
    )[0]
    attention_gate_forward = executor_source.split(
        "def run_attention_gate_afd_forward(",
        1,
    )[1].split("def run_async_moe_ubatch_afd_forward(", 1)[0]

    assert 'if afd_role == "attention":' in source
    assert "afd_plugin.model_executor.models.npu" not in module_imports
    assert "def _forward_attention(" not in source
    assert "return super().forward(" in model_forward
    assert "deepseek_v2_async_cam_forward.run_model_forward(" in model_forward
    assert "compute_gate_topk(" in gate_proxy
    assert "topk_weights=topk_weights" in gate_proxy
    assert "topk_ids=topk_ids" in gate_proxy
    assert "router_logits=router_logits" in gate_proxy
    assert "layer.compute_attn_output(" in attention_gate_forward
    assert "pending_ffn_recv" in attention_gate_forward
    assert "topk_weights" in attention_gate_forward
    assert "topk_ids" in attention_gate_forward


def test_deepseek_afd_attention_gate_can_force_balanced_topk_ids():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    module_imports = source.split("logger = init_logger(__name__)", 1)[0]
    compute_attn_output = source.split("    def compute_attn_output(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]

    assert "compute_attention_gate_topk(" in compute_attn_output
    assert "afd_plugin.model_executor.models.npu" not in module_imports
    assert "from afd_plugin.model_executor.models.npu import (" in compute_attn_output
    assert "deepseek_v2_attention_gate," in compute_attn_output
    assert "force_balanced_topk_ids_enabled" in gate_source
    assert "def _force_balanced_topk_ids(" in gate_source
    assert "topk_ids.copy_(balanced_topk_ids)" in gate_source
    assert "topk_weights, topk_ids = afd_connector.select_experts(" in (gate_source)
    assert "if force_balanced_topk_ids_enabled():" in gate_source
    assert (
        gate_source.index(
            "topk_weights, topk_ids = afd_connector.select_experts(",
        )
        < gate_source.index("if force_balanced_topk_ids_enabled():")
        < gate_source.index("topk_weights = topk_weights.to(torch.float32)")
    )


def test_deepseek_afd_gate_on_attention_keeps_dense_layers_local():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    executor_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py",
    ).read_text()

    assert "self.is_moe_layer = is_moe_layer" in source
    assert "self.compute_gate_on_attention and not self.is_moe_layer" in source
    assert "if not layer.is_moe_layer:" in executor_source
    assert (
        "return _ATTENTION_ROLE if compute_gate_on_attention else _FFN_ROLE" in source
    )


def test_deepseek_compute_gate_on_attention_selects_backend_boundary():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'device_type not in ("cuda", "npu")' in source
    assert "self.mlp = AFDDeepseekV2RemoteExpertsMoE(" in source
    assert "self.mlp = GateOnlyRemoteMoE(" in source
    assert 'prefix=f"{prefix}.mlp"' in source
    assert (
        "# NPU-only: Attention-side gate/topk is implemented in the NPU helper."
        in source
    )
    assert (
        "# NPU-only: gated MoE FFN compute consumes Attention-side topk payloads."
        in source
    )


def test_deepseek_async_moe_ubatching_runs_attention_inside_stage_context():
    executor_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py",
    ).read_text()
    model_forward = executor_source.split("def run_model_forward(", 1)[1].split(
        "def run_attention_gate_afd_forward(",
        1,
    )[0]
    async_ubatch_forward = executor_source.split(
        "def run_async_moe_ubatch_afd_forward(",
        1,
    )[1].split(
        "_MISSING_FORWARD_CONTEXT_ATTR = object()",
        1,
    )[0]

    assert "async_moe_ubatch_metadata" in model_forward
    assert "run_async_moe_ubatch_afd_forward(" in model_forward
    assert "_log_async_moe_forward_step(" not in async_ubatch_forward
    assert "first_moe_layer = int(model.config.first_k_dense_replace)" in (
        async_ubatch_forward
    )
    assert "dense_end_layer = min(model.end_layer, first_moe_layer)" in (
        async_ubatch_forward
    )
    assert "stage_hidden_states = [" in async_ubatch_forward
    assert (
        "moe_layers = list(islice(model.layers, moe_start_layer, model.end_layer))"
        in async_ubatch_forward
    )
    assert "def compute_stage_attention(" in async_ubatch_forward
    assert "def send_stage_attention(" in async_ubatch_forward
    assert "def recv_stage_ffn(" in async_ubatch_forward
    assert "for moe_layer_offset in range(last_moe_layer_offset):" in (
        async_ubatch_forward
    )
    assert "def flush_pending_ffn_outputs()" not in async_ubatch_forward
    assert "torch.cat(stage_hidden_states, dim=0)" in async_ubatch_forward
    assert "_run_async_moe_ubatch_layer(" not in executor_source
    assert "_recv_async_moe_ubatch_outputs(" not in executor_source
    assert "forward_context.attn_metadata = attn_metadata[stage_idx]" in executor_source
    assert async_ubatch_forward.index(
        "with _use_async_moe_ubatch_forward_context(",
    ) < (async_ubatch_forward.index("layer.compute_attn_output("))
    assert async_ubatch_forward.index(") = layer.compute_attn_output(") < (
        async_ubatch_forward.index("def send_stage_attention(")
    )
    assert async_ubatch_forward.index(
        "first_layer = moe_layers[0]",
    ) < async_ubatch_forward.index(
        "for moe_layer_offset in range(last_moe_layer_offset):",
    )
    assert async_ubatch_forward.index("recv_stage_ffn(0)") < (
        async_ubatch_forward.index(
            "send_stage_attention(\n            current_layer,\n            1",
        )
    )
    assert async_ubatch_forward.index("recv_stage_ffn(1)") < (
        async_ubatch_forward.index(
            "send_stage_attention(\n            next_layer,\n            0",
        )
    )
    assert async_ubatch_forward.index(
        "send_stage_attention(\n        last_layer,\n        1",
    ) < (async_ubatch_forward.rindex("recv_stage_ffn(1)"))


def test_deepseek_afd_ffn_path_reuses_ascend_moe_mlp_after_attention_gate():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    compute_ffn_output = source.split(
        "    def compute_ffn_output(",
        1,
    )[1].split("\n\n@native.support_torch_compile", 1)[0]
    compute_moe = gate_source.split(
        "def compute_attention_gate_moe_ffn(",
        1,
    )[1].split("\ndef _dequantize_int8_activation(", 1)[0]

    assert "compute_attention_gate_moe_ffn(" in compute_ffn_output
    assert "from afd_plugin.model_executor.models.npu import (" in compute_ffn_output
    assert "deepseek_v2_attention_gate," in compute_ffn_output
    assert "AFDF2ATransferPayload(" in compute_moe
    assert "MoEMlpComputeInput(" in compute_moe
    assert "unified_apply_mlp(" in compute_moe
    assert "routed_output, _ = unified_apply_mlp(" in compute_moe
    assert "quant_type == QuantType.W8A8" in compute_moe
    assert 'experts.get_eplb_parameter("w13_weight")' in compute_moe
    assert 'experts.get_eplb_parameter("w2_weight")' in compute_moe
    assert "experts.w13_weight" not in compute_moe
    assert "experts.w2_weight" not in compute_moe
    assert "w13_weight_scale_fp32" in compute_moe
    assert "w13_weight_scale_fp32_list" in compute_moe
    assert "w2_weight_scale_list" in compute_moe
    assert "MoEQuantParams(quant_type=quant_type)" in compute_moe
    assert "_gmmswigluquant_fusion_enabled()" in compute_moe
    assert "fusion=use_gmmswigluquant_fusion" in compute_moe
    assert "_compute_w8a8_shared_experts_from_int8(" in compute_moe
    assert "shared_input.dtype == torch.int8" in compute_moe
    assert "fusion=False" not in compute_moe


@pytest.mark.parametrize(
    ("num_routed_tokens", "num_shared_tokens"),
    [(2, 2), (2, 0), (0, 2), (0, 0)],
)
def test_deepseek_afd_ffn_skips_empty_rank_local_moe_work(
    monkeypatch,
    num_routed_tokens,
    num_shared_tokens,
):
    from afd_plugin.model_executor.models.npu import deepseek_v2_attention_gate

    class FakeQuantType:
        NONE = "none"
        W8A8 = "w8a8"

    class KeywordArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    routed_calls = []

    def fake_unified_apply_mlp(*, mlp_compute_input):
        routed_calls.append(mlp_compute_input.hidden_states)
        return (
            torch.zeros_like(
                mlp_compute_input.hidden_states,
                dtype=torch.bfloat16,
            ),
            None,
        )

    fake_moe_mlp = ModuleType("vllm_ascend.ops.fused_moe.moe_mlp")
    fake_moe_mlp.unified_apply_mlp = fake_unified_apply_mlp
    fake_stage_contracts = ModuleType(
        "vllm_ascend.ops.fused_moe.moe_stage_contracts",
    )
    fake_stage_contracts.MoEMlpComputeInput = KeywordArguments
    fake_stage_contracts.MoEWeights = KeywordArguments
    fake_stage_params = ModuleType(
        "vllm_ascend.ops.fused_moe.moe_stage_params",
    )
    fake_stage_params.MoEQuantParams = KeywordArguments
    fake_quant_type = ModuleType("vllm_ascend.quantization.quant_type")
    fake_quant_type.QuantType = FakeQuantType
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_mlp",
        fake_moe_mlp,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_stage_contracts",
        fake_stage_contracts,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_stage_params",
        fake_stage_params,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.quant_type",
        fake_quant_type,
    )

    shared_calls = []

    def fake_compute_shared(
        shared_experts,
        hidden_states,
        dynamic_scales,
        *,
        output_dtype,
    ):
        shared_calls.append((shared_experts, hidden_states, dynamic_scales))
        return torch.zeros_like(hidden_states, dtype=output_dtype)

    monkeypatch.setattr(
        deepseek_v2_attention_gate,
        "_compute_w8a8_shared_experts_from_int8",
        fake_compute_shared,
    )
    monkeypatch.setattr(
        deepseek_v2_attention_gate,
        "_gmmswigluquant_fusion_enabled",
        lambda: False,
    )

    shared_experts = object()
    experts = SimpleNamespace(
        quant_type=FakeQuantType.W8A8,
        dynamic_eplb=False,
        get_eplb_parameter=lambda name: name,
        activation="silu",
        _shared_experts=shared_experts,
    )
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            experts=experts,
            routed_scaling_factor=1.0,
        ),
    )
    hidden_states = torch.zeros((num_routed_tokens, 4), dtype=torch.int8)
    expand_x_shared = torch.zeros((num_shared_tokens, 4), dtype=torch.int8)

    output = deepseek_v2_attention_gate.compute_attention_gate_moe_ffn(
        layer,
        hidden_states=hidden_states,
        group_list=torch.zeros(2, dtype=torch.int64),
        dynamic_scales=torch.ones(num_routed_tokens),
        expand_x_shared=expand_x_shared,
        dynamic_scales_shared=torch.ones(num_shared_tokens),
        topk_scales=None,
        group_list_type=1,
    )

    assert len(routed_calls) == int(num_routed_tokens > 0)
    assert output.routed_output.shape == hidden_states.shape
    assert output.routed_output.dtype == torch.bfloat16
    assert len(shared_calls) == int(num_shared_tokens > 0)
    if num_shared_tokens > 0:
        assert output.shared_output is not None
        assert output.shared_output.shape == expand_x_shared.shape
    else:
        assert output.shared_output is None


def test_deepseek_afd_ffn_compute_omits_stub_io_diagnostics():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    compute_ffn_output = source.split(
        "    def compute_ffn_output(",
        1,
    )[1].split("\n\n@native.support_torch_compile", 1)[0]
    compute_moe = gate_source.split(
        "def compute_attention_gate_moe_ffn(",
        1,
    )[1].split("\ndef _dequantize_int8_activation(", 1)[0]

    assert "camp2p_stub_io_enabled()" not in source
    assert "_log_ffn_compute_step(" not in compute_ffn_output
    assert '"dense_mlp_begin"' not in compute_ffn_output
    assert '"dense_scaling_begin"' not in compute_ffn_output
    assert "_log_ffn_compute_step(" not in compute_moe
    assert '"routed_scaling_begin"' not in compute_moe
    assert '"shared_scaling_begin"' not in compute_moe
