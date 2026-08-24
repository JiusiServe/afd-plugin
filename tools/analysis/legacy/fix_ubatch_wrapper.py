import io

path = "afd-plugin/afd_plugin/v1/worker/npu/npu_ubatch_wrapper.py"
with io.open(path, "r", encoding="utf-8") as f:
    text = f.read()

old = """        self.comm_stream = torch.npu.Stream(device=device)
        assert self.vllm_config.parallel_config.num_ubatches == AFD_NPU_NUM_UBATCHES
        self.ready_barrier = threading.Barrier(_READY_BARRIER_PARTIES)"""

new = """        self.comm_stream = torch.npu.Stream(device=device)
        # vLLM native ubatching/DBO is rejected for CAMAsyncAFDConnector by
        # fail_if_unsupported_npu_afd_features; the AFD async MoE ubatching
        # path drives stage counts through connector_extra_config
        # (async_moe_num_ubatches), which is validated there as well.
        if self.vllm_config.parallel_config.use_ubatching:
            raise RuntimeError(
                "AscendUBatchWrapper does not support vLLM native "
                "ubatching/DBO; use the AFD async MoE ubatching path.",
            )
        self.ready_barrier = threading.Barrier(_READY_BARRIER_PARTIES)"""

assert old in text, "target block not found"
text = text.replace(old, new)
with io.open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("patched ok")
