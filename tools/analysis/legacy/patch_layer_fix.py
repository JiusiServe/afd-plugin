import sys

path = "/mnt/d/cyj/afd/afd-plugin/afd_plugin/connectors/npu/async_cam.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

old = """        # Preserve the vendor-produced count/rank fields and repair only its
        # stale layer field before returning the header to combine-send.
        layer_idx = int(expected_layer_idx)
        token_nums_rankid_layeridx = token_nums_rankid_layeridx.clone()
        token_nums_rankid_layeridx[2] = layer_idx
        states.token_nums_rankid_layeridx = token_nums_rankid_layeridx
        states.cam_dp_group_index = cam_dp_group_index"""

new = """        # CAM returns the true decoder-layer index in the dispatch-recv header
        # (field 2 of token_nums_rankid_layeridx).  Use it for both the local
        # FFN layer computation and the combine-send header so the completion
        # matches the attention-side combine-recv.  Do NOT overwrite field 2
        # with expected_layer_idx: the shared-FFN-pool scheduler alternates
        # DP groups per layer, so expected_layer_idx is the busy-loop
        # iteration index, not the true decoder layer.
        layer_idx = max(0, int(token_nums_rankid_layeridx[2].item()))
        token_nums_rankid_layeridx = token_nums_rankid_layeridx.clone()
        states.token_nums_rankid_layeridx = token_nums_rankid_layeridx
        states.cam_dp_group_index = cam_dp_group_index"""

count = src.count(old)
assert count == 1, f"expected 1 occurrence, got {count}"
src = src.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print("patched OK")
