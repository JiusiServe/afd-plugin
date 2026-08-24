import os
print("exists:", os.path.exists("/tmp/afd_dsv4_async/ffn.log"))
print("size:", os.path.getsize("/tmp/afd_dsv4_async/ffn.log"))
with open("/tmp/afd_dsv4_async/ffn.log", "r", errors="replace") as f:
    for i, line in enumerate(f):
        if "first5=" in line:
            print("line", i, line[:200])
            break
print("done")
