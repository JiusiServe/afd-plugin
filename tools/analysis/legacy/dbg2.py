import collections
lines = open("/tmp/afd_dsv4_async/ffn.log", "r", errors="replace").read().splitlines()
print("total lines:", len(lines))
marker = "first5=["
cnt = 0
recv = 0
comb = 0
for line in lines:
    if marker in line:
        cnt += 1
        if "async_dispatch_recv outputs" in line:
            recv += 1
        if "async_combine_send inputs" in line:
            comb += 1
print("marker hits:", cnt, "recv:", recv, "comb:", comb)
