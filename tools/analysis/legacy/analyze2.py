import collections
lines = open("/tmp/afd_dsv4_async/ffn.log", "r", errors="replace").read().splitlines()
recv = collections.Counter()
comb = collections.Counter()
section = None
marker = "first5=["
for line in lines:
    if "async_dispatch_recv outputs" in line:
        section = "recv"
        continue
    if "async_combine_send inputs" in line:
        section = "comb"
        continue
    if "async_dispatch_recv inputs" in line or "async_combine_recv" in line or "async_dispatch_send" in line:
        section = None
        continue
    i = line.find(marker)
    if i < 0 or section is None:
        continue
    j = line.find("]", i)
    vals = [int(x.strip()) for x in line[i + len(marker):j].split(",")]
    key = (vals[1], vals[2])
    if section == "recv":
        recv[key] += 1
    elif section == "comb":
        comb[key] += 1
print("=== dispatch_recv outputs (rankid, layer) ===")
for k in sorted(recv):
    print(recv[k], k)
print("=== combine_send inputs (rankid, layer) ===")
for k in sorted(comb):
    print(comb[k], k)
