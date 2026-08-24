import collections
txt = open("/tmp/afd_dsv4_async/ffn.log", "r", errors="replace").read().splitlines()
recv = collections.Counter()
comb = collections.Counter()
marker = "first5=["
for line in txt:
    i = line.find(marker)
    if i < 0:
        continue
    j = line.find("]", i)
    if j < 0:
        continue
    vals = [int(x.strip()) for x in line[i + len(marker):j].split(",")]
    key = (vals[1], vals[2])
    if "async_dispatch_recv outputs" in line:
        recv[key] += 1
    elif "async_combine_send inputs" in line:
        comb[key] += 1
print("=== dispatch_recv outputs (rankid, layer) ===")
for k in sorted(recv):
    print(recv[k], k)
print("=== combine_send inputs (rankid, layer) ===")
for k in sorted(comb):
    print(comb[k], k)
