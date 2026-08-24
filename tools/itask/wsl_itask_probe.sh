#!/bin/bash
for i in 1 2 3 4 5 6; do
  out=$(itask exec afd_bjf_26_a3_new -- bash -c 'python3 - <<"PYEOF"
import os, glob
pids = [336413,336477,336587,336692,336802,336907,337016,337121,  # FFN workers
        341047,341172,341382,341596,                               # attn dp0
        341060,341204,341424,341638]                               # attn dp1
# map inode -> (local, remote, state)
socks = {}
for f in ["/proc/net/tcp", "/proc/net/tcp6"]:
    try:
        with open(f) as fh:
            lines = fh.read().strip().splitlines()[1:]
    except Exception:
        continue
    for ln in lines:
        parts = ln.split()
        if len(parts) < 10: continue
        local, remote, state, inode = parts[1], parts[2], parts[3], parts[9]
        socks[inode] = (local, remote, state)
def dec(addr):
    hexpart, port = addr.split(":")
    port = int(port, 16)
    # ipv4-mapped ipv6
    if hexpart.startswith("0000000000000000FFFF0000"):
        h = hexpart[24:]
        ip = ".".join(str(int(h[i:i+2],16)) for i in (6,4,2,0))
    elif len(hexpart) == 8:
        ip = ".".join(str(int(hexpart[i:i+2],16)) for i in (6,4,2,0))
    else:
        ip = hexpart
    return f"{ip}:{port}"
for p in pids:
    try:
        fds = os.listdir(f"/proc/{p}/fd")
    except Exception:
        continue
    conns = []
    for fd in fds:
        try:
            target = os.readlink(f"/proc/{p}/fd/{fd}")
        except Exception:
            continue
        if target.startswith("socket:["):
            ino = target[8:-1]
            if ino in socks:
                l, r, s = socks[ino]
                if r.split(":")[1] != "0000":
                    conns.append((dec(l), dec(r), s))
    print(f"PID {p}: " + (", ".join(f"{l}->{r} {s}" for l,r,s in conns) if conns else "no established"))
PYEOF' 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "$out"
    exit 0
  fi
  echo "retry $i rc=$rc" >&2
  sleep 4
done
exit 1
