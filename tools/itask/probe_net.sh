echo START
echo "=== /proc/net/tcp lines ==="
wc -l /proc/net/tcp 2>&1
head -n 3 /proc/net/tcp 2>&1
echo "=== established tcp (local->remote state inode) ==="
awk 'NR>1 {print $2, $3, $4, $10}' /proc/net/tcp 2>&1 | grep -v '0100007F' | head -n 50
echo "=== listening ==="
awk 'NR>1 && $4=="0A" {print $2, $4}' /proc/net/tcp 2>&1 | head -n 30
echo "=== tcp6 ==="
awk 'NR>1 {print $2, $3, $4}' /proc/net/tcp6 2>&1 | head -n 20
echo "=== pgrep python ==="
pgrep -a -f 'vllm serve' | head -n 10
echo "=== children of 336159 ==="
cat /proc/336159/task/*/children 2>/dev/null | tr ' ' '\n' | grep -v '^$' | head -n 40
echo DONE