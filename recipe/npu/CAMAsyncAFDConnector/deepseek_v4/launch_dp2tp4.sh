#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/afd_dsv4_async_dp2tp4}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-120}"
mkdir -p "$LOG_DIR"

: >"$LOG_DIR/ffn.log"
setsid nohup env AFD_DP_SIZE=8 AFD_TP_SIZE=1 AFD_ATTN_RANKS_PER_DP=4 AFD_SHARED_FFN_POOL=true \
  bash "$SCRIPT_DIR/ffn_ep8.sh" >"$LOG_DIR/ffn.log" 2>&1 < /dev/null &
echo $! >"$LOG_DIR/ffn.pid"

sleep "$START_DELAY_SECONDS"

: >"$LOG_DIR/attention.log"
setsid nohup env AFD_DP_SIZE=2 AFD_TP_SIZE=4 AFD_SHARED_FFN_POOL=true \
  bash "$SCRIPT_DIR/attention_tp8.sh" >"$LOG_DIR/attention.log" 2>&1 < /dev/null &
echo $! >"$LOG_DIR/attention.pid"

printf 'FFN PID: %s\nAttention PID: %s\nLogs: %s\n' \
  "$(cat "$LOG_DIR/ffn.pid")" \
  "$(cat "$LOG_DIR/attention.pid")" \
  "$LOG_DIR"
