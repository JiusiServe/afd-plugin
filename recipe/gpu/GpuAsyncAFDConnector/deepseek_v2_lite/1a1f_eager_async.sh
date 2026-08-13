#!/usr/bin/env bash
# 1A1F async GPU connector, eager, prefill-only.
#
# Launch under a GPU reservation, which sets CUDA_VISIBLE_DEVICES:
#   gpu run --gpu-ids 3,7 -- bash recipe/gpu/GpuAsyncAFDConnector/deepseek_v2_lite/1a1f_eager_async.sh
#
# The two roles are separate vllm serve processes: the AFD process group hosts
# its own TCPStore, which cannot be created under a single torchrun/torchelastic
# launcher.
set -u

MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}
LOG_DIR=${LOG_DIR:-.}
mkdir -p "$LOG_DIR"
export VLLM_USE_V2_MODEL_RUNNER=0
# Single node over NVLink: skip the IB transport probe.
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-none}
# Two servers on one box spawn a lot of threads; the HF tokenizer's rayon pool
# is the first thing to fail when thread creation gets refused.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [ "${#DEVICES[@]}" -lt 2 ]; then
    echo "need 2 visible GPUs, got ${#DEVICES[@]}: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
fi
ATTN_DEVICES="${DEVICES[0]}"
FFN_DEVICES="${DEVICES[1]}"
echo "attention on ${ATTN_DEVICES}, ffn on ${FFN_DEVICES}"

# Lower this when sharing a box: vLLM refuses to start if the desired
# fraction exceeds what is actually free.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
# Prefill batch size drives whether each MoE call clears the compute-bound
# inflection point, so it is the knob to raise when benchmarking.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-512}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
AFD_PORT=${AFD_PORT:-6275}
API_PORT=${API_PORT:-18311}

CUDA_VISIBLE_DEVICES="$ATTN_DEVICES" uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "role": "attention",
            "connector": "GpuAsyncAFDConnector",
            "async": true,
            "compute_gate_on_attention": true,
            "host": "127.0.0.1",
            "port": '"$AFD_PORT"',
            "num_attention_ranks": 1,
            "num_ffn_ranks": 1
        }
    }' \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --api-server-count 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enforce-eager \
    --host 127.0.0.1 \
    --port "$API_PORT" \
    --trust-remote-code > "$LOG_DIR/attn.log" 2>&1 &
ATTN_PID=$!

CUDA_VISIBLE_DEVICES="$FFN_DEVICES" uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "role": "ffn",
            "connector": "GpuAsyncAFDConnector",
            "async": true,
            "compute_gate_on_attention": true,
            "host": "127.0.0.1",
            "port": '"$AFD_PORT"',
            "num_attention_ranks": 1,
            "num_ffn_ranks": 1
        }
    }' \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --api-server-count 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enforce-eager \
    --host 127.0.0.1 \
    --port "$API_PORT" \
    --trust-remote-code > "$LOG_DIR/ffn.log" 2>&1 &
FFN_PID=$!

cleanup() {
    kill "$ATTN_PID" "$FFN_PID" 2>/dev/null
    wait "$ATTN_PID" "$FFN_PID" 2>/dev/null
}
trap cleanup EXIT

for _ in $(seq 1 "${READY_TIMEOUT:-600}"); do
    if curl -sf "http://127.0.0.1:$API_PORT/health" > /dev/null 2>&1; then
        echo "server ready on http://127.0.0.1:$API_PORT"
        echo
        echo "curl -s http://127.0.0.1:$API_PORT/v1/completions \\"
        echo "  -H 'Content-Type: application/json' \\"
        echo "  -d '{\"model\":\"$MODEL_PATH\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}'"
        echo
        if [ -n "${SMOKE:-}" ]; then
            curl -s "http://127.0.0.1:$API_PORT/v1/completions" \
                -H 'Content-Type: application/json' \
                -d '{"model":"'"$MODEL_PATH"'","prompt":"The capital of France is",
                     "max_tokens":16,"temperature":0}'
            echo
            exit 0
        fi
        # Stay up so the servers can take requests; Ctrl-C tears both down.
        wait "$ATTN_PID" "$FFN_PID"
        exit 0
    fi
    if ! kill -0 "$ATTN_PID" 2>/dev/null || ! kill -0 "$FFN_PID" 2>/dev/null; then
        echo "a server exited early; see $LOG_DIR/attn.log and $LOG_DIR/ffn.log" >&2
        exit 1
    fi
    sleep 1
done
echo "timed out waiting for the server" >&2
exit 1
