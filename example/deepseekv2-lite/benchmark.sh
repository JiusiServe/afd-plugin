MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}
RESULT_DIR=${RESULT_DIR:-/path/results}
RESULT_FILENAME=${RESULT_FILENAME:-2p1a1f_graph_dbo.json}

mkdir -p "$RESULT_DIR"

uv run vllm bench serve \
    --host 127.0.0.1 --port 18300 \
    --model "$MODEL_PATH" \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 1024 \
    --request-rate inf \
    --max-concurrency 32 \
    --result-dir "$RESULT_DIR" \
    --result-filename "$RESULT_FILENAME" \
    --save-result
