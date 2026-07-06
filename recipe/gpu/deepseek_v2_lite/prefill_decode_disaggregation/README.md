# Prefill/Decode Disaggregation — 2P1A1F

End-to-end prefill/decode (PD) disaggregation for DeepSeek-V2-Lite on the AFD
plugin. Two prefill producers compute KV and push it over NIXL into the decode
tier; a proxy stitches the halves into one OpenAI-compatible endpoint.

## Topology (one GPU each)

| GPU | Role             | Worker / HTTP            | LMCache role            |
|-----|------------------|--------------------------|-------------------------|
| 0   | Prefill producer #1 | default · 18301       | PD `sender`             |
| 1   | Prefill producer #2 | default · 18302       | PD `sender`             |
| 2   | Decode attention | `AFDAttentionWorker` · 18303 | PD `receiver` (KV owner) |
| 3   | Decode FFN       | `AFDFFNWorker` · 18304   | none (holds no KV cache)|

Client traffic goes to the **proxy on `18305`**, never to `18303` directly.

## How the halves connect (the part that was missing before)

This uses LMCache's **PD mode** over NIXL (`enable_pd: true`,
`transfer_channel: nixl`, `pd_role: sender|receiver`), driven by
`disagg_proxy_server.py`, a minimal proxy serving `/v1/completions`. Per request
it:

1. tokenizes the prompt via the prefiller's `/tokenize`;
2. runs prefill with `max_tokens=1`, injecting the decoder's NIXL endpoint via
   `kv_transfer_params["disagg_spec"]` (`receiver_host` + `init_port` 7300 +
   `alloc_port` 7400) and `ret_first_tok`;
3. the prefill **sender** pushes the KV straight into the decoder's PD buffer
   and notifies the proxy's **ZMQ PULL socket** (`7500`) when it lands;
4. the proxy waits for that notification, prepends the prefill's first token,
   and streams the decode — which now finds the KV already in its buffer.

It is intentionally small: only `/v1/completions`, no TTFT stats, no PD-buffer
semaphore (the decoder does its own reservation-based admission control), no
session affinity. Streaming is kept because `vllm bench serve` requires an SSE
response.

Two requirements are easy to miss and both are handled by the launch scripts:

- **`PYTHONHASHSEED` must match across every participant.** LMCache keys KV
  chunks by Python's string hash; if the senders and the receiver hash
  differently the receiver can't match the pushed chunks and silently
  recomputes (`hit/load = 0`). The scripts `export PYTHONHASHSEED=0`.
- Only the **attention** worker is a PD receiver (it loads `decode.yaml`); the
  FFN worker holds no KV cache, gets no `LMCACHE_CONFIG_FILE`, and uses a
  distinct `lmcache_rpc_port` so its connector IPC socket doesn't collide.

## Run

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
bash 2p1a1f_graph_dbo.sh        # or 2p1a1f_eager_dbo.sh
```

The script launches the four vLLM servers + the proxy (foreground; `wait`s on
the servers). Wait until `attn.log` (`afd_prefill0.log` and `afd_prefill1.log`) print `Application startup complete` and the proxy is up on `18300`, then:

```bash
MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
curl http://127.0.0.1:18300/v1/completions   -H "Content-Type: application/json"   -d '{"model": $MODEL_PATH,"prompt": "The capital of France is","max_tokens": 16,"temperature": 0}'
```

Confirm KV actually transferred — `attn.log` should show a real hit, e.g.
`LMCache hit tokens: 40, need to load: 40` with `computed tokens: 0`
(a `hit/load = 0` line means the prompt was recomputed, i.e. transfer failed).

Per-process logs: `afd_prefill.log`, `afd_prefill1.log`, `attn.log`, `ffn.log`,
`proxy.log`.

## Ports

| Port  | Purpose                                  |
|-------|------------------------------------------|
| 18305 | proxy (client-facing endpoint)           |
| 18301 / 18302 | prefill producers (HTTP)         |
| 18303 | decode attention (HTTP)                  |
| 6269  | AFD attention↔FFN p2p connector          |
| 7300 / 7400 | decoder NIXL init / alloc ports    |
| 7500  | proxy ZMQ PULL (sender → proxy notifies)  |

## Requirements

- 4 GPUs, DeepSeek-V2-Lite weights.
- `lmcache` (PD/NIXL build) and `nixl` installed in the vLLM + `afd-plugin` env.
- The proxy needs `httpx`, `msgspec`, `pyzmq` (plus `lmcache` for the PD wire types).
