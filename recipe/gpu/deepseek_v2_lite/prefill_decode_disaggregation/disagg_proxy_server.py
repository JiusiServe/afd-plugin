# SPDX-License-Identifier: Apache-2.0
"""Minimal prefill/decode disaggregation proxy for the AFD 2P1A1F example.

Keeps only what the LMCache PD protocol needs to serve ``/v1/completions`` and
run ``vllm bench serve``:

  1. tokenize the prompt on a prefiller (round-robin),
  2. prefill one token with the decoder's NIXL endpoint injected via
     ``kv_transfer_params["disagg_spec"]``; the prefiller pushes the prompt KV
     into the decoder's PD buffer and returns the first sampled token,
  3. wait on a ZMQ PULL socket for the sender's notification that the KV landed,
  4. decode the remaining tokens on the decoder (prompt + first token) and return
     the completion (prompt + first token prepended) as a plain JSON response.

Dropped vs the upstream vendored proxy: the chat endpoint, TTFT stats, the
PD-buffer semaphore (the decoder already does reservation-based admission
control), session affinity and multi-decoder routing.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
from contextlib import asynccontextmanager

import httpx
import msgspec
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, Request
from lmcache.v1.storage_backend.pd_backend import PDMsg, ProxyNotif

logger = logging.getLogger("disagg_proxy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Minimal AFD prefill/decode disaggregation proxy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18300)
    p.add_argument("--prefiller-host", default="127.0.0.1")
    p.add_argument("--prefiller-port", default="18301",
                   help="comma-separated prefiller ports, scheduled round-robin")
    p.add_argument("--decoder-host", default="127.0.0.1")
    p.add_argument("--decoder-port", type=int, default=18305)
    p.add_argument("--decoder-init-port", default="7300",
                   help="comma-separated NIXL init ports, one per decoder TP rank")
    p.add_argument("--decoder-alloc-port", default="7400",
                   help="comma-separated NIXL alloc ports, one per decoder TP rank")
    p.add_argument("--proxy-host", default="127.0.0.1")
    p.add_argument("--proxy-port", type=int, default=7500)
    return p.parse_args()


args = parse_args()
INIT_PORTS = [int(x) for x in str(args.decoder_init_port).split(",")]
ALLOC_PORTS = [int(x) for x in str(args.decoder_alloc_port).split(",")]
NUM_TP_RANKS = len(INIT_PORTS)

_req_counter = itertools.count(1)
# req_id -> number of sender notifications seen (one per decoder TP rank)
_kv_ready: dict[str, int] = {}
_running = True


async def _zmq_pull_loop() -> None:
    """Count ProxyNotif messages senders emit once KV lands in the decoder."""
    sock = zmq.asyncio.Context.instance().socket(zmq.PULL)
    sock.bind(f"tcp://{args.proxy_host}:{args.proxy_port}")
    while _running:
        try:
            raw = await asyncio.wait_for(sock.recv(), timeout=0.5)
        except (asyncio.TimeoutError, TimeoutError, zmq.Again):
            continue
        except zmq.ZMQError:
            break
        try:
            msg = msgspec.msgpack.decode(raw, type=PDMsg)
        except msgspec.DecodeError as e:
            logger.warning("[DEBUG-PROXY] ZMQ DecodeError: %s  raw=%s", e, raw[:64])
            continue
        if isinstance(msg, ProxyNotif):
            _kv_ready[msg.req_id] = _kv_ready.get(msg.req_id, 0) + 1
    sock.close()


async def _wait_kv_ready(req_id: str) -> None:
    while _kv_ready.get(req_id, 0) < NUM_TP_RANKS:
        await asyncio.sleep(1e-4)
    _kv_ready.pop(req_id, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    prefillers = [
        httpx.AsyncClient(
            base_url=f"http://{args.prefiller_host}:{int(port)}",
            timeout=None, limits=limits,
        )
        for port in str(args.prefiller_port).split(",")
    ]
    app.state.prefillers = prefillers
    app.state.prefiller_cycle = itertools.cycle(prefillers)
    app.state.decoder = httpx.AsyncClient(
        base_url=f"http://{args.decoder_host}:{args.decoder_port}",
        timeout=None, limits=limits,
    )
    app.state.zmq_task = asyncio.create_task(_zmq_pull_loop())
    yield
    global _running
    _running = False
    await app.state.zmq_task
    for client in (*prefillers, app.state.decoder):
        await client.aclose()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/completions")
async def completions(request: Request) -> dict:
    req = await request.json()
    req_id = str(next(_req_counter))
    prefiller = next(request.app.state.prefiller_cycle)
    decoder = request.app.state.decoder

    # 1) tokenize so prefill and decode share identical token ids (and so we can
    #    append the prefill's first token to the decode prompt).
    resp = await prefiller.post("/tokenize", json={"prompt": req["prompt"]})
    resp.raise_for_status()
    tokens = resp.json()["tokens"]

    org_max_tokens = int(req.get("max_tokens", 16))

    # 2) prefill one token; push the prompt KV into the decoder's PD buffer.
    prefill_req = dict(req)
    prefill_req.pop("stream_options", None)
    prefill_req.update(
        prompt=tokens,
        max_tokens=1,
        stream=False,
        kv_transfer_params={
            "ret_first_tok": True,
            "disagg_spec": {
                "req_id": req_id,
                "receiver_host": args.decoder_host,
                "receiver_init_port": INIT_PORTS,
                "receiver_alloc_port": ALLOC_PORTS,
            },
        },
    )
    resp = await prefiller.post("/v1/completions", json=prefill_req)
    resp.raise_for_status()
    prefill = resp.json()
    first_token = prefill["kv_transfer_params"]["first_tok"]

    # 3) wait until every decoder TP rank reports the KV has landed.
    await _wait_kv_ready(req_id)

    # 4) decode the rest on the decoder, prompt extended by the first token.
    decode_req = dict(req)
    decode_req.pop("kv_transfer_params", None)
    decode_req.pop("stream_options", None)
    decode_req.update(
        prompt=tokens + [first_token],
        max_tokens=max(org_max_tokens - 1, 0),
        stream=False,
    )
    resp = await decoder.post("/v1/completions", json=decode_req)
    resp.raise_for_status()
    completion = resp.json()

    # prepend the prefill's first token to the decoded text.
    completion["choices"][0]["text"] = (
        prefill["choices"][0]["text"] + completion["choices"][0]["text"]
    )
    return completion


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
