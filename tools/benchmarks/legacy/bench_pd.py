#!/usr/bin/env python3
"""独立 PD serving 测压脚本：直打 OpenAI /v1/completions(流式)，测 TTFT / TPOT / 吞吐。
不依赖 vllm 的 benchmark 代码，只要 aiohttp + 一个可达的 endpoint。
用法见文件底部注释。
"""
import argparse
import asyncio
import json
import random
import string
import time

import aiohttp


def gen_prompt(approx_words: int) -> str:
    # 造一段随机"英文"，长度近似 approx_words 个词
    return "".join(random.choice(string.ascii_letters + " ") for _ in range(approx_words * 6))


async def one_request(session, url, model, prompt, out_len, idx, results):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": out_len,
        "temperature": 1.0,
        "stream": True,
    }
    t_send = time.perf_counter()
    ttft = None
    n_tok = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                results.append({"idx": idx, "error": f"HTTP {resp.status}: {body[:120]}"})
                return
            async for raw in resp.content:
                line = raw.decode(errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices", [])
                if not choices:
                    continue
                text = choices[0].get("text") or ""
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - t_send
                    n_tok += 1  # vllm 流式通常每 chunk 1 token，近似计数
    except Exception as e:
        results.append({"idx": idx, "error": f"{type(e).__name__}: {e}"})
        return
    t_end = time.perf_counter()
    lat = t_end - t_send
    results.append({
        "idx": idx, "ttft": ttft, "lat": lat, "n_tok": n_tok,
        "tpot": (lat - ttft) / (n_tok - 1) if (ttft and n_tok > 1) else None,
    })


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="proxy 的地址，如 http://33.215.116.107:8000")
    ap.add_argument("--model", required=True, help="服务端 served-model-name（没设就是权重路径）")
    ap.add_argument("--endpoint", default="/v1/completions")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--request-rate", type=float, default=10.0, help="请求/秒；inf 表示一次性全发")
    ap.add_argument("--input-len", type=int, default=512, help="输入近似词数")
    ap.add_argument("--output-len", type=int, default=128, help="max_tokens")
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + args.endpoint
    prompts = [gen_prompt(args.input_len) for _ in range(args.num_prompts)]
    results = []
    delay = 0.0 if args.request_rate in (0, float("inf")) else 1.0 / args.request_rate

    async with aiohttp.ClientTimeout(total=None) as _t, aiohttp.ClientSession(timeout=_t) as session:
        t0 = time.perf_counter()
        tasks = []
        for i in range(args.num_prompts):
            tasks.append(asyncio.create_task(one_request(session, url, args.model, prompts[i], args.output_len, i, results)))
            if delay:
                await asyncio.sleep(delay)
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]
    tot_tok = sum(r["n_tok"] for r in ok)
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    tpots = [r["tpot"] for r in ok if r["tpot"] is not None]
    lats = [r["lat"] for r in ok]

    def stat(xs):
        if not xs:
            return (0, 0, 0)
        xs = sorted(xs)
        n = len(xs)
        return (xs[n // 2], sum(xs) / n, xs[int(n * 0.95)])

    ttft_p50, ttft_mean, ttft_p95 = stat([x * 1000 for x in ttfts])
    tpot_p50, tpot_mean, tpot_p95 = stat([x * 1000 for x in tpots])
    lat_p50, lat_mean, lat_p95 = stat([x * 1000 for x in lats])

    print("\n================ PD Serving Benchmark ================")
    print(f"endpoint        : {url}")
    print(f"requests        : {len(results)} (ok={len(ok)}, err={len(err)})")
    print(f"input/output    : ~{args.input_len} words / {args.output_len} tokens")
    print(f"request rate    : {args.request_rate} req/s")
    print(f"wall time       : {wall:.2f} s")
    print("------------------------------------------------------")
    print(f"output tokens   : {tot_tok}")
    print(f"output tput     : {tot_tok / wall:.2f} tok/s")
    print(f"request tput    : {len(ok) / wall:.2f} req/s")
    print(f"TTFT  p50/mean/p95 : {ttft_p50:.1f} / {ttft_mean:.1f} / {ttft_p95:.1f} ms")
    print(f"TPOT  p50/mean/p95 : {tpot_p50:.1f} / {tpot_mean:.1f} / {tpot_p95:.1f} ms")
    print(f"Lat   p50/mean/p95 : {lat_p50:.1f} / {lat_mean:.1f} / {lat_p95:.1f} ms")
    if err:
        print("------------------------------------------------------")
        print(f"errors ({len(err)}):")
        for e in err[:5]:
            print(f"  req{e['idx']}: {e['error']}")
    print("======================================================")


if __name__ == "__main__":
    # 示例：
    #   python bench_pd.py --base-url http://33.215.116.107:8000 \
    #     --model /home/admin/model-csi/model --num-prompts 200 --request-rate 10
    asyncio.run(main())
