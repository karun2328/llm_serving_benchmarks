import time, asyncio, aiohttp, statistics

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

async def stream_once(session, prompt, max_tokens=256):
    payload = {
        "model": MODEL,
        "messages": [{"role":"user","content": prompt}],
        "temperature": 0.2,
        "stream": True,
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()
    ttft_ms = None
    token_events = 0

    async with session.post(URL, json=payload) as r:
        async for raw in r.content:
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                break
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
            token_events += 1

    t1 = time.perf_counter()
    dur_s = t1 - t0
    tps = token_events / dur_s if dur_s > 0 else 0.0
    return ttft_ms, tps, dur_s, token_events

async def main():
    prompts = [
        "Explain KV cache in simple terms.",
        "Give 6 bullets on why batching increases GPU throughput.",
        "What is paged attention? Explain briefly.",
    ]

    # Warmup
    async with aiohttp.ClientSession() as session:
        await stream_once(session, "Say hello in one sentence.", max_tokens=32)

        ttfts, tpss, durs = [], [], []
        for p in prompts:
            for _ in range(3):  # 3 runs each
                ttft_ms, tps, dur_s, events = await stream_once(session, p, max_tokens=256)
                ttfts.append(ttft_ms)
                tpss.append(tps)
                durs.append(dur_s)
                print(f"prompt='{p[:28]}...'  TTFT={ttft_ms:.1f} ms | tokens/sec={tps:.2f} | dur={dur_s:.2f}s | events={events}")

        print("\n=== SUMMARY (streaming) ===")
        ttfts_sorted = sorted(ttfts)
        tpss_sorted = sorted(tpss)
        print(f"TTFT ms:  p50={statistics.median(ttfts_sorted):.1f}  p95={ttfts_sorted[int(0.95*len(ttfts_sorted))-1]:.1f}")
        print(f"Tokens/s: p50={statistics.median(tpss_sorted):.2f}  p95={tpss_sorted[int(0.95*len(tpss_sorted))-1]:.2f}")

if __name__ == "__main__":
    asyncio.run(main()) 
