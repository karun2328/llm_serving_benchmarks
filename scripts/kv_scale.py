import time, asyncio, aiohttp, statistics

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

def make_prompt(repeats: int) -> str:
    base = "This is filler text to increase the prompt context length. "
    return (base * repeats) + "\nIn 3 bullets, explain KV cache."

async def stream_once(session, prompt: str, max_tokens: int = 60):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()
    ttft_ms = None
    events = 0

    async with session.post(URL, json=payload) as r:
        async for raw in r.content:
            if not raw:
                continue
            s = raw.decode("utf-8", errors="ignore").strip()
            if not s.startswith("data:"):
                continue
            if s == "data: [DONE]":
                break
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
            events += 1

    dur = time.perf_counter() - t0
    tps = events / dur if dur > 0 else 0.0
    return ttft_ms, tps

async def main():
    steps = [
        ("~512", 60),
        ("~1024", 120),
        ("~2048", 240),
        ("~3072", 360),
    ]

    async with aiohttp.ClientSession() as session:
        # warmup
        await stream_once(session, "Hello", max_tokens=16)

        for label, reps in steps:
            prompt = make_prompt(reps)
            ttfts, tpss = [], []

            for _ in range(3):
                ttft, tps = await stream_once(session, prompt)
                if ttft is not None:
                    ttfts.append(ttft)
                tpss.append(tps)

            ttft_p50 = statistics.median(ttfts) if ttfts else float("nan")
            tps_p50 = statistics.median(tpss)

            print(
                f"context={label}  "
                f"TTFT_p50={ttft_p50:.1f}ms  "
                f"tok/s_p50={tps_p50:.2f}"
            )

if __name__ == "__main__":
    asyncio.run(main())
