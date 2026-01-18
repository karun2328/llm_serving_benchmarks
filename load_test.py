import asyncio
import time
import statistics
import subprocess
import aiohttp

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

BASE = (
    "Explain batching in one paragraph. Keep it short and technical."
)
PROMPT=BASE*1200

MAX_TOKENS = 60     
TEMPERATURE = 0.0     
TOTAL_REQUESTS = 128  
TIMEOUT_S = 180        
CONCURRENCY_STEPS = [8, 16, 32, 64]  


def gpu_mem_mib():
    """
    Returns used GPU memory in MiB (int) for GPU 0.
    If nvidia-smi is unavailable, returns None.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        first = out.splitlines()[0].strip()
        return int(first)
    except Exception:
        return None


async def one_request(session):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    t0 = time.perf_counter()
    try:
        async with session.post(URL, json=payload) as r:
            _ = await r.json()
            if r.status != 200:
                return None
    except Exception:
        return None

    return (time.perf_counter() - t0) * 1000.0  


async def run_step(concurrency, total, timeout_s):
    sem = asyncio.Semaphore(concurrency)

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(limit=0)  

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def gated():
            async with sem:
                return await one_request(session)

        t_start = time.perf_counter()
        results = await asyncio.gather(*[gated() for _ in range(total)])
        t_end = time.perf_counter()

    times = [x for x in results if x is not None]
    failures = total - len(times)
    wall = t_end - t_start
    qps = (len(times) / wall) if wall > 0 else 0.0

    if times:
        times.sort()
        p50 = statistics.median(times)
        p95 = times[int(0.95 * len(times)) - 1]
        return qps, p50, p95, failures, len(times), wall
    else:
        return 0.0, float("nan"), float("nan"), failures, 0, wall


async def main():
    print("\n=== vLLM LOAD TEST (non-streaming) ===")
    print(f"URL={URL}")
    print(f"MODEL={MODEL}")
    print(f"prompt_len_chars={len(PROMPT)}  max_tokens={MAX_TOKENS}  temp={TEMPERATURE}")
    print(f"total_per_step={TOTAL_REQUESTS}  timeout={TIMEOUT_S}s")
    print(f"concurrency_steps={CONCURRENCY_STEPS}\n")

   
    try:
        timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            _ = await one_request(session)
    except Exception:
        pass

    for c in CONCURRENCY_STEPS:
        mem_before = gpu_mem_mib()
        if mem_before is not None:
            print(f"[GPU] before concurrency={c}: mem_used={mem_before} MiB")

        qps, p50, p95, failures, ok, wall = await run_step(
            concurrency=c, total=TOTAL_REQUESTS, timeout_s=TIMEOUT_S
        )

        mem_after = gpu_mem_mib()
        if mem_after is not None:
            print(f"[GPU] after  concurrency={c}:mem_used={mem_after} MiB")

        if ok > 0:
            print(
                f"concurrency={c:<3}  ok={ok:<3} fail={failures:<3}  "
                f"QPS={qps:.2f}  p50={p50:.0f}ms  p95={p95:.0f}ms  wall={wall:.1f}s"
            )
        else:
            print(
                f"concurrency={c:<3}  ok=0   fail={failures:<3}  "
                f"QPS=0.00  (all failed or timed out)  wall={wall:.1f}s"
            )

        print("-" * 72)

    print("\nDone.\n")


if __name__ == "__main__":
    asyncio.run(main())
