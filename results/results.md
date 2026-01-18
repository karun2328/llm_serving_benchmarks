# vLLM Inference Benchmark Results

## Experimental Setup

**Model**
- mistralai/Mistral-7B-Instruct-v0.2

**Inference Engine**
- vLLM (OpenAI-compatible API server)

**Precision**
- FP16

**Attention Mechanism**
- PagedAttention (vLLM default)

**Hardware**
- GPU: NVIDIA L4 (24 GB)
- CUDA: 12.7
- Driver: 565.57.01

**Serving Configuration**
- max_model_len: 4096
- gpu_memory_utilization: 0.90
- Streaming enabled

---

## Objective

Evaluate vLLM serving behavior under:
- Increasing **context length** (512 → 4096 tokens)
- Increasing **concurrency** (8 → 64 requests)
- Measure latency, throughput, GPU utilization, memory behavior, and failure modes

---

## 1. GPU Utilization & Memory Stability

Across all valid workloads, GPU usage remained saturated and memory usage stable.

| Context Length (tokens) | GPU Utilization | GPU Memory Used | Power |
|-------------------------|----------------|-----------------|-------|
| 512                     | 97–99%         | ~21.2 GB        | ~71 W |
| 1024                    | 98–99%         | ~21.2 GB        | ~71 W |
| 2048                    | 98–99%         | ~21.2 GB        | ~72 W |
| 3072                    | 97–99%         | ~21.2 GB        | ~72 W |
| 4096                    | 97–99%         | ~21.2 GB        | ~72 W |

**Observation**
- KV cache memory is allocated once and reused
- No incremental memory growth with longer contexts
- System remains compute-bound, not memory-bound

---

## 2. Time To First Token (TTFT)

TTFT remains stable even at maximum context length.

| Context Length | TTFT p50 (ms) | TTFT p95 (ms) |
|---------------|--------------|--------------|
| 1024          | ~66–69        | ~70–73       |
| 2048          | ~65–67        | ~70–71       |
| 3072          | ~66–69        | ~71–72       |
| 4096          | ~66–70        | ~71–73       |

**Observation**
- Prompt ingestion latency does not scale with input length
- vLLM parallelizes prompt processing efficiently
- PagedAttention avoids quadratic attention cost

---

## 3. Token Generation Throughput

Token throughput remains effectively constant across context sizes.

| Context Length | Tokens/sec (p50) | Tokens/sec (p95) |
|---------------|------------------|------------------|
| 1024          | ~17.71           | ~17.73           |
| 2048          | ~17.71           | ~17.71           |
| 3072          | ~17.70           | ~17.72           |
| 4096          | ~17.71           | ~17.72           |

**Observation**
- Decode phase dominates inference cost
- Throughput is decoupled from prompt length
- Long-context workloads do not reduce generation speed

---

## 4. Load & Concurrency Testing (Non-Streaming)

**Test Configuration**
- Requests per step: 128
- max_tokens: 60
- Prompt length: ~63 chars
- Timeout: 180s

### Results

| Concurrency | QPS  | p50 Latency (ms) | p95 Latency (ms) | Wall Time |
|------------|------|------------------|------------------|-----------|
| 8          | 2.23 | 3583             | 3713             | 57.5 s    |
| 16         | 4.42 | 3616             | 3648             | 29.0 s    |
| 32         | 7.98 | 4002             | 4031             | 16.0 s    |
| 64         | 14.90| 4277             | 4299             | 8.6 s     |

**Observation**
- Near-linear QPS scaling with concurrency
- Latency increases moderately due to queueing
- GPU memory usage remains constant (~21.2 GB)
- GPU utilization remains ~98–99%

---

## 5. Failure Case: Context Length Violation

**Input**
- Prompt length: ~16,800 tokens
- Model limit: 4096 tokens

**Observed Behavior**
- Requests rejected with HTTP 400
- Error raised during input validation
- GPU utilization dropped to ~0%
- No wasted GPU compute

**Conclusion**
- vLLM enforces hard context limits early
- Invalid requests do not consume GPU resources

---

## 6. Key Findings

1. Context length up to 4096 tokens does **not** reduce throughput
2. TTFT remains stable across all valid input sizes
3. GPU memory usage plateaus due to KV cache reuse
4. Throughput scales cleanly with concurrency
5. vLLM rejects invalid workloads safely and early

---

## 7. Production Implications

- Suitable for long-context workloads (RAG, agents, chat history)
- Prompt length does not complicate capacity planning
- PagedAttention enables predictable scaling
- Strong candidate for high-throughput LLM serving

---
