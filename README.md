# LLM Serving Benchmarks with vLLM

This project explores **LLM inference serving using vLLM**, with a focus on
**throughput, concurrency, GPU memory behavior, and KV cache effects**.

The goal is to understand how large language models behave in **real serving
environments**, rather than measuring model accuracy or training performance.

---

## What This Project Covers

- Running an OpenAI-compatible LLM server using **vLLM**
- Measuring inference performance under different concurrency levels
- Observing **GPU memory residency vs utilization**
- Understanding how **KV cache and context length** impact serving
- Identifying failure modes such as context-length violations

This project reflects **systems-level LLM engineering**, not prompt engineering.

---

## Model & Serving Stack

- **Model**: Mistral-7B-Instruct
- **Serving Engine**: vLLM
- **Precision**: FP16
- **Hardware**: NVIDIA GPU (cloud-hosted)
- **API**: OpenAI-compatible REST interface

---

## Project Structure

llm_serving_benchmarks/
├── bench_vllm.py # Single-request and streaming benchmarks
├── load_test.py # Concurrent load testing for throughput & latency
├── kv_scale.py # Experiments related to KV cache & context length
├── README.md
└── .gitignore

---

## Script Overview

### `bench_vllm.py`
Runs single-request and streaming inference tests to measure:
- Time to first token (TTFT)
- Token generation speed
- End-to-end request duration

Useful for understanding **baseline latency characteristics**.

---

### `load_test.py`
Simulates concurrent client requests against the vLLM server to observe:
- Throughput scaling
- Latency under load
- Request scheduling behavior

This reflects **production-style inference workloads**.

---

### `kv_scale.py`
Explores how increasing context length affects:
- KV cache memory usage
- GPU memory residency
- Request feasibility within model limits

Used to understand **why KV cache becomes the bottleneck** in LLM serving.

---

## Key Learnings

- GPU memory can remain fully allocated even when utilization is low
- Throughput scales with concurrency due to batching and scheduling
- Context length directly impacts feasibility and memory usage
- Input validation occurs before GPU compute is triggered

These behaviors are expected in **production LLM inference systems**.

---

## Why This Matters

Modern LLM systems are limited by **serving efficiency**, not model size.
This project demonstrates practical knowledge of:

- LLM serving internals
- GPU inference constraints
- Production failure cases
- Performance debugging mindset

---

## Author

**Karun**  
Interests: LLM Serving Systems, GPU Inference, AI Infrastructure
