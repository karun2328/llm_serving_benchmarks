# Benchmarks Overview

This document describes the **benchmarking methodology** and **experiments executed**
as part of the vLLM serving evaluation.

Only **completed benchmarks** are documented here.
Planned or future experiments are listed separately without placeholder metrics.

---

## Completed Benchmarks

### 1. Context Length Scaling (512 → 4096)

**Objective**

Evaluate whether increasing input context length impacts:
- GPU utilization
- Memory usage (KV cache)
- Time to first token (TTFT)
- Token generation throughput

**Result Summary**

- GPU utilization remained saturated (≈97–99%)
- GPU memory usage plateaued at ~21.2 GB
- TTFT remained stable across all tested lengths
- Token throughput remained invariant

This confirms that vLLM’s paged attention mechanism decouples
generation cost from prompt length within supported limits.

---

### 2. Concurrency Scaling (8 → 64)

**Objective**

Measure throughput and latency behavior under increasing concurrent requests.

**Result Summary**

- Throughput scaled near-linearly with concurrency
- Latency increased moderately due to queueing
- GPU memory usage remained constant
- No instability observed under valid workloads

This reflects expected batching and scheduling behavior in production LLM servers.

---

### 3. Context Length Violation (Failure Case)

**Objective**

Validate system behavior when requests exceed model context limits.

**Result Summary**

- Requests rejected during input validation
- No GPU compute triggered
- GPU utilization dropped to idle
- No memory corruption or instability

This confirms correct early-rejection behavior.

---

## GPU Observability

Across all benchmarks:
- Model weights and KV cache remained resident in GPU memory
- Memory residency did not imply active inference
- GPU utilization correlated only with valid inference workloads

This behavior matches production-grade inference systems.

---

## Future Work

The following experiments are planned but **not yet executed**:

- INT8 / INT4 quantized inference
- TensorRT-LLM comparison
- Streaming vs non-streaming latency comparison
- Long-running soak tests
- Multi-GPU tensor parallelism

These will be added once measurements are completed.

---
