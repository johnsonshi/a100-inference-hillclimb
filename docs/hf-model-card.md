---
license: apache-2.0
base_model:
- Nanbeige/Nanbeige4.3-33B
library_name: transformers
pipeline_tag: text-generation
language:
- en
- zh
tags:
- llm-compressor
- compressed-tensors
- int8
- w8a8
- quantized
- vllm
- sglang
- nanbeige
- a100
---

# Nanbeige4.3-33B-W8A8-INT8

This repository contains a **data-free INT8 W8A8 (`compressed-tensors`)** quantization of [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B), produced with [`vllm-project/llm-compressor`](https://github.com/vllm-project/llm-compressor) for high-throughput serving on **NVIDIA Ampere (A100, A10G, RTX 3090)** and **Ada (L40S, RTX 4090)** GPUs in **vLLM** and **SGLang**.

## Why INT8 W8A8 on Ampere (A100)?

The official `Nanbeige4.3-33B` quantized checkpoints (`INT4 W4A16`, `INT8 W8A16`, and `FP8`) all execute as **weight-only (`WxA16`)** on NVIDIA A100 GPUs:
- A100 GPUs do not have native FP8 tensor cores, so FP8 checkpoints unpack weights to BF16 before matrix multiplication.
- Weight-only `INT4 (W4A16)` and `INT8 (W8A16)` checkpoints save HBM bandwidth at batch size 1, but dequantize weights back to BF16 inside every GEMM kernel and run compute on **BF16 tensor cores** (312 TFLOPS peak). Under concurrent batched load or long 8k-token prefills, unpacking INT4 weights makes an 8k prefill slower than unquantized BF16 (`3.47s` vs. `2.74s`).

By quantizing **both weights (per-channel symmetric INT8) and activations (dynamic per-token symmetric INT8)** while keeping `lm_head` in BF16, this **INT8 W8A8** checkpoint runs matrix multiplications directly on the **A100's INT8 tensor cores (624 TOPS peak, 2× BF16)** while cutting weight memory in half (≈31 GiB, freeing ≈28 GiB of HBM for KV cache).

## Benchmark & Quality Results (1× NVIDIA A100 80GB PCIe)

Evaluated in a 337-trial automated hill-climbing benchmark across **vLLM 0.34.0** and **SGLang 0.5.23** on a single Azure `Standard_NC24ads_A100_v4` node (A100 80GB PCIe):

### 1. Quality Verification (Zero Measurable Accuracy Loss on GSM8K & Passkey Retrieval)

| Evaluation | Unquantized BF16 (`Nanbeige/Nanbeige4.3-33B`) | `Nanbeige4.3-33B-W8A8-INT8` (BF16 KV) | `Nanbeige4.3-33B-W8A8-INT8` (FP8 KV) |
|---|---|---|---|
| **GSM8K** (200 problems, thinking on, SE ±1.1%) | 97.5% (vLLM) / 98.0% (SGLang) | **97.5%** (vLLM) / **98.5%** (SGLang) | **97.5–98.0%** (vLLM) / **97.0%** (SGLang) |
| **Long-Context Passkey Retrieval** (8k, 16k, 28k tokens) | 9 / 9 (100%) | **9 / 9 (100%)** | **9 / 9 (100%)** |
| **Top-1 Next-Token Agreement with BF16** (3,827 greedy tokens) | 99.1% (BF16 re-run noise floor) | **94.1–94.5%** (+0.032–0.037 nats/tok NLL) | **93.8–94.3%** (+0.037–0.039 nats/tok NLL) |

### 2. Serving Performance on 1× A100 80GB PCIe (TTFT ≤ 2s, TPOT ≤ 100ms SLO)

| Workload | vLLM BF16 Defaults | `W8A8-INT8` (vLLM + FP8 KV) | `W8A8-INT8` (SGLang) | Improvement |
|---|---|---|---|---|
| **8k-Token Prompt Prefill (TTFT at `c=1`)** | 2.74 s (INT4 W4A16: 3.47 s) | **1.68 s** | **1.58 s** | **1.6–1.7× faster prefill** |
| **Batched ShareGPT Chat (2-min capacity)** | 1.72 req/s | **3.83 req/s** (`3.27` with BF16 KV) | **4.40 req/s** (`--enable-mixed-chunk`) | **2.2–2.6× capacity** |
| **Batched ShareGPT Chat (9-min sustained)** | 1.45–1.50 req/s | **3.21 req/s** | **3.24 req/s** | **2.2× sustained capacity** |
| **Synthetic Load (1,024 in / 512 out, sustained)** | 1.13 req/s (2-min) | **2.31 req/s** (`2.61` in 2-min) | **2.25 req/s** (`2.31` in 2-min) | **2.3× (2-min)** |
| **Multi-Turn Agentic (48 sessions × 6 turns, p90 TTFT)** | 114.4 s (70k KV pool, 11 preemptions) | **1.05 s** (418k KV pool, 0 preemptions) | **0.94 s** (with 64 GiB host HiCache) | **≈110–120× lower p90 TTFT** |
| **Single-User + Batched (with batch-scheduled suffix spec)** | 23.4 tok/s (`c=1`) | **55.1 tok/s (`c=1`) / 3.84 chat req/s** | 43.0 tok/s (`c=1`, no spec) | **2.4× single-user + 2.2× chat** |

## Usage

### vLLM

**1. High-throughput batched serving & multi-turn agentic workloads:**

```bash
vllm serve __REPO_ID__ \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --safetensors-load-strategy prefetch
```

**2. Combined single-user speed (`55.1 tok/s`) + full batched chat capacity (`3.84 req/s`) using batch-scheduled Suffix Decoding:**

```bash
vllm serve __REPO_ID__ \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --safetensors-load-strategy prefetch \
  --speculative-config '{"method": "suffix", "num_speculative_tokens": 4, "num_speculative_tokens_per_batch_size": [[1, 4, 4], [5, 16, 2], [17, 4096, 0]]}'
```

### SGLang

**1. High-throughput batched chat:**

```bash
python3 -m sglang.launch_server \
  --model-path __REPO_ID__ \
  --context-length 32768 \
  --mem-fraction-static 0.91 \
  --enable-mixed-chunk \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

**2. Multi-turn agentic sessions with host-DRAM KV caching (`0.94s` p90 TTFT):**

```bash
python3 -m sglang.launch_server \
  --model-path __REPO_ID__ \
  --context-length 32768 \
  --mem-fraction-static 0.91 \
  --enable-hierarchical-cache \
  --hicache-size 64 \
  --hicache-write-policy write_through \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

## Quantization Recipe (`llm-compressor`)

This checkpoint was created data-free from [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) using `llmcompressor`:
- **Weights:** Per-channel symmetric INT8 (`Linear` layers, memoryless minmax observer)
- **Activations:** Dynamic per-token symmetric INT8
- **Unquantized layers:** `lm_head` kept in BF16

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Nanbeige/Nanbeige4.3-33B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
oneshot(
    model=model,
    recipe=QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
)
model.save_pretrained("Nanbeige4.3-33B-W8A8-INT8", save_compressed=True)
AutoTokenizer.from_pretrained("Nanbeige/Nanbeige4.3-33B").save_pretrained("Nanbeige4.3-33B-W8A8-INT8")
```

## License

This model inherits the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0) from the base model [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B).
