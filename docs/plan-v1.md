# Hill-climbing harness for LLM serving on A100: plan (draft for your review)

> **Status: plan only.** Nothing has been built, installed, or changed. 18 Nov 2026.
>
> **Scope:** one model, one A100 80GB per node (`<cluster>`, 2 nodes), vLLM and SGLang. A multi-day automated search for higher tokens/s and lower time to first token (TTFT) that stops when it plateaus.

---

## Decisions I need from you

| # | Decision | My recommendation |
|---|---|---|
| 1 | Model | **Nanbeige4.3-33B**, released 24 Oct 2026 ([§2](#2-the-model-nanbeige43-33b)) |
| 2 | What "better" means | Three tracks scored separately: interactivity, throughput within a latency target, and multi-turn agentic ([§1](#1-what-better-means-three-tracks)) |
| 3 | Latency target for the throughput track | MLPerf-style: TTFT p99 ≤ 2 s and time per output token p99 ≤ 100 ms, plus an "interactive" tier of 0.5 s / 30 ms |
| 4 | Quality bar for quantization and speculative decoding | Mini-eval within noise of BF16: average drop ≤ 1.5 points, no single eval worse by more than 3 points |
| 5 | Who drives the climb | A scripted climber first. Optionally, once the harness is trusted, I run it as an agent on a schedule ([§4.5](#45-who-drives-it)). |
| 6 | Community artifacts (DFlash2 draft, MTP head, NVFP4 quants) | Allow them, labeled "community", safetensors only, no `trust_remote_code` |
| 7 | Day 0 cluster changes | Needs your approval: GPU device plugin, driver check, namespace, storage, ACR images. Also confirm you're cleared to run multi-day jobs in `<cluster-rg>` (both nodes ≈ $200/day at pay-as-you-go; they're already billing while idle). |
| 8 | Where the code lives | A new repo, `<repo>/` |

---

## TL;DR

- **The machine:** a harness launches one server config on one A100, runs a fixed, versioned test suite against it, and scores it. A climber changes one setting at a time and keeps a change only if it beats the current best by more than measurement noise *and* passes a quality check. When no single change helps and random restarts don't either, that track stops.
- **The subject: Nanbeige4.3-33B.**
  - Released 25 days ago. Tied with Qwen3.8-27B for the highest Artificial Analysis (AA) Intelligence Index among open models of 40B parameters or fewer.
  - Its maker publishes BF16, FP8, INT8, and INT4 checkpoints, and the community has published speculative-decoding drafts.
  - The vendor's own serving example uses 4 GPUs. **Fitting it on one A100 is the hill.**
- **The hill is big.** At BF16, the A100's memory bandwidth caps single-user decode at ≈30 tok/s and leaves room for only ≈65k tokens of KV cache. INT4 raises the cap to ≈98 tok/s with ≈4× the KV room, and speculative decoding can multiply single-user speed again.
- **Two nodes = two evaluation lanes**, doubling search speed, plus a final two-replica routing experiment. No tensor parallelism and no prefill/decode split across nodes, since they're connected by Ethernet only.
- **Timeline:** about 5 working days after a Day 0 for prerequisites. Each track stops at its plateau or when the budget runs out.

---

## Contents

1. [What "better" means: three tracks](#1-what-better-means-three-tracks)
2. [The model: Nanbeige4.3-33B](#2-the-model-nanbeige43-33b)
3. [The search space](#3-the-search-space)
4. [The climber](#4-the-climber)
5. [The tests](#5-the-tests)
6. [Harness architecture](#6-harness-architecture)
7. [Timeline](#7-timeline)
8. [Hypotheses to confirm or kill](#8-hypotheses-to-confirm-or-kill)
9. [Risks](#9-risks)
10. [Deliverables](#10-deliverables)
11. [Sources](#sources)

---

## 1. What "better" means: three tracks

Each track is its own hill with one score. Every trial records all metrics, so trade-offs between tracks stay visible.

| Track | Question | Score | Workload |
|---|---|---|---|
| **T1: Interactivity** | How fast does one user see tokens? | Decode tok/s at concurrency 1 (= 1000 ÷ time per output token in ms). Guard: TTFT p50 must not get more than 10 % worse. | W1 and W3 ([§5.2](#52-workloads-fixed-versioned-seeded)) |
| **T2: Goodput** | How much traffic can one A100 serve within the latency target? | Highest output tok/s, across concurrency levels, counting only requests that meet TTFT p99 ≤ 2 s and time per output token p99 ≤ 100 ms | W1 and W2, concurrency 1 → 256 |
| **T3: Agentic** | How fast are long, multi-turn sessions? | TTFT p90 at a fixed session arrival rate (lower is better), with throughput as a guard | W4 |

- Every trial also reports AA-style output speed (output tokens ÷ end-to-end time) and time to first *answer* token, so results compare with AA's numbers.
- I read "high TTFT" in your message as *low* TTFT, since lower is better.
- **Any accepted config must** have zero request errors, be stable across repeats, and pass the quality gate ([§5.3](#53-quality-gate)).

---

## 2. The model: Nanbeige4.3-33B

**Why this one**

- **Recent:** released 24 Oct 2026. The official FP8, INT8, and INT4 versions followed on 31 Oct.
- **Strong:** tied with Qwen3.8-27B for the top AA Intelligence Index among open models ≤ 40B (≈ 20 on v5.0.1). AA lists no hosted API speed numbers for it yet, so your A100 numbers would be new.
- **Easy to serve fairly:** a plain `LlamaForCausalLM` architecture, fully supported by vLLM and SGLang on Ampere. No exotic kernels block either engine, so the engine comparison is fair.
- **Many knobs to climb:**
  - Four official precisions.
  - Community NVFP4 and AutoRound quantizations.
  - A community DFlash2 draft (its card reports 1.5–1.8× single-user decode in SGLang, measured on an RTX PRO 6000 rather than an A100) and a community MTP head.
  - n-gram and suffix speculation, which need no draft model.
- **Reasoning model:** thinking is on by default. Its card serves it with `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`, so time to first answer token matters, as it does for AA.
- **Vendor claims** (not verified by us): SWE-Bench Pro 60.3, GPQA Diamond 89.5, AIME 2026 95.2.

**Architecture:** 33.1B dense parameters; 52 layers; 56 query heads and 8 KV heads; head dimension 128; a 166,912-token vocabulary with a separate (untied) output head; 512K context.
**KV cache: 208 KiB per token in BF16.** That's heavy, so KV capacity is a first-class constraint.

**Memory math on one A100 80GB.** Rough figures: ≈79 GiB usable, ≈4 GiB reserved for activations and CUDA graphs.

| Variant | Format | Weights | KV pool | KV tokens (BF16 KV) | Single-user decode ceiling* |
|---|---|---|---|---|---|
| BF16 | — | ≈61.7 GiB | ≈13 GiB | ≈65k | ≈30 tok/s |
| FP8 (official) | compressed-tensors, block-scaled weights. Expected to run weight-only (W8A16, Marlin kernel) on Ampere; verify on Day 1. | ≈33 GiB | ≈42 GiB | ≈210k | ≈58 tok/s |
| INT8 (official) | Weight-only (W8A16), group size 128 | ≈33 GiB | ≈42 GiB | ≈210k | ≈58 tok/s |
| INT4 (official) | Weight-only (W4A16), group size 32 | ≈20.5 GiB | ≈54 GiB | ≈275k | ≈98 tok/s |
| Any of the above + FP8 KV cache | — | same | same | ×2 | same |

\* Memory-bandwidth roof: 1,935 GB/s ÷ bytes read per token (model body plus output head). Real kernels land below this; speculative decoding can beat it per user.

**Alternatives, if you prefer a different subject**

- **Qwen3.8-27B** (Aug 2026): tied on quality, with the richest speculative-decoding ecosystem (a built-in MTP head, NVIDIA's draft model from 11 Nov, Tencent's DSpine draft from 7 Nov). But it's older than your one-month window, and its hybrid linear-attention architecture isn't listed as verified on A100 in vLLM's recipe.
- **An MoE model as a control** for kernel tuning, such as Qwen3-Coder-30B-A3B from the main report. It exercises tuning of vLLM's fused-MoE Triton kernels.

---

## 3. The search space

Concurrency is part of the *test*, not a knob: every trial sweeps load and reports its best operating point. Server-side limits on admitted requests are knobs.

| Layer | Knobs (vLLM v0.34 / SGLang v0.5.23 names) | Expected impact |
|---|---|---|
| **L0: Engine** | vLLM vs SGLang (pinned image digests) | Medium |
| **L1: Weights** | Official BF16, FP8, INT8 (W8A16), INT4 (W4A16). Community NVFP4 and AutoRound W4A16/W3A16. **Our own INT8 W8A8 checkpoint**, built with llm-compressor: the only way to use the A100's double-rate INT8 tensor cores, and so the only weight format that should speed up prefill. | **High** (T1, T2) |
| **L2: KV cache** | `--kv-cache-dtype auto/fp8` (in vLLM on Ampere, FP8 KV needs FlashInfer or Triton attention); block/page size; prefix caching on or off; CPU offload `--kv-offloading-size` 0, 32, 64, or 128 GiB | **High** (T2, T3) |
| **L3: Speculative decoding** | vLLM `--speculative-config` methods: `ngram`, `ngram_gpu`, `suffix`, `dflash`, `mtp`, `draft_model`. SGLang `--speculative-algorithm`, e.g. `NGRAM` or `DFLASH` (per the draft's card). Draft length 1–8. | **High** at low concurrency, likely negative at high |
| **L4: Scheduler** | `--max-num-seqs` / `--max-running-requests`; `--max-num-batched-tokens` / `--chunked-prefill-size`; `--scheduling-policy` / `--schedule-policy`; `--async-scheduling`. SGLang: `--disable-overlap-schedule`, `--num-continuous-decode-steps`, `--enable-mixed-chunk`, `--schedule-conservativeness`. `--api-server-count`. | Medium–high (T2) |
| **L5: Kernels and compilation** | **Attention backend:** vLLM `FLASH_ATTN` (FlashAttention-2), `FLASHINFER`, `TRITON_ATTN`, `FLEX_ATTENTION`; SGLang `flashinfer`, `triton`, and `fa3` if it runs on Ampere (verify). **CUDA graphs:** mode `FULL_AND_PIECEWISE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`, or `NONE`; `--cudagraph-capture-sizes`. **Compile level:** `-O0` through `-O3`. **Sampler:** `VLLM_USE_FLASHINFER_SAMPLER`. **SGLang:** `--cuda-graph-max-bs-decode`, `--enable-torch-compile`. | Medium |
| **L6: Custom kernels and model surgery** (stretch, driven by profiling) | Tuned Triton configs via `VLLM_TUNED_CONFIG_FOLDER`. A quantized output head (with a 166,912-token vocabulary, the head is ≈12 % of the bytes read per token at INT4). Custom fused kernels, only if profiling shows a hotspot the engines don't cover. | Unknown, evidence-driven |
| **L7: Topology (two nodes)** | One replica; two replicas with round-robin; two replicas with prefix- and load-aware routing (llm-d Optimized Baseline) | High, but only for T3 |

- **Validity rules** are encoded so the climber never wastes a run. Examples: FP8 KV with FlashAttention-2 in vLLM on Ampere is invalid; the DFlash2 draft targets SGLang, and its compatibility with vLLM's `dflash` method is unverified; engine-specific flags apply only to that engine. An unknown combination is tried once and blacklisted if it crashes.
- **The space is far too big to search exhaustively** (millions of combinations), hence staged search plus hill climbing.

---

## 4. The climber

### 4.1 Stages

1. **Calibrate the noise floor.** Run each engine's default BF16 config 5× on each node. This gives the run-to-run noise for each track's score and any bias between the two nodes. The acceptance threshold is max(2 %, 2 × the standard deviation).
2. **Macro screen, to find the promising regions.** Short screening runs over the high-impact combinations: engine × weights × KV dtype × speculative method, about 40–60 valid combos. Use successive halving: every combo gets a quick screen, and the top quarter gets a full evaluation. The top 3 become starting points.
3. **Coordinate ascent: the climb itself.**
   1. From each starting point, go through the knobs in that track's order (below).
   2. For each knob, try its neighboring values.
   3. Keep the best change only if it beats the current best by more than the threshold *and* passes the quality gate.
   4. Repeat full passes until a pass changes nothing.
4. **Kicks and restarts, to escape local optima.** Randomly change 2–3 knobs of the best config and climb again. Stop after 3 restarts in a row fail to beat the best.
5. **Verify.** Re-run the winners 5× on both nodes, and run the full quality evals.

**Knob order by track**

- **T1:** weights → speculative method → draft length → CUDA graph mode and sizes → attention backend → compile level → sampler → async scheduling.
- **T2:** weights → KV dtype → max concurrent sequences → prefill chunk budget → speculative decoding (expect off) → attention backend → CUDA graph sizes → async/overlap scheduling → API-server count.
- **T3:** prefix caching → CPU KV offload size → schedule policy (SGLang `lpm`) → prefill chunk budget → weights and KV dtype → routing (two nodes).

### 4.2 Acceptance rule (so we don't fool ourselves)

- The challenger and the current best run **on the same node, interleaved** (A, B, A, B) to cancel out drift.
- Accept only if the challenger beats the current best by more than the threshold in both repeats (or a bootstrap confidence interval excludes zero) *and* it passes the quality gate.
- Re-measure the current best every ≈10 trials. If it has drifted more than the threshold, pause and recalibrate.

### 4.3 Stopping rule (per track)

Stop when **all** of these hold:
- a full coordinate pass accepted nothing;
- the last 3 restarts failed to beat the best;
- the best-so-far improved by less than 1 % over the last 24 trials.

Or stop when the budget runs out (default: 5 days total). The ledger shows the plateau.

### 4.4 How fast the search runs (estimates)

| Item | Time |
|---|---|
| Server start: weights from local NVMe or a PVC, plus compilation and CUDA graph capture (with caches persisted) | ≈2–4 min |
| Screening run: 2 load points, 1 workload | ≈4 min |
| Full evaluation: W1 and W2 at 8 concurrency levels, W3 at 3 levels | ≈25 min |
| Per lane, per day | ≈150–200 screening runs, or ≈40–50 full evaluations |

With two lanes: ≈300–400 screening runs or ≈80–100 full evaluations per day.

### 4.5 Who drives it

- **Mode A: scripted (default).** The deterministic stages above. Fully reproducible, good coverage.
- **Mode B: agent-assisted (optional, after Mode A works).**
  - I act as the climber from a scheduled session that wakes every 2–3 hours.
  - Each time, I read the ledger, metrics, and profiler summaries, and propose moves within the knob registry based on what the data shows. Example: "KV usage is at 100 % and requests are queueing at concurrency 64, so try FP8 KV."
  - I launch those runs and post a daily digest.
  - Same acceptance and stopping rules, with budget caps and a kill switch.

---

## 5. The tests

### 5.1 The measuring instrument

- **AIPerf** (pinned version) is the single client for both engines. It gives consistent metric definitions, its multiprocess client won't become the bottleneck, and it reports TTFT, time to first answer token, inter-token latency, goodput, and server-side token counts.
- `vllm bench serve` serves as a periodic cross-check.
- AIPerf runs as a **sidecar container in the server pod**, hitting `localhost` with a CPU limit (e.g. 4 cores). That removes network noise and keeps the client off the critical path.

### 5.2 Workloads (fixed, versioned, seeded)

| ID | Shape | Purpose |
|---|---|---|
| W1 | 1k tokens in / 1k out, random tokens, `ignore_eos` | Raw engine speed (InferenceX-style) |
| W2 | 8k in / 1k out | Pressure from long-prompt prefill |
| W3 | Real chat and coding prompts with the chat template, model-card sampling (temperature 0.6 or 1.0), thinking on, capped output | Speculative decoding needs real text to be measured honestly; also measures time to first answer token |
| W4 | Multi-turn sessions (AIPerf `--conversation-turn-mean`), 8k–64k context, with prompt-prefix reuse across turns | T3 agentic |

**Load:** closed-loop concurrency of 1, 2, 4, 8, 16, 32, 64, 128, and 256, plus one open-loop Poisson arrival rate near the point where latency starts climbing, to measure tail latency.

### 5.3 Quality gate

Run only for configs that are about to be accepted.

| Level | What | Time |
|---|---|---|
| **Q0: smoke test** | 50 fixed prompts with greedy decoding, compared token-by-token with the BF16 reference. Catches broken kernels, templates, and parsers. | ≈3 min |
| **Q1: mini-evals** | Subsets of GSM8K or AIME, IFEval, GPQA Diamond, and a coding set (LiveCodeBench or HumanEval+), thinking on with a token cap, run via lm-evaluation-harness or Inspect | ≈30–45 min |
| **Q2: final** | Winners only: longer evals, e.g. a 50-task SWE-bench slice with mini-SWE-agent | Hours |

- Speculative decoding is lossless by design, so in Q0 it should match its non-speculative twin.
- Quantized configs must stay within the tolerance you pick (decision 4).
- Configs that are fast but fail the gate stay in the ledger, labeled "fast but lossy".

### 5.4 What every trial records

- Client metrics.
- Server `/metrics`: KV usage, running and waiting requests, prefix-cache hits, preemptions, speculative-decoding acceptance rate.
- GPU telemetry from DCGM: compute and memory-bandwidth activity, power, clocks, temperature.
- Engine CPU usage, startup time, and peak memory.
- The exact config, image digest, model revision, node, and an environment fingerprint.

---

## 6. Harness architecture

```text
knob registry (YAML) ──► climber (stages, acceptance, stop rule) ──► trial queue ─┬─► lane 0 (node 0)
        ▲                                                                        └─► lane 1 (node 1)
        │                                                                                │
        └──── ledger (SQLite + artifacts) ◄── scorer + quality gate ◄── trial runner (K8s Pod: server + AIPerf sidecar)
                                                                                         │
                                                                     metrics scrape (/metrics, DCGM) + logs
```

- **Trial runner:**
  1. Renders a pod from the config (server container plus AIPerf sidecar) and pins it to its lane's node.
  2. Waits for `/health`, runs the workload plan, and scrapes metrics.
  3. Tears the pod down. Timeouts, out-of-memory errors, and crashes mark the config "infeasible".
- **Caches:**
  - Model weights for all variants (≈180–220 GB) on each node's 960 GiB local NVMe disk, or on a PVC per node.
  - torch.compile and FlashInfer JIT caches on a PVC, to cut restart time.
- **Ledger:** one row per trial. Each row holds the trial id, its parent, the knob change, a config hash, image digests, the node, metrics, score, confidence interval, and verdict. Every accepted change links to its evidence.
- **Reports:**
  - A climb curve (best-so-far vs trial number).
  - A Pareto plot (tok/s per user vs tok/s per GPU).
  - A waterfall showing what each accepted change bought.
  - Heatmaps per knob.
- **Controls:** `climb run --track T2 --budget 48h`, `climb status`, `climb pause`, and a kill switch (deleting the namespace).

---

## 7. Timeline

| Day | Work | Done when |
|---|---|---|
| **0** | Prerequisites (need your approval): expose the GPUs (managed GPU or device plugin), confirm the driver is R580+ for the CUDA 13 images, import the vLLM, SGLang, and AIPerf images into ACR, create a namespace and storage, download the model variants | Both nodes show `nvidia.com/gpu: 1`; images and weights are cached |
| **1** | Build a minimal harness; validate its numbers against `vllm bench serve`; calibrate noise; record BF16 baselines for both engines | Noise floor known; baseline report written |
| **2** | Macro screen (engine × weights × KV dtype × speculative method), with quality gates | Top 3 starting points per track |
| **3** | Coordinate ascent on T1 and T2 | First plateau, or a clear trend |
| **4** | Restarts. T3: CPU offload, plus the two-replica routing experiment. Optionally build the INT8 W8A8 checkpoint and try agent mode. | T3 plateau |
| **5** | Verify winners (5× repeats, both nodes, Q2 evals); write up | Final report and reproducible configs |

A track can plateau early; the stopping rule ends it as soon as it's met.

---

## 8. Hypotheses to confirm or kill

- **T1:**
  - The BF16 default lands near its ≈30 tok/s ceiling.
  - INT4 gives the biggest single jump.
  - A working draft (DFlash2 or MTP) adds roughly 1.4–1.8× at concurrency 1, if the A100 behaves like the RTX PRO 6000 numbers on the draft's card.
- **T2:**
  - BF16 is starved for KV space (≈65k tokens).
  - FP8 or INT4 weights plus FP8 KV raise capacity ≈3–8×, so goodput should rise sharply.
  - Speculative decoding probably hurts at high concurrency.
- **Prefill (TTFT):** only INT8 W8A8 can use the faster tensor cores on an A100; weight-only formats won't speed prefill much.
- **T3:** CPU KV offload and prefix-aware routing cut TTFT for long multi-turn sessions.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Measurement noise and drift | Calibration; paired, interleaved A/B runs on the same node; regular re-measurement of the current best; record clocks and temperatures |
| Overfitting to synthetic prompts | Real-text and multi-turn workloads (W3, W4); the quality gate |
| Lossy quantization or kernels | The quality gate; a "fast but lossy" label |
| Community artifacts (drafts, NVFP4) | Provenance labels; safetensors only; no `trust_remote_code` |
| Invalid or crashing configs | Validity rules, timeouts, a blacklist |
| Engine version churn | Pin image digests; treat upgrades as separate experiments |
| Slow restarts | Weights on local NVMe, persisted compile caches, screening runs before full evaluations |
| Image policy (currently warns on non-ACR images) | Import images into ACR |
| Cost and ownership | Confirm budget and owner. Both nodes ≈ $8.30/h at pay-as-you-go, and they're billing while idle today. |
| Unattended multi-day runs | Budget caps, auto-pause after repeated failures, a daily digest, a kill switch |

---

## 10. Deliverables

- `<repo>/`: harness code, knob registry, workloads, and a runbook.
- The best config per track, as YAML plus exact `vllm serve` / SGLang launch commands.
- The ledger (SQLite) and raw artifacts.
- A final report: climb curves, Pareto frontier, waterfall, quality table, $/1M tokens within the latency target, and lessons learned. Plus the enterprise story: "a 33B model its vendor serves on 4 GPUs, running on one A100."

---

## Sources

All verified 18 Nov 2026.

- Nanbeige4.3-33B card and official variants: <https://huggingface.co/Nanbeige/Nanbeige4.3-33B> (plus `-FP8`, `-INT8`, `-INT4`)
- DFlash2 draft (community): <https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2>
- AA leaderboard: <https://artificialanalysis.ai/leaderboards/models>
- Qwen3.8-Flash-Next is 176B total (excluded): `vllm-project/recipes/models/Qwen/Qwen3.8-Flash-Next.yaml`
- vLLM v0.34.0 source:
  - `vllm/config/speculative.py`: speculative-decoding methods
  - `vllm/config/compilation.py`: CUDA graph modes
  - `vllm/v1/attention/backends/registry.py`: attention backends
  - `vllm/engine/arg_utils.py` and `vllm/envs.py`: flags and environment variables
- SGLang v0.5.23 source: `python/sglang/srt/arg_groups/fields/*.py`
- AIPerf CLI reference: <https://github.com/ai-dynamo/aiperf/blob/main/docs/cli-options.md>
- Companion docs in `<local-dir>/`:
  - Main report (memory math, engine flags)
  - Provider-measurement doc (metric definitions)
  - Single-vs-two-node doc (networking)
