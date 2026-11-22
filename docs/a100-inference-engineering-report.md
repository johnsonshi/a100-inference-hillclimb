# Inference engineering on a single A100

**How Artificial Analysis, SWE-bench, and industry perf benchmarks actually measure models, and how to reproduce that with one small open model on one A100 80GB using vLLM or SGLang.**

> Research report · 18 Nov 2026 · **Research only. Nothing was changed on `<cluster>`.** Cluster facts come from read-only `kubectl get` and `az aks show` calls.
>
> Versions checked against source or docs today:
> - Engines: vLLM **v0.34.0** (16 Nov 2026), SGLang **v0.5.23** (2 Nov 2026)
> - Benchmark clients: guidellm **v0.8.1**, NVIDIA AIPerf **v0.13.0**
> - Artificial Analysis: performance methodology **v3.0.0** (3 Nov 2026), Intelligence Index **v5.0.1**

---

## TL;DR

- **AA speed is not GPU speed.**
  - Artificial Analysis (AA) benchmarks *public API endpoints*. It uses the OpenAI Python client from Google Cloud Run `us-central1`, every 6 hours.
  - Workloads are ≈1k / ≈10k / ≈100k-token prompts plus multi-turn agentic sessions, all under concurrent load. Per-run medians are averaged over 5 days.
  - So every number includes the provider's hardware, engine, quantization, batching, and network.
  - Since v3.0.0 (3 Nov 2026), **output speed = output tokens ÷ full request time, *including* time to first token (TTFT)**. Older screenshots used tokens/s *after* the first token, so they aren't comparable.
- **AA's intelligence score isn't reproducible locally.** Intelligence Index v5.0.1 is 9 mostly agentic evals, several private and graded by LLM judges. Reproduce the *public* ingredients instead (GPQA Diamond, MMLU-Pro, AIME, IFBench, LiveCodeBench, Terminal-Bench, SWE-bench) and use them as quality guardrails.
- **How SWE-bench works:**
  - Each task is a real GitHub issue. An agent scaffold drives the model to write a patch, and a Docker harness grades it with hidden `FAIL_TO_PASS` + `PASS_TO_PASS` tests.
  - Scores swing with the scaffold and budget: gpt-oss-20b goes from 37.4 % to 60.7 % just by raising reasoning effort.
  - OpenAI stopped reporting *SWE-bench Verified* in Feb 2026 (contamination, flawed tests) and now recommends **SWE-bench Pro**.
- **Perf benchmarking is a sweep, not a single number.**
  - Fix the workload shape (input/output lengths), sweep concurrency, and plot **tok/s per user vs tok/s per GPU**, the way InferenceX does.
  - Report TTFT and time per output token (TPOT) at p50/p99, plus **goodput** under service-level objectives (SLOs). MLPerf's Llama-3.1-8B targets: TTFT ≤ 2 s and TPOT ≤ 100 ms (server), ≤ 0.5 s and ≤ 30 ms (interactive).
  - **Neither InferenceX nor MLPerf includes A100**, so your data would fill a real gap.
- **One A100 80GB is enough for 8–30B-class models, with no distributed inference.**
  - Decode (generating tokens) is limited by memory bandwidth (1.94 TB/s on your PCIe card). That makes **mixture-of-experts (MoE) models with ≈3–4B active parameters A100's sweet spot**.
  - A100 has no FP8/FP4 tensor cores, but vLLM and SGLang run FP8, INT4, and MXFP4 checkpoints as *weight-only* quantization via Marlin kernels.
- **The one model to start with: `Qwen/Qwen3-Coder-30B-A3B-Instruct`.**
  - Apache-2.0, 30.5B total / 3.3B active parameters.
  - Fits in BF16 with ≈200k tokens of KV cache, or in FP8 with ≈500k.
  - First-class in both engines on Ampere, with tool calling for SWE-bench.
  - External yardsticks: AA tracks it (128 tok/s median), and a community OpenHands run scored 51.6 % on SWE-bench Verified.
- **Economics:**
  - H100 NVL has ≈2.0× the bandwidth for ≈2.07× the on-demand price of A100 80GB PCIe, so **\$/token is roughly at parity for BF16 small-model decode**. H100 pulls ahead with native FP8.
  - The defensible A100 pitch: *capacity you can get (or already own) at comparable \$/token.* Prove it with your own sweep.
- **Your cluster is idle, but no pod can get a GPU yet.**
  - AKS installs the driver, but nodes don't advertise `nvidia.com/gpu` because the device plugin is missing.
  - Both current images (vLLM v0.34.0, SGLang v0.5.23) are **CUDA 13**, so they need an **R580+ driver**.
  - The cluster's image policy only allows `*.azurecr.io` / `mcr.microsoft.com` images (warn/dry-run today), so import images into ACR.

---

## Contents

0. [Your starting point: the cluster](#0-your-starting-point-the-cluster)
1. [Mental model: quality vs. performance](#1-mental-model-quality-vs-performance)
2. [How Artificial Analysis measures speed](#2-how-artificial-analysis-measures-speed)
3. [How Artificial Analysis measures intelligence](#3-how-artificial-analysis-measures-intelligence)
4. [How SWE-bench (and friends) work](#4-how-swe-bench-and-friends-work)
5. [How performance benchmarks are done](#5-how-performance-benchmarks-are-done)
6. [Why A100, and where it's weak](#6-why-a100-and-where-its-weak)
7. [The one model: Qwen3-Coder-30B-A3B-Instruct](#7-the-one-model-qwenqwen3-coder-30b-a3b-instruct)
8. [vLLM on one A100](#8-vllm-on-one-a100-v0340)
9. [SGLang on one A100](#9-sglang-on-one-a100-v0523)
10. [Running it on AKS (example, not applied)](#10-running-it-on-aks-example-only--not-applied)
11. [Two-week experiment plan](#11-two-week-experiment-plan)
12. [AA-style probe script](#12-aa-style-probe-script)
13. [Pitfalls checklist](#13-pitfalls-checklist)
14. [Gaps and open questions](#14-gaps-and-open-questions)
15. [Sources](#sources)

---

## 0. Your starting point: the cluster

| | Read-only finding (18 Nov 2026) |
|---|---|
| Cluster | `<cluster-rg>` / `<cluster>`, **centralus**, Kubernetes 1.37.1, Running |
| Nodes | 2 × `Standard_NC24ads_A100_v4`. Each has **1× A100 80GB PCIe**, 24 vCPU (AMD EPYC 7V13), 220 GiB RAM, 64 GiB temp disk. One GPU per node, so "one model on one A100" = one pod per node. |
| Workloads | Idle. Only `kube-system` and Gatekeeper/Azure Policy pods; nodes at ≈2 % CPU and memory. |
| GPU plumbing | Node pool has `gpuProfile.driver: Install` and nodes are labeled `kubernetes.azure.com/accelerator=nvidia`, **but there's no `nvidia.com/gpu` capacity**. The device plugin is missing, so no pod can request a GPU. |
| Image policy | An Azure Policy allowed-images constraint, `^([^/]+\.azurecr\.io\|mcr\.microsoft\.com)/.+$`, applies to Deployments, DaemonSets, StatefulSets, and CronJobs outside system namespaces. Enforcement is **warn + dry-run** today. Docker Hub images (`vllm/vllm-openai`, `lmsysorg/sglang`) would trigger warnings now and be **blocked** if the policy flips to deny. |
| Blip | The API server returned HTTP 503 for ≈6 min (≈13:09–13:15 PT) during this check, then recovered. |

**Prerequisites before you can serve anything (your call; none of this was done):**

1. **Expose the GPU.** Either:
    - create an AKS **managed-GPU** node pool (preview; AKS manages the driver, device plugin, and DCGM exporter), or
    - install the NVIDIA device plugin or GPU Operator on the existing pool.

   Success looks like `kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu` showing `1`. See [AKS GPU docs](https://learn.microsoft.com/azure/aks/use-nvidia-gpu) and [managed GPU nodes](https://learn.microsoft.com/azure/aks/aks-managed-gpu-nodes).
2. **Check the driver version** with `nvidia-smi`. Both current images need an **R580+ driver**:
    - vLLM v0.34.0's image is built on CUDA **13.0.3**.
    - SGLang retired its CUDA 12 builds after v0.5.19.

   If AKS installed an older driver, use `--gpu-driver none` plus the GPU Operator with R580+, or pin older images.
3. **Import images into ACR**, which also satisfies the policy:

   ```bash
   az acr import -n <acr> --source docker.io/vllm/vllm-openai:v0.34.0 --image vllm/vllm-openai:v0.34.0
   az acr import -n <acr> --source docker.io/lmsysorg/sglang:v0.5.23 --image lmsysorg/sglang:v0.5.23
   ```

   The images are ≈8–11 GB. An ACR artifact-cache rule for `docker.io` and artifact streaming would cut cold starts.
4. **Model storage.** BF16 Qwen3-Coder-30B-A3B is 61 GB and the temp disk is only 64 GiB, so use a PVC (e.g. `managed-csi-premium`, 200 Gi).

---

## 1. Mental model: quality vs. performance

| | Quality: "how smart?" | Performance: "how fast, how cheap?" |
|---|---|---|
| Question | Does the model solve the task? | How fast, and at what cost, does **this deployment** serve tokens? |
| Units | % solved, Elo, index points | TTFT, TPOT/ITL, tok/s per user, tok/s per GPU, $/1M tokens, goodput |
| Depends on | Weights, chat template, sampling, agent scaffold, token budget | Hardware, engine, kernels, quantization, concurrency, workload shape, network |
| Examples | AA Intelligence Index, SWE-bench, Terminal-Bench, GPQA | AA speed charts, InferenceX, MLPerf Inference, `vllm bench serve` |

Two consequences:

- **Performance is a property of (model × engine × hardware × config × workload × load)**, not of the model alone.
- **Serving choices can silently damage quality.** Quantization, a wrong chat template, or a missing tool or reasoning parser can all degrade answers. So an inference engineer always measures *both*. AA itself runs a separate "Endpoint Accuracy Index" to catch providers whose serving choices hurt quality.

Anatomy of one streamed request:

```text
send ─► queue ─► PREFILL (all input tokens; compute-bound) ─► t1 ─► t2 ─► … ─► tN   (DECODE; memory-bandwidth-bound)
|◄──────────────────── TTFT ───────────────────────────────►|◄ITL►|
|◄──────────────────────────── end-to-end latency (E2E) ───────────────────────────►|

TPOT               = (E2E − TTFT) / (N − 1)          → per-user decode speed = 1 / TPOT
AA v3 output speed = N / E2E                         (includes TTFT; N counts reasoning tokens)
System throughput  = Σ output tokens of all concurrent requests / wall time   (tok/s per GPU)
```

TTFT = time to first token, ITL = inter-token latency, TPOT = time per output token, E2E = end-to-end latency.

---

## 2. How Artificial Analysis measures speed

### 2.1 Method (Performance Benchmarking v3.0.0, 3 Nov 2026)

| Aspect | What AA does |
|---|---|
| What's measured | **Public API endpoints**: the provider's whole stack, including hardware, engine, quantization, batching, speculative decoding, region, and network |
| Client | Official **OpenAI Python library** (for OpenAI-compatible APIs), running on **Google Cloud Run, `us-central1`** |
| Workloads | Single requests of ≈**1k**, ≈**10k**, and ≈**100k** input tokens, plus **multi-turn agentic** sessions with tool calls drawn from AA's coding, knowledge-work, and automation evals |
| Load | "Every workload under concurrent load". The concurrency level is **not published**. |
| Prompts | Drawn fresh each run from a large, diverse set; every endpoint gets the same prompts in a run. Diversity matters because speculative decoding makes speed depend on content. |
| Sampling | `temperature` 0 for non-reasoning models, 0.6 for reasoning models, unless the lab specifies otherwise. `top_p` 1. `max_tokens` up to the endpoint cap (≤ 65,536, or 128,000 for long reasoners). |
| Frequency | Every **6 hours** (every 18 hours for deprecated models) |
| Aggregation | Median per workload per run → **average of run medians over 5 days** → headline = equal-weight average across workloads. Before v3 it was the median of the 10k workload over 72 hours. |
| Token counting | **Provider-reported** output tokens, including hidden reasoning tokens. Before v3 AA used `tiktoken o200k_base`. |
| Failures | Endpoint-side failures (429, 5xx, timeouts) count as **speed 0** and feed the *Endpoint Error Rate*. AA-side failures are excluded. |
| Integrity | Providers may not fingerprint AA traffic, route it to special capacity, or serve it at unrepresentative batch sizes or concurrency. AA cross-checks with anonymous accounts. |

Definitions (paraphrased from AA's methodology page):

- **TTFT**: send → first token. For reasoning models, this is the first *reasoning* token.
- **TTFAT** (time to first *answer* token): includes thinking time, so reasoning and non-reasoning models are compared on the same "how long until I see an answer".
- **Output speed**: output tokens (including reasoning) ÷ **full request duration, including TTFT**. For multi-turn sessions, tool runtime is excluded.
- **End-to-end time**: send → final token, measured per request.

### 2.2 What AA's numbers are, and aren't

- They're a **user-experienced, per-request** number from a public API at the provider's production load.
- They're **not hardware numbers**. Provider GPUs (H100/H200/B200/TPU…), quantization, speculative decoding, and network round trips to `us-central1` are all baked in.
- AA model pages on 18 Nov 2026:
    - **gpt-oss-20b**: 141 tok/s
    - **Qwen3-Coder-30B-A3B-Instruct**: 128 tok/s
    - **Qwen3-30B-A3B-Instruct-2507**: 57 tok/s

  These average the 1k, 10k, 100k, and agentic workloads. Long-prompt TTFT drags the v3 "output speed" down.
- **To get an AA-comparable number on your A100:**
    1. Use the same workload sizes with unique prompts and temperature 0.
    2. Stream through the OpenAI client and use the server-reported token counts (`usage`).
    3. Compute `N / E2E` per request.
    4. Take the median per workload, then average across workloads.
    5. **State the concurrency you used**, since AA doesn't publish theirs.

  A script for this is in [§12](#12-aa-style-probe-script).

### 2.3 AA's hardware benchmarks (they don't cover A100)

- **Legacy "System Load Test"** (used in the 2025 MI300X vs H100 vs H200 study):
  - Workload: 1k input / 1k output tokens, streaming.
  - Concurrency 1, 2, 4 … 64, then +50 until throughput plateaus, in 3-minute phases, on SGLang.
  - Models: DeepSeek R1 and Llama 4 Maverick.
  - Reports system throughput vs per-query speed.
- **AA-AgentPerf** (current, June 2026):
  - Replays real agentic-coding trajectories: 5k–131k input tokens (mean ≈27k), with simulated tool delays.
  - Plots a Pareto curve of server throughput vs per-request output speed (P25). Headline metrics are "agents per megawatt" and $/h.
  - Hardware is rack-scale GB300, B300, MI355X, H200, and Vera Rubin systems running frontier MoE models.
- **Neither covers A100 or models of 40B parameters and under.** Borrow the *method* (fixed shapes, concurrency sweep, Pareto curve, percentiles, cost normalization), not the numbers.

---

## 3. How Artificial Analysis measures intelligence

**Intelligence Index v5.0.1** is 9 evals across 4 equal-weight categories:

| Category (25 % each) | Eval | Items × repeats | Scoring | Weight |
|---|---|---|---|---|
| Knowledge work | AA-Briefcase v1.1 | 91 tasks × 1 | Elo from pairwise + rubric grading (agentic, file outputs) | 15 % |
| | AutomationBench-AA | 657 × 1 | Programmatic; zero credit on a guardrail violation | 5 % |
| | GDP.pdf | 100 × 5 | Rubric LLM grading, all-pass | 5 % |
| Coding | FrontierCode-AA 2.0 | 132 × 3 | Tests + LLM-judge rubric on repo code changes | 15 % |
| | Terminal-Bench 4.0 | 66 × 3 | Test suite pass/fail, pass@1 | 10 % |
| Science | Terminal-Bench-Science 0.1 | 70 × 3 | Test suite pass/fail, pass@1 | 15 % |
| | CritPt v2 (hard subset) | 40 × 5 | Official grading server, pass@1 | 10 % |
| General | AA-Omniscience v1.1 | 3,000 × 1 | Accuracy (10 %) + 1 − hallucination rate (5 %) | 15 % |
| | HLE-Diamond | 1,000 × 1 | LLM equality checker, pass@1 | 10 % |

**Global settings:**
- Zero-shot instructions.
- Temperature 0 (non-reasoning) or 0.6 (reasoning).
- Max output 16,384 tokens for non-reasoning models; the creator's cap for reasoning models.
- pass@1, aggregated over repeats.
- Up to 30 retries on API errors.
- Agentic evals run in **E2B** sandboxes with AA's open-source **Stirrup** harness.

**What this means for you:**

- **You can't reproduce the Index.**
  - Several datasets are private, grading uses frontier LLM judges and Elo against other models, and versions aren't comparable.
  - Many 2025-era small models currently show **N/A** on v5. Among scored open-weight models of 40B and under, 2026 releases lead; for example, Nanbeige4.3-33B scores ≈ 20 on today's leaderboard.
  - v5 is a much harder scale than v2–v4, so don't compare its numbers with older ones.
- **You can reproduce the public evals.** AA used MMLU-Pro, GPQA Diamond, HLE, LiveCodeBench, SciCode, AIME 2025, IFBench, τ²-Bench, Terminal-Bench 2.0, and AA-LCR in earlier versions, and you can add SWE-bench. Use them as **regression guardrails for serving changes**: a "faster" config that drops GPQA or LiveCodeBench beyond noise isn't actually better.

---

## 4. How SWE-bench (and friends) work

### 4.1 Anatomy of an instance

- **Given to the model:** the GitHub issue text and the repo at the pre-fix commit.
- **Hidden from the model:**
  - the **gold patch** (the real fix PR),
  - **`FAIL_TO_PASS`** tests (fail before the fix, pass after),
  - **`PASS_TO_PASS`** tests (regression tests that must keep passing).
- **Resolved** means the model's patch applies **and** both test sets pass. The headline score is % resolved (pass@1).

### 4.2 The harness

Each instance has its own Docker image containing the repo and its environment. The harness applies the predicted patch, runs the tests, and grades.

```bash
pip install swebench
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path preds.jsonl \
  --max_workers 8 \
  --run_id a100-qwen3coder-bf16
# results: logs/evaluation/<run_id>/... ; re-runs need a new --run_id (cached per run_id+instance)
```

- **Requirements:** x86_64, ≥ 120 GB free disk, 16 GB RAM, 8 cores, Docker. Keep `--max_workers` below `min(0.75 × cores, 24)`. Run the harness on a separate VM or dev box, **not on the GPU nodes**.
- **Cloud grading options:**
  - `sb-cli submit swe-bench_verified test --predictions_path preds.json --run_id <id>`: free, ≈20 minutes.
  - Modal: add `--modal true` to `run_evaluation`.

### 4.3 Variants

| Variant | Size | Notes |
|---|---|---|
| Full | 2,294 | Original set, 12 Python repos |
| Lite | 300 | Cheaper subset |
| **Verified** | 500 | Human-filtered with OpenAI (Aug 2024). The most-cited variant, now contaminated (see [§4.5](#45-contamination)). |
| Multilingual | 300 | 42 repos, 9 languages |
| Multimodal | 480 | Issues with images |
| **SWE-bench Pro** (Scale AI) | 731 (v1) / 642 (v2) | Long-horizon tasks, 11 repos, runs via **Harbor**. The recommended replacement. |
| Verified "bash-only" leaderboard | 500 | Every model runs the same minimal **mini-SWE-agent** loop, so the scaffold is held constant |

### 4.4 The scaffold is half the score

A model doesn't "take" SWE-bench by itself. An **agent scaffold** (mini-SWE-agent, SWE-agent, OpenHands, or a vendor harness) runs a tool loop: read files, run commands, edit, test, submit. Step limits, cost limits, and reasoning budgets move scores a lot:

- **gpt-oss-20b:** 37.4 / 53.2 / 60.7 % at low / medium / high reasoning effort, with the same weights.
- **Seed-OSS-36B:** 56 % with OpenHands vs 47 % with Agentless.

Always report **model + scaffold + limits + number of instances**.

### 4.5 Contamination

OpenAI's Feb 2026 post *"Why we no longer evaluate SWE-bench Verified"* found that:
- at least 59 % of 138 audited hard problems had flawed tests, and
- frontier models could **reproduce the gold patches**, which means they'd seen them in training.

OpenAI now recommends **SWE-bench Pro**. For your purposes, Verified is still fine as a **relative** signal (for example BF16 vs FP8 on the same model), but not as an absolute capability claim.

### 4.6 Running it against your A100 endpoint (mini-SWE-agent)

mini-SWE-agent calls the model through LiteLLM and requires a native `bash` **tool call** on every turn. So the server must have tool parsing enabled:
- vLLM: `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
- SGLang: `--tool-call-parser qwen3_coder`

```yaml
# my_local_config.yaml (merge into mini-swe-agent's SWE-bench config)
model:
  model_name: "qwen3-coder"              # must match --served-model-name
  model_kwargs:
    custom_llm_provider: "openai"
    api_base: "http://<endpoint>:8000/v1"
    api_key: "EMPTY"
    temperature: 0.7                     # model-card recommendation
  cost_tracking: "ignore_errors"         # a local model has no price entry
```

```bash
pip install mini-swe-agent
# smoke test on one instance
mini-extra swebench-single --config my_local_config.yaml --subset verified --split test \
  -m qwen3-coder -i sympy__sympy-15599
# batch run; then grade with the harness in §4.2
mini-extra swebench --model qwen3-coder --config my_local_config.yaml \
  --subset verified --split test --workers 4 -o ./preds
```

Start with a 25–50 instance slice (see `mini-extra swebench --help`), then run all 500. Track **$ per resolved issue** = GPU-hours × $/h ÷ resolved count. It makes a strong enterprise metric.

### 4.7 Terminal-Bench 2.0 and Harbor

- A Terminal-Bench task is a container + an instruction + tests + a reference solution. TB 2.0 has 89 tasks.
- **Harbor** is the official harness and also runs SWE-bench Pro. Each task writes a reward file; the score is the resolution rate.

```bash
uv tool install harbor
export OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://<endpoint>:8000/v1
harbor run -d terminal-bench@2.0 -a terminus-2 -m openai/qwen3-coder -n 4   # confirm flags with: harbor run --help
```

### 4.8 The other common evals, one line each

| Eval | Measures | Scoring |
|---|---|---|
| MMLU-Pro | 14-domain knowledge and reasoning, 10-option multiple choice (≈12k questions) | Accuracy |
| GPQA Diamond | 198 "Google-proof" PhD-level science multiple-choice questions | Accuracy |
| Humanity's Last Exam | Frontier academic questions across 100+ subjects | Accuracy (+ calibration) |
| LiveCodeBench | Dated contest problems (resists contamination) | pass@1 on hidden tests |
| SciCode | Research-level scientific coding | Sub-problem and main-problem pass rate |
| AIME 2025 | 30 competition math problems with integer answers | Exact match (avg@k) |
| IFBench | Precise instruction following (successor to IFEval) | Rule-verified prompt accuracy |
| τ²-Bench (telecom) | Multi-turn customer-service tool use with a simulated user | Final-state success, pass^k |
| AA-LCR | ≈100k-token multi-document reasoning | LLM-judged accuracy |

### 4.9 Harnesses that work with an OpenAI-compatible endpoint

| Harness | Best for | Example |
|---|---|---|
| **lm-evaluation-harness** | MMLU-Pro, IFEval, AIME, GSM8K | `lm_eval --model local-chat-completions --model_args model=qwen3-coder,base_url=http://<ep>:8000/v1/chat/completions,num_concurrent=32 --tasks ifeval --apply_chat_template` |
| **Inspect AI** (UK AISI) | GPQA Diamond and many agentic evals | `inspect eval inspect_evals/gpqa_diamond --model vllm/qwen3-coder --model-base-url http://<ep>:8000/v1` |
| **evalscope** | LiveCodeBench over an API (also has a perf mode) | `evalscope eval --model qwen3-coder --api-url http://<ep>:8000/v1 --api-key EMPTY --eval-type openai_api --datasets gsm8k` |
| **lighteval** | Hugging Face-style suites | `lighteval endpoint litellm "model_name=hosted_vllm/qwen3-coder,base_url=http://<ep>:8000/v1" "gpqa:diamond"` (verify the syntax) |
| OpenAI simple-evals | Historical reference only | Deprecated (Jul 2025) |

### 4.10 Published SWE-bench Verified scores for small open models

| Model (params) | SWE-bench Verified | Scaffold | Source |
|---|---|---|---|
| Devstral Small 2 (24B dense, FP8) | 68.0 % | Not stated | Mistral, Dec 2025 |
| gpt-oss-20b (21B / 3.6B active) | 37.4 / 53.2 / 60.7 % | OpenAI internal harness, low / medium / high effort | Model card, Aug 2025 |
| Granite-4.2-30B | 57 % | Not stated | IBM Hugging Face card, Aug 2026 |
| Seed-OSS-36B | 56 % / 47 % | OpenHands / Agentless | ByteDance Hugging Face card |
| Devstral Small 1.1 (24B) | 53.6 % | OpenHands | Mistral Hugging Face card, Jul 2025 |
| **Qwen3-Coder-30B-A3B-Instruct** | **51.6 %** | OpenHands | Community submission (SWE-bench/experiments) |

These use different scaffolds, so don't rank models across rows. Run your candidates through one scaffold yourself.

---

## 5. How performance benchmarks are done

### 5.1 Metrics (as the vLLM and SGLang bench clients compute them)

| Metric | Definition | Gotcha |
|---|---|---|
| TTFT | Send → first streamed chunk | Includes queueing and prefill (and network) |
| TPOT | (E2E − TTFT) / (output tokens − 1) | Excludes the first token; averaged per request |
| ITL | Gap between consecutive streamed **chunks** | With multi-token chunks (e.g. speculative decoding), ITL ≠ TPOT |
| E2E latency | Send → last token | |
| Output throughput | Σ output tokens / wall time | Your "tok/s per GPU" |
| Goodput | Requests/s that meet **all** SLOs (e.g. `ttft:2000 tpot:100`) | The number that maps to user experience |
| tok/s per user | ≈ 1000 / TPOT(ms) | InferenceX's "interactivity" axis |

### 5.2 Workload design: the knobs that change the answer

| Knob | Options | Advice |
|---|---|---|
| Input/output lengths (ISL/OSL) | 1k/1k (chat), 8k/1k (summarization/RAG), 1k/8k (reasoning), agentic long-prefix | Run at least 1k/1k and 8k/1k, InferenceX's current pair |
| Data | `random` tokens, ShareGPT, prefix-repetition, real traces | `random` for engine comparisons; prefix-heavy data for agentic realism |
| Arrival pattern | Closed loop (`--max-concurrency`) or open-loop Poisson (`--request-rate`) | Closed loop for capacity curves; open loop for tail-latency claims, because closed loop hides stalls ("coordinated omission") |
| Output length control | `--ignore-eos` | Forces exact output length; vLLM turns it on automatically for `random` |
| Warmup | `--num-warmups` (vLLM), `--warmup-requests` (SGLang) | Excludes compile and CUDA-graph capture time |
| Prefix cache | Unique prompts, or flush caches between runs | Repeated prompts inflate results |
| Sample size | ≥ 5× concurrency; ≥ 1,000 requests for a stable p99 | |
| Client placement | Same cluster or zone; never through `kubectl port-forward` | Port-forward adds latency and caps throughput |

### 5.3 Industry references

| | SemiAnalysis **InferenceX** (formerly InferenceMAX) | MLCommons **MLPerf Inference v6.1** (Sep 2026) |
|---|---|---|
| What it is | Open-source nightly sweeps on real hardware | Audited submissions under fixed rules |
| Shapes / models | 1k/1k and 8k/1k random tokens on frontier models | Llama 3.1 8B, Llama 2 70B, DeepSeek-R1, gpt-oss-120b, Qwen3-VL-235B |
| Output | Pareto curve of **tok/s/GPU vs tok/s/user**; TCO model → $/1M tokens | Offline / Server / Interactive throughput under latency SLOs, at 99 % (or 99.9 %) of reference accuracy |
| Example SLO | — | Llama 3.1 8B: Server **TTFT ≤ 2 s, TPOT ≤ 100 ms**; Interactive **TTFT ≤ 0.5 s, TPOT ≤ 30 ms** |
| A100 included? | **No** (Hopper, Blackwell, MI3xx, TPU, RTX PRO 6000 only) | **No** A100 submissions in v5.0–v6.1 |

### 5.4 Tools

| Tool | Use it for | Notes |
|---|---|---|
| `vllm bench serve` / `vllm bench sweep` | Default client for both engines | A plain OpenAI streaming client. `--backend openai-chat --endpoint /v1/chat/completions` works against SGLang too. `vllm bench sweep … plot_pareto` automates the curves. |
| `python -m sglang.benchmark.serving` | SGLang's client (can also drive vLLM) | `sglang.bench_serving` is a deprecated shim in v0.5.23 and is removed on main |
| **guidellm** (`vllm-project/guidellm`) | One-command sweep (synchronous → max throughput → interpolated rates) | `guidellm run --backend kind=openai_http,target=http://<ep>:8000 --profile kind=sweep --data kind=synthetic_text,prompt_tokens=1024,output_tokens=1024` |
| **NVIDIA AIPerf** | Very high concurrency (multiprocess client), Pareto recipes | Successor to GenAI-Perf, which is deprecated |
| `kubernetes-sigs/inference-perf` | Kubernetes-native load stages (llm-d ecosystem) | For later, when you test gateways and routing |
| evalscope perf | Quick stress tests | No Poisson mode |
| ~~LLMPerf~~ | — | Archived Dec 2025 |

---

## 6. Why A100, and where it's weak

### 6.1 Hardware

| GPU | Memory | Bandwidth | BF16 dense | FP8 dense | INT8 dense | TDP |
|---|---|---|---|---|---|---|
| **A100 80GB PCIe** (your nodes) | 80 GB HBM2e | **1,935 GB/s** | 312 TFLOPS | **none** | 624 TOPS | 300 W |
| A100 80GB SXM | 80 GB HBM2e | 2,039 GB/s | 312 TFLOPS | none | 624 TOPS | 400 W |
| H100 SXM | 80 GB HBM3 | 3,350 GB/s | 989 TFLOPS | 1,979 TFLOPS | 1,979 TOPS | 700 W |
| H100 NVL (Azure NC H100 v5) | 94 GB HBM3 | 3,938 GB/s | ≈835 TFLOPS | ≈1,671 TFLOPS | ≈1,671 TOPS | 400 W |
| L40S | 48 GB GDDR6 | 864 GB/s | 362 TFLOPS | 733 TFLOPS | 733 TOPS | 350 W |
| H200 | 141 GB HBM3e | 4,800 GB/s | 989 TFLOPS | 1,979 TFLOPS | 1,979 TOPS | 700 W |

### 6.2 The physics you'll be measuring

- **Decode is memory-bandwidth-bound.** Every decode step reads the active weights (plus KV cache) from HBM, so the batch-1 ceiling ≈ bandwidth ÷ bytes read per token. Batching reuses each weight read across many sequences, which is why throughput grows with concurrency until compute or KV memory runs out.
- **Prefill is compute-bound.** It costs ≈ 2 × active params × prompt tokens FLOPs, plus attention that grows with the square of prompt length.

| Model on 1× A100 80GB PCIe | Weights | Batch-1 decode ceiling¹ | KV pool² | KV capacity |
|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B, BF16** | 56.9 GiB | ≈293 tok/s (3.3B active × 2 bytes) | ≈18 GiB | ≈198k tokens |
| **Qwen3-Coder-30B-A3B, FP8 → W8A16** | 29.1 GiB | ≈586 tok/s | ≈46 GiB | ≈502k tokens |
| Llama-3.1-8B, BF16 | 15.0 GiB | ≈120 tok/s | ≈60 GiB | ≈492k tokens |
| Qwen3-8B, BF16 | 15.3 GiB | ≈118 tok/s | ≈60 GiB | ≈435k tokens |
| Devstral-Small-2-24B, FP8 → W8A16 | 22.4 GiB | ≈81 tok/s | ≈53 GiB | ≈345k tokens |
| Qwen3.8-27B (dense, hybrid attention), BF16 | 51.8 GiB | ≈35 tok/s | ≈23 GiB | ≈380k tokens³ |
| gpt-oss-20b, MXFP4 | ≈13 GiB | (≈ 2–3 GB/token; measure it) | ≈62 GiB | > 1M tokens³ |

¹ Upper bound = bandwidth ÷ bytes of active weights per token. Real kernels land well below it; measuring your "% of roof" is exercise #1.
² ≈ 79 GiB visible − weights − ≈4 GiB for activations and CUDA graphs (rough). vLLM prints the exact `GPU KV cache size: N tokens` at startup.
³ Hybrid or sliding-window attention means real KV use is lower than this naive estimate.

**Three takeaways:**

1. **MoE with ≈3–4B active parameters is A100's sweet spot.** Qwen3-Coder-30B-A3B has about 8× the decode ceiling of a 27B dense model. As batch size grows, nearly every expert gets used each step, so MoE decode approaches reading all 61 GB per step, spread across the batch.
2. **Weight-only quantization buys both speed and capacity on A100.** FP8 halves the bytes read per token (a higher ceiling) and frees ≈28 GiB for KV cache.
3. **Long prompts are A100's weak spot.** Back-of-envelope prefill time for Qwen3-Coder-30B-A3B at 40 % model-FLOPs utilization:
   - ≈0.06 s for a 1k-token prompt
   - ≈0.8 s for 10k
   - **≈35 s for 100k** (attention dominates)

   Prefix caching (on by default in both engines) is what makes long-context agentic workloads viable.

### 6.3 What A100 can't do natively, and how the engines compensate

| Feature | A100 (sm_80) reality |
|---|---|
| FP8 compute (W8A8) | ❌ No FP8 tensor cores. FP8 checkpoints run **weight-only (W8A16) via Marlin** in vLLM (`MarlinFP8ScaledMMLinearKernel`) and SGLang (automatic on sm80–89). |
| INT8 W8A8 | ✅ Native INT8 tensor cores, in both engines |
| AWQ/GPTQ INT4 | ✅ Marlin kernels. In SGLang use `gptq_marlin`; plain `gptq` was removed. |
| MXFP4 (gpt-oss) | ✅ vLLM, with an official A100 recipe (`TRITON_ATTN` + Marlin MXFP4 MoE). ⚠️ SGLang on A100 is unclear (its kernels are Hopper-gated). |
| NVFP4 checkpoints | ✅ Load via Marlin W4A16 (weight-only). Native FP4 compute is Blackwell-only. |
| FlashAttention-3 / FA4 | ❌ Hopper / Blackwell only. vLLM uses FlashAttention-2 on A100; SGLang defaults to FlashInfer. |
| FP8 KV cache | ⚠️ Depends on the attention backend. vLLM's default FA2 backend rejects it on Ampere (use FlashInfer or Triton). SGLang supports `fp8_e4m3` / `fp8_e5m2`. |

### 6.4 Economics (Cost per 1M Output Tokens vs. Sustained Throughput)

**Cost per 1M output tokens = hourly VM rate ÷ (output tok/s × 3600) × 10⁶.** Illustrative on-demand values for one A100 80GB PCIe across sustained output token rates; plug in *your* measured throughput at *your* SLO:

| Sustained output tok/s | On-demand $/1M output tokens |
|---|---|
| 500 | \$2.31 |
| 1,000 | \$1.15 |
| 2,000 | \$0.58 |
| 4,000 | \$0.29 |

**A100 vs H100, honestly:**

- **Measured performance gaps:**
  - Microsoft's Azure HPC blog (Llama-3.1-8B, vLLM, 1 GPU, chat profile): A100 ≈2,622 vs H100 ≈6,067 generated tok/s, about 2.3×. The blog named A100 (40GB) the **best value for chat**.
  - An independent Qwen3-8B BF16 test put H100 at about 2.29× A100.
- **Price and bandwidth gaps:**
  - On-demand list price ratio (H100 NVL ÷ A100 PCIe) ≈ **2.07×**.
  - Bandwidth ratio (H100 NVL ÷ A100 PCIe) = **2.04×**.
  - So for BF16, bandwidth-bound small-model serving, **\$/token is roughly a wash**. H100 wins once you use native FP8 (W8A8 plus FP8 KV cache) and on long-prompt, compute-bound workloads.
- **The defensible enterprise claim:** *"For 8–30B-class models, one A100 gives production-grade latency at roughly H100-comparable \$/token, on capacity many enterprises can get or already own."* Prove it with [Phase 7](#11-two-week-experiment-plan).

### 6.5 Are small models good enough?

- Epoch AI (Oct 2025): open-weight models trail the best closed model by only **≈3 months** on their capability index.
- Stanford AI Index 2026: the gap between the top closed and top open model on Chatbot Arena was **≈3.3 %** (Mar 2026).
- Small models on SWE-bench Verified: gpt-oss-20b reaches 60.7 % (high effort, vendor harness), and Devstral Small 2 (24B) claims 68 %.

---

## 7. The one model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`

**Why this one:**

- **It fits one A100 comfortably, in BF16 or FP8.** BF16 vs W8A16 becomes your first real inference-engineering experiment.
- **It's first-class in both vLLM and SGLang on Ampere.** Standard grouped-query attention (GQA) plus MoE, no exotic kernels, so the engine head-to-head is fair.
- **It's a non-thinking model**, so reasoning tokens don't distort speed numbers.
- **It does native tool calling** (`qwen3_coder` parser in both engines), so SWE-bench via mini-SWE-agent and Terminal-Bench via Harbor work out of the box.
- **External yardsticks exist:** AA output speed of 128 tok/s (18 Nov 2026) and 51.6 % SWE-bench Verified with OpenHands.
- **It's Apache-2.0 and ungated**, so there's no license friction for an enterprise demo.

| Fact | Value |
|---|---|
| Params | 30.5B total / 3.3B active (128 experts, 8 routed per token) |
| Attention | 48 layers, 32 query / **4 KV** heads, head_dim 128 → **96 KiB of KV cache per token** (BF16) |
| Context | 262,144 tokens native (YaRN extends to 1M) |
| Checkpoints | BF16: 61.1 GB. **`Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`**: 31.2 GB, block-wise FP8, runs as W8A16 on A100. |
| Sampling (model card) | temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05 |
| Sibling | `Qwen/Qwen3-30B-A3B-Instruct-2507`: same architecture and memory, for general chat. Has an EAGLE-3 speculative-decoding head: `RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3`. |

**Capacity planning (rough):**

| | KV tokens | Concurrent sequences @ 8k context | @ 32k context |
|---|---|---|---|
| BF16 | ≈198k | ≈24 | ≈6 |
| FP8 (W8A16) | ≈502k | ≈61 | ≈15 |
| FP8 weights + FP8 KV cache | ≈1M | ≈120 | ≈30 |

**Alternates (swap in once your harness works):**

| Model | Why | Watch out |
|---|---|---|
| `openai/gpt-oss-20b` | OpenAI brand; ≈13 GiB in MXFP4; strong at reasoning and tools; AA's 141 tok/s is a comparator | It's a reasoning model (harmony format; vLLM parser `openai_gptoss`). vLLM has an official A100 path; SGLang's is uncertain. A third-party A100 test showed SGLang far behind vLLM, probably on a fallback path. |
| `meta-llama/Llama-3.1-8B-Instruct` | The **MLPerf reference model**, so you can compare against published SLOs; best EAGLE-3 coverage | Gated license; older |
| `mistralai/Devstral-Small-2-24B-Instruct-2512` | Best claimed SWE-bench score (68 %) at ≤ 24B | Dense, so ≈81 tok/s ceiling even in FP8; ships as FP8 (runs W8A16 on A100) |
| `Qwen/Qwen3.8-27B` | Newest Qwen dense hybrid (Aug 2026) | vLLM recipe is verified on Blackwell/RTX, not A100. Dense, so decode is slow on A100 (≈35 tok/s ceiling in BF16). |
| `google/gemma-4-26B-A4B-it` | Apache-2.0 MoE, 3.8B active, 5:1 sliding-window attention (small KV) | Newer; verify engine support on Ampere |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | Hybrid Mamba + MoE with a tiny KV cache | NVIDIA license. The BF16 checkpoint is 72 GB, so use FP8. |

Doesn't fit on one A100: **Mistral Small 4** (a 119B MoE; Mistral states ≥ 4× H100).

---

## 8. vLLM on one A100 (v0.34.0)

### 8.1 Run it

```bash
# Docker on a box with 1× A100. The image is CUDA 13.0.3, so the host needs an R580+ driver.
docker run --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.34.0 \
  Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --served-model-name qwen3-coder \
  --max-model-len 65536 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder

# FP8 (runs W8A16 on Ampere): use the model ID Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
```

In the startup logs, check:
- the attention backend (FlashAttention-2 on Ampere),
- `Marlin` (FP8 build only),
- `GPU KV cache size: … tokens`, which is your real capacity,
- CUDA-graph capture time.

### 8.2 What vLLM does by default on A100 (verified in the v0.34.0 source)

| Setting | Default on A100 | Why it matters |
|---|---|---|
| `--gpu-memory-utilization` | **Unset → 1.0, with the new extensible KV cache** (0.92 if that's disabled) | New in v0.34: the KV cache is sized from memory actually free after warmup. Release notes warn TPOT can rise 6–18 % on some models at high concurrency. Opt out with `--gpu-memory-utilization 0.9` or `--no-enable-extensible-kv-cache`. |
| `--max-num-seqs` | 256 | Max concurrent sequences per batch |
| `--max-num-batched-tokens` | **2048** (H100/H200 get 8192) | Deliberately small on A100: PR #17885 found larger values reduced A100 throughput. Trades TTFT against TPOT under load. |
| Chunked prefill | On | Long prompts are split so decodes keep flowing |
| Prefix caching | On | Shared prompt prefixes are reused, a big win for agentic and multi-turn traffic |
| `--block-size` | 64 (was 16) | KV cache page size |
| Compilation | `-O2` (torch.compile + full/piecewise CUDA graphs) | Use `-O0` or `--enforce-eager` to see what graphs buy you |
| Attention | FlashAttention-2 (FA3 is Hopper-only) | Override with `--attention-backend`; the old env var was removed |
| `--generation-config` | `auto` → the model's `generation_config.json` | Server-side default sampling follows the model card unless the client overrides it |

### 8.3 Knobs worth one experiment each

| Experiment | Flag(s) | What to test |
|---|---|---|
| BF16 vs FP8 weights | Model ID `…-FP8` | Higher batch-1 tok/s and ≈2.5× KV capacity; check quality per [§4.9](#49-harnesses-that-work-with-an-openai-compatible-endpoint) |
| Extensible vs fixed KV cache | (unset) vs `--gpu-memory-utilization 0.9` | More capacity vs a possible TPOT penalty; quantify it on A100 |
| Prefill chunk budget | `--max-num-batched-tokens 2048/4096/8192` | Bigger usually improves TTFT under load but worsens TPOT spikes |
| Concurrency cap | `--max-num-seqs 32/64/128/256` | Protects the TPOT SLO at the cost of queueing (higher TTFT) |
| FP8 KV cache | `--kv-cache-dtype fp8 --attention-backend FLASHINFER` | ≈2× KV tokens; verify accuracy. FA2 rejects FP8 KV on Ampere. |
| N-gram speculative decoding | `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}'` | Should win on code edits and repetitive output at low concurrency, and fade at high concurrency |
| EAGLE-3 (on the 2507 sibling) | `--speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3","num_speculative_tokens":3}'` | Lower latency at concurrency 1–8; little or negative gain at 64+ |
| Async scheduling | `--async-scheduling` | Hides CPU scheduling overhead |
| CUDA graphs off | `--enforce-eager` | Shows what graphs buy you, especially at small batch sizes |

**Watch:** `GET /metrics` exposes
- `vllm:time_to_first_token_seconds`
- `vllm:inter_token_latency_seconds`
- `vllm:request_time_per_output_token_seconds`
- `vllm:kv_cache_usage_perc`
- `vllm:num_requests_running`
- `vllm:num_requests_waiting`

Pair them with DCGM metrics: `DCGM_FI_DEV_GPU_UTIL`, and `DCGM_FI_PROF_DRAM_ACTIVE` (how close decode is to the bandwidth roof).

### 8.4 Benchmark it

```bash
# Concurrency sweep, InferenceX-style 1k/1k random tokens
for c in 1 2 4 8 16 32 64 128 256; do
  vllm bench serve --backend openai-chat --base-url http://<ep>:8000 --endpoint /v1/chat/completions \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct --served-model-name qwen3-coder \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 \
    --num-prompts $(( c*8 > 64 ? c*8 : 64 )) --max-concurrency "$c" --ignore-eos --num-warmups 4 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --goodput ttft:2000 tpot:100 \
    --save-result --result-dir ./results/bf16-1k1k --result-filename "c${c}.json"
done
# Repeat with --random-input-len 8192 --random-output-len 1024.
# Then plot tok/s per user (1000/TPOT) against output tok/s.
```

`vllm bench sweep serve_workload` and `vllm bench sweep plot_pareto` automate the sweep and the Pareto plot.

---

## 9. SGLang on one A100 (v0.5.23)

### 9.1 Run it

```bash
# The image is CUDA 13 (SGLang dropped CUDA 12 builds after v0.5.19), so the host needs an R580+ driver.
docker run --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:v0.5.23 \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --served-model-name qwen3-coder \
    --context-length 65536 \
    --tool-call-parser qwen3_coder \
    --enable-metrics \
    --host 0.0.0.0 --port 30000
# `sglang serve --model-path …` is equivalent
```

### 9.2 Defaults and A100 notes

| Setting | Default | Note |
|---|---|---|
| `--attention-backend` | Auto → **FlashInfer** on Ampere (Triton fallback) | Don't force `fa3`, `fa4`, or `trtllm_*`; they target Hopper or Blackwell |
| `--mem-fraction-static` | Auto (≈ 0.88) | Fraction for **weights + KV pool only**; activations live outside it. Not the same meaning as vLLM's memory flag. |
| Radix (prefix) cache | On | Use `--disable-radix-cache` for no-cache baselines |
| `--chunked-prefill-size` | Auto (`-1` disables) | SGLang's equivalent of vLLM's prefill budget |
| `--max-prefill-tokens` | 16384 | |
| `--schedule-policy` | `fcfs` | Try `lpm` (longest prefix match) for prefix-heavy agentic traffic |
| CUDA graphs | On; size them with `--cuda-graph-max-bs-decode` / `--cuda-graph-max-bs-prefill` | The old `--cuda-graph-max-bs` was split in two, with no compatibility shim |
| `--enable-torch-compile` | Off | SGLang's docs call it "out of maintenance" |
| FP8 checkpoints | Automatically run as Marlin W8A16 on sm80–89 | Same story as vLLM |
| `--kv-cache-dtype` | `auto` | `fp8_e4m3` / `fp8_e5m2` work on A100 |
| Speculative decoding | `--speculative-algorithm NGRAM\|EAGLE3\|…` | SGLang's SpecForge project publishes an EAGLE-3 head for Qwen3-Coder-30B-A3B |

**Watch:** with `--enable-metrics`, `/metrics` exposes
- `sglang:time_to_first_token_seconds`
- `sglang:inter_token_latency_seconds`
- `sglang:request_time_per_output_token_seconds`
- `sglang:token_usage` (KV cache)
- `sglang:num_running_reqs`
- `sglang:num_queue_reqs`

### 9.3 Benchmark it

```bash
python3 -m sglang.benchmark.serving --backend sglang-oai-chat --host <ep> --port 30000 \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts 512 --max-concurrency 64 --warmup-requests 4 --flush-cache
```

`--random-range-ratio` is one-sided (lengths drawn from [ratio × len, len]), so use 1.0 for fixed lengths.

**For a fair vLLM vs SGLang comparison:**
- Use **one** client against both, e.g. `vllm bench serve --backend openai-chat --base-url http://<sglang>:30000 --endpoint /v1/chat/completions …`.
- Keep the model, precision, context length, datasets, and seeds identical.
- Flush prefix caches between runs: SGLang `POST /flush_cache`; vLLM `POST /reset_prefix_cache` (requires `VLLM_SERVER_DEV_MODE=1`). Or just restart the servers.

---

## 10. Running it on AKS (example only — not applied)

This assumes the [§0 prerequisites](#0-your-starting-point-the-cluster) are done. The server runs on node 0 and the benchmark client on node 1, so traffic crosses the VNet like a real in-region client would.

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: inference
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: hf-cache
  namespace: inference
spec:
  accessModes: ["ReadWriteOnce"]
  storageClassName: managed-csi-premium
  resources:
    requests:
      storage: 200Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen3-coder-vllm
  namespace: inference
spec:
  replicas: 1
  strategy:
    type: Recreate              # one GPU per node: never run two replicas during a rollout
  selector:
    matchLabels: {app: qwen3-coder-vllm}
  template:
    metadata:
      labels: {app: qwen3-coder-vllm}
    spec:
      nodeSelector:
        kubernetes.io/hostname: <node-0>
      containers:
        - name: vllm
          image: <your-acr>.azurecr.io/vllm/vllm-openai:v0.34.0   # entrypoint is `vllm serve`
          args:
            - Qwen/Qwen3-Coder-30B-A3B-Instruct
            - --served-model-name=qwen3-coder
            - --max-model-len=65536
            - --enable-auto-tool-choice
            - --tool-call-parser=qwen3_coder
          env:
            - {name: HF_HOME, value: /models}
          ports:
            - containerPort: 8000
          resources:
            requests: {cpu: "16", memory: 120Gi, nvidia.com/gpu: "1"}
            limits: {memory: 180Gi, nvidia.com/gpu: "1"}
          volumeMounts:
            - {name: hf-cache, mountPath: /models}
            - {name: dshm, mountPath: /dev/shm}
          startupProbe:              # first start downloads 61 GB, compiles, and captures CUDA graphs
            httpGet: {path: /health, port: 8000}
            periodSeconds: 10
            failureThreshold: 180
          readinessProbe:
            httpGet: {path: /health, port: 8000}
            periodSeconds: 10
      volumes:
        - name: hf-cache
          persistentVolumeClaim: {claimName: hf-cache}
        - name: dshm
          emptyDir: {medium: Memory, sizeLimit: 16Gi}
---
apiVersion: v1
kind: Service
metadata:
  name: qwen3-coder
  namespace: inference
spec:
  selector: {app: qwen3-coder-vllm}
  ports:
    - port: 8000
      targetPort: 8000
---
apiVersion: batch/v1
kind: Job
metadata:
  name: bench-c32-1k1k
  namespace: inference
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        kubernetes.io/hostname: <node-1>
      containers:
        - name: bench
          image: <your-acr>.azurecr.io/vllm/vllm-openai:v0.34.0
          command: ["vllm", "bench", "serve"]
          args:
            - --backend=openai-chat
            - --base-url=http://qwen3-coder:8000
            - --endpoint=/v1/chat/completions
            - --model=Qwen/Qwen3-Coder-30B-A3B-Instruct
            - --served-model-name=qwen3-coder
            - --dataset-name=random
            - --random-input-len=1024
            - --random-output-len=1024
            - --num-prompts=320
            - --max-concurrency=32
            - --ignore-eos
            - --num-warmups=4
          resources:
            requests: {cpu: "8", memory: 16Gi}
```

Notes:

- Read results with `kubectl logs -n inference job/bench-c32-1k1k`. Add a volume plus `--save-result --result-dir` if you want the JSON files.
- `kubernetes.io/hostname` pins specific VMSS instances; update it if nodes get reimaged or replaced.
- **Don't benchmark through `kubectl port-forward`.**
- For SGLang, swap the container to `lmsysorg/sglang:v0.5.23` with `command: ["python3", "-m", "sglang.launch_server"]`, the args from [§9.1](#91-run-it), and port 30000.
- Later options, not needed for single-GPU serving:
  - **KAITO** (AKS AI toolchain operator) deploys with one custom resource and uses vLLM by default.
  - **llm-d**, **KServe**, and the **Gateway API Inference Extension** matter once you go multi-replica.

---

## 11. Two-week experiment plan

| Phase | Goal | Do | Output |
|---|---|---|---|
| 0. Bring-up | A known-good baseline | Serve BF16 with defaults. Smoke-test a chat request and a tool call. Record the attention backend, KV tokens, startup time, and driver/CUDA versions. | Config fingerprint |
| 1. Roofline | Learn the physics | Concurrency 1, 1k/1k, `--ignore-eos`. Decode tok/s = 1000 / TPOT. Compare with the ceiling (293 BF16 / 586 FP8). | "% of bandwidth roof" |
| 2. Capacity curve | **The core artifact** | Concurrency sweep 1 → 256 at 1k/1k and at 8k/1k | Pareto plot (tok/s/user vs tok/s/GPU), the knee, max goodput at MLPerf-style SLOs |
| 3. AA-style probe | Compare with AA | Run the [§12](#12-aa-style-probe-script) script at 1k / 10k / 100k with unique prompts, temperature 0, at your knee concurrency | AA-v3 output speed and TTFT vs AA's 128 tok/s |
| 4. One knob at a time | Build intuition | FP8 vs BF16; extensible KV vs 0.9; prefill budget; FP8 KV cache; n-gram speculation at concurrency 1/8/64; a prefix-heavy workload (`prefix_repetition` / `generated-shared-prefix`) | Delta table against the baseline |
| 5. vLLM vs SGLang | A fair head-to-head | Same model, precision, and context; same client and seeds; caches flushed | Two Pareto curves on one chart |
| 6. Quality guardrails | Don't ship a fast-but-dumb config | Quick evals (IFEval, GPQA Diamond, a LiveCodeBench subset), then a 50-instance SWE-bench Verified slice, then all 500 | Score deltas, $ per resolved issue |
| 7. The A100 story | Evidence for the narrative | Run the same scripts on one NC40ads_H100_v5 for a few hours (with native FP8 there) | $/1M tokens at SLO: A100 vs H100 |

**Record every run as one row:**

```csv
date,engine,engine_version,model,precision,kv_dtype,max_model_len,max_num_seqs,max_batched_tokens,spec,dataset,isl,osl,concurrency,request_rate,num_prompts,ttft_p50_ms,ttft_p99_ms,tpot_p50_ms,tpot_p99_ms,itl_p99_ms,output_tok_s,req_s,goodput_req_s,kv_usage_peak,gpu_util,notes
```

---

## 12. AA-style probe script

This mirrors AA v3.0.0 where practical:
- unique prompts per request,
- the official OpenAI client with streaming,
- server-reported token counts,
- TTFT and TTFAT,
- output speed = N / E2E,
- failures counted as speed 0.

It also prints the classic decode speed, (N − 1) / (E2E − TTFT). The script was syntax-checked and functionally tested against a local mock OpenAI-compatible streaming server; it hasn't been run against vLLM or SGLang yet. A copy is saved next to this report as `aa_probe.py`.

```python
"""AA-style per-request speed probe for any OpenAI-compatible endpoint (vLLM, SGLang, ...).

Mirrors Artificial Analysis performance methodology v3.0.0 where practical:
unique prompts per request, streaming via the official OpenAI client, temperature 0
(non-reasoning), provider-reported token counts (usage), TTFT / TTFAT, and
output speed = output tokens / full request time (includes TTFT). Endpoint-side
failures count as speed 0. Run once per workload size and concurrency level.

    pip install openai
    BASE_URL=http://<endpoint>:8000/v1 MODEL=qwen3-coder CONCURRENCY=8 \
      INPUT_WORDS=750 REQUESTS=64 python aa_probe.py
"""
import asyncio
import os
import random
import string
import time

from openai import AsyncOpenAI

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("MODEL", "qwen3-coder")
CONCURRENCY = int(os.getenv("CONCURRENCY", "8"))
REQUESTS = int(os.getenv("REQUESTS", "64"))
INPUT_WORDS = int(os.getenv("INPUT_WORDS", "750"))  # calibrate with the printed prompt_tokens
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))  # AA: 0 non-reasoning, 0.6 reasoning

client = AsyncOpenAI(base_url=BASE_URL, api_key=os.getenv("API_KEY", "EMPTY"), timeout=600)


def make_prompt(i: int) -> str:
    # Unique filler per request defeats prefix caching, like AA's per-run prompt sampling.
    rnd = random.Random(i)
    words = (
        "".join(rnd.choices(string.ascii_lowercase, k=rnd.randint(3, 9)))
        for _ in range(INPUT_WORDS)
    )
    return (
        f"Document #{i}:\n{' '.join(words)}\n\n"
        "Describe any patterns you notice in the document above, then write a detailed "
        "technical essay about how large language model inference works on GPUs."
    )


async def one_request(i: int) -> dict:
    t0 = time.perf_counter()
    ttft = ttfat = None
    usage = None
    err = None
    try:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": make_prompt(i)}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=1,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            now = time.perf_counter()
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            if ttft is None and (delta.content or reasoning):
                ttft = now - t0
            if ttfat is None and delta.content:
                ttfat = now - t0
    except Exception as e:  # endpoint-side failure -> speed 0, like AA
        err = repr(e)
    e2e = time.perf_counter() - t0
    out = usage.completion_tokens if usage else 0
    return {
        "err": err,
        "inp": usage.prompt_tokens if usage else None,
        "out": out,
        "ttft": ttft,
        "ttfat": ttfat,
        "e2e": e2e,
        "speed_aa_v3": out / e2e if out and not err else 0.0,
        "decode_tok_s": (out - 1) / (e2e - ttft) if out > 1 and ttft and not err else 0.0,
    }


def pct(values, p):
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


async def main():
    await one_request(-1)  # warmup (CUDA graphs, caches); excluded from stats
    sem = asyncio.Semaphore(CONCURRENCY)

    async def guarded(i):
        async with sem:
            return await one_request(i)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(guarded(i) for i in range(REQUESTS)))
    wall = time.perf_counter() - t0

    ok = [r for r in results if not r["err"]]
    print(f"model={MODEL} concurrency={CONCURRENCY} requests={REQUESTS} "
          f"errors={len(results) - len(ok)} prompt_tokens_p50={pct([r['inp'] for r in ok], 50)}")
    for key, unit in [("ttft", "s"), ("ttfat", "s"), ("e2e", "s"),
                      ("speed_aa_v3", "tok/s"), ("decode_tok_s", "tok/s")]:
        vals = [r[key] for r in results]
        print(f"{key:>13} ({unit:5}) p50={pct(vals, 50):9.3f}  p90={pct(vals, 90):9.3f}  p99={pct(vals, 99):9.3f}")
    print(f"system output throughput: {sum(r['out'] for r in ok) / wall:.1f} tok/s over {wall:.1f}s")
    for r in results:
        if r["err"]:
            print("first error:", r["err"])
            break


if __name__ == "__main__":
    asyncio.run(main())
```

To mimic AA's three workload sizes, run it three times with `INPUT_WORDS` tuned until the printed `prompt_tokens_p50` is ≈1k, ≈10k, and ≈100k. Take the p50 `speed_aa_v3` and `ttft` from each run, average the three, and report the concurrency you used.

---

## 13. Pitfalls checklist

1. **Prefix-cache inflation.** Repeated prompts make you look fast. Use unique prompts or flush caches.
2. **Warmup.** The first requests pay for torch.compile and CUDA-graph capture; exclude them.
3. **Variable output lengths.** Without `--ignore-eos`, output length varies and throughput comparisons break.
4. **Closed vs open loop.** Fixed concurrency hides stalls; use Poisson arrivals for tail-latency claims.
5. **Client bottleneck.** A single Python client can saturate before the server does. Watch client CPU, and use AIPerf at very high concurrency.
6. **Network.** Benchmark in-cluster, never through `kubectl port-forward`.
7. **Tokenizer mismatch.** Use the model's tokenizer on the client side, and prefer server-reported `usage`.
8. **Chunks aren't tokens.** With speculative decoding, ITL measures chunks; report TPOT too.
9. **Sampling, templates, and parsers.** Use model-card sampling for evals, and verify the chat template and parsers. In vLLM's Kimi-K2 post-mortem, a template bug dropped tool-call success below 20 %.
10. **Changed defaults.** vLLM v0.34 changed the GPU-memory and block-size defaults. Record the engine version and full arguments with every result.
11. **A100 PCIe isn't A100 SXM.** Most published "A100" numbers are SXM (400 W, 2.04 TB/s); expect your PCIe cards (300 W, 1.94 TB/s) to come in somewhat lower.
12. **Quantization.** Re-run the quality guardrails whenever precision changes.

---

## 14. Gaps and open questions

- **No credible single-A100 numbers** turned up for Qwen3-Coder-30B-A3B, Qwen3-30B-A3B, or Gemma 4 on current engine versions, so your Phase 2 data would be new. A GPUStack A100-SXM test of gpt-oss-20b showed vLLM far ahead of SGLang, likely because SGLang hit a fallback path.
- **SGLang MXFP4 on A100** (needed for gpt-oss) is unclear from the source code; test it before relying on it.
- **FP8 KV cache on Ampere** depends on the attention backend; verify it per engine.
- **AA's exact concurrency** isn't published, so any AA-comparable number needs its concurrency stated.
- **The driver version AKS installed** on your nodes wasn't checked; that needs `nvidia-smi` from a pod, which would have been a change. CUDA 13 images need R580+.
- **Market price trends** in §6.4 come from secondary reporting. The Azure prices are first-party retail list prices.
- **AA small-model rankings** under Index v5 were only partly visible in the server-rendered leaderboard (e.g. Nanbeige4.3-33B ≈ 20). Treat any ranking as indicative.

---

## Sources

**Artificial Analysis**
- Performance methodology v3.0.0: <https://artificialanalysis.ai/methodology/performance-benchmarking>
- Intelligence methodology v5.0.1: <https://artificialanalysis.ai/methodology/intelligence-benchmarking>
- AgentPerf methodology: <https://artificialanalysis.ai/methodology/agentperf>
- GPU comparison (MI300X / H100 / H200): <https://artificialanalysis.ai/articles/independent-analysis-of-leading-gpus-amd-nvidia>
- Model pages: <https://artificialanalysis.ai/models/qwen3-coder-30b-a3b-instruct>, <https://artificialanalysis.ai/models/gpt-oss-20b>, <https://artificialanalysis.ai/models/qwen3-30b-a3b-2507>
- Leaderboard: <https://artificialanalysis.ai/leaderboards/models>
- Stirrup harness: <https://github.com/ArtificialAnalysis/Stirrup>

**SWE-bench and evals**
- SWE-bench: <https://www.swebench.com/>, <https://github.com/SWE-bench/SWE-bench>, evaluation guide `docs/guides/evaluation.md`
- OpenAI: <https://openai.com/index/introducing-swe-bench-verified/>, <https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/>
- SWE-bench Pro: <https://github.com/scaleapi/SWE-bench_Pro-os>
- sb-cli: <https://github.com/swe-bench/sb-cli>
- mini-SWE-agent: <https://mini-swe-agent.com/latest/models/local_models/>, <https://mini-swe-agent.com/latest/usage/swebench/>
- Terminal-Bench and Harbor: <https://www.tbench.ai/>, <https://github.com/harbor-framework/harbor>
- gpt-oss model card: <https://arxiv.org/abs/2508.10925>
- lm-evaluation-harness: <https://github.com/EleutherAI/lm-evaluation-harness>
- Inspect AI providers: <https://inspect.aisi.org.uk/providers.html>
- evalscope: <https://github.com/modelscope/evalscope>
- lighteval: <https://github.com/huggingface/lighteval>
- vLLM Kimi-K2 accuracy post-mortem: <https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy>

**Performance benchmarking**
- vLLM benchmark CLI: <https://docs.vllm.ai/en/latest/benchmarking/cli/>
- vLLM sweeps: <https://docs.vllm.ai/en/latest/benchmarking/sweeps/>
- SGLang bench_serving: <https://docs.sglang.io/docs/developer_guide/bench_serving>
- guidellm: <https://github.com/vllm-project/guidellm>
- AIPerf: <https://github.com/ai-dynamo/aiperf>
- inference-perf: <https://github.com/kubernetes-sigs/inference-perf>
- InferenceX: <https://github.com/SemiAnalysisAI/InferenceX>
- MLPerf Inference rules: <https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc>
- MLPerf v6.1 results: <https://github.com/mlcommons/inference_results_v6.1>

**Engines**
- vLLM v0.34.0 release notes: <https://github.com/vllm-project/vllm/releases/tag/v0.34.0>
- vLLM A100 defaults in source: `vllm/engine/arg_utils.py` (A100 batch defaults, PR #17885) and `vllm/config/cache.py` @ v0.34.0
- vLLM gpt-oss recipe (A100 section): <https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html>
- vLLM quantization: <https://docs.vllm.ai/en/latest/features/quantization/>
- vLLM attention backends: <https://docs.vllm.ai/en/latest/design/attention_backends/>
- vLLM speculative decoding: <https://docs.vllm.ai/en/stable/features/speculative_decoding/>
- vLLM metrics: <https://docs.vllm.ai/en/stable/usage/metrics/>
- vLLM Qwen3.8-27B recipe: `vllm-project/recipes/models/Qwen/Qwen3.8-27B.yaml`
- SGLang v0.5.20 release notes (CUDA 12 builds retired): <https://github.com/sgl-project/sglang/releases/tag/v0.5.20>
- SGLang attention backends: <https://docs.sglang.io/docs/advanced_features/attention_backend>
- SGLang quantization: <https://docs.sglang.io/docs/advanced_features/quantization>
- SGLang speculative decoding: <https://docs.sglang.io/docs/advanced_features/speculative_decoding>
- Red Hat, EAGLE-3 on vLLM (A100): <https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding>

**Hardware, Azure, and market**
- NVIDIA product pages: <https://www.nvidia.com/en-us/data-center/a100/>, <https://www.nvidia.com/en-us/data-center/h100/>, <https://www.nvidia.com/en-us/data-center/h200/>
- Azure NC A100 v4: <https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series>
- Azure NC H100 v5: <https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/ncadsh100v5-series>
- Azure Retail Prices API (queried 18 Nov 2026, `armRegionName eq 'centralus'`): <https://prices.azure.com/api/retail/prices>
- AKS GPUs: <https://learn.microsoft.com/azure/aks/use-nvidia-gpu>
- AKS managed GPU nodes: <https://learn.microsoft.com/azure/aks/aks-managed-gpu-nodes>
- AKS MIG: <https://learn.microsoft.com/azure/aks/gpu-multi-instance>
- KAITO: <https://learn.microsoft.com/azure/aks/ai-toolchain-operator>
- Microsoft Azure HPC blog, Llama-3.1-8B across GPUs: <https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/inference-performance-of-llama-3-1-8b-using-vllm-across-various-gpus-and-cpus/4448420>
- JarvisLabs, H100 vs A100: <https://jarvislabs.ai/blog/h100-vs-a100>
- GPUStack, gpt-oss-20b on A100: <https://docs.gpustack.ai/2.3/performance-lab/gpt-oss-20b/a100/>
- Epoch AI, open vs closed weights: <https://epoch.ai/data-insights/open-weights-vs-closed-weights-models>
- Stanford AI Index 2026: <https://hai.stanford.edu/ai-index/2026-ai-index-report/technical-performance>
- Model cards on Hugging Face: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (+ `-FP8`), `Qwen/Qwen3-30B-A3B-Instruct-2507`, `openai/gpt-oss-20b`, `mistralai/Devstral-Small-2-24B-Instruct-2512`, `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3.8-27B`, `google/gemma-4-26B-A4B-it`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`
