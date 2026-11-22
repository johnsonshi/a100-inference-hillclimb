# How inference providers measure LLM speed, and one-line tools to measure yours

> Companion to `a100-inference-engineering-report.md` · 18 Nov 2026 · Research only; nothing was deployed or changed.
>
> Tool behavior and flags were checked against each project's GitHub README or source code today. Provider claims were checked on the provider's own page today. I couldn't search X directly, so this covers blogs, docs, and GitHub.

---

## TL;DR

- **Everyone measures speed the same way: a client sends a streaming request and timestamps the chunks.**
  - First chunk with text → time to first token (TTFT).
  - First chunk with *answer* text (not reasoning) → time to first answer token (TTFAT).
  - Last chunk → end-to-end latency (E2E).
  - Token counts come from the server's `usage` field or a client-side tokenizer.
  - Every other metric is arithmetic on those numbers. Results differ because of workload shape, concurrency, client location, and **naming** ([§2](#2-same-name-different-formula)).
- **Fireworks publishes its actual harness**: [`fw-ai-external/benchmark`](https://github.com/fw-ai-external/benchmark), Locust-based, with commits as recent as 10 Nov 2026. It handles reasoning models (`--first-visible-token`), Poisson or bursty arrivals, production-shaped prompt mixes with controlled cache hits, and trace replay. It also logs server-side TTFT next to client-side TTFT, so you can see network overhead.
- **Baseten publishes a methodology guide, not a harness.**
  - Its "Understanding performance benchmarks" post defines TTFT and three kinds of tokens/s, and lists what every benchmark must disclose.
  - Its speed-record posts (Kimi K2 Thinking, Kimi K2.5) cite **Artificial Analysis (AA)** results rather than a self-run setup.
  - On GitHub it maintains a fork of SGLang's `genai-bench` (95 commits ahead of upstream), and NVIDIA's AIPerf can replay Baseten-format production traces.
- **Most public speed claims are really AA's numbers.**
  - Fireworks and Baseten cite AA for headline results.
  - Groq's docs say its console shows server-side latency only, and point to AA for end-to-end numbers.
  - OpenRouter publishes its own per-provider TTFT and throughput (p50–p99 over a rolling 5-minute window).
- **Same name, different formula.**
  - vLLM's "ITL" is the gap between streamed chunks; AIPerf's and guidellm's "ITL" is average decode time per token (what vLLM calls "TPOT").
  - guidellm's "TPOT" and LLMPerf's "ITL" *include* the first token.
  - Translate before comparing numbers across tools.
- **Yes, one-line tools exist** ([§4](#4-one-line-tools-verified-18-nov-2026)):
  - Interactive back-and-forth, with TTFT and tok/s printed per turn: `aiperf chat --model <name> --url http://<endpoint>:8000`
  - Automatic load sweep: `guidellm run --backend kind=openai_http,target=http://<endpoint>:8000 --profile kind=sweep --data kind=synthetic_text,prompt_tokens=1024,output_tokens=1024`
  - Concurrency sweep with TTFT / ITL / answer-token percentiles: `aiperf profile … --isl 1024 --osl 1024 --concurrency 1,8,32,64`

---

## Contents

1. [How the measurement works](#1-how-the-measurement-works)
2. [Same name, different formula](#2-same-name-different-formula)
3. [What providers and benchmarkers publish](#3-what-providers-and-benchmarkers-publish)
4. [One-line tools (verified 18 Nov 2026)](#4-one-line-tools-verified-18-nov-2026)
5. [Back-and-forth: multi-turn and replayed traffic](#5-back-and-forth-multi-turn-and-replayed-traffic)
6. [What to use for the A100 project](#6-what-to-use-for-the-a100-project)
7. [How to read a provider's speed claim](#7-how-to-read-a-providers-speed-claim)
8. [Sources](#sources)

---

## 1. How the measurement works

Every tool in this doc, and AA itself, boils down to this:

```text
t0 = now()
POST /v1/chat/completions   {stream: true, stream_options: {include_usage: true}, ...}
for each streamed chunk (t = arrival time):
    first chunk with any text or reasoning   → TTFT  = t − t0
    first chunk with answer (non-reasoning)  → TTFAT = t − t0   (AIPerf: TTFO; Fireworks: "first visible token")
    keep t for chunk-gap statistics
last content chunk                            → E2E   = t_last − t0
N = usage.completion_tokens   (or count with a tokenizer)

decode time per token     = (E2E − TTFT) / (N − 1)      per-user decode speed = 1 / that
AA-style output speed     = N / E2E                      (includes TTFT)
system output throughput  = Σ N over all requests / wall-clock time
```

The hard part is the details. What mature harnesses handle that a naive script doesn't:

| Gotcha | Why it matters | Who handles it, and how |
|---|---|---|
| Empty first chunk | SGLang sends an empty first chunk, so naive TTFT looks near zero | Fireworks starts the clock only on a chunk with text or a non-zero token count. Its code comment: "some providers (SGLang) send an empty chunk first skewing the TTFT". AIPerf's TTFT also skips empty chunks. |
| Reasoning fields differ | vLLM streams thinking in `delta.reasoning`, SGLang in `delta.reasoning_content`. A tool that reads only `content` silently reports TTFAT as "TTFT". | vLLM's and SGLang's bench clients read both fields and count reasoning toward TTFT. AIPerf (TTFO), guidellm (time to first output token), and Fireworks (`--first-visible-token`) also report the first *answer* token. |
| Chunks aren't tokens | Speculative decoding and gateways can pack several tokens into one chunk | AIPerf reports both a per-token ITL and a per-chunk ICL; vLLM's "ITL" is per chunk |
| Token counting | A client tokenizer may not match the model's | AIPerf `--use-server-token-count` and AA use server-reported counts. LLMPerf counted every provider's output with a Llama tokenizer. |
| Network and gateway time | Client-side TTFT includes the network | Fireworks also records server-reported TTFT; the gap is network + gateway time. AA's client runs in GCP `us-central1`. |
| Prompt caching | Repeated prompts inflate speed; zero reuse is unrealistic for chat and agents | Fireworks `--shape-mix` builds shared prefixes plus a unique per-request nonce, with a new `--request-seed` per run. genai-bench has prefix-ratio scenarios. vLLM/SGLang `random` datasets are unique per request. |
| Output length | Models stop early, which breaks tokens/s comparisons | Force exact lengths with `ignore_eos` (automatic in vLLM for `random`; AIPerf `--extra-inputs ignore_eos:true`). Fireworks sets `min_tokens = max_tokens` and warns on mismatches. |
| Arrival pattern | Fixed concurrency (closed loop) hides queueing stalls | Open-loop arrivals: Fireworks `--qps` with `exponential` or `gamma` gaps, vLLM `--request-rate`, AIPerf `--request-rate`, guidellm `poisson` and `constant` profiles |

---

## 2. Same name, different formula

The same raw timestamps produce different "ITL" and "TPOT" numbers depending on the tool. Verified in each tool's source or docs:

| You want | vLLM `bench serve` / SGLang bench | AIPerf | guidellm | Fireworks `load_test.py` | LLMPerf (archived) | AA |
|---|---|---|---|---|---|---|
| First token, including reasoning | TTFT | TTFT | `time_to_first_token_ms` | `time_to_first_token` (default) | `ttft_s` | TTFT |
| First **answer** token | — | **TTFO** | `time_to_first_output_token_ms` | TTFT with `--first-visible-token` | — | **TTFAT** |
| Decode time per token, first token excluded | **TPOT**, = (E2E − TTFT)/(N − 1) | **ITL**, same formula | **ITL**, same formula | `latency_per_token`, = (E2E − TTFT)/N | — | — |
| Gaps between streamed chunks | **ITL** (one sample per chunk) | **ICL** | — | — | — | — |
| Time per token, first token included | — | — | **TPOT** (!), = E2E/N | `overall_latency_per_token`, = E2E/N | **ITL** (!), ≈ E2E/N | — |
| Per-user speed | Derive as 1000 / TPOT | Output token throughput per user, = 1/ITL (excludes TTFT) | `output_tokens_per_second`, = N/E2E (includes TTFT) | `generation_tokens_per_second`, = N/(E2E − TTFT) | `request_output_throughput_token_per_s`, = N/E2E | Output speed, = N/E2E (includes TTFT since v3) |

How to compare two sources: convert both to two numbers you compute yourself. Decode time per token, (E2E − TTFT)/(N − 1), describes the engine. Output speed, N/E2E, describes what the user experiences.

Anyscale chose to fold TTFT into ITL on purpose. Its LLMPerf post says: *"We've decided to include the TTFT in the inter-token latency computation. We've seen some systems that start streaming very late."* That's a defensible choice, but it's a different number from vLLM's.

---

## 3. What providers and benchmarkers publish

| Who | Methodology published | Open-source code | Where headline numbers come from |
|---|---|---|---|
| **Fireworks** | Harness README; speed blog | [`fw-ai-external/benchmark`](https://github.com/fw-ai-external/benchmark), active | Blog cites AA |
| **Baseten** | "Understanding performance benchmarks" guide | `genai-bench` fork; `sa_spec` speculator; fork of Moonshot's K2-Vendor-Verifier | Speed-record posts cite AA |
| **Modal** | LLM Engineer's Almanac, "How to benchmark" | [`modal-labs/stopwatch`](https://github.com/modal-labs/stopwatch), a guidellm wrapper | Its own runs |
| **Together AI** | Coding-agent benchmark post describes its workload | None found | Its own runs |
| **Anyscale** | LLMPerf post (2023) | [`ray-project/llmperf`](https://github.com/ray-project/llmperf) and its leaderboard, both **archived** | — |
| **Databricks** | "LLM Inference Performance Engineering: Best Practices" (2023) | — | — |
| **OpenRouter** | Provider-performance docs and blog | — | Its own p50–p99 stats, rolling 5-minute window |
| **Groq** | Latency docs (formulas; console metrics are server-side only) | — | Points to AA |
| **NVIDIA** | "LLM Benchmarking: Fundamental Concepts"; AIPerf metrics reference | [`ai-dynamo/aiperf`](https://github.com/ai-dynamo/aiperf) | — |
| **Artificial Analysis** | Full methodology, Integrity Terms, Endpoint Accuracy Index | (Stirrup, for evals) | Itself |

### Fireworks: publishes the actual harness

- **Repo:** [`fw-ai-external/benchmark`](https://github.com/fw-ai-external/benchmark); the old `fw-ai/benchmark` URL redirects here. The harness is `llm_bench/load_test.py`, run with Locust. Commits on 6–10 Nov 2026 added production-shaped workloads (`--shape-mix`, `--replay-schedule`).
- **What it records:** overall latency, generated tokens, QPS, and, when streaming, TTFT and per-token latency. Results go to CSV via `--summary-file`. For Fireworks deployments it also logs *server-side* TTFT and total latency; per its README, client minus server ≈ network + gateway time.
- **How it times** (from `load_test.py`):
  - TTFT starts on the first chunk carrying text or a non-zero token count, which skips SGLang's empty first chunk.
  - `--first-visible-token` moves "first token" to the first non-reasoning chunk, so reasoning and non-reasoning models compare fairly. It requires `--chat --stream`.
  - `latency_per_token` = (E2E − TTFT)/N, and `overall_latency_per_token` = E2E/N.
  - Output length is forced exact by default, and the script warns when the count is off.
- **Load models:**
  - Fixed concurrency: Locust users (`-u`, `-r`).
  - Open loop: `--qps` with `constant`, `exponential` (Poisson), or `gamma` (bursty, `--qps-cv2`) arrivals, plus optional rate modulation to mimic production bursts.
  - `--shape-mix`: traffic classes with shared prefixes and unique tails, so cache hits are designed rather than accidental.
  - `--replay-schedule`: replays recorded traffic, including multi-turn sessions.
- **Public claims:** the Fireworks speed blog says it's *"fastest across the top open models among GPU-based providers, as benchmarked by Artificial Analysis."*

### Baseten: methodology guide plus AA for records

- **Guide:** ["Understanding performance benchmarks for LLM inference"](https://www.baseten.co/blog/understanding-performance-benchmarks-for-llm-inference/).
    - Factors every benchmark must state: hardware, streaming, quantization, input size, output size, batch size (concurrency), percentile, and network speed. Also whether latency includes a cold start.
    - Three tokens/s definitions:

    | Metric | Start | End | Tokens counted |
    |---|---|---|---|
    | Total tokens/s | Request sent | Last token | Input + output |
    | Output tokens/s | Request sent | Last token | Output only |
    | **Perceived** tokens/s | First token | Last token | Output only |

    - Rules of thumb: TTFT under ≈200 ms feels snappy, and ≈30 output tok/s is a good minimum for chat.
- **Speed records:**
    - The Kimi K2 Thinking post opens: *"best-performing provider on Artificial Analysis with a 300 millisecond time to first token and 140+ tokens per second."*
    - The Kimi K2.5 post is titled "How we built the fastest Kimi K2.5 on Artificial Analysis."
    - Both explain the optimizations; the measurement is AA's.
- **GitHub** (`basetenlabs`):
    - [`genai-bench`](https://github.com/basetenlabs/genai-bench), a fork of `sgl-project/genai-bench`: 95 commits ahead and 59 behind upstream. Recent commits add GPU types (L4, A10G, RTX PRO 6000), a "TokenSpeed" engine option, VLM scenarios, and tokenizer fallbacks. This is likely the in-house load tester, though not documented as such.
    - [`sa_spec`](https://github.com/basetenlabs/sa_spec): a suffix-automaton speculator from their speculative-decoding work.
    - A fork of [MoonshotAI's `K2-Vendor-Verifier`](https://github.com/MoonshotAI/K2-Vendor-Verifier), which checks Kimi K2 tool-calling accuracy across vendors to catch silent quality loss.
- **Traces:** NVIDIA AIPerf ships a `baseten_trace` loader (Parquet) for session-preserving replay. I didn't find an official public Baseten trace dataset.

### Others worth reading

- **Modal, LLM Engineer's Almanac.** Uses guidellm in three steps:
    1. `throughput` mode, all requests at once → maximum requests/s.
    2. `synchronous` mode, one at a time → minimum latency.
    3. A sweep of constant rates in between.

  It reports **requests/s rather than tokens/s**, calling tokens/s a *"false fungibility of tokens."* Its code is `modal-labs/stopwatch`, which pins guidellm, benchmarks vLLM, SGLang, and TensorRT-LLM, and can capture PyTorch-profiler traces. Last pushed Aug 2025.
- **Databricks, "LLM Inference Performance Engineering: Best Practices".** The classic definitions: TTFT, TPOT, latency = TTFT + TPOT × output tokens, and throughput. It also defines **MBU (model bandwidth utilization)** = achieved memory bandwidth ÷ peak bandwidth, where achieved = (weight bytes + KV-cache bytes) ÷ TPOT. That's the "% of roof" metric in the main report's Phase 1.
- **Anyscale, LLMPerf (2023).**
    - Workload: 550 input and 150 output tokens on average.
    - Every provider's tokens were counted with one Llama tokenizer.
    - ITL deliberately includes TTFT.
    - The tool and its provider leaderboard are both archived on GitHub; the leaderboard's last update was Jan 2024. Treat it as history.
- **OpenRouter.** Latency = TTFT; throughput = output tok/s over the whole request. Both are tracked per model and provider *"using percentile statistics over a rolling 5-minute window"* (p50, p75, p90, p99), and you can route on them (`preferred_max_latency`, `preferred_min_throughput`).
- **Groq.** Its latency docs give the formulas and note that console metrics are server-side only: user-experienced latency = network + server-side latency. Then: *"We recommend visiting Artificial Analysis for third-party performance benchmarks."*
- **Together AI.** Its coding-agent benchmark post documents the workload: prompts from ≈45k tokens up, modeled on production coding-agent traffic. Headline: 31 % more tokens/s than TensorRT-LLM and 2× better TTFT at saturation, from its own runs.
- **NVIDIA.**
    - "LLM Benchmarking: Fundamental Concepts" separates load testing from performance benchmarking and warns that client tools define and compute metrics differently.
    - The AIPerf metrics reference gives exact formulas: TTFT skips empty chunks, ITL excludes TTFT, and TTFO covers reasoning models.
- **Artificial Analysis.** Covered in the main report §2. Two pieces relevant here:
    - **Integrity Terms:** providers may not fingerprint AA traffic, route it to special capacity, or serve it at unrepresentative batch sizes.
    - **Endpoint Accuracy Index:** re-runs evals against each provider's endpoint to catch quality lost to quantization or bugs.

---

## 4. One-line tools (verified 18 Nov 2026)

All of these speak the OpenAI API, so they work against vLLM, SGLang, or a hosted provider.

| Tool | Install | Best for | Load pattern | Multi-turn | First-answer-token metric | Status |
|---|---|---|---|---|---|---|
| **AIPerf** (NVIDIA) | `pip install aiperf` | All-rounder: interactive chat, sweeps, trace replay | Closed (`--concurrency`) or open (`--request-rate`) | Yes: synthetic turns and trace replay (Mooncake, Bailian, Baseten formats) | TTFO | Active, v0.13.0 |
| **guidellm** (vLLM project) | `pip install guidellm` | One-command sweep, HTML/CSV/JSON reports | sweep, synchronous, throughput, concurrent, constant, poisson | Not checked | Time to first output token | Active, v0.8.1 |
| **`vllm bench serve`** | Ships with vLLM (Linux), or use the `vllm/vllm-openai` image | Engine comparisons; Pareto plots via `vllm bench sweep` | Closed (`--max-concurrency`) or Poisson (`--request-rate`) | Separate script (§5) | No; reasoning counts toward TTFT | Active, v0.34.0 |
| **`sglang.benchmark.serving`** | Ships with SGLang | Engine comparisons, shared-prefix datasets | Closed or rate | Trace datasets (`agentic-trace`, `mooncake`) | No; reasoning counted as content | Active, v0.5.23 |
| **genai-bench** (SGLang project; Baseten maintains a fork) | `pip install genai-bench` | Scenario × concurrency grids, Excel and plots, live dashboard | Closed loop only (no request-rate control) | Not checked | Not checked | Active |
| **Fireworks `load_test.py`** | `git clone`, then `bash scripts/setup.sh` | Production-shaped load, bursts, trace replay | Locust users or open-loop `--qps` | Yes, `--replay-schedule` sessions | `--first-visible-token` | Active (Nov 2026 commits) |
| **HF `inference-benchmarker`** | `cargo install --git …` (Rust) | Quick sweep with realistic profiles; used in Microsoft's Azure A100/H100/H200 blog | sweep, rate, throughput | `--profile chat` resends the full history each turn | Not checked | Active |
| **inference-perf** (Kubernetes SIG) | `pip install inference-perf` | Kubernetes-native load stages, llm-d ecosystem | constant, poisson, concurrent stages | Conversation replay | Not checked | Active |
| **stopwatch** (Modal) | `pip install -e .` from the repo | Provision and benchmark on Modal | synchronous, constant, throughput | — | Inherits guidellm | Last push Aug 2025 |
| ~~LLMPerf~~ (Anyscale) | — | History only | Fixed concurrency | — | — | Archived |

**Copy-paste commands.** Set the variables first; SGLang's default port is 30000.

```bash
EP="http://<endpoint>:8000"           # your endpoint; SGLang default port is 30000
NAME="qwen3-coder"                    # the server's --served-model-name
HF="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# 1) Interactive back-and-forth: per-turn TTFT, tok/s, token count, E2E (history resent each turn)
aiperf chat --model "$NAME" --url "$EP"
aiperf chat --model "$NAME" --url "$EP" --quick "Explain the KV cache in three sentences."

# 2) AIPerf concurrency sweep, fixed 1k-in/1k-out shape, server-side token counts
aiperf profile --model "$NAME" --tokenizer "$HF" --url "$EP" --endpoint-type chat --streaming \
  --isl 1024 --osl 1024 --extra-inputs ignore_eos:true --use-server-token-count \
  --concurrency 1,8,32,64 --request-count 256 --warmup-request-count 8

# 3) guidellm automatic sweep: synchronous → max throughput → constant rates in between
guidellm run --backend kind=openai_http,target="$EP" --profile kind=sweep \
  --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=1024,output_tokens=1024

# 4) vLLM's client (also works against SGLang)
vllm bench serve --backend openai-chat --base-url "$EP" --endpoint /v1/chat/completions \
  --model "$HF" --served-model-name "$NAME" --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --max-concurrency 32 --num-prompts 320 --ignore-eos

# 5) SGLang's client
python3 -m sglang.benchmark.serving --backend sglang-oai-chat --base-url "$EP" \
  --model "$HF" --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1.0 --num-prompts 320 --max-concurrency 32

# 6) genai-bench grid: 2 scenarios × 3 concurrencies, up to 2 minutes per run
genai-bench benchmark --api-backend openai --api-base "$EP" --api-key EMPTY \
  --api-model-name "$NAME" --model-tokenizer "$HF" --task text-to-text \
  --traffic-scenario "D(1024,1024)" --traffic-scenario "N(480,240)/(300,150)" \
  --num-concurrency 1 --num-concurrency 8 --num-concurrency 32 \
  --max-time-per-run 2 --max-requests-per-run 200

# 7) Fireworks harness: open-loop Poisson at 2 req/s, chat + streaming
#    (git clone https://github.com/fw-ai-external/benchmark && cd benchmark && bash scripts/setup.sh && source .venv/bin/activate)
cd llm_bench && locust -t 3min -u 100 -r 100 -H "$EP" --provider vllm -m "$NAME" \
  --chat --stream -p 1024 -o 512 --tokenizer "$HF" \
  --qps 2 --qps-distribution exponential --summary-file results.csv
#    add --first-visible-token for reasoning models

# 8) Hugging Face inference-benchmarker: default sweep with the multi-turn chat profile
inference-benchmarker --tokenizer-name "$HF" --url "$EP" --profile chat

# 9) Kubernetes SIG inference-perf: constant 10 req/s for 60 s
inference-perf --server.type vllm --server.base_url "$EP" --data.type random \
  --load.type constant --load.stages '[{"rate": 10, "duration": 60}]' --api.streaming true
```

Notes:

- `genai-bench`'s `--max-time-per-run` is in **minutes**, and `--api-key` is required for the `openai` backend (any string works for a local server).
- Fireworks' `--chat` / `--no-chat` is required, and its default "limericks" prompt generator needs `--tokenizer`. `--provider` accepts `fireworks`, `vllm`, `sglang`, `openai`, `anyscale`, `together`, and `tgi`.
- `vllm bench serve` needs a vLLM install, and vLLM wheels are Linux-only. Run it inside the `vllm/vllm-openai` image, as the main report's AKS Job does.
- None of these commands were run against a live server today; they were checked against current docs and source.

---

## 5. Back-and-forth: multi-turn and replayed traffic

| What you want | How |
|---|---|
| **Chat with the endpoint yourself** and see stats per turn | `aiperf chat --model "$NAME" --url "$EP"`. History is kept and resent each turn; use `--no-history` for stateless turns. |
| **Synthetic multi-turn load** | AIPerf: `aiperf profile … --conversation-num 50 --conversation-turn-mean 4` (turns are normally distributed around the mean; comma-separated lists sweep) |
| | HF `inference-benchmarker --profile chat`: resends the whole conversation each turn, so prefix caching matters a lot |
| | vLLM: from `benchmarks/multi_turn/` in the vLLM repo, run `python benchmark_serving_multi_turn.py --url "$EP" --model "$HF" --served-model-name "$NAME" --input-file generate_multi_turn.json --num-clients 2 --max-active-conversations 6` |
| | `inference-perf`: conversation-replay data type, for concurrent multi-turn agentic sessions |
| **Replay real production traffic** | AIPerf: `--custom-dataset-type mooncake_trace` (or `bailian_trace`, `baseten_trace`) with `--input-file <trace>`. Replay is open-loop by default; `--no-open-loop-replay` waits a think-time after each reply. |
| | Fireworks: `--replay-schedule FILE.jsonl` keeps each session's turn order and gaps; `--replay-retime poisson\|gamma:CV2` redraws only the session start times |

Why it matters: multi-turn traffic resends ever-longer prefixes, so the prefix cache (on by default in vLLM and SGLang) dominates TTFT. Single-turn random-prompt benchmarks can't show that. AA added multi-turn agentic workloads to its speed benchmark in Nov 2026 (v3.0.0) for this reason.

---

## 6. What to use for the A100 project

| Need | Use |
|---|---|
| "Does it work, and how fast does it feel?" | `aiperf chat` |
| A number comparable to AA's | `aa_probe.py` (main report §12) |
| Capacity curve / Pareto plot | `aiperf profile --concurrency 1,8,32,64,128`, `guidellm run --profile kind=sweep`, or `vllm bench sweep` |
| A reasoning model (e.g. gpt-oss-20b) | A tool with a first-answer-token metric: AIPerf (TTFO), guidellm, or Fireworks `--first-visible-token` |
| Realistic traffic (bursts, shared prefixes, multi-turn) | Fireworks `--shape-mix` with `--qps-distribution gamma`, or AIPerf trace replay |
| vLLM vs SGLang | The same client and settings against both |
| Hardware efficiency | Compute MBU from measured TPOT (see below) |

**MBU example (illustrative TPOT).** Qwen3-Coder-30B-A3B in BF16 reads ≈3.3B active params × 2 bytes ≈ 6.6 GB per token at batch 1. If you measure TPOT = 5 ms, achieved bandwidth = 6.6 GB / 0.005 s = 1.32 TB/s. That's **≈68 %** of the A100 PCIe's 1.935 TB/s, ignoring KV-cache reads.

---

## 7. How to read a provider's speed claim

1. **Which tokens/s?**
    - Output speed *including* TTFT: AA, OpenRouter, Baseten's "output tokens/s".
    - Decode speed *excluding* TTFT: Baseten's "perceived tokens/s", Fireworks' `generation_tokens_per_second`, AIPerf's per-user throughput.

   The second is always higher, and much higher for long prompts.
2. **Reasoning models.** "TTFT" may be the first *thinking* token. Ask for time to first answer token: AA's TTFAT, AIPerf's TTFO, or Fireworks' first visible token.
3. **Workload and load.** Check input/output lengths, concurrency or request rate, and the percentile (p50 or p90/p99). Baseten's own guide lists all of these as must-disclose.
4. **Who ran it.** "As benchmarked by Artificial Analysis" means AA's setup: a client in `us-central1`, AA's workloads, 5-day averages. It doesn't describe the provider's own configuration.
5. **Precision.** FP8, FP4, and INT4 change speed *and* quality. AA's Endpoint Accuracy Index and Moonshot's K2-Vendor-Verifier exist because quality varies by provider.
6. **Speculative decoding.** Speed depends on content; code edits speculate better than creative text. AA varies prompts for this reason.
7. **Network and cold starts.** Where's the client? Is a cold start included? Groq's console numbers are server-side only.
8. **Tokenizers.** "Tokens" differ between models. LLMPerf normalized with a Llama tokenizer; Modal reports requests/s instead of tokens/s.

---

## Sources

**Provider and benchmarker methodology**
- Baseten, Understanding performance benchmarks: <https://www.baseten.co/blog/understanding-performance-benchmarks-for-llm-inference/>
- Baseten, Kimi K2 Thinking at 140 TPS: <https://www.baseten.co/blog/kimi-k2-thinking-at-140-tps-on-nvidia-blackwell/>
- Baseten, How we built the fastest Kimi K2.5: <https://www.baseten.co/blog/how-we-built-the-fastest-kimi-k2-5-on-artificial-analysis/>
- Fireworks, open-model speed benchmarks: <https://fireworks.ai/blog/open-model-inference-speed-benchmarks>
- Modal, LLM Engineer's Almanac, How to benchmark: <https://modal.com/llm-almanac/how-to-benchmark>
- Databricks, LLM inference performance engineering: <https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices>
- Anyscale, Reproducible performance metrics for LLM inference: <https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference>
- OpenRouter, Evaluating LLM provider performance: <https://openrouter.ai/blog/insights/evaluate-llm-provider-performance/>
- OpenRouter, Provider selection: <https://openrouter.ai/docs/guides/routing/provider-selection>
- Groq, Optimizing latency: <https://console.groq.com/docs/production-readiness/optimizing-latency>
- Together AI, Coding-agent benchmarks: <https://www.together.ai/blog/coding-agent-benchmarks>
- NVIDIA, LLM Benchmarking: Fundamental Concepts: <https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/>
- NVIDIA NIM benchmarking metrics: <https://docs.nvidia.com/nim/benchmarking/llm/metrics>
- AIPerf metrics reference: <https://github.com/ai-dynamo/aiperf/blob/main/docs/metrics-reference.md>
- AIPerf CLI options: <https://github.com/ai-dynamo/aiperf/blob/main/docs/cli-options.md>
- Artificial Analysis performance methodology and Integrity Terms: <https://artificialanalysis.ai/methodology/performance-benchmarking>
- Artificial Analysis Endpoint Accuracy Index: <https://artificialanalysis.ai/methodology/endpoint-accuracy-index>

**Code**
- Fireworks harness: <https://github.com/fw-ai-external/benchmark> (`llm_bench/load_test.py`, `llm_bench/README.md`)
- Baseten: <https://github.com/basetenlabs/genai-bench>, <https://github.com/basetenlabs/sa_spec>
- Moonshot K2-Vendor-Verifier: <https://github.com/MoonshotAI/K2-Vendor-Verifier>
- Modal stopwatch: <https://github.com/modal-labs/stopwatch>
- AIPerf: <https://github.com/ai-dynamo/aiperf>
- guidellm: <https://github.com/vllm-project/guidellm> (metric formulas in `src/guidellm/schemas/base/request_stats.py`)
- genai-bench: <https://github.com/sgl-project/genai-bench> (scenarios in `docs/user-guide/scenario-definition.md`)
- Hugging Face inference-benchmarker: <https://github.com/huggingface/inference-benchmarker>
- inference-perf: <https://github.com/kubernetes-sigs/inference-perf>
- vLLM benchmarks: <https://docs.vllm.ai/en/latest/benchmarking/cli/>
- vLLM multi-turn benchmark script: <https://github.com/vllm-project/vllm/tree/main/benchmarks/multi_turn>
- SGLang benchmark client: <https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/serving.py>
- LLMPerf (archived): <https://github.com/ray-project/llmperf>
- LLMPerf leaderboard (archived): <https://github.com/ray-project/llmperf-leaderboard>
- Microsoft Azure HPC blog using inference-benchmarker: <https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/inference-performance-of-llama-3-1-8b-using-vllm-across-various-gpus-and-cpus/4448420>
