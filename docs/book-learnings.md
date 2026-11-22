# What two books (and Inferact) say about the A100 hill climber

> **Status: research only.** No changes were made to `hill-climb-plan.md`, the `a100-hillclimb` repo, the running campaign, or the cluster. Everything below is a backlog for the *next* iteration. 18 Nov 2026.
>
> **Update, 19 Nov 2026 09:51 UTC:** a subset is now applied as climber v2, `a100-hillclimb@16ed3d8`, deployed at the T2-climb boundary:
> - P0-1: drain-free T2 score
> - P0-2: fresh-run acceptance (noise calibration is not yet done)
> - P0-3: power and clock telemetry per step
> - P0-4: resolved-config logging (deduplicating trials is not yet done)
> - P0-5: 8k-TTFT guard
> - P1-8: T2 probe screen at the incumbent's capacity × 1.05, paired confirmation, reuse of results under 12 h old
>
> Also: `spec` is out of the T2 order, and the chunk budget gains 512. See the repo's `docs/decisions.md`. Still open: P0-6, P0-7, P0-8, the rest of P1, and all of P2.
>
> **Update, 19 Nov 2026 (harness v3, campaign `main-v3`, redoes every climb on both engines):**
> - **Applied:** P0-1 (plus interpolated capacity), P0-2 (calibrated thresholds = max(floor, 3σ); paired C,I confirm), P0-3 (plus clock-mismatch events), P0-4 (plus no-op skipping), P0-5, P0-6 (TTFT-only / TPOT-only pass fractions, p99 max rate reported, 3 × 300 s verification seeds), P0-7 a/b/c/e (teacher-forced NLL + top-1 vs BF16, GSM8K with thinking, passkey at 8k/16k/28k, a gate class per speculation method), P0-8 (ShareGPT Poisson track `t2c`, 85/15 mixed-length verification, 32 SPEC-Bench prompts with natural stopping, natural vs forced-512 in `t1v`), P1-2, P1-3 (method × draft length coupled, per-position acceptance, draft length by batch size), P1-4, P1-5, P1-6, P1-8 (all seven steps except roofline pruning), P1-9, P1-11, P1-12 (shared prefix, growing context, think time, working set above the GPU KV pool).
> - **Partial:** P1-1 (torch-profiler trace per winner; no nsys or DCGM, which need a privileged pod), P1-10 (prefetch load only), P0-7d (28k, the 32k context limit), P0-5 (hard 15% guard, not a soft penalty).
> - **Infeasible here:** P1-7 (vLLM quantizes the LM head only for GPTQ/AWQ, not compressed-tensors), P2-2 (no W4A8 kernel for SM80 in either engine).
> - **Deferred:** P0-7f (quality under load), P2-1 (two-node bake-off, a later campaign), P2-3, P2-4, P2-5 (partly covered by verify and ablation stages).
> - **Early signal while validating:** with natural stopping (32 prompts, thinking on), INT4 + suffix decoding ran at 93 tok/s with 0.53 accepted tokens per draft, against 129 tok/s in v2's forced-512-token T1 (10 prompts). The runs differ in prompts and knobs, so this only *suggests* that text past the natural end (`ignore_eos`) is repetitive and inflates acceptance (P0-8). `t1v` measures both on one server.
>
> **Sources**
> - Philip Kiely, *Inference Engineering* (Baseten, Jan 2026, 259 pp.)
> - Chris Fregly, *AI Systems Performance Engineering* (O'Reilly, 2026, 1,063 pp.)
> - Simon Mo, ["Building an Autonomous Inference System"](https://x.com/simon_mo_/status/2123133742063706343) (X, 18 Nov 2026), plus Inferact's linked [pipeline-parallelism post](https://inferact.ai/news/pipeline-parallelism-in-agentic-era)
>
> **Method**
> - I split the books into 12 slices. Parallel agents read 11 of them and I read the twelfth.
> - Each slice was checked against the harness code and the ledger, read-only.
> - Numbers marked *(ledger)* I recomputed from `results/ledger.sqlite` and the trial artifacts.
> - Citations use printed page numbers: [IE p.N] and [AISPE p.N].
> - The 214 per-slice recommendations are in [`book-review/`](book-review/).

---

## TL;DR

1. **Verdict: the search space is right; scoring and acceptance need a v2.**
   - Layers L0–L7 cover every technique family in both books: quantization, KV cache, speculation, batching and scheduling, kernels and compilation, parallelism and disaggregation. Nothing big is missing from *what* the plan tunes.
   - The weak spots are *how* it measures and accepts. Four of them can change which config wins ([§3](#3-is-the-plan-good-enough-scorecard)):
     - The T2 score is skewed by drain time.
     - Challengers are accepted partly on their own best-of-N screening run.
     - T2 uses random-token, fixed-shape prompts.
     - The quality gate can't see long-context, KV-cache, or speculative-decoding errors.
   - The running campaign is still a useful coarse screen. Don't publish its T2 numbers or any accepted gain under 5% until the P0 fixes land.
2. **Biggest new insight (ledger): single-user decode time ≈ 4.6 ms fixed + bytes ÷ 1.69 TB/s.**
   - This fits BF16, FP8, INT8, and INT4 to within 0.6 ms.
   - The fixed part is **28% of an INT4 token** but only 11% of a BF16 token. That alone explains why INT4 reaches only 62% of the bandwidth roof.
   - The next T1 gains come from removing overhead (CUDA graphs, scheduling, sampler, speculative decoding), not from smaller weights. With zero overhead, INT4 would reach about 85 tok/s.
3. **T2 is limited by GPU time, not KV memory.**
   - At INT4's limit (≈1.2 req/s), prefill takes about half the GPU.
   - TPOT is what collapses SLO attainment: the median reaches 103 ms at 1.245 req/s.
   - KV usage stays at only ≈15–25% of capacity.
   - So W8A8 (plus FP8 KV) should beat INT4 on T2, and KV-capacity knobs can't help INT4 there.
4. **All three sources endorse the method.**
   - Kiely: a Baseten engineer scripted 77 configurations and found a non-obvious one that doubled TPS [IE p.119]. Test each optimization "individually as well as collectively" [IE p.114].
   - Fregly: hypothesis → reproducible benchmark → adjust → rerun [AISPE p.8].
   - Simon Mo (today): Inferact's product is autonomous "Hillclimb" rounds around vLLM.
5. **Two nodes, revisited.**
   - Ethernet bandwidth isn't what makes prefill/decode disaggregation lose for this dense 33B model: a 1k-token prompt's KV is 0.22 GB, about 87 ms over 20 Gb/s. Splitting only two GPUs 1:1 is the problem.
   - Worth one measured run of W8A8 prefill → INT4 decode, plus a lossless BF16 pipeline-parallel run.
   - Two independent replicas remain the default ([§7](#7-two-nodes-over-ethernet-revisited)).

---

## Contents

1. [The sources in one paragraph each](#1-the-sources-in-one-paragraph-each)
2. [What I learned](#2-what-i-learned)
3. [Is the plan good enough? Scorecard](#3-is-the-plan-good-enough-scorecard)
4. [What the plan already gets right](#4-what-the-plan-already-gets-right)
5. [Backlog for the next iteration](#5-backlog-for-the-next-iteration-nothing-applied)
6. [The numbers behind the headline findings](#6-the-numbers-behind-the-headline-findings)
7. [Two nodes over Ethernet, revisited](#7-two-nodes-over-ethernet-revisited)
8. [Metric definitions for Part 0](#8-metric-definitions-for-part-0-book-vs-harness)
9. [Blog-worthy facts](#9-blog-worthy-facts)
10. [Simon Mo's article and this project](#10-simon-mos-article-and-this-project)
- [Appendix A: Per-slice findings](#appendix-a-per-slice-findings)
- [Appendix B: The 175+ item checklist, audited](#appendix-b-the-175-item-checklist-audited)

---

## 1. The sources in one paragraph each

**_Inference Engineering_ (Kiely, Baseten).** A practitioner's map of three layers: the runtime (batching, caching, quantization, speculation, parallelism, disaggregation), infrastructure, and tooling. It is best on vocabulary and trade-offs: TTFT, perceived vs total TPS, percentiles, inference-only vs end-to-end timing, when each technique helps, and how to benchmark. It is Hopper/Blackwell-centric and lists the A100 among GPUs still seen "in low-traffic or legacy systems" [IE p.79]. That is exactly the perception this series tests.

**_AI Systems Performance Engineering_ (Fregly, O'Reilly).** A bottom-up tour in 20 chapters: hardware, OS/Kubernetes, networking, storage, CUDA, PyTorch and compilers, inference at scale, prefill/decode disaggregation and KV tuning, adaptive engines, AI-assisted optimization, and a 175+ item checklist. Mechanisms transfer to the A100; most of its numbers don't, since they come from Blackwell or Grace examples the book often labels illustrative. In the preface, Fregly compares his old car to the A100, "a timeless classic" [AISPE p.xxvii], which is a nice hook for this series.

**Simon Mo, "Building an Autonomous Inference System" (Inferact/vLLM, today).** Inferact is building a system around vLLM that profiles new models, has agents write kernels or tune configs in "Hillclimb" rounds, and promotes only what survives end-to-end serving benchmarks and shadowed production traffic. Winning changes go upstream to vLLM. It is the same loop as this harness, at GB300 scale and with kernel-writing agents ([§10](#10-simon-mos-article-and-this-project)).

---

## 2. What I learned

### From *Inference Engineering*

1. **Say which TPS you mean.** *Perceived* TPS is per-user speed after the first token, which is what T1 measures. *Total* TPS is service throughput. An ITL of 10 ms equals 100 TPS per user [IE p.35]. The book never uses "TPOT" or "goodput", so Part 0 must define both.
2. **Use percentiles, and separate inference-only from end-to-end time.** Latency is right-skewed, so the mean sits above P50; manage P90 and P99 [IE p.36–37]. If inference is fast but end-to-end is slow, fix the infrastructure, not the model [IE p.37]. For a tool call, measure total response time, not TTFT or TPS [IE p.36].
3. **VRAM rule: weights plus at least 50% headroom for KV cache** [IE p.77]. BF16 Nanbeige on one 80 GB A100 breaks the rule, which is why it gets only ≈59k KV tokens.
4. **One precision step usually buys 30–50%, not 2×.** Halving precision speeds prefill *only* if the math runs on lower-precision tensor cores [IE p.120]. On the A100 that means only INT8 W8A8 helps TTFT, which the ledger confirms: weight-only formats made 8k TTFT *worse*, while W8A8 cut it from 2.7 s to 1.7 s.
5. **Quantization sensitivity rises from weights to activations to KV cache to attention.** Output layers are often left at high precision [IE p.126–127]. The quality bar is "zero perceptible quality loss": differences indistinguishable from noise across perplexity, benchmarks, and custom evals [IE p.128]. KV and attention errors compound over long outputs.
6. **Speculative decoding** depends on draft cost, draft length, and acceptance rate:
   - Acceptance falls with draft depth, and temperature is the biggest factor [IE p.130–131].
   - It must be switched off at higher batch sizes [IE p.131].
   - It "only improves TPS/ITL, not TTFT" [IE p.129].
7. **Caching:**
   - Prefix caching is strictly positional, so put novel tokens last.
   - KV is a tiered resource: VRAM, then host RAM, SSD, and network [IE p.137–140].
   - Long context changes which knobs matter [IE p.141].
8. **Parallelism and disaggregation are tools for scale.** Disaggregate at ≥100M–1B tokens/day, with models ≥100B parameters, on prefill-heavy traffic [IE p.149–150]. Otherwise use extra GPUs as replicas.
9. **Benchmark like an experiment** [IE p.112–114]:
   - Simulate production on four axes: sequence lengths, traffic pattern, request content, and request parameters.
   - Take a baseline first.
   - Test optimizations "individually as well as collectively", since they can work against each other (e.g., speculation at large batch).
   - Change one variable at a time.
   - The 77-configuration script that doubled TPS is the book's case for automated search [IE p.119].
10. **torch.compile can't fuse plugin kernels** such as FlashAttention or custom GEMMs, which limits it for LLM serving [IE p.103].
11. **Production:**
    - Utilization is a lagging, misleading signal [IE p.185–186].
    - Don't reverse-engineer a $/token figure from GPU rent; compare total cost over a week or more [IE p.202–203].
    - A cold start = provisioning + image + weights + compilation [IE p.189–190].

### From *AI Systems Performance Engineering*

1. **Method:** hypothesis → reproducible benchmark → adjust → rerun, and publish every step [AISPE p.8].
   - Its CI rule fails a build on a ≈5% regression that holds over 3 runs [AISPE p.594].
   - Lock clocks when benchmarking, or heat will masquerade as a regression [AISPE p.84].
2. **Goodput:**
   - It is the minimum over phases, divided by GPUs, as in the DistServe example [AISPE p.763–764].
   - MLPerf uses p99 SLOs, e.g., ≈450 ms TTFT and 40 ms TPOT for Llama 2 70B [AISPE p.761].
   - Plan capacity at ≈90% of peak [AISPE p.730].
3. **Roofline first.** Classify the bottleneck (memory, compute, or launch/idle) before picking a technique [AISPE p.521–523; ch. 6–8]. nvidia-smi "utilization" is a duty cycle, not efficiency; use SM-active and DRAM-active counters instead [AISPE p.227].
4. **Launch overhead is structural:**
   - ≈1,000 small kernels at ≈20 µs each can leave a GPU only ≈35% busy [AISPE p.392–393].
   - CUDA graphs remove the launch cost (rule of thumb: 20–30% latency) but make no kernel faster [AISPE p.491, p.523].
   - Megakernels report 1.2–6.7× lower latency but remain research code [AISPE p.394–395].
5. **Compilation is low-yield for dense models.** Compile gains are under 10% in total, and single fusions give "a few percent" [AISPE p.547, p.600]. Most single L5 knobs therefore fall below a 3–5% acceptance bar.
6. **Speculative decoding** gives about 2× in practice, with 1.5–2.5× common [AISPE p.682–684]. Structured output costs double digits at small batch [AISPE p.690].
7. **Chunked prefill:** size each chunk to ≈50–100 ms of GPU time [AISPE p.895]. On this A100 that is a few hundred tokens, vs today's 2,048 default.
8. **Prefill/decode disaggregation** pays only if the KV handoff is tiny next to prefill compute: milliseconds against hundreds of milliseconds [AISPE p.812–817].
   - Using a different precision per phase is legitimate [AISPE p.826].
   - Splitwise paired H100 prefill with A100 decode [AISPE p.820–821].
9. **Routing is a cost function** that weighs prefix-cache matches against load, and it barely matters at low load [AISPE p.789–790].
10. **Auto-tuning:**
    - Soft penalties beat hard thresholds [AISPE p.879].
    - Verify everything AI generates (FP32 reference, shape sweep, anti-cheat checks) before trusting it [AISPE p.887–889, p.926–929].
11. **OS tuning matters less here.** The book's 5–20% OS/container gains come from multi-socket, bare-metal training. For single-GPU serving on a VM, expect at most 1–3%; the value is mostly lower variance (AI01).

### From Simon Mo and Inferact

1. "Several candidates win in isolation and lose end to end." Only full serving benchmarks count, which the harness already does.
2. **Fail fast.** Their system noticed that most failures died at the same gate, so it moved that check earlier.
3. **Silent fallbacks** are a first-class failure mode: a new attention variant "falls back to a generic path and misses the latency target."
4. **Quality bugs can depend on batch composition.** A Kimi K3.1 accuracy loss came from an attention-routing switch that depended on batch composition. Test quality under load, not only at idle.
5. **Their rate protocol:** 3 seeds per rate, each on a freshly started server; rate steps of ≤4%; and ≥95% of requests completed. Before any load, every config passes GSM8K plus a long-prompt first-token check.
6. **Results they claim:**
   - A 24 h hillclimb made TTFT 2.78× faster at 800k context.
   - A 30 h hillclimb cut p99 TPOT by 35%.
   - Production hillclimbing raised throughput per GPU ≈1.8× over six weeks.

---

## 3. Is the plan good enough? Scorecard

✅ = right as is · ⚠️ = fix before trusting the numbers · ➕ = worth adding

| Area | Harness today | What the books (and the ledger) say | Verdict |
|---|---|---|---|
| Tracks | T1 interactivity, T2 goodput; T3 agentic designed but not built | Separate latency-optimized and throughput-optimized deployments [IE p.29–30]. Goodput is SLO-gated and per GPU [AISPE p.763]. | ✅ Right framing. ➕ T3 still to build. |
| Search space (L0–L7) | 5 weight formats, KV, speculation, scheduler, kernels and compile, topology | Same technique families in both books (IE ch. 5; AISPE ch. 15–19) | ✅ Complete at the family level. ➕ A few missing knobs (P1). |
| T2 score | `request_goodput × 512` at the best passing rate | *(ledger)* vLLM divides by the arrival window *plus drain* (134.6 s vs ≈120 s), so the score reads 8–12% below the offered SLO-meeting rate, and the gap grows with TPOT ([§6.2](#62-t2-score-vs-offered-rate)). | ⚠️ P0-1 |
| T2 resolution | Rate ladder ×1.5 plus 4 bisections; 120 s windows; pass = ≥90% attainment | Rate steps of ≈2.6% and attainment noise of ±2.5–4 points, vs a 5% threshold. Ramp-up and drain dominate a 120 s window. MLPerf uses p99 and ≥600 s (general knowledge). | ⚠️ P0-1, P0-6 |
| Acceptance rule | Accept if the screening run (best of N) plus one confirm run, averaged, beats the incumbent's mean; fixed 3%/5% thresholds | Winner's curse: the best-of-N screening run is biased upward. The book's CI rule uses 3 runs [AISPE p.594]. One identical T1 pair agrees to ≈0.03%, so 3% may be coarse for T1 (calibrate). | ⚠️ P0-2 |
| Workload content | T1: 10 SPEC-Bench prompts with `ignore_eos`. T2: random tokens, fixed 1k in / 512 out. | Request content drives acceptance and cache hits [IE p.113]. "Jitter" means varying shapes, not just arrival times [IE p.220]. Head-of-line blocking needs mixed lengths [AISPE p.761]. | ⚠️ P0-8 |
| Quality gate | GSM8K-500, thinking off, ≤1,024 tokens, ±2.5 points, per numerics class | "Zero perceptible loss" [IE p.128]. KV and attention errors compound [IE p.126–127]. Swapped kernels must match a reference within tolerance [AISPE p.887]. | ⚠️ P0-7 |
| Noise control | nvidia-smi every 2 s; one throttle flag per trial | *(ledger)* The 300 W SW power cap is active in ≈86–91% of busy samples, and T2 SM clocks run ≈12% lower. The flag always includes the "idle" bit, so it can't flag anything. | ⚠️ P0-3 |
| Attribution and profiling | None (nvidia-smi only) | Profile before optimizing; the roofline classifies the bottleneck (AISPE ch. 6–8, 13, 16) | ➕ P1-1 |
| Climber algorithm | Coordinate ascent; plateau = a full pass with no accepts | Right for 40–50 trials. Add coupled moves and cheap screens. RL and Bayesian optimization would be starved for samples at this budget (AI08). | ✅ Keep. ➕ P1-8 |
| Topology | Replicas only; no TP or P/D over Ethernet | Ethernet bandwidth doesn't block P/D at 1k prompts; the 1:1 split of two GPUs does. | ✅ Default stays. ➕ P2-1 experiments |
| Cost story | $/1M tokens at peak goodput | Price at ≈90% of peak [AISPE p.730]. State the duty cycle and compare total cost [IE p.202–203]. | ⚠️ Refine in the finale |

**Bottom line:**
- The plan is *good enough to keep running* as a screen.
- It is *not yet good enough to publish* T2 results or small-delta accepts.
- Every gap is a measurement or acceptance fix, not a missing dimension.

---

## 4. What the plan already gets right

- **Reproducibility:** pinned image digests, seeded and versioned workloads, a ledger with config hashes and parent links [AISPE p.8; checklist §"Reproducibility"].
- **Open-loop Poisson T2 with SLO attainment** (DistServe-style) rather than closed-loop bursts [AISPE p.763]. Warmup runs use different prompts, so they don't pre-fill the prefix cache.
- **Same-lane paired confirmation, plateau stop, validity rules, and a crash blacklist.**
- **A custom W8A8 checkpoint for compute-bound prefill.** This is the book's lever, "lower-precision tensor cores" [IE p.120], and it's the only A100 format that cut TTFT.
- **Replicas instead of tensor parallelism over Ethernet** [IE p.80, p.143–148].
- **Every trial is a full serving benchmark.** Inferact's lesson is that only end-to-end wins count.
- **Checklist coverage:** of the 89 checklist items that apply here, 40 are covered and 31 partly covered ([Appendix B](#appendix-b-the-175-item-checklist-audited)).

---

## 5. Backlog for the next iteration (nothing applied)

Effort is S, M, or L. **Offline** means it can be computed from the existing ledger without touching the running campaign. IDs in parentheses point to the per-slice files in [`book-review/`](book-review/).

### P0: fix before trusting or publishing numbers

| # | Change | Evidence | Effort |
|---|---|---|---|
| **P0-1** | **Score T2 as sustainable offered rate × attainment × output tokens.** Interpolate the rate at the attainment target instead of taking the last passing rung. Keep vLLM's goodput as a diagnostic, and re-score the existing ledger. | *(ledger)* [§6.2](#62-t2-score-vs-offered-rate). The book prefers soft scores to hard cut-offs [AISPE p.879]. (AI01-R1, AI05-03, AI08-04) | S, offline |
| **P0-2** | **Decide on fresh runs only.** Run challenger and incumbent interleaved on one lane (C, I, C, I), and leave the screening value out of the estimate. **Calibrate thresholds per metric** with plan-v1's 5× incumbent stage: T1 may support ≈1%, while T2 near the knee needs more repeats. | 5% regression sustained over 3 runs [AISPE p.594] (AI05-01/02, AI08-01, AI01-R4/R5, IE2-11) | S |
| **P0-3** | **Check power and clocks per step.** Log SM clock, power, energy per token, and the fraction of time power-capped for each benchmark step. Compare challenger and incumbent clocks, and drop the "idle" bit from the throttle flag. | Lock clocks or log them [AISPE p.84]. *(ledger)* Capped 86–91% of the time. (AI09-01, AI03-9, AI01-R3) | S |
| **P0-4** | **Record what actually ran:** resolved attention backend, CUDA-graph mode and sizes, quantization kernel, compile-cache hit, image digest, and driver. Skip trials that resolve to the same effective config. Purge the contaminated W8A8 "0.89 s" 8k TTFT from the ledger ([§6.4](#64-prefill-efficiency-at-8k-tokens-c1)). | Inferact: "falls back to a generic path". Engines silently downgrade incompatible combinations. (AI04-R3, AI02-15, AI09-07) | S |
| **P0-5** | **Fix the T1 guard.** Guard 8k TTFT with a soft penalty instead of capping short-prompt TTFT at +15% of ≈38 ms; a few-ms speculative draft can trip that cap. | λ-penalty recipe [AISPE p.876] (IE2-02, AI09-08, AI08-06) | S |
| **P0-6** | **Report which SLO binds, at which percentile.** For each rate, log the TTFT-only and TPOT-only pass fractions at p90 and p99. Pick P90 (DistServe) or p99 (MLPerf and plan-v1) and use it everywhere. Verify the final rate with ≥300 s windows minus the ramp-up, 3 seeds each (Inferact protocol). | [AISPE p.761, p.797] (AI07-01, IE3-02, AI06-14) | S |
| **P0-7** | **Quality gate v2:**<br>(a) KL divergence and top-1 agreement vs BF16 on fixed prompts; takes minutes and catches LM-head and kernel problems.<br>(b) A thinking-on, long-output eval.<br>(c) A long-prompt first-token check at 8k and 32k (Inferact).<br>(d) A 32–64k context check for FP8 KV.<br>(e) An equivalence check for each speculative-decoding method.<br>(f) A quality check under load: eval prompts mixed into a Poisson stream. | [IE p.126–128]; [AISPE p.887]; the K3.1 batch-composition bug (IE2-07/08/09, IE3-07, AI07-10, AI06-06, X1) | M |
| **P0-8** | **Use realistic content where content matters:**<br>- Real-text Poisson arrivals before any T2 decision on speculative decoding or prefix caching.<br>- A mixed-length T2 guard, e.g. 85% 1k-token and 15% 8k-token prompts.<br>- At least 30 stratified T1 prompts.<br>- A check on how much `ignore_eos` inflates draft acceptance. | [IE p.113, p.220]; [AISPE p.761] (AI06-02/03, IE3-03, AI07-03, IE2-03) | M |

### P1: highest-value additions

| # | Change | Why, and expected effect | Effort |
|---|---|---|---|
| **P1-1** | **Profile each new incumbent once, unscored.** Use an nsys timeline, the engine's profiler, and DCGM DRAM-active and SM-active counters. Split the 4.6 ms fixed cost into idle gaps, small kernels, sampler, and LM head. Use a separate pod template, since profiling needs elevated permissions. | Explains the 62%-of-roof gap and decides whether L6 kernel work is worth it. nvidia-smi suggests ≈2–2.6 ms of idle per token in vLLM vs ≈1.1 ms in SGLang (AI03-2, AI04-R1, AI05-05, AI06-11, IE2-01). | M |
| **P1-2** | **For INT4 in T1, try overhead knobs first:**<br>- A one-off CUDA-graph mode diagnostic per format: NONE, PIECEWISE, FULL_DECODE_ONLY, and the default.<br>- Async scheduling, the FlashInfer sampler, and SGLang's overlap scheduler.<br>- Then speculative decoding. | Overhead is 28% of an INT4 token, so halving it ≈ +16% on T1 (AI04-R2, AI02-10). | S |
| **P1-3** | **Run speculative decoding properly:**<br>- Move the method and the draft length together.<br>- Log acceptance per draft position and choose the length analytically.<br>- Add a "disable above batch size" knob for T2.<br>- Re-test at long context in T3. | [IE p.130–131]; [AISPE p.682–684] (AI06-01/04/05, AI08-03) | M |
| **P1-4** | **Change T2's starting points.**<br>- Screen W8A8 + FP8 KV first.<br>- Treat FP8 KV as a *bandwidth* knob: KV is ≈35–45% of the bytes per step at T2 load.<br>- Skip KV-capacity knobs for INT4 in T2. | INT4 becomes compute-bound past a batch of ≈45; W8A8 stays memory-bound up to ≈160 ([§6.3](#63-where-decode-turns-compute-bound-on-a100-pcie)) (AI03-4/5, IE2-11). | S |
| **P1-5** | **Pin the KV pool size.** Add vLLM `--gpu-memory-utilization`, or `--kv-cache-memory-bytes` (verify the flag on v0.34). This removes the 203k vs 213k KV-token jitter between restarts. Each +0.01 adds ≈4k BF16 KV tokens. | (AI07-06) | S |
| **P1-6** | **Add scheduler knobs that need mixed lengths:** chunk budgets of 256 and 512 tokens (the book says chunks of ≈50–100 ms) and `--long-prefill-token-threshold`. Score them on ITL p99 and the mixed-length guard. | [AISPE p.895] (AI08-10/11, AI07-04, AI03-8) | S |
| **P1-7** | **Try an INT8 LM head** using existing kernels, gated by P0-7a. The gain is capped at ~+6–9% on T1. The book says output layers are sensitive, so treat it as high risk. | [IE p.126] (AI08-15, AI03-7, IE2-14) | M |
| **P1-8** | **Climber v2** (from AI08 §8):<br>1. Calibrate noise.<br>2. Drop knobs whose roofline-bounded gain is below the threshold.<br>3. Screen cheaply. For T2, probe once or twice at the incumbent's capacity, so a loser is rejected in \≈3 min instead of a 20–40 min ladder.<br>4. Confirm on fresh runs only.<br>5. Move known interacting knobs together: speculation × draft length, KV dtype × attention backend, max sequences × chunk size.<br>6. Kick from near-misses at a plateau.<br>7. Finish with a leave-one-out ablation.<br>Keep coordinate ascent; RL and Bayesian optimization are starved at 40–50 trials. | [IE p.114]; [AISPE p.877–879]; Inferact's fail-fast gate ordering (AI08, AI05-10, X1) | M |
| **P1-9** | **Give SGLang a fair setup:**<br>- Persist its compile cache (`TORCHINDUCTOR_CACHE_DIR` isn't set).<br>- Make prefill CUDA-graph capture a knob; it took 373 s of a 465 s INT4 startup.<br>- Sweep `mem_fraction_static` together with an allocator setting; BF16 got 16.8k KV tokens vs vLLM's 59.4k.<br>- Add HiCache for T3. | (AI05-07, AI02-01/02, AI05-13, AI07-07) | S–M |
| **P1-10** | **Faster restarts mean more trials per day:**<br>- `--safetensors-load-strategy=prefetch` (vLLM's own log suggests it).<br>- Raise NVMe read-ahead from 128 KB to ≈8 MB.<br>- Sync compile caches across nodes. | (AI02-03/04) | S |
| **P1-11** | **Prune dead knobs and add one:**<br>- Drop `-O3`, which equals `-O2`.<br>- Drop vLLM fusion passes, which target FP8/NVFP4 only and never fire on Ampere.<br>- Try FLEX_ATTENTION last.<br>- Add `compile_sizes: [1]` for T1. | FlexAttention ≈ 85–90% of FlashAttention [AISPE p.619] (AI05-09/10/11) | S |
| **P1-12** | **Build T3 the books' way:**<br>- A system/tool prefix shared across sessions.<br>- Contexts growing from 8k to 64k (`MAX_MODEL_LEN` is 32k today).<br>- Think-time gaps between turns.<br>- A working set sized to overflow GPU KV.<br>- Tune KV dtype and offload size jointly.<br>- Reject any preemption of a long sequence. | [AISPE p.807–809] (AI07-09/15/16, IE2 caution 12) | M |

### P2: stretch goals and blog experiments

| # | Experiment | Why | Effort |
|---|---|---|---|
| **P2-1** | **Two-node bake-off at equal GPU count:**<br>(a) Two replicas with routing arms (round-robin, least-loaded, session affinity, prefix + load), run on T3 at two load levels.<br>(b) 1P1D: W8A8 prefill feeding INT4 decode, with KV over TCP (NIXL/UCX).<br>(c) BF16 with PP=2 across the nodes, as the lossless option. | [§7](#7-two-nodes-over-ethernet-revisited) (AI07-12/14, AI02-08, IE3-08, X1) | L |
| **P2-2** | **W4A8 on SM80** (INT4 weights, INT8 activations), only if the pinned engines ship a kernel. It's the only A100 format that cuts both decode bytes and prefill FLOPs; the model predicts ~+40% on T2. Check for a kernel before building anything. | (AI03-14, IE2 caution 14) | S to check |
| **P2-3** | **Different precision per phase on one GPU:** W8A8 for prefill steps, W4A16 for decode. The model predicts ~+25% on T2, but the KV pool shrinks. This would need a research fork. | [AISPE p.826] (AI07-17) | L |
| **P2-4** | **Agent mode (plan-v1 Mode B):**<br>- Loop: generate → verify → refine, behind the same acceptance rule.<br>- Allow agent-written kernels only after P1-1 shows a hotspot.<br>- Require an FP32 reference, a shape sweep, and anti-cheat checks. | [AISPE p.887–889, p.926–929]; Inferact (AI08-25/26) | L |
| **P2-5** | **Finale hygiene:**<br>- Leave-one-out ablation for the waterfall.<br>- A 60-minute soak at 90% of capacity.<br>- One true cold start.<br>- One end-to-end run through a K8s Service.<br>- $/1M tokens at 90% of peak, with a stated duty cycle. | [AISPE p.730]; [IE p.202] (AI08-13/24, IE3-09/12/14, IE1-3) | M |

### Skip: the books or the A100 say no

- **RL or Bayesian optimization as the main search:** 40–50 samples is too few (AI08).
- **OS tuning below the pod spec** (CPU governor, IRQ affinity, NUMA): not available on Azure VMs, and expect ≤1–3% (AI01).
- **CPU limits on the bench sidecar** without the static CPU Manager: they create fake latency tails through throttling (AI01).
- **vLLM fusion passes and `-O3`:** the fusions target FP8/NVFP4, and `-O3` equals `-O2` (AI05).
- **2:4 sparsity without retraining:** likely fails the quality gate (AI03, AI05).
- **Custom INT4 GEMV kernels or megakernels before profiling:** INT4's residual off the fit is only +0.1 ms, so the cost is fixed overhead, not the GEMV (AI03, AI04).
- **Cross-node tensor parallelism:** it moves ≈1.5 MB per token, which caps prefill near 1.7k tok/s, slower than one GPU (AI02).
- **Hybrid CPU prefill:** 24 vCPUs is ≥200× slower than the GPU (AI07).
- **Length bucketing and batching-delay timers:** these belong to padded batching. vLLM and SGLang batch ragged sequences step by step (AI06, AI08).
- **MIG/MPS split of one GPU for prefill/decode:** two copies of the weights, and MIG statically splits bandwidth (AI07).
- **Hopper/Blackwell features:** FP8/FP4 tensor cores, FlashAttention-3, TMA, thread-block clusters, PDL, wgmma.

---

## 6. The numbers behind the headline findings

### 6.1 Single-user decode = fixed overhead + bytes

Least-squares fit over the four vLLM single-user runs: **time per token ≈ 4.59 ms + bytes ÷ 1.69 TB/s**. That bandwidth is 87% of the 1,935 GB/s spec. Bytes per token come from the plan's roofline ceilings. *(ledger)*

| Variant | Measured ms/token | Fit | Residual | Fixed share of a token | % of spec roof | tok/s with zero overhead |
|---|---|---|---|---|---|---|
| BF16 | 42.80 | 42.76 | +0.04 | 10.7% | 77.9% | 26 |
| FP8 (W8A16) | 23.74 | 24.33 | −0.59 | 19.3% | 72.6% | 52 |
| INT8 (W8A16) | 24.78 | 24.33 | +0.45 | 18.5% | 69.6% | 50 |
| INT4 (W4A16) | 16.37 | 16.27 | +0.10 | **28.0%** | 62.3% | **85** |
| INT8 W8A8 (custom) | 26.32 | 24.33 | **+1.99** | 17.4% | 65.5% | 46 |

**What this tells us:**
- Shrinking the weights made the fixed per-token cost the bottleneck. Its likely sources are launch gaps, small kernels, the sampler over a 166,912-token vocabulary, and scheduler time.
- INT4's GEMV residual is only +0.1 ms, so faster INT4 kernels are not the lever.
- W8A8 sits 2 ms above the line, which is consistent with its extra activation-quantization kernels. That makes it a fusion target, but only on the T2 and TTFT side; it's slower than W8A16 at batch 1.
- Against the *achievable* roof (≈1.69 TB/s rather than the spec), BF16 is at ≈90% and INT4 at ≈72%.
- **Caveat:** the fit is from four points with assumed byte counts. Confirm it with P1-1 profiling before publishing.

### 6.2 T2 score vs offered rate

INT4 rate search (trial `t261119034148-9bcec6-n1`). Each step offers ≈120 s of Poisson arrivals. vLLM's goodput divides by the run's full duration, *including* the drain after the last arrival. *(ledger)*

| Offered req/s | Duration | Attainment | Score today (vLLM goodput × 512) | Offered × attainment × 512 | Gap |
|---|---|---|---|---|---|
| 0.500 | 129.8 s | 100% | 237 | 256 | −7.6% |
| 1.125 | 132.7 s | 100% | 521 | 576 | −9.5% |
| **1.183** | 134.6 s | 100% | **540** (today's T2 score) | **606** | −10.8% |
| 1.214 | 134.3 s | 93.8% | 518 | 583 | −11.1% |
| 1.245 | 135.6 s | 38.3% | 215 | 244 | −11.9% |

**What this tells us:**
- The gap grows with load because slower TPOT lengthens the drain.
- Two configs that sustain the *same* rate can therefore score ≈5% apart on TPOT alone, the size of the T2 acceptance threshold.
- Pass/fail per rate is unaffected; only the score's magnitude is wrong.
- At 1.245 req/s the median TPOT is 103 ms, so TPOT collapses attainment while TTFT tails degrade more gradually.

### 6.3 Where decode turns compute-bound on A100 PCIe

The A100's ridge point is 312 TFLOPS ÷ 1.935 TB/s ≈ **161 FLOP/byte** for BF16. For INT8 tensor cores it is 624 TOPS ÷ 1.935 TB/s ≈ 322 op/byte. A decode GEMM at batch B does about 2·B FLOPs per weight read. The table divides by bytes per parameter.

| Weights | Math | Arithmetic intensity | Turns compute-bound near |
|---|---|---|---|
| BF16 | BF16 tensor cores | ≈ 1·B | B ≈ 160 |
| W8A16 (FP8 or INT8, Marlin) | BF16 after dequant | ≈ 2·B | B ≈ 80 |
| W4A16 g32 (≈0.56 B/param) | BF16 after dequant | ≈ 3.6·B | **B ≈ 45** |
| W8A8 INT8 | INT8 tensor cores (2× rate) | ≈ 2·B against a 322 ridge | **B ≈ 160** |

**What this tells us:**
- At c=1 INT4 wins (2.6× BF16).
- T2's running batch reaches ≈50–90, past INT4's knee, so INT4 steps grow with batch while W8A8 steps stay flat.
- Attention stays memory-bound at any batch (≈7 FLOP per BF16 KV byte), so FP8 KV keeps helping under load.
- This is why `climb-t2-vllm` starting from W8A8 is the right call.

### 6.4 Prefill efficiency at 8k tokens, c=1

An 8k-token prefill costs ≈ 542 TFLOP of GEMMs plus ≈ 50 TFLOP of attention. *(ledger, computed)*

| Variant | 8k TTFT | Effective throughput | Utilization (MFU) |
|---|---|---|---|
| BF16 | 2.72 s | 218 TFLOPS | ≈70% of 312 |
| FP8 (W8A16) | 2.99 s | 198 TFLOPS | ≈64% |
| INT4 (W4A16) | 3.48 s | 170 TFLOPS | ≈55% |
| INT8 (W8A16) | 3.96 s | 150 TFLOPS | ≈48% |
| INT8 W8A8 | 1.71 s | ≈ 370 TOPS on the GEMMs | ≈60% of 624 |

- The ledger also holds a W8A8 8k TTFT of **0.89 s**. That is faster than the ≈1.03 s physical floor at 100% of peak, so it's an artifact: a prefix-cache hit from the old warmup (AI09). Use 1.7 s.
- Weight-only formats lose 10–30% of prefill to dequantization.

---

## 7. Two nodes over Ethernet, revisited

The earlier topology doc reached the right default, two independent replicas, but partly for the wrong reason. Its prefill/decode numbers came from Qwen3-Coder-30B-A3B, where prefilling 1k tokens takes ≈57 ms. On dense Nanbeige-33B, prefill is expensive: ≈0.33–0.42 s per 1k tokens, and about half the GPU's time at T2 load. That changes the arithmetic.

**What crosses the ≈20 Gb/s (≈2.5 GB/s) link per request:**

| Option | What moves | 1k-token prompt | 8k-token prompt | Compared with |
|---|---|---|---|---|
| Two replicas (default) | HTTP only | ≈0 | ≈0 | n/a |
| Prefill/decode split (1P1D) | KV cache (BF16 / FP8) | 0.22 / 0.11 GB → **87 / 44 ms** | 1.74 / 0.87 GB → **0.70 / 0.35 s** | Prefill compute: ≈0.35 s (1k), ≈2.7 s (8k) in BF16 |
| Pipeline parallel (PP=2) | Hidden states at one stage boundary | 15 MB → **6 ms** | 117 MB → **47 ms** | ≈15× less than BF16 KV |
| Tensor parallel (TP=2) | Two all-reduces per layer, every token | ≈1.5 MB per token, which caps prefill near **1.7k tok/s** | n/a | Slower than one GPU (AI02) |

**What the books and agents conclude:**
- **The book's test for splitting prefill and decode fails here.** It wants a KV handoff of a few milliseconds next to hundreds of milliseconds of prefill [AISPE p.812–817]. Over this link the ratio is ≈0.26, roughly 10–40× worse than the book assumes. *Inference Engineering* adds that splitting pays off only with ≥100B models, ≥100M tokens/day, and prefill-heavy traffic [IE p.149–150].
- **Bandwidth is not what kills it at 1k prompts, though.** The real cost is the 1:1 split: with two GPUs, one prefill node gets half the prefill capacity. The book's min-per-phase goodput method [AISPE p.763–764] predicts:
  - **INT4 on both nodes:** ~−10% vs two replicas.
  - **W8A8 prefill feeding INT4 decode** (a different precision per phase [AISPE p.826]): ~+20%, *if* NIXL over TCP sustains ≥1–1.5 GB/s and quality holds. llm-d calls NIXL over TCP test/dev only (AI07, AI02).
  - Agents disagree on whether this arm can win (AI07/AI02: maybe; IE2: no, for the book's reasons). That makes it a good measured blog result either way.
- **Pipeline parallelism over Ethernet is cheap on the wire, and it is the lossless option** *(estimate)*.
  - Each GPU holds ≈26 layers (≈33 GB of BF16), leaving room for **≈400k KV tokens instead of 59k**.
  - Per-user decode stays about at single-GPU BF16 speed (≈23 tok/s), so it can't win T1.
  - Its case is "no quantization allowed": compare BF16 PP=2 against two BF16 replicas (starved for KV) and against two INT4/W8A8 replicas.
  - Inferact found PP2 beats TP2 by 17% under a 2 s TTFT target, because PP's bubbles shrink as load grows.
  - Setup: `--pipeline-parallel-size 2`, plus `VLLM_PP_LAYER_PARTITION` if needed and NCCL over sockets (`NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME`, `NCCL_NSOCKS_PERTHREAD`, `NCCL_SOCKET_NTHREADS`). This is the one place AISPE ch. 4's NCCL tuning applies. Verify vLLM v0.34's multi-node launcher on AKS first.
- **Routing (L7) needs shared prefixes and high load to show anything.** On T2's random prompts, cache-aware routing will look identical to round-robin [AISPE p.790; IE p.191]. Run it on T3 at two load levels, normalize per GPU, and include a single replica behind the same gateway as a control (IE3-08, AI07-12). Set `PYTHONHASHSEED=0` or use SHA-256 block hashing so the router and replicas agree on prefix hashes (AI07-13).

---

## 8. Metric definitions for Part 0 (book vs harness)

| Term | *Inference Engineering* | Harness / vLLM bench | Say this in the blog |
|---|---|---|---|
| TTFT | Time to the first output token. Separates inference-only time from end-to-end time (network + queue) [IE p.35, p.37]. | Measured client-side over localhost, so it includes queueing and tokenization but no network. For a thinking model, the first token is a *thinking* token. | "Server-side TTFT (localhost)". Add time-to-first-*answer*-token. |
| TPS (perceived) / TPOT | TPS = per-user speed *after* the first token [IE p.35]. "TPOT" never appears in the book. | T1 = 1000 ÷ median TPOT, which is the book's perceived TPS. | "Per-user decode speed" |
| AA output speed | n/a | `aa_speed_c1` = output tokens ÷ end-to-end time, *including* TTFT | Never compare directly with T1 |
| ITL | Time between tokens (10 ms = 100 TPS) [IE p.35, p.220] | vLLM measures per streamed *chunk*, so it differs from TPOT under speculative decoding | Use TPOT for speed and chunk-ITL p99 for smoothness |
| Throughput / total TPS | Total tokens per second for the service [IE p.35] | `output_throughput` over the whole run, including ramp-up and drain | State output vs total, and whether it's SLO-filtered |
| Goodput | Not in IE. AISPE ch. 1 defines it as useful work ÷ peak, in a training context [AISPE p.17]. Serving goodput is SLO-gated [AISPE p.763]. | T2: SLO-meeting output tok/s at the highest rate with ≥90% joint attainment | Give the units and the attainment level |
| Percentiles | P50/P90/P95/P99 [IE p.37] | T1 reports the median; T2's ≥90% joint attainment ≈ P90; plan-v1 says p99 | Name the percentile every time |
| Concurrency | A replica's concurrency target ≈ its batch size [IE p.186–188] | In open-loop T2, concurrency is an *outcome* (Little's law) | Report the concurrency actually reached |
| Cold start | Provisioning + image + weights + compile [IE p.189–190] | `startup_s` is a warm restart on a provisioned node with warm caches | "Warm restart"; measure one true cold start |

---

## 9. Blog-worthy facts

1. A Baseten engineer scripted **77 configurations** to find a non-obvious one that doubled TPS for a customer [IE p.119]. This is the case for hill climbing.
2. The two books frame the A100 differently: still seen "in low-traffic or legacy systems" [IE p.79] vs "a timeless classic" [AISPE p.xxvii]. The series tests which is right.
3. One precision step usually buys 30–50% [IE p.120]. At c=1 on this A100, BF16 → 8-bit gave **+80%** and 8-bit → INT4 gave **+45%**, because batch-1 decode is pure bandwidth. *(ledger)*
4. **On Ampere, weight-only quantization trades prefill for decode.** INT4 decodes 2.6× faster than BF16 but has 28% slower 8k TTFT. W8A8 is the only format that speeds prefill (2.7 → 1.7 s). *(ledger)*
5. **Shrinking weights turns overhead into the bottleneck:** decode ≈ 4.6 ms + bytes ÷ 1.69 TB/s, so fixed overhead is 28% of an INT4 token vs 11% for BF16. *(ledger fit)*
6. The A100's ridge point is ≈161 FLOP/byte. INT4 decode becomes compute-bound around batch ≈45; BF16 and W8A8 around ≈160. *(computed)*
7. Every trial ran into the **300 W power cap**, and SM clocks drop ≈12% under load. *(ledger)*
8. VRAM should hold weights plus ≥50% headroom for KV [IE p.77]. BF16 33B on 80 GB leaves ≈13 GiB, about 59k tokens. **1 GiB of BF16 KV holds ≈5,041 Nanbeige tokens; one 64k conversation is ≈14 GB.** *(computed)*
9. Moving an 8k prompt's KV over 20 Gb/s takes **0.70 s**, vs LMCache's 8 ms for 7.5k tokens over RDMA [AISPE p.815]. Pipeline parallelism moves 15× less data. *(computed)*
10. DistServe reports up to 7.4× more requests within SLO, but results "greatly depend on the workload and network fabric" [AISPE p.665–666].
11. Speculative decoding gives ≈2× in practice (1.5–2.5×) [AISPE p.682–684]. It "only improves TPS/ITL, not TTFT" [IE p.129]. Structured output costs double digits at small batch [AISPE p.690].
12. torch.compile buys dense models under 10% [AISPE p.547], which is why L5 gains after quantization are small.
13. Plan production at ≈90% of peak; running at 90% with efficient kernels can outperform 100% with throttling [AISPE p.730].
14. People read about 4–7 tokens/s [AISPE p.752]. For a reasoning model, the real wait is hidden thinking: 2,000 thinking tokens at 61 tok/s is ≈33 s.
15. MLPerf v5.0's Llama 2 70B SLOs are p99 TTFT ≈450 ms and p99 TPOT 40 ms [AISPE p.761]. Our T2 SLO is far looser, so say so.
16. Inferact reports a 30-hour hillclimb cut p99 TPOT by 35%, and production hillclimbing raised throughput per GPU ≈1.8× over six weeks (vendor claims, X, 18 Nov 2026).

---

## 10. Simon Mo's article and this project

**How the harness maps onto Inferact's "autonomous inference system":**

| Inferact AIS capability | This harness today | Gap, or what to borrow |
|---|---|---|
| **Sees** production workloads and profiles new models | `/metrics`, nvidia-smi, synthetic workloads | P1-1 profiling. No production traffic exists, so replay a public trace instead, e.g. Azure's published LLM inference traces (general knowledge; verify format and license). That's also a good Azure tie-in. |
| **Adapts** to new models, hardware, and workloads | Knob registry plus validity rules | P0-4: log the config that actually resolved, so silent fallbacks become visible |
| **Evolves** its own code and checks | Static climber | A daily retrospective over the ledger that moves the gate where most trials die to the front (fail fast) |
| **Drives** itself toward goals | In-cluster controller and campaigns | Mode B agent (P2-4) |
| **Optimizes** against each workload's targets | T1 and T2 (T3 pending) | T3 plus mixed-length workloads |
| Agents write kernels and megakernels | No (L6 stretch) | Only after profiling. The fixed-overhead fit caps a perfect megakernel at ≈1.4× for INT4 T1. |
| Promotes only what survives full serving benchmarks and shadow traffic | Every trial is a full serving benchmark | Add quality under load (P0-7f) and the 3-seed rate verification (P0-6) |

**How to describe it:** "the same loop at a smaller scope." It is a reproducible, open config-space hill climb on capacity enterprises already have. It is *not* a superset of Inferact's system, which also writes kernels, shadows production traffic, and modifies itself. Relative to the two books' tuning toolkits, the harness does cover essentially the whole tunable surface for one A100 (§3).

---

## Appendix A: Per-slice findings

Every file uses the same seven sections: chapter map, key learnings, recommendations (NEW, PARTIAL, or COVERED), already covered, not applicable, cautions, and blog-worthy facts. Agents read the harness and ledger read-only.

| File | Slice | Recs | Standout content |
|---|---|---|---|
| [IE1.md](book-review/IE1.md) | IE Preface and Ch 0–3 (read by me) | 7 | Metric vocabulary, ops:byte ratio, VRAM headroom rule, A100 framed as "legacy" |
| [IE2.md](book-review/IE2.md) | IE Ch 4–5: software and techniques | 26 | Quantization, speculative decoding, and caching rules; benchmarking method; the 77-config story; fixed-overhead fit |
| [IE3.md](book-review/IE3.md) | IE Ch 6–7 and Appendices A/B | 17 | Production, cost modeling, book vs harness glossary, reading list |
| [AI01.md](book-review/AI01.md) | AISPE Ch 1–3: intro, hardware, OS/K8s | 15 | Two definitions of goodput, **T2 drain bias**, OS/K8s triage for AKS |
| [AI02.md](book-review/AI02.md) | AISPE Ch 4–5: networking, storage | 15 | Where startup time goes, TCP math for TP/PP/PD, faster weight loading |
| [AI03.md](book-review/AI03.md) | AISPE Ch 6–9: GPU architecture, roofline | 15 | **Four-point roofline fit**, prefill MFU, W8A8/W4A8/2:4 on A100 |
| [AI04.md](book-review/AI04.md) | AISPE Ch 10–12: streams, CUDA graphs | 8 | Launch overhead, CUDA-graph policy, megakernel gate |
| [AI05.md](book-review/AI05.md) | AISPE Ch 13–14: PyTorch, compile, Triton | 20 | **Acceptance statistics**, compile knobs, SGLang cache, allocator |
| [AI06.md](book-review/AI06.md) | AISPE Ch 15–16: inference tuning | 24 | Speculative-decoding accounting, structured output, quantization claims vs the ledger |
| [AI07.md](book-review/AI07.md) | AISPE Ch 17–18: P/D, KV cache | 17 | **P/D math for this model**, T3 design, L7 routing arms |
| [AI08.md](book-review/AI08.md) | AISPE Ch 19–20: adaptive, AI-assisted | 26 | **Climber v2 procedure**, soft scores, agent verification |
| [AI09.md](book-review/AI09.md) | AISPE Appendix: 175+ item checklist | 24 | Audit of all 180 items, power-cap evidence, ledger fixes |
| [X1_simon_mo_inferact.md](book-review/X1_simon_mo_inferact.md) | Simon Mo's article and Inferact's PP post | n/a | Fail-fast gates, quality under load, 3-seed rate protocol, PP over Ethernet |
| [00-agent-brief.md](book-review/00-agent-brief.md) | The context brief every agent received | n/a | Hardware, model, harness, and results as of this review |

---

## Appendix B: The 175+ item checklist, audited

Audit by AI09 of all 180 items in AISPE's appendix [AISPE p.947–988] against the harness:

| Checklist section | Items | Covered | Partial | New | N/A |
|---|---|---|---|---|---|
| Performance tuning and cost mindset | 12 | 6 | 6 | 0 | 0 |
| Reproducibility and documentation | 9 | 6 | 3 | 0 | 0 |
| System architecture and hardware planning | 8 | 4 | 1 | 0 | 3 |
| Unified CPU-GPU superchip | 6 | 0 | 0 | 0 | 6 |
| Multi-GPU scaling and interconnect | 8 | 0 | 0 | 0 | 8 |
| OS and driver | 14 | 2 | 3 | 6 | 3 |
| GPU resource management and scheduling | 10 | 3 | 1 | 1 | 5 |
| I/O | 11 | 2 | 0 | 1 | 8 |
| Data pipelines | 12 | 3 | 1 | 1 | 7 |
| Profiling, debugging, and monitoring | 11 | 0 | 5 | 3 | 3 |
| GPU programming and CUDA | 23 | 3 | 4 | 0 | 16 |
| Kernel scheduling and execution | 5 | 2 | 0 | 0 | 3 |
| Arithmetic and mixed precision | 10 | 2 | 2 | 1 | 5 |
| Advanced tuning strategies | 8 | 2 | 1 | 1 | 4 |
| Distributed training and network | 9 | 0 | 0 | 2 | 7 |
| Efficient inference and serving | 11 | 3 | 2 | 0 | 6 |
| Multinode inference and serving | 4 | 1 | 1 | 1 | 1 |
| Power and thermal | 9 | 1 | 1 | 1 | 6 |
| **Total** | **180** | **40** | **31** | **18** | **91** |

**What the tally says:**
- About half the checklist doesn't apply to one PCIe A100 per node over Ethernet: training pipelines, NVLink, Grace, RDMA, and kernel authoring.
- Of the 89 items that do apply, 45% are covered and 35% partly covered.
- Coverage is strongest where the harness was designed to be strong: the search space, end-to-end measurement, and ledgered experiments.
- It is weakest in three areas, which match the P0 and P1 items above:
  - Observability below the API: no profiler, DCGM counters, or CPU telemetry.
  - Reproducibility metadata: no image digests or drivers per trial, and no noise calibration.
  - Power: the 300 W cap binds and nothing records it usefully.
