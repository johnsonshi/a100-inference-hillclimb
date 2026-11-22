"""Versioned workload plans. Each step is one `vllm bench serve` run (random tokens, fixed
lengths via ignore_eos, fixed seeds so every trial sees identical prompts)."""
import json

SLO = {"ttft_ms": 2000, "tpot_ms": 100}  # MLPerf-server-style target for goodput (T2)


def _sweep(cs, isl=1024, osl=1024, base_seed=2000):
    return [dict(isl=isl, osl=osl, c=c, n=max(16, 2 * c), warmups=min(4, c), seed=base_seed + i,
                 stop_on_error=False) for i, c in enumerate(cs)]


PLANS = {
    # quick functional check after a new engine/variant
    "smoke": {"steps": [dict(isl=512, osl=128, c=1, n=2, warmups=1, seed=900),
                        dict(isl=512, osl=128, c=8, n=16, seed=901)],
              "timeout_s": 2400},
    # macro screen: one single-user point and one mid-load point
    "screen": {"steps": [dict(isl=1024, osl=1024, c=1, n=3, warmups=1, seed=1000),
                         dict(isl=1024, osl=1024, c=32, n=64, seed=1001, stop_on_error=False)],
               "timeout_s": 3600},
    # T1 interactivity: single-user decode speed and TTFT (1k and 8k prompts)
    "t1": {"steps": [dict(isl=1024, osl=1024, c=1, n=5, warmups=1, seed=1100),
                     dict(isl=8192, osl=256, c=1, n=3, warmups=1, seed=1101)],
           "timeout_s": 3600},
    # T1 (realistic, used by the climber): SPEC-Bench chat prompts at model-card sampling are the
    # primary score, since speculative decoding acceptance depends on real text; plus the
    # random-token engine baseline and 8k-prompt TTFT.
    "t1s": {"steps": [dict(dataset="spec_bench", osl=512, c=1, n=10, warmups=1, seed=1200, primary=True,
                           temperature=0.6, top_p=0.95),
                      dict(isl=1024, osl=1024, c=1, n=3, warmups=1, seed=1100),
                      dict(isl=8192, osl=256, c=1, n=3, warmups=1, seed=1101)],
            "timeout_s": 3600},
    # T2 goodput: concurrency sweep with early stop once goodput falls well below its best
    "t2": {"steps": _sweep([8, 16, 32, 64, 128, 256]), "timeout_s": 7200,
           "stop_when_goodput_drops": 0.6},
    # T2 (used by the climber): open-loop Poisson arrivals at increasing request rates (MLPerf
    # "Server"-style), ~2 minutes of arrivals per rate; goodput = SLO-meeting output tokens/s.
    # Stops once goodput falls below 60% of its best (the server is past saturation).
    "t2p": {"steps": [dict(isl=1024, osl=512, c=1024, request_rate=r, n=max(60, int(r * 120)), warmups=2,
                           seed=3000 + i, stop_on_error=False)
                      for i, r in enumerate([0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16])],
            "timeout_s": 7200, "stop_when_goodput_drops": 0.6},
    # T2 (used by the climber): MLPerf-style peak search under Poisson arrivals. Geometric ladder
    # (x1.5 from 0.5 req/s) until <90% of requests meet the SLO, then 4 geometric bisections
    # (~3% resolution). Score = SLO-meeting output tokens/s at the highest sustainable rate.
    "t2s": {"steps": [], "timeout_s": 9000,
            "rate_search": {"step": dict(isl=1024, osl=512, c=1024, warmups=2), "start": 0.5, "factor": 1.5,
                            "attainment": 0.9, "bisect_iters": 4, "duration_s": 120, "min_n": 60,
                            "max_rate": 64, "seed": 3000}},
    # quality gate per numerics class: GSM8K accuracy (temperature 0, thinking off) + greedy refs
    "quality": {"driver": "quality_driver.py", "steps": [], "timeout_s": 3600,
                "gsm8k_n": 500, "gsm8k_max_tokens": 1024, "concurrency": 32},
    # --- verification of the climbs' winners (verify stages) ---
    # T1 with natural stopping (thinking on, up to 2048 tokens) on 32 SPEC-Bench prompts, then 32 *other*
    # prompts forced to 512 tokens like t1s. Different seeds on purpose: suffix decoding caches earlier
    # outputs across requests, so re-running the same prompts would flatter the second step.
    # (v3) prompt_offset selects explicit, disjoint SPEC-Bench slices: vLLM's SpecBench loader ignores --seed,
    # so seeds alone gave both steps (and the warmup) the same prompts. The climb (t1n) uses prompts 0-31;
    # verification uses held-out prompts 64-95 and 96-127, which also checks the climb did not overfit.
    "t1v": {"steps": [dict(dataset="spec_bench", osl=2048, c=1, n=32, warmups=1, seed=1300, primary=True,
                           temperature=0.6, top_p=0.95, ignore_eos=False, prompt_offset=64),
                      dict(dataset="spec_bench", osl=512, c=1, n=32, warmups=1, seed=1301,
                           temperature=0.6, top_p=0.95, ignore_eos=True, prompt_offset=96)],
            "timeout_s": 7200},
    # quality as served (thinking on, long outputs): KV/attention error compounds over long generations
    "quality-think": {"driver": "quality_driver.py", "steps": [], "timeout_s": 5400, "thinking": True,
                      "gsm8k_n": 200, "gsm8k_max_tokens": 4096, "concurrency": 32},
    # ===================== harness v3 plans =====================
    # T1 v3: per-user decode on 32 SPEC-Bench chat prompts with *natural stopping* (thinking on, at most 768
    # tokens, model-card sampling), plus the 8k-prompt TTFT step used by the T1 guard.
    "t1n": {"steps": [dict(dataset="spec_bench", osl=768, c=1, n=32, warmups=1, seed=1400, primary=True,
                           temperature=0.6, top_p=0.95, ignore_eos=False, prompt_offset=0),
                      dict(isl=8192, osl=256, c=1, n=3, warmups=1, seed=1101)],
            "timeout_s": 7200},
    # T2 chat (real text): Poisson arrivals of ShareGPT first turns with natural stopping (thinking on, at most
    # 1024 tokens). Content-dependent knobs (speculative decoding) can only be judged on real text. ShareGPT
    # (a fixed 30k subset, tools/stage_sharegpt.sh) rather than SPEC-Bench: a rate search sends ~1000 requests,
    # and SPEC-Bench's 480 prompts would repeat, flattering prefix caching and suffix decoding's output cache.
    "t2c": {"steps": [], "timeout_s": 9000,
            "rate_search": {"step": dict(dataset="sharegpt", osl=1024, c=1024, warmups=2, temperature=0.6,
                                         top_p=0.95, ignore_eos=False),
                            "start": 0.25, "factor": 1.5, "attainment": 0.9, "bisect_iters": 4, "duration_s": 120,
                            "min_n": 40, "max_rate": 32, "seed": 5000}},
    # quality gate v2 (per numerics class): GSM8K-200 with thinking on (4096 tokens), passkey retrieval at
    # 8k/16k/28k context, greedy outputs for speculative-equivalence, and teacher-forced NLL / top-1 agreement
    # on BF16's greedy continuations. "make_refs" (BF16 only) produces those continuations.
    "quality2-refs": {"driver": "quality_driver.py", "steps": [], "timeout_s": 7200, "thinking": True,
                      "gsm8k_n": 200, "gsm8k_max_tokens": 4096, "concurrency": 32, "passkey": True,
                      "make_refs": True},
    "quality2": {"driver": "quality_driver.py", "steps": [], "timeout_s": 7200, "thinking": True,
                 "gsm8k_n": 200, "gsm8k_max_tokens": 4096, "concurrency": 32, "passkey": True, "use_refs": True},
    # T2 mixed-length capacity (verification, not climbed): rate search over a Poisson stream of 85% 1k-token
    # and 15% 8k-token prompts (512 output tokens each), 180-s windows so each rung sees enough long prompts.
    # Long prompts roughly double prefill work per request, so this capacity sits well below t2s's.
    "t2m": {"steps": [], "timeout_s": 10800,
            "rate_search": {"step": dict(isl=1024, osl=512, c=1024, warmups=2,
                                         mix_fractions=[[0.85, dict(isl=1024, osl=512, c=1024)],
                                                        [0.15, dict(isl=8192, osl=512, c=1024)]]),
                            "start": 0.4, "factor": 1.5, "attainment": 0.9, "bisect_iters": 4, "duration_s": 180,
                            "min_n": 60, "max_rate": 16, "seed": 6100}},
    # one-off profile of single-user decode (torch profiler): 2 requests x 256 tokens after a 1k prompt
    # (prefill is subtracted using the measured TTFT; the trace is ~45 MB).
    "profile": {"steps": [], "timeout_s": 3600, "profile": {"isl": 1024, "osl": 256, "n": 2}},
    # T3 agentic multi-turn: Poisson session arrivals, shared ~3k-token system prompt, a ~1.2k-token document
    # per turn, 6 turns, tool "think time" Exp(15 s) between turns; score = TTFT p90 over all turns (lower is
    # better). Sized at the edge of the KV pool: ~30 concurrent sessions x ~8k tokens ~= 240k live tokens,
    # ~1.1x W8A8's BF16 KV pool (213k) and ~0.6x its FP8 pool, with ~1.75k tok/s of new prefill. (A first
    # sizing at ~580k live tokens collapsed: 98% KV use, 25 queued, re-prefill of evicted sessions.)
    "t3": {"driver": "t3_driver.py", "steps": [], "timeout_s": 5400,
           "t3": {"sessions": 48, "turns": 6, "session_rate": 0.2, "think_mean_s": 15.0, "system_tokens": 3000,
                  "doc_tokens": 1200, "max_tokens": 256, "seed": 7000}},
}
ATTAINMENT = PLANS["t2s"]["rate_search"]["attainment"]
# capacity families: ladder plan -> (probe prefix, fine-ladder prefix, verification prefix)
CAPACITY_FAMILIES = {"t2s": ("t2probe", "t2s", "t2v"), "t2c": ("t2cprobe", "t2c", "t2cv")}
# Steady-state capacity (verification). At the T2 climbs' capacities, 300-s windows met the SLO for only 23-89% of
# requests, although the climbs' 120-s rungs had passed: requests arriving in the first minute or two meet a batch
# that is still ramping up, those in the last minute one that drains after arrivals stop, and in between attainment
# fell as low as 0. A 120-s window is mostly these transients. Chat is worse (~870 output tokens per request): vLLM's
# chat winner failed even at 0.9x its capacity. These plans search with long windows and judge only the requests
# that arrive inside `measure` [from_s, until_s), after the ramp-up and while arrivals continue for about one request
# lifetime more. The search starts at 0.9x the measured capacity; it steps up by 4% until a rung fails, or down by
# 15% (to 0.25x at most) until one passes, then bisects until the bracket is within 1%.
# prefix -> (ladder family, window s, measure from s, measure until s)
STEADY = {"t2ss": ("t2s", 420, 120, 360), "t2css": ("t2c", 540, 180, 450)}


def _window(rs, rate, seed, duration_s):
    return dict(rs["step"], request_rate=rate, n=max(rs["min_n"], int(rate * duration_s)), seed=seed,
                stop_on_error=False)


def get(name):
    """Plan by name. Parameterized plans (the name is stored as-is in the ledger, so it stays traceable):
    - "t2probe@<r>" / "t2cprobe@<r>": one Poisson rung of the t2s / t2c rate search at r req/s (same step
      shape, duration and prompt seed as the ladder), used to screen a challenger against a capacity bar.
    - "t2s@<r>" / "t2c@<r>": the rate search starting at r req/s with a finer ladder (x1.1, 3 bisections),
      used to measure a newly accepted incumbent's capacity from the bar upward.
    - "t2v@<r>" / "t2cv@<r>": verification at a measured capacity r: three 300-s Poisson windows at r plus
      one at 0.9 r (the operating headroom used for $/1M tokens).
    - "t2mv@<r>": mixed-length verification at r req/s for 300 s: 85% 1k-token and 15% 8k-token prompts
      (two concurrent Poisson streams; their superposition is Poisson at r).
    - "t2ss@<r>" / "t2css@<r>": steady-state capacity search around a measured capacity r (see STEADY)."""
    if "@" not in name:
        return PLANS[name]
    base, arg = name.split("@", 1)
    rate = float(arg)
    if base in STEADY:
        fam, window_s, from_s, until_s = STEADY[base]
        p = json.loads(json.dumps(PLANS[fam]))
        rs = p["rate_search"]
        rs["step"]["measure"] = {"from_s": from_s, "until_s": until_s}
        rs.update(start=0.9 * rate, factor=1.04, down_factor=1.15, resolution=0.01, max_bisect=5, duration_s=window_s,
                  seed=rs["seed"] + 4300, max_rate=1.2 * rate, min_rate=0.25 * rate)
        p["timeout_s"] = 8 * 3600
        return p
    for fam, (probe, fine, verify) in CAPACITY_FAMILIES.items():
        rs = PLANS[fam]["rate_search"]
        if base == probe:
            return {"steps": [_window(rs, rate, rs["seed"], rs["duration_s"])], "timeout_s": 3600}
        if base == fine:
            p = json.loads(json.dumps(PLANS[fam]))
            p["rate_search"].update(start=rate, factor=1.1, bisect_iters=3)
            return p
        if base == verify:
            s = rs["seed"] + 1000
            return {"steps": [_window(rs, rate, s, 300), _window(rs, rate, s + 1, 300), _window(rs, rate, s + 2, 300),
                              _window(rs, 0.9 * rate, s + 3, 300)], "timeout_s": 9000}
    if base == "t2mv":
        n1, n2 = int(0.85 * rate * 300), int(0.15 * rate * 300)
        return {"steps": [{"mix": [dict(isl=1024, osl=512, c=1024, request_rate=0.85 * rate, n=n1, seed=6000),
                                   dict(isl=8192, osl=512, c=1024, request_rate=0.15 * rate, n=n2, seed=6001)],
                           "request_rate": rate, "osl": 512, "n": n1 + n2, "warmups": 2, "seed": 6002}],
                "timeout_s": 5400}
    raise KeyError(name)
