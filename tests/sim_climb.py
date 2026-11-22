"""Offline simulation of the climber against a fake cluster (no Kubernetes, temporary ledger).
Checks the control flow of both tracks: reuse of recent results, T2 capacity screening with paired
confirmation, fresh-run acceptance for T1, and the plateau stop. Run: python3 -m tests.sim_climb"""
import json
import random
import tempfile
from pathlib import Path

from harness import climber, ledger, plans, runner, score, space

rng = random.Random(7)
TMP = Path(tempfile.mkdtemp(prefix="hc-sim-"))
ledger.DB = TMP / "ledger.sqlite"
climber.BEST_DIR = TMP / "best"
_seq = [0]

# Hidden "true" capacity (req/s) of vLLM configs relative to the W8A8 default (2.0 req/s).
T2_EFFECT = {("variant", "bf16"): 0.57, ("variant", "fp8"): 0.62, ("variant", "int8"): 0.57,
             ("variant", "int4"): 0.62, ("kv_cache_dtype", "fp8"): 1.10, ("max_num_seqs", 128): 0.97,
             ("max_num_batched_tokens", 1024): 1.06, ("max_num_batched_tokens", 512): 1.08,
             ("max_num_batched_tokens", 4096): 0.95, ("attention_backend", "TRITON_ATTN"): 0.90,
             ("attention_backend", "FLEX_ATTENTION"): 0.80, ("attention_backend", "FLASHINFER"): 1.02,
             ("cudagraph_mode", "NONE"): 0.70}
T1_EFFECT = {("spec_k", 6): 1.15, ("spec_k", 8): 1.05, ("spec_k", 3): 0.93, ("spec", None): 0.55,
             ("spec", "ngram"): 0.85, ("cudagraph_mode", "NONE"): 0.40, ("attention_backend", "FLEX_ATTENTION"): 0.8,
             ("block_size", 32): 1.02}  # a real +2% gain, below the 3% threshold


def true_cap(cfg):
    v = 2.0
    for (k, val), f in T2_EFFECT.items():
        if cfg.get(k) == val:
            v *= f
    return v


def true_tps(cfg):
    v = 112.0
    for (k, val), f in T1_EFFECT.items():
        if cfg.get(k) == val:
            v *= f
    return v


def attain(rate, cap):
    # sharp but noisy transition around capacity
    return max(0.0, min(1.0, 0.97 - 2.5 * max(0.0, rate - cap) / cap + rng.gauss(0, 0.012)))


def step(rate, cap, osl=512):
    n = max(60, int(rate * 120))
    a = attain(rate, cap)
    thr = rate * 0.9
    return {"isl": 1024, "osl": osl, "c": 1024, "n": n, "completed": n, "request_rate": rate,
            "request_throughput": thr, "request_goodput": thr * a, "dataset": "random", "primary": False}


def ladder(cap, start=0.5, factor=1.5, bisect=4):
    steps, lo, hi, r = [], None, None, start
    while r <= 64:
        s = step(r, cap)
        steps.append(s)
        if s["request_goodput"] / s["request_throughput"] >= plans.ATTAINMENT:
            lo, r = r, r * factor
        else:
            hi = r
            break
    if lo is not None and hi is not None:
        for _ in range(bisect):
            mid = (lo * hi) ** 0.5
            s = step(mid, cap)
            steps.append(s)
            if s["request_goodput"] / s["request_throughput"] >= plans.ATTAINMENT:
                lo = mid
            else:
                hi = mid
    return steps


def fake_metrics(cfg, plan_name, lane):
    lane_f = 1.0 if lane == "n0" else 1.01
    if plan_name.startswith("t2"):
        cap = true_cap(cfg) * lane_f
        if plan_name.startswith("t2probe@"):
            steps = [step(float(plan_name.split("@")[1]), cap)]
        elif plan_name.startswith("t2s@"):
            steps = ladder(cap, start=float(plan_name.split("@")[1]), factor=1.1, bisect=3)
        else:
            steps = ladder(cap)
    else:
        tps = true_tps(cfg) * lane_f * (1 + rng.gauss(0, 0.002))
        steps = [{"dataset": "spec_bench", "primary": True, "c": 1, "isl": None, "osl": 512, "n": 10, "completed": 10,
                  "median_tpot_ms": 1000 / tps, "median_ttft_ms": 46, "median_e2el_ms": 46 + 512 * 1000 / tps},
                 {"dataset": "random", "c": 1, "isl": 1024, "osl": 1024, "n": 3, "completed": 3,
                  "median_tpot_ms": 1000 / (tps * 0.6), "median_ttft_ms": 40},
                 {"dataset": "random", "c": 1, "isl": 8192, "osl": 256, "n": 3, "completed": 3,
                  "median_tpot_ms": 20,
                  "median_ttft_ms": 3480 * (1.3 if cfg.get("attention_backend") == "TRITON_ATTN" else 1)}]
    m = {"steps": steps}
    m["scores"] = score.scores(m)
    return m


def fake_run_many(jobs, campaign, lanes=("n0", "n1"), log=print):
    out = []
    for i, (cfg, plan_name, meta) in enumerate(jobs):
        lane = lanes[i % len(lanes)]
        cfg = space.full(cfg)
        _seq[0] += 1
        tid = f"t{_seq[0]:08d}-{space.config_hash(cfg)[:6]}-{lane}"
        ledger.start(tid, campaign, plan_name, lane, "node", cfg, space.config_hash(cfg), None,
                     (meta or {}).get("move"), (meta or {}).get("notes"))
        reason = space.validity(cfg)
        if reason:
            ledger.finish(tid, "invalid", reason, None)
            out.append((tid, "invalid", None))
            continue
        m = fake_metrics(cfg, plan_name, lane)
        ledger.finish(tid, "done", None, m)
        out.append((tid, "done", m))
    return out


runner.run_many = fake_run_many


def calibration_case():
    """The main-v3 SGLang T3 climb's runs (t3_score = 1 s / p90 TTFT). Its first four incumbent runs put 3 sigma at
    37.5%; FP8 KV screened +38.4% and its fresh confirmation run +37.2%, while the confirmation pair added a fifth
    incumbent run that brings 3 sigma to 31.1%. The decision must use the estimate from all five, the same one a
    restarted controller computes from the same runs."""
    inc = space.full({"engine": "sglang", "variant": "int8w8a8", "mem_fraction_static": 0.91})
    script = {("auto", "n0"): [0.6330, 0.7591, 0.6757], ("auto", "n1"): [0.7041, 0.5922],
              ("fp8_e4m3", "n0"): [0.9636, 0.9460], ("fp8_e4m3", "n1"): [0.9500, 0.9480]}

    def scripted(jobs, campaign, lanes=("n0", "n1"), log=print):
        out = []
        for i, (cfg, plan_name, meta) in enumerate(jobs):
            lane = lanes[i % len(lanes)]
            cfg = space.full(cfg)
            _seq[0] += 1
            tid = f"t{_seq[0]:08d}-{space.config_hash(cfg)[:6]}-{lane}"
            ledger.start(tid, campaign, plan_name, lane, "node", cfg, space.config_hash(cfg), None,
                         (meta or {}).get("move"), (meta or {}).get("notes"))
            v = script[(cfg["kv_cache_dtype"], lane)].pop(0)
            m = {"t3": {"ttft_p90_ms": 1000.0 / v, "errors": 0, "turns_done": 288, "turns_planned": 288}}
            m["scores"] = score.scores(m)
            ledger.finish(tid, "done", None, m)
            out.append((tid, "done", m))
        return out

    def thresholds(sub):
        with ledger.conn() as c:
            return [json.loads(r["detail"])["threshold"] for r in
                    c.execute("select detail from events where campaign=? and kind='calibration' order by ts", (sub,))]

    saved, runner.run_many = runner.run_many, scripted
    try:
        opts = dict(threshold=0.05, knob_order=["kv_cache_dtype"], calibrate=True, coupled=False, kicks=False,
                    gate=False, log=lambda *a: print("  ", *a))
        best = climber.climb(inc, "t3", "cal:climb-t3-sglang", max_trials=35, **opts)
        thr = thresholds("cal:climb-t3-sglang")
        # a restart of the climb before its accept: reloads the same five incumbent runs and only calibrates
        climber.climb(inc, "t3", "cal:restart", max_trials=0, **opts)
        restart = thresholds("cal:restart")
        # a restart after the accept: the new incumbent has too few runs to replace the estimate in use, so the
        # climb resumes it instead of calibrating on the new incumbent's runs
        before = _seq[0]
        climber.climb(best, "t3", "cal:climb-t3-sglang", max_trials=8, **opts)
        resumed = thresholds("cal:climb-t3-sglang")[-1]
    finally:
        runner.run_many = saved
    print(f"T3 calibration: thresholds {[round(t, 4) for t in thr]}, restart {[round(t, 4) for t in restart]}, "
          f"after the accept {resumed:.4f}")
    assert abs(thr[0] - 0.3754) < 5e-4 and abs(thr[-1] - 0.3110) < 5e-4, thr
    assert len(restart) == 1 and abs(restart[0] - thr[-1]) < 1e-12, \
        "a restarted climb must judge with the threshold the running climb used"
    assert best["kv_cache_dtype"] == "fp8_e4m3", "+37.2% on the fresh run beats the five-run threshold of 31.1%"
    assert abs(resumed - thr[-1]) < 1e-12 and _seq[0] == before, \
        "a climb restarted after an accept must resume its noise estimate without calibration runs"


def main():
    # Results from an earlier stage, to be reused rather than re-run: the incumbent and one variant neighbour.
    w8 = space.full({"engine": "vllm", "variant": "int8w8a8"})
    fake_run_many([(w8, "t2s", {"notes": "t2-variants"}), (dict(w8, variant="int4"), "t2s", {"notes": "t2-variants"})],
                  "sim:t2-variants")
    used_before = _seq[0]
    best = climber.climb(w8, "t2", "sim:climb-t2-vllm", threshold=0.05, max_trials=40, log=lambda *a: print("  ", *a))
    print("T2 final:", {k: best[k] for k in ("variant", "kv_cache_dtype", "max_num_batched_tokens", "attention_backend")},
          f"true capacity {true_cap(best):.3f} (start {true_cap(w8):.3f}); trials run {_seq[0] - used_before}")
    with ledger.conn() as c:
        ev = [(r["kind"], json.loads(r["detail"])) for r in
              c.execute("select kind, detail from events where campaign='sim:climb-t2-vllm' order by ts")]
    for k, d in ev:
        if k in ("accept", "bar-raised", "reject-confirm", "plateau", "capacity"):
            print("   event", k, {x: d.get(x) for x in ("knob", "to", "before", "after", "bar", "fresh_att",
                                                        "incumbent_att", "incumbent", "capacity") if x in d})
    assert best["kv_cache_dtype"] == "fp8", "the +10% FP8-KV gain must be accepted"
    assert best["max_num_batched_tokens"] in (512, 1024), "the chunk-budget gain must be accepted"
    assert best["variant"] == "int8w8a8"

    t1_start = space.full({"engine": "vllm", "variant": "int4", "spec": "suffix", "spec_k": 4})
    best1 = climber.climb(t1_start, "t1", "sim:climb-t1-vllm", threshold=0.03, max_trials=60,
                          log=lambda *a: print("  ", *a))
    print("T1 final:", {k: best1[k] for k in ("spec", "spec_k", "block_size")}, f"true tps {true_tps(best1):.1f}")
    assert best1["spec_k"] == 6
    calibration_case()
    print("SIMULATION OK", TMP)


if __name__ == "__main__":
    main()
