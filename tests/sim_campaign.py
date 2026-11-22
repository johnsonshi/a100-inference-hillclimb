"""End-to-end offline simulation of a campaign file (default campaigns/main-v3.json) against a fake cluster
and a temporary ledger: every stage type (quality batches, screens with reuse, climbs of all five tracks
with calibration / coupled passes / kicks / the quality gate, verify stages with @capacity, ablations).
Checks control flow and stage wiring, not performance. Run: python3 -m tests.sim_campaign [campaign.json]"""
import contextlib
import io
import json
import random
import sys
import tempfile
import time
from pathlib import Path

from harness import campaign, climber, ledger, plans, runner, score, space

rng = random.Random(11)
TMP = Path(tempfile.mkdtemp(prefix="hc-simcamp-"))
ledger.DB = TMP / "ledger.sqlite"
climber.BEST_DIR = TMP / "best"
_seq = [0]
CALLS = []

# ---- hidden "truth" (rough shapes of the real measurements; only relative effects matter) ----
T2_BASE = {("vllm", "int8w8a8"): 2.0, ("vllm", "int4"): 1.2, ("vllm", "fp8"): 1.2, ("vllm", "int8"): 1.1,
           ("vllm", "bf16"): 1.1, ("sglang", "int8w8a8"): 2.3, ("sglang", "int4"): 1.3, ("sglang", "bf16"): 1.0,
           ("sglang", "int8"): 1.0}
T2_EFFECT = {("kv_cache_dtype", "fp8"): 1.24, ("kv_cache_dtype", "fp8_e4m3"): 1.12, ("max_num_seqs", 128): 0.97,
             ("max_num_batched_tokens", 1024): 1.06, ("max_num_batched_tokens", 4096): 0.95,
             ("attention_backend", "TRITON_ATTN"): 0.9, ("attention_backend", "FLEX_ATTENTION"): 0.8,
             ("cudagraph_mode", "NONE"): 0.7, ("chunked_prefill_size", 4096): 1.07, ("max_running_requests", 128): 1.04}
T1_BASE = {("vllm", "int4"): 61.0, ("vllm", "int8w8a8"): 40.0, ("vllm", "int8"): 38.0, ("vllm", "fp8"): 38.0,
           ("vllm", "bf16"): 23.0, ("sglang", "int4"): 66.0, ("sglang", "int8w8a8"): 42.0, ("sglang", "bf16"): 24.0,
           ("sglang", "int8"): 39.0}
T1_EFFECT = {("spec", "suffix"): 2.1, ("spec", "ngram"): 1.3, ("spec", "ngram_gpu"): 1.25, ("spec", "dflash2"): 1.9,
             ("spec_k", 8): 0.93, ("spec_k", 6): 1.06, ("cudagraph_mode", "NONE"): 0.4,
             ("attention_backend", "FLEX_ATTENTION"): 0.8, ("kv_cache_dtype", "fp8"): 0.95,
             ("performance_mode", "interactivity"): 1.04, ("block_size", 32): 1.02, ("torch_compile", True): 1.05}
T3_EFFECT = {("prefix_caching", False): 3.0, ("radix_cache", False): 3.0, ("kv_cache_dtype", "fp8"): 0.9,
             ("kv_offloading_gib", 64): 0.85, ("hicache_gib", 64): 0.85, ("max_num_batched_tokens", 4096): 0.92,
             ("chunked_prefill_size", 4096): 0.93}
QUAL = {"bf16": 0.95, "fp8": 0.945, "int8": 0.945, "int4": 0.93, "int8w8a8": 0.94}
BAD_PASSKEY = {"sglang/int8w8a8/fp8_e4m3/nospec"}  # one class fails the gate on purpose


def eff(cfg, table):
    v = 1.0
    for (k, val), f in table.items():
        if cfg.get(k) == val:
            v *= f
    return v


def lane_f(lane):
    return 1.0 if lane == "n0" else 1.01


def cap_true(cfg, chat=False):
    c = T2_BASE.get((cfg["engine"], cfg["variant"]), 1.0) * eff(cfg, T2_EFFECT)
    if chat:
        c *= 0.7
        # measured in main-v3's chat screen: speculation always on collapses capacity (-63% to -73%); switched
        # off above batch 16 ("cutoff") it ties the plain engine
        if cfg.get("spec") and cfg.get("spec_schedule") != "cutoff":
            c *= 0.3
    return c


def att(rate, cap):
    return max(0.0, min(1.0, 0.97 - 2.5 * max(0.0, rate - cap) / cap + rng.gauss(0, 0.012)))


def window(rate, cap, osl, out_tok):
    n = max(40, int(rate * 120))
    a = att(rate, cap)
    thr = rate * 0.9
    return {"dataset": "random", "osl": osl, "c": 1024, "n": n, "completed": n, "request_rate": rate,
            "request_throughput": thr, "request_goodput": thr * a, "total_output_tokens": n * out_tok,
            "p99_ttft_ms": 900 + 3000 * max(0, rate / cap - 0.9), "p99_tpot_ms": 60 + 80 * max(0, rate / cap - 0.9),
            "step_gpu_sm_mhz_mean_busy": 1300 + rng.gauss(0, 8)}


def ladder(rs, cap, out_tok):
    steps, lo, hi, r = [], None, None, rs["start"]
    while r <= rs["max_rate"]:
        s = window(r, cap, rs["step"]["osl"], out_tok)
        steps.append(s)
        if s["request_goodput"] / s["request_throughput"] >= rs["attainment"]:
            lo, r = r, r * rs["factor"]
        else:
            hi = r
            break
    if lo is not None and hi is not None:
        for _ in range(rs["bisect_iters"]):
            mid = (lo * hi) ** 0.5
            s = window(mid, cap, rs["step"]["osl"], out_tok)
            steps.append(s)
            if s["request_goodput"] / s["request_throughput"] >= rs["attainment"]:
                lo = mid
            else:
                hi = mid
    return steps


def fake_metrics(cfg, plan_name, lane):
    plan = plans.get(plan_name)
    key = space.numerics_key(cfg)
    m = {"steps": [], "resolved": {"attention_backend": cfg.get("attention_backend") or (
        ("FLASHINFER" if cfg.get("kv_cache_dtype") == "fp8" else "FLASH_ATTN") if cfg["engine"] == "vllm" else "flashinfer")}}
    if plan.get("driver") == "quality_driver.py":
        q = QUAL[cfg["variant"]] - (0.005 if cfg["kv_cache_dtype"] != "auto" else 0) + rng.gauss(0, 0.004)
        m["quality"] = {"gsm8k_acc": q, "passkey_acc": 0.67 if key in BAD_PASSKEY else 1.0, "thinking": True,
                        "nll_delta": 0.0 if cfg["variant"] == "bf16" else 0.02, "top1_agree": 0.97}
    elif plan.get("driver") == "t3_driver.py":
        p90 = 900 * eff(cfg, T3_EFFECT) * (1.3 if cfg["variant"] == "int4" else 1.0) * (1 + rng.gauss(0, 0.01))
        m["t3"] = {"ttft_p90_ms": p90, "ttft_p50_ms": p90 / 2, "errors": 0, "turns_done": 288, "turns_planned": 288}
    elif plan.get("profile"):
        m["profile"] = {"span_per_token_ms": 16.0, "busy_per_token_ms": 13.0, "idle_per_token_ms": 3.0,
                        "gpu_idle_frac": 0.19, "kernels_per_token": 700}
    elif plan.get("rate_search"):
        rs = plan["rate_search"]
        chat = rs["step"].get("dataset") == "sharegpt"
        m["steps"] = ladder(rs, cap_true(cfg, chat) * lane_f(lane), 700 if chat else rs["step"]["osl"])
    else:
        tps = T1_BASE.get((cfg["engine"], cfg["variant"]), 30.0) * eff(cfg, T1_EFFECT) * lane_f(lane)
        for st in plan["steps"]:
            if st.get("request_rate") or st.get("mix"):
                chat = st.get("dataset") == "sharegpt"
                m["steps"].append(window(st["request_rate"], cap_true(cfg, chat) * lane_f(lane), st.get("osl", 512),
                                         700 if chat else st.get("osl", 512)))
                m["steps"][-1]["n"] = m["steps"][-1]["completed"] = st["n"]
            elif st.get("isl") == 8192:
                m["steps"].append({"dataset": "random", "c": 1, "isl": 8192, "osl": 256, "n": 3, "completed": 3,
                                   "median_tpot_ms": 20, "median_ttft_ms": 2700 * (1.3 if cfg.get("attention_backend")
                                                                                   == "TRITON_ATTN" else 1)})
            else:
                t = tps * (1 + rng.gauss(0, 0.004))
                m["steps"].append({"dataset": st.get("dataset", "random"), "primary": bool(st.get("primary")), "c": 1,
                                   "isl": st.get("isl"), "osl": st["osl"], "n": st["n"], "completed": st["n"],
                                   "median_tpot_ms": 1000 / t, "median_ttft_ms": 45,
                                   "median_e2el_ms": 45 + st["osl"] * 0.8 * 1000 / t,
                                   "step_gpu_sm_mhz_mean_busy": 1300 + rng.gauss(0, 8)})
    m["scores"] = score.scores(m)
    return m


class ControllerKilled(Exception):
    """Simulates the controller pod being replaced mid-trial (the trial's ledger row stays 'running')."""


CRASH_AT = [None]  # trial sequence number at which the fake controller dies


def fake_run_many(jobs, campaign_name, lanes=("n0", "n1"), log=print):
    out = []
    for i, (cfg, plan_name, meta) in enumerate(jobs):
        lane = lanes[i % len(lanes)]
        cfg = space.full(cfg)
        plans.get(plan_name)  # every plan name the harness generates must resolve
        _seq[0] += 1
        tid = f"t{_seq[0]:08d}-{space.config_hash(cfg)[:6]}-{lane}"
        CALLS.append((campaign_name, plan_name))
        ledger.start(tid, campaign_name, plan_name, lane, "node", cfg, space.config_hash(cfg), None,
                     (meta or {}).get("move"), (meta or {}).get("notes"))
        if CRASH_AT[0] is not None and _seq[0] >= CRASH_AT[0]:
            CRASH_AT[0] = None
            raise ControllerKilled(tid)
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


def seed_previous_campaign():
    """A recent v2-style result in campaign `main` that main-v3's t2-screen may reuse (reuse_from)."""
    cfg = space.full({"engine": "vllm", "variant": "int8w8a8", "kv_cache_dtype": "fp8"})
    fake_run_many([(cfg, "t2s", {"notes": "v2 climb"})], "main:climb-t2-vllm")


def events(kind, like="%"):
    with ledger.conn() as c:
        return [(r["campaign"], json.loads(r["detail"])) for r in
                c.execute("SELECT campaign, detail FROM events WHERE kind=? AND campaign LIKE ? ORDER BY ts",
                          (kind, like))]


def screen_dupes(sub):
    """Challenger screens (not confirmations or incumbent runs) re-run although an identical screen had already
    completed (an aborted screen, e.g. the one a controller restart killed, must be re-run)."""
    seen, dupes = set(), []
    for r in ledger.rows("campaign=? AND notes IS NULL AND move IS NOT NULL", (sub,)):
        key = (space.config_hash(json.loads(r["config_json"])), r["plan"].split("@")[0])
        if key in seen:
            dupes.append(r["trial_id"])
        if r["state"] == "done":
            seen.add(key)
    return dupes


def main(path="campaigns/main-v3.json", crash_at=70):
    """crash_at: kill the fake controller when it starts this trial (mid climb-t1n-vllm by default), then
    restart the campaign; the restarted run must abort the orphan, resume the climb within its budget, and
    reuse screens instead of re-running them."""
    camp = json.loads(Path(path).read_text())
    name = camp["name"]
    seed_previous_campaign()
    t0 = time.time()
    buf = io.StringIO()
    CRASH_AT[0] = int(crash_at) if crash_at else None
    crashed = None
    with contextlib.redirect_stdout(buf):
        try:
            campaign.run(path)
        except ControllerKilled as e:
            crashed = str(e)
            print(f"=== controller killed while starting {crashed}; restarting ===")
            campaign.run(path)
    (TMP / "campaign.log").write_text(buf.getvalue())
    if crashed:
        r = ledger.rows("trial_id=?", (crashed,))[0]
        print(f"crash at {crashed} ({r['campaign']}): orphan state after restart = {r['state']}")
        assert r["state"] == "aborted", "the restarted controller must abort the orphaned trial"
        assert "resuming:" in buf.getvalue(), "the interrupted climb must resume with its spent budget"
    done = campaign.done_stages(name)
    want = [s["id"] for s in camp["stages"]]
    missing = [s for s in want if s not in done]
    print(f"stages done {len(done)}/{len(want)}; trials {_seq[0]}; {time.time() - t0:.1f}s; log {TMP / 'campaign.log'}")
    assert not missing, f"stages not done: {missing}"
    per_stage = {}
    for cn, _ in CALLS:
        per_stage[cn] = per_stage.get(cn, 0) + 1
    for st in camp["stages"]:
        sub = f"{name}:{st['id']}"
        acc = events("accept", sub)
        extra = ""
        if st["type"] == "climb":
            spent = len(ledger.rows("campaign=? AND state NOT IN ('invalid')", (sub,)))
            # a pass can overshoot by its last screen batch plus a confirmation pair and a quality gate run
            assert spent <= st["max_trials"] + 40, f"{sub}: {spent} trials for max_trials {st['max_trials']}"
            assert not screen_dupes(sub), f"{sub}: re-screened configs {screen_dupes(sub)}"
        if st["type"] == "climb":
            start = events("climb-start", sub)
            plateau = events("plateau", sub)
            budget = events("budget-exhausted", sub)
            assert len(plateau) + len(budget) == 1, f"{sub}: plateau {len(plateau)} / budget {len(budget)} events"
            cal = events("calibration", sub)
            extra = (f"start {space.canonical(start[0][1]['start']) if start else None}; accepts "
                     f"{[list(d['move']) if 'move' in d else d.get('knob') for _, d in acc]}; "
                     f"{'plateau' if plateau else 'budget'}{'; calibrated %.2f%%' % (100 * cal[0][1]['threshold']) if cal else ''}")
            assert start, f"{sub}: no climb-start"
        print(f"  {st['id']:20} {per_stage.get(sub, 0):4} trials  {extra}")
    # wiring checks
    reuse = events("reuse", f"{name}:t2-screen")
    assert reuse, "t2-screen should have reused the recent main-campaign ladder"
    s2 = events("climb-start", f"{name}:climb-t2-vllm")[0][1]["start"]
    assert s2["kv_cache_dtype"] == "fp8", "the reused W8A8+FP8-KV ladder is the best screen result -> climb start"
    s2s = events("climb-start", f"{name}:climb-t2-sglang")[0][1]["start"]
    assert s2s["kv_cache_dtype"] == "auto", "SGLang FP8 KV fails passkey -> must not be the climb start"
    t1s = [d["start"] for _, d in events("climb-start", f"{name}:climb-t1n-vllm")]
    assert all(space.config_hash(x) == space.config_hash(t1s[0]) for x in t1s[1:]) or len(t1s) > 1, t1s
    qr = events("quality-reject")
    print(f"quality rejects: {[(c.split(':')[-1], d.get('key')) for c, d in qr]}")
    assert not any(space.numerics_key(json.loads((climber.BEST_DIR / f).read_text())["config"]) in BAD_PASSKEY
                   for f in [p.name for p in climber.BEST_DIR.glob('*.json')]), "a quality-failing class won a climb"
    ver = [p for cn, p in CALLS if cn.startswith((f"{name}:verify", f"{name}:steady"))]
    print(f"verify plans: {sorted(set(p.split('@')[0] for p in ver))}")
    for fam in ("t2v", "t2m", "t2cv", "t2ss", "t2css", "t1v", "quality2"):
        assert any(p.split("@")[0] == fam for p in ver), f"verify stage never ran {fam}"
    cap256 = ledger.rows("campaign=?", (f"{name}:steady-t2c-cap256",))
    assert len(cap256) == 1 and cap256[0]["plan"].startswith("t2css@"), "the cap-256 chat config needs a steady run"
    assert space.full(json.loads(cap256[0]["config_json"]))["max_num_seqs"] == 256
    # T2 chat climbs never touch speculation (it can't add capacity; the "both" stages measure its value)
    for eng in ("vllm", "sglang"):
        sub = f"{name}:climb-t2c-{eng}"
        assert not [r for r in ledger.rows("campaign=?", (sub,)) if json.loads(r["config_json"]).get("spec")], \
            f"{sub} ran a speculative config"
    # "both" stages: the T2c vLLM winner + the T1n vLLM winner's speculation + cutoff (and the T1n winner + cutoff)
    t2c_final = space.full(campaign.climb_result(name, "climb-t2c-vllm")[0])
    t1n_final = space.full(campaign.climb_result(name, "climb-t1n-vllm")[0])
    for sid in ("both-t1", "both-t2c", "both-quality"):
        rows = ledger.rows("campaign=?", (f"{name}:{sid}",)) + [
            ledger.rows("trial_id=?", (d["trial"],))[0] for _, d in events("reuse", f"{name}:{sid}")]
        assert rows, f"{sid} measured nothing"
        cfgs = [space.full(json.loads(r["config_json"])) for r in rows]
        assert all(c["spec_schedule"] == "cutoff" and c["spec"] == t1n_final["spec"]
                   and c["spec_k"] == t1n_final["spec_k"] for c in cfgs), (sid, cfgs)
        assert any(c["variant"] == t2c_final["variant"] and c["kv_cache_dtype"] == t2c_final["kv_cache_dtype"]
                   for c in cfgs), (sid, "the T2c winner was not the base")
    # cross stages: one measurement per distinct winner config
    sources = {s["id"]: s["from_climbs"] for s in camp["stages"] if s["id"] in ("cross-t1", "cross-t2c")}
    for sid, srcs in sources.items():
        distinct = {space.config_hash(campaign.climb_result(name, x)[0]) for x in srcs}
        got = len(ledger.rows("campaign=?", (f"{name}:{sid}",))) + len(events("reuse", f"{name}:{sid}"))
        assert got == len(distinct), (sid, got, len(distinct))
    for sid in ("baseline-t2c", "baseline-t3"):
        assert ledger.rows("campaign=? AND state='done'", (f"{name}:{sid}",)), f"{sid} measured nothing"
    # the SGLang single-user winner re-measured on chat with more memory headroom (its n-gram path OOMed at 0.91)
    t1n_sg = space.full(campaign.climb_result(name, "climb-t1n-sglang")[0])
    mem88 = [space.full(json.loads(r["config_json"])) for r in ledger.rows("campaign=?", (f"{name}:cross-t2c-sglang-mem88",))]
    assert len(mem88) == 2 and all(c == dict(t1n_sg, mem_fraction_static=0.88) for c in mem88), mem88
    print("cross/both/baseline stages wired correctly; T2c climbs ran no speculative configs")
    abl = events("ablation")
    print(f"ablations: {[(d['climb'], d['reverted_scores']) for _, d in abl]}")
    print("CAMPAIGN SIMULATION OK", TMP)


if __name__ == "__main__":
    main(*sys.argv[1:])
