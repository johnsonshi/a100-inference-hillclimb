"""Coordinate-ascent hill climber with noise-aware, paired acceptance and a plateau stop rule (harness v3).

Score tracks (T1, T3): evaluate the incumbent's valid neighbours for each knob (track-specific order).
The best challenger whose gain over the incumbent *on the same lane* exceeds the threshold is re-run
back-to-back with the incumbent on that lane, and is accepted only if the *fresh* run alone still wins by
more than the threshold (the best-of-N screening value never enters the decision: no winner's curse).
With `calibrate`, the threshold is max(floor, 3 x the incumbent's run-to-run noise), pooled over all of its repeated
runs and re-estimated as confirmations add runs; a restarted climb resumes the estimate it was using.

Capacity tracks (T2 random-token, T2 chat): the incumbent's sustainable Poisson rate r_inc (>= ATTAINMENT
of requests meet the SLO) comes from a full rate search. Each challenger is screened with ONE probe at the
bar r_bar = r_inc x (1 + threshold). The best passer is confirmed on one lane by a fresh paired probe at the
bar (challenger must pass, incumbent must fail); its capacity is then measured with a fine ladder from the
bar upward. If the incumbent also passes the bar, r_inc was underestimated: the bar is raised and the
challenger is retried next pass. Score: offered rate x attainment x output tokens (no drain bias).

Every track:
- identical configs (canonical identity: non-default knobs) measured with the same plan family in the last
  12 h are reused, re-scored with the current scoring code (incumbent baselines and challenger screens), and
  the trial budget counts trials already run under the climb's sub-campaign, so a controller restart resumes
  a climb instead of re-screening it;
- a challenger that changes the numerics class (engine/weights/KV dtype/speculation) must pass the quality
  gate v2 (GSM8K with thinking, passkey) against the same engine's BF16 before it is accepted;
- moves that resolve to what the incumbent already runs (e.g. an explicit attention backend equal to the
  backend the default resolved to) are skipped;
- after a single-knob pass with no accept, a coupled pass tries known interacting pairs; then up to two
  guided kicks combine the best near-miss moves; then the climb stops (plateau)."""
import itertools
import json
import statistics
import time
from pathlib import Path

from . import ledger, plans, runner, score, space

TRACKS = {
    # T1 guard: reject a challenger whose 8k-prompt TTFT regresses >15% (the short-prompt TTFT is ~40 ms,
    # where a few ms of drafter overhead would trip a relative guard)
    "t1": {"kind": "score", "plan": "t1s", "metric": "t1_decode_tps", "guard": ("t1_ttft8k_ms", 1.15)},
    "t1n": {"kind": "score", "plan": "t1n", "metric": "t1_decode_tps", "guard": ("t1_ttft8k_ms", 1.15)},
    "t3": {"kind": "score", "plan": "t3", "metric": "t3_score", "guard": None},
    "t2": {"kind": "capacity", "family": "t2s", "metric": "t2_slo_tps"},
    "t2c": {"kind": "capacity", "family": "t2c", "metric": "t2_slo_tps"},
}
_T1_VLLM = ["variant", "spec", "spec_k", "performance_mode", "cudagraph_mode", "async_scheduling",
            "flashinfer_sampler", "attention_backend", "compile_sizes", "optimization_level", "kv_cache_dtype",
            "block_size"]
_T1_SGLANG = ["variant", "spec", "spec_k", "attention_backend", "cuda_graph_max_bs_decode", "prefill_cuda_graph",
              "torch_compile", "torch_compile_max_bs", "overlap_schedule", "num_continuous_decode_steps",
              "kv_cache_dtype", "alloc_expandable"]
KNOB_ORDER = {
    ("t1", "vllm"): _T1_VLLM, ("t1n", "vllm"): _T1_VLLM,
    ("t1", "sglang"): _T1_SGLANG, ("t1n", "sglang"): _T1_SGLANG,
    # T2 random-token prompts: speculative decoding acceptance is meaningless on random tokens (suffix k=4
    # measured 148.6 tok/s but k=6 69.1), so `spec` is only climbed in T2 chat (real text)
    ("t2", "vllm"): ["variant", "kv_cache_dtype", "max_num_seqs", "max_num_batched_tokens", "long_prefill_threshold",
                     "long_prefill_adaptive", "gpu_memory_utilization", "performance_mode", "attention_backend",
                     "cudagraph_mode", "async_scheduling", "api_server_count", "block_size", "optimization_level",
                     "flashinfer_sampler"],
    ("t2", "sglang"): ["variant", "kv_cache_dtype", "max_running_requests", "chunked_prefill_size",
                       "mem_fraction_static", "schedule_policy", "schedule_conservativeness", "attention_backend",
                       "cuda_graph_max_bs_decode", "prefill_cuda_graph", "overlap_schedule",
                       "num_continuous_decode_steps", "mixed_chunk", "page_size", "alloc_expandable", "torch_compile"],
    # T2 chat: no speculative decoding. The main-v3 screen showed it can't add capacity: always on it cost
    # 63-73% (vLLM suffix, SGLang DFlash2), and switched off above batch 16 (`cutoff`) it ties the plain engine,
    # because capacity is decided at batches of ~140. Its value (speed below capacity) is measured by the
    # campaign's "both" verify stages instead.
    ("t2c", "vllm"): ["variant", "kv_cache_dtype", "max_num_seqs", "max_num_batched_tokens", "long_prefill_threshold",
                      "long_prefill_adaptive", "performance_mode", "gpu_memory_utilization", "attention_backend",
                      "cudagraph_mode", "async_scheduling", "block_size"],
    ("t2c", "sglang"): ["variant", "kv_cache_dtype", "max_running_requests", "chunked_prefill_size",
                        "mem_fraction_static", "schedule_policy", "schedule_conservativeness", "attention_backend",
                        "cuda_graph_max_bs_decode", "overlap_schedule", "mixed_chunk"],
    ("t3", "vllm"): ["prefix_caching", "kv_cache_dtype", "kv_offloading_gib", "max_num_batched_tokens",
                     "long_prefill_threshold", "long_prefill_adaptive", "gpu_memory_utilization", "block_size",
                     "variant", "performance_mode"],
    ("t3", "sglang"): ["radix_cache", "kv_cache_dtype", "hicache_gib", "chunked_prefill_size", "schedule_policy",
                       "page_size", "mem_fraction_static", "variant"],
}
# interacting knobs, tried together when single-knob moves stall
COUPLED = {
    ("t1", "vllm"): [("spec", "spec_k"), ("performance_mode", "cudagraph_mode")],
    ("t1n", "vllm"): [("spec", "spec_k"), ("performance_mode", "cudagraph_mode")],
    ("t1", "sglang"): [("spec", "spec_k"), ("torch_compile", "torch_compile_max_bs")],
    ("t1n", "sglang"): [("spec", "spec_k"), ("torch_compile", "torch_compile_max_bs")],
    ("t2", "vllm"): [("kv_cache_dtype", "attention_backend"), ("max_num_seqs", "max_num_batched_tokens"),
                     ("max_num_batched_tokens", "long_prefill_threshold")],
    ("t2", "sglang"): [("max_running_requests", "chunked_prefill_size"), ("kv_cache_dtype", "attention_backend")],
    ("t2c", "vllm"): [("kv_cache_dtype", "attention_backend"), ("max_num_seqs", "max_num_batched_tokens"),
                      ("max_num_batched_tokens", "long_prefill_threshold")],
    ("t2c", "sglang"): [("max_running_requests", "chunked_prefill_size"), ("kv_cache_dtype", "attention_backend")],
    ("t3", "vllm"): [("kv_cache_dtype", "kv_offloading_gib"), ("max_num_batched_tokens", "long_prefill_threshold")],
    ("t3", "sglang"): [("kv_cache_dtype", "hicache_gib"), ("chunked_prefill_size", "schedule_policy")],
}
BEST_DIR = Path(__file__).resolve().parents[1] / "results" / "best"
REUSE_MAX_AGE_S = 12 * 3600
# Earlier top-level campaigns whose results may be reused, for plan families whose workload is unchanged
# (set from the campaign file's "reuse_from"; e.g. v3 reuses v2's random-token T2 ladders and probes).
REUSE_FROM = {}  # {campaign top-level: ([other top-levels], {plan families})}
QUALITY_TOL = {"gsm8k_acc": 0.03, "passkey_acc": 0.12}  # vs same-engine BF16 (n=200: SE ~1.5 pts; 1 of 9 passkeys)
MAX_KICKS = 2


def _val(m, key):
    return ((m or {}).get("scores") or {}).get(key)


def _lane(trial_id):
    return trial_id.rsplit("-", 1)[1] if trial_id else None


def _family(plan):
    return (plan or "").split("@", 1)[0]


def _rows_like(cfg, campaign, max_age_s=REUSE_MAX_AGE_S, states=("done",)):
    """Ledger rows of a config with the same canonical identity (recomputed: older rows were hashed with
    every knob), within the same top-level campaign."""
    c = space.full(cfg)
    top = campaign.split(":", 1)[0]
    target = space.config_hash(c)
    extra_tops, extra_fams = REUSE_FROM.get(top, ([], set()))
    out = []
    for t in [top] + list(extra_tops):
        q = (f"engine=? AND variant=? AND state IN ({','.join('?' * len(states))}) AND finished_at>? "
             "AND (campaign=? OR campaign LIKE ?)")
        for r in ledger.rows(q, (c["engine"], c["variant"], *states, time.time() - max_age_s, t, t + ":%")):
            if t != top and _family(r["plan"]) not in extra_fams:
                continue
            try:
                if space.config_hash(json.loads(r["config_json"])) == target:
                    out.append(r)
            except (ValueError, KeyError):
                continue
    return out


def _recent(cfg, plan_families, campaign):
    """Done trials of an identical config whose plan family is in `plan_families`, finished in the last
    REUSE_MAX_AGE_S within the same top-level campaign, re-scored with the current scoring code."""
    out = []
    for r in _rows_like(cfg, campaign):
        if _family(r["plan"]) not in plan_families:
            continue
        try:
            m = json.loads(r["metrics_json"] or "{}")
            m["scores"] = score.scores(m)
        except (ValueError, TypeError, KeyError):
            continue
        out.append((r["trial_id"], r["lane"], r["plan"], m))
    return out


def _spent(campaign):
    """Trials this climb already ran under its sub-campaign (before a controller restart, too), so a restarted
    climb continues with the remaining budget instead of a fresh one."""
    return len(ledger.rows("campaign=? AND state!='invalid'", (campaign,)))


def _last_calibration(campaign):
    """The climb's latest noise estimate (a calibration event), which a restarted climb resumes."""
    with ledger.conn() as c:
        r = c.execute("SELECT detail FROM events WHERE campaign=? AND kind='calibration' ORDER BY ts DESC LIMIT 1",
                      (campaign,)).fetchone()
    return json.loads(r["detail"]) if r else None


def _end(campaign, used, max_trials, log, **detail):
    """Record why a climb stopped when it ran out of budget rather than reaching a plateau."""
    if used >= max_trials:
        log(f"BUDGET exhausted after {used} trials (max {max_trials})")
        ledger.event(campaign, "budget-exhausted", {"used": used, "max_trials": max_trials, **detail})


def _save_best(campaign, track, metric, value, cfg, **extra):
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    (BEST_DIR / f"{campaign}.json").write_text(json.dumps({"track": track, "metric": metric, "score": value,
                                                          "config": cfg, **extra}, indent=1))


def _primary_clock(m):
    steps = (m or {}).get("steps") or []
    prim = [x for x in steps if x.get("primary")] or steps
    return prim[0].get("step_gpu_sm_mhz_mean_busy") if prim else None


def _clock_check(campaign, knob, cm, im, log):
    a, b = _primary_clock(cm), _primary_clock(im)
    if a and b and abs(a - b) / b > 0.05:
        log(f"  clock covariate: challenger {a:.0f} MHz vs incumbent {b:.0f} MHz ({a / b - 1:+.1%})")
        ledger.event(campaign, "clock-mismatch", {"knob": knob, "challenger_mhz": a, "incumbent_mhz": b})


def _noop(inc_resolved, knob, value):
    """True if the move asks for what the incumbent already resolves to (e.g. its default backend)."""
    if knob == "attention_backend" and value and inc_resolved.get("attention_backend"):
        return str(value).lower() == str(inc_resolved["attention_backend"]).lower()
    return False


def _fill_resolved(inc_resolved, cfg, campaign):
    """Add what `cfg` resolved to in its other recent trials of this campaign, keeping what the incumbent's own
    measurements recorded. A long rate search can push the server's startup lines (attention backend, kernels)
    out of the captured log tail, so one trial alone may lack them (and miss a no-op move)."""
    for r in _rows_like(cfg, campaign):
        try:
            res = json.loads(r["metrics_json"] or "{}").get("resolved") or {}
        except (ValueError, TypeError):
            continue
        for k, v in res.items():
            inc_resolved.setdefault(k, v)


# ------------------------------------------------------------------ quality gate v2
def _quality_runs(key, campaign):
    out = []
    top = campaign.split(":", 1)[0]
    for r in ledger.rows("state='done' AND (plan='quality2' OR plan='quality2-refs') AND (campaign=? OR campaign LIKE ?)",
                         (top, top + ":%")):
        try:
            if space.numerics_key(json.loads(r["config_json"])) == key:
                out.append(json.loads(r["scores_json"] or "{}"))
        except (ValueError, KeyError):
            continue
    return out


def quality_verdict(cfg, campaign):
    """(ok, detail) from existing quality2 results of the config's numerics class; ok is None if it has none."""
    c = space.full(cfg)
    runs = _quality_runs(space.numerics_key(c), campaign)
    if not runs:
        return None, {"key": space.numerics_key(c)}
    return _quality_compare(c, campaign, runs)


def quality_gate(cfg, campaign, lane, log=print):
    """(ok, detail, trials_used). Runs quality2 for a new numerics class if the ledger has none."""
    c = space.full(cfg)
    key = space.numerics_key(c)
    used = 0
    runs = _quality_runs(key, campaign)
    if not runs:
        log(f"  quality gate: no quality2 result for {key}; running it on {lane}")
        tid, st, m = runner.run_many([(c, "quality2", {"notes": "quality-gate"})], campaign, lanes=(lane,), log=log)[0]
        used = 1
        runs = [m["scores"]] if st == "done" and m else []
    if not runs:
        return False, {"key": key, "reason": "quality2 run failed"}, used
    ok, detail = _quality_compare(c, campaign, runs)
    return ok, detail, used


def _quality_compare(c, campaign, runs):
    key = space.numerics_key(c)
    ref_key = space.numerics_key({"engine": c["engine"], "variant": "bf16"})
    refs = _quality_runs(ref_key, campaign) or _quality_runs("vllm/bf16/auto/nospec", campaign)
    got = {k: statistics.mean(r[k] for r in runs if r.get(k) is not None) for k in QUALITY_TOL
           if any(r.get(k) is not None for r in runs)}
    if not refs:
        return True, {"key": key, "got": got, "warning": "no BF16 reference; gate not enforced"}
    ref = {k: statistics.mean(r[k] for r in refs if r.get(k) is not None) for k in QUALITY_TOL
           if any(r.get(k) is not None for r in refs)}
    fails = {k: (got[k], ref[k]) for k in QUALITY_TOL if k in got and k in ref and got[k] < ref[k] - QUALITY_TOL[k]}
    return not fails, {"key": key, "got": got, "ref": ref, "fails": fails}


def _known_quality_fail(c, inc, campaign, log):
    """True if `c` changes the numerics class to one whose existing quality2 results fail the gate."""
    if space.numerics_key(c) == space.numerics_key(inc):
        return False
    ok, detail = quality_verdict(c, campaign)
    if ok is False:
        log(f"  {space.numerics_key(c)}: skipped, fails the quality gate {detail.get('fails')}")
        return True
    return False


def climb(start, track, campaign, threshold=0.03, max_trials=80, knob_order=None, lanes=("n0", "n1"), log=print,
          calibrate=False, coupled=True, kicks=True, gate=True):
    inc = space.full(start)
    order = knob_order or KNOB_ORDER[(track, inc["engine"])]
    opts = dict(coupled=coupled, kicks=kicks, gate=gate)
    if TRACKS[track]["kind"] == "capacity":
        return _climb_capacity(inc, track, campaign, threshold, max_trials, order, tuple(lanes), log, **opts)
    return _climb_score(inc, track, campaign, threshold, max_trials, order, tuple(lanes), log, calibrate=calibrate, **opts)


COUPLED_PRUNE = -0.15  # skip pairs containing a move that alone lost >15% (score) / failed badly (capacity)


def _coupled_candidates(inc, track, tried, single=None):
    """Untried valid pairs of the track's interacting knobs. Each knob's values are taken before the validity
    check and the *pair* is validated, so dependent moves are reachable (a speculation schedule or draft
    length is only valid together with a speculation method). `single` maps (knob, value) to that move's own
    screened result (relative gain, or attainment - target for capacity tracks); pairs containing a clear loser
    are pruned, except when the partner is a dependent knob that has no valid move of its own: it may be
    exactly the fix (e.g. a schedule that switches a collapsing speculation method off under load)."""
    out = []
    single = single or {}
    for a, b in COUPLED.get((track, inc["engine"]), []):
        if a not in inc or b not in inc:
            continue
        va, vb = space.neighbor_values(inc, a), space.neighbor_values(inc, b)
        dependent = not space.neighbors(inc, a) or not space.neighbors(inc, b)
        for x, y in itertools.product(va, vb):
            if not dependent and any(single.get((k, repr(v)), 0) < COUPLED_PRUNE for k, v in ((a, x), (b, y))):
                continue
            c = dict(inc, **{a: x, b: y})
            if space.validity(c) is None and space.config_hash(c) not in tried:
                out.append(((a, b), c))
    return out


def _combine(inc, moves):
    """Apply several (knob, value) near-miss moves at once (later moves win on the same knob)."""
    c = dict(inc)
    for knob, value in moves:
        c[knob] = value
    return c if space.validity(c) is None else None


def _climb_score(inc, track, campaign, threshold, max_trials, order, lanes, log, calibrate=False, coupled=True,
                 kicks=True, gate=True):
    tr = TRACKS[track]
    plan, metric, guard = tr["plan"], tr["metric"], tr["guard"]
    used = _spent(campaign)
    if used:
        log(f"resuming: {used} of {max_trials} trials already used by this climb")
    inc_scores, inc_guard, inc_resolved = {}, {}, {}
    near = []  # (gain, knob, value) of screened moves that improved but not by the threshold
    single = {}  # (knob, repr(value)) -> screened relative gain of that single move vs the current incumbent

    def note(tid, m):
        ln = _lane(tid)
        inc_scores.setdefault(ln, []).append(_val(m, metric))
        if guard and _val(m, guard[0]) is not None:
            inc_guard.setdefault(ln, []).append(_val(m, guard[0]))
        inc_resolved.update((m or {}).get("resolved") or {})

    def measure_incumbent(on_lanes, why):
        nonlocal used
        res = runner.run_many([(inc, plan, {"notes": why}) for _ in on_lanes], campaign, lanes=on_lanes, log=log)
        used += len(on_lanes)
        for tid, st, m in res:
            if st == "done" and _val(m, metric):
                note(tid, m)

    def guard_ok(m, ln):
        if not guard or ln not in inc_guard:
            return True
        g = _val(m, guard[0])
        return g is None or g <= guard[1] * statistics.mean(inc_guard[ln])

    floor, cal = threshold, {"dof": 0, "key": None}

    def recalibrate():
        """threshold = max(floor, 3 sigma), sigma pooled from the relative within-lane deviations of every run of
        the current incumbent. Re-estimated whenever a confirmation adds an incumbent run, so a decision uses the
        estimate a restarted controller would compute from the same runs (a restart reloads all recent incumbent
        runs and resumes the estimate in use). An estimate with fewer degrees of freedom than the one in use (a
        new incumbent's first runs) does not replace it."""
        nonlocal threshold
        groups = [vs for vs in inc_scores.values() if len(vs) >= 2]
        n = sum(len(vs) for vs in groups)
        dof = n - len(groups)
        key = (space.config_hash(inc), n)
        if dof < max(2, cal["dof"]) or key == cal["key"]:
            return
        ss = sum(((v - statistics.mean(vs)) / statistics.mean(vs)) ** 2 for vs in groups for v in vs)
        sigma = (ss / dof) ** 0.5
        new_thr = max(floor, 3 * sigma)
        log(f"{'re' if cal['key'] else ''}calibration: run-to-run noise {sigma:.2%} ({n} runs, {dof} dof) -> "
            f"threshold {new_thr:.2%} (floor {floor:.2%})")
        ledger.event(campaign, "calibration", {"sigma_rel": sigma, "threshold": new_thr, "floor": floor, "dof": dof,
                                               "runs": {k: v for k, v in inc_scores.items()}})
        cal.update(dof=dof, key=key)
        threshold = new_thr

    ledger.event(campaign, "climb-start", {"track": track, "start": inc, "threshold": threshold, "order": order})
    for tid, ln, _, m in _recent(inc, {plan}, campaign):
        if ln in lanes and _val(m, metric):
            note(tid, m)
    if inc_scores:
        log(f"reusing recent incumbent runs: { {k: [round(v, 2) for v in vs] for k, vs in inc_scores.items()} }")
    missing = [ln for ln in lanes if ln not in inc_scores]
    if missing:
        measure_incumbent(missing, "incumbent-baseline")
    if calibrate:
        last = _last_calibration(campaign)
        if last and last.get("dof"):
            # a restarted climb resumes the estimate it was using (after an accept, the new incumbent's few runs
            # would otherwise replace it), so it needs no calibration runs
            threshold = max(floor, 3 * last["sigma_rel"])
            cal["dof"] = last["dof"]
            log(f"calibration: resuming noise {last['sigma_rel']:.2%} ({last['dof']} dof) -> threshold {threshold:.2%}")
            ledger.event(campaign, "calibration", {**last, "threshold": threshold, "floor": floor, "resumed": True})
        else:
            short = [ln for ln in lanes if len(inc_scores.get(ln, [])) < 2]
            if short:
                measure_incumbent(short, "calibration")
        recalibrate()
    if not inc_scores:
        ledger.event(campaign, "abort", {"reason": "baseline failed"})
        log("baseline failed; aborting")
        return inc
    tried = {space.config_hash(inc)}

    def evaluate(cands, label):
        """Screen [(moves, cfg)], confirm the best on fresh paired runs; returns True if accepted."""
        nonlocal inc, used, inc_scores, inc_guard
        cands = [(mv, c) for mv, c in cands if space.config_hash(c) not in tried]
        for _, c in cands:
            tried.add(space.config_hash(c))
        if gate:
            cands = [(mv, c) for mv, c in cands if not _known_quality_fail(c, inc, campaign, log)]
        if not cands:
            return False
        # an identical challenger measured with this plan in the last 12 h (e.g. before a controller restart) is
        # reused as its screen; acceptance still needs the fresh paired confirmation below
        res, todo = [None] * len(cands), []
        for i, (mv, c) in enumerate(cands):
            prev = [x for x in _recent(c, {plan}, campaign) if x[1] in inc_scores and _val(x[3], metric)]
            if prev:
                tid, ln, _, m = prev[-1]
                log(f"  {','.join(f'{k}={c[k]!r}' for k in mv)}: reusing recent result {tid}")
                res[i] = (tid, "done", m)
            else:
                todo.append(i)
        if todo:
            jobs = [(cands[i][1], plan, {"move": {k: [inc[k], cands[i][1][k]] for k in cands[i][0]}}) for i in todo]
            for i, r in zip(todo, runner.run_many(jobs, campaign, lanes=lanes, log=log)):
                res[i] = r
            used += len(jobs)
        best = None
        for (mv, c), (tid, st, m) in zip(cands, res):
            ln, v = _lane(tid), _val(m, metric)
            if st != "done" or v is None or ln not in inc_scores:
                continue
            base = statistics.mean(inc_scores[ln])
            gain = (v - base) / base
            desc = ",".join(f"{k}={c[k]!r}" for k in mv)
            if not guard_ok(m, ln):
                log(f"  guard: {desc} {guard[0]} {_val(m, guard[0]):.0f} regressed")
                continue
            log(f"  {desc}: {metric} {v:.2f} vs {base:.2f} ({gain:+.1%}) on {ln}")
            if len(mv) == 1:
                single[(mv[0], repr(c[mv[0]]))] = gain
            if 0 < gain <= threshold and len(mv) == 1:
                near.append((gain, mv[0], c[mv[0]]))
            if gain > threshold and (best is None or gain > best[0]):
                best = (gain, mv, c, tid, ln, v)
        if not best:
            return False
        gain, mv, c, tid, ln, v = best
        move = {k: [inc[k], c[k]] for k in mv}
        (ctid, cst, cm), (itid, ist, im_) = runner.run_many(
            [(c, plan, {"notes": "confirm-challenger", "move": move}),
             (inc, plan, {"notes": "confirm-incumbent"})], campaign, lanes=(ln,), log=log)
        used += 2
        if ist == "done" and _val(im_, metric):
            note(itid, im_)
            if calibrate:
                recalibrate()
        fresh = _val(cm, metric) if cst == "done" else None
        inc_mean = statistics.mean(inc_scores[ln])
        fresh_gain = (fresh - inc_mean) / inc_mean if fresh is not None else None
        if cst == "done" and ist == "done":
            _clock_check(campaign, label, cm, im_, log)
        if not (fresh_gain is not None and fresh_gain > threshold and guard_ok(cm, ln)):
            log(f"reject after confirm {move}: fresh {fresh} vs incumbent {inc_mean:.2f}")
            ledger.event(campaign, "reject-confirm", {"move": move, "screen": v, "fresh": fresh,
                                                      "incumbent": inc_scores[ln]})
            return False
        if gate and space.numerics_key(c) != space.numerics_key(inc):
            ok, detail, q_used = quality_gate(c, campaign, ln, log)
            used += q_used
            if not ok:
                log(f"quality gate FAILED for {move}: {detail}")
                ledger.event(campaign, "quality-reject", {"move": move, **detail})
                return False
            ledger.event(campaign, "quality-pass", {"move": move, **detail})
        log(f"ACCEPT {move}  {metric} {inc_mean:.2f} -> {fresh:.2f} ({fresh_gain:+.1%} on the fresh run; "
            f"screen {gain:+.1%}; threshold {threshold:.1%})")
        ledger.event(campaign, "accept", {"knob": label, "move": move, "before": inc_mean, "after": fresh,
                                          "screen_gain": gain, "threshold": threshold, "lane": ln, "trial": ctid,
                                          "screen_trial": tid})
        inc = c
        inc_scores, inc_guard = {}, {}
        inc_resolved.clear()
        single.clear()
        note(ctid, cm)
        _fill_resolved(inc_resolved, inc, campaign)
        _save_best(campaign, track, metric, fresh, inc)
        others = [x for x in lanes if x != ln]
        if others:
            measure_incumbent(others, "incumbent-remeasure")
        return True

    _fill_resolved(inc_resolved, inc, campaign)
    passes, kicks_done, stop = 0, 0, "budget"
    while used < max_trials:
        accepted = 0
        for knob in order:
            if used >= max_trials:
                break
            cands = []
            for c in space.neighbors(inc, knob):
                if _noop(inc_resolved, knob, c[knob]):
                    log(f"  {knob}={c[knob]!r}: no-op (the incumbent already resolves to it)")
                    tried.add(space.config_hash(c))
                    continue
                cands.append(((knob,), c))
            accepted += evaluate(cands, knob)
        passes += 1
        ledger.event(campaign, "pass-done", {"pass": passes, "accepted": accepted, "used": used})
        if accepted:
            continue
        if used >= max_trials:
            break  # the pass was cut short by the budget: not a plateau
        if coupled and used < max_trials:
            cc = _coupled_candidates(inc, track, tried, single)
            if cc:
                log(f"coupled pass: {len(cc)} paired moves")
                if evaluate([(pair, c) for pair, c in cc], "coupled"):
                    continue
        if kicks and kicks_done < MAX_KICKS and used < max_trials:
            picks, seen = [], set()
            for g, knob, value in sorted(near, reverse=True):
                if knob not in seen and inc.get(knob) != value:
                    picks.append((knob, value))
                    seen.add(knob)
            c = _combine(inc, picks[:3]) if len(picks) >= 2 else None
            kicks_done += 1
            if c is not None:
                log(f"guided kick {kicks_done}: {picks[:3]}")
                ledger.event(campaign, "kick", {"moves": picks[:3]})
                near.clear()
                if evaluate([(tuple(k for k, _ in picks[:3]), c)], "kick"):
                    continue
        if used >= max_trials:
            break  # coupled moves or kicks were cut short by the budget
        log(f"PLATEAU after {passes} passes, {used} trials. Best: {inc_scores}")
        ledger.event(campaign, "plateau", {"passes": passes, "used": used, "config": inc, "threshold": threshold})
        stop = "plateau"
        break
    if stop == "budget":
        _end(campaign, used, max_trials, log, config=inc)
    return inc


# ------------------------------------------------------------------ capacity climbs (T2)
def _probe_att(m, rate):
    """SLO attainment of the Poisson step that ran at `rate` (None if no such completed step)."""
    for r, a, _ in score.t2_points(m or {}):
        if abs(r - rate) <= 1e-3 * max(rate, 1.0):
            return a
    return None


def _t2_known(cfg, r_bar, campaign, families=("t2s", "t2probe")):
    """From recent identical-config results: (True, attainment, lane) if some step sustained >= r_bar,
    (False, None, None) if some step at <= r_bar missed the SLO, else (None, None, None)."""
    verdict = (None, None, None)
    # plan names round probe rates to 4 decimals, so compare with a 0.1% relative tolerance
    for tid, ln, _, m in _recent(cfg, set(families), campaign):
        for r, a, _ in score.t2_points(m):
            if a >= plans.ATTAINMENT and r >= r_bar * (1 - 1e-3):
                return True, a, ln
            if a < plans.ATTAINMENT and r <= r_bar * (1 + 1e-3):
                verdict = (False, None, None)
    return verdict


def _capacity_of(m):
    return _val(m, "t2_capacity") or _val(m, "t2_max_rate")


def _climb_capacity(inc, track, campaign, threshold, max_trials, order, lanes, log, coupled=True, kicks=True,
                    gate=True):
    tr = TRACKS[track]
    fam, metric = tr["family"], tr["metric"]
    probe_prefix, fine_prefix, _ = plans.CAPACITY_FAMILIES[fam]
    rs = plans.PLANS[fam]["rate_search"]
    osl = rs["step"]["osl"]
    families = (fam, probe_prefix)
    used = _spent(campaign)
    if used:
        log(f"resuming: {used} of {max_trials} trials already used by this climb")
    inc_resolved = {}
    near = []  # (attainment at the bar, knob, value) of challengers that came close
    raised = [0]  # bar raises in the current pass (their challengers are retried in another pass)
    single = {}  # (knob, repr(value)) -> attainment at the bar minus the target, for coupled-pair pruning

    def probe(r):
        return f"{probe_prefix}@{r:.4f}"

    ledger.event(campaign, "climb-start", {"track": track, "start": inc, "threshold": threshold, "order": order,
                                           "mode": "capacity", "family": fam})
    caps = {}
    for tid, ln, p, m in _recent(inc, {fam}, campaign):
        if _capacity_of(m):
            caps.setdefault(ln, []).append(_capacity_of(m))
            inc_resolved.update(m.get("resolved") or {})
    if caps:
        log(f"reusing recent incumbent capacity (req/s): {caps}")
    else:
        res = runner.run_many([(inc, fam, {"notes": "incumbent-baseline"}) for _ in lanes], campaign,
                              lanes=lanes, log=log)
        used += len(lanes)
        for tid, st, m in res:
            if st == "done" and _capacity_of(m):
                caps.setdefault(_lane(tid), []).append(_capacity_of(m))
                inc_resolved.update(m.get("resolved") or {})
    if not caps:
        ledger.event(campaign, "abort", {"reason": "baseline failed"})
        log("baseline failed; aborting")
        return inc
    r_inc = statistics.mean(v for vs in caps.values() for v in vs)
    ledger.event(campaign, "capacity", {"incumbent": r_inc, "per_lane": caps})
    log(f"incumbent capacity {r_inc:.3f} req/s; a challenger must sustain x{1 + threshold:.2f}")
    tried = {space.config_hash(inc)}
    _fill_resolved(inc_resolved, inc, campaign)

    def evaluate(cands, label):
        nonlocal inc, r_inc, used
        cands = [(mv, c) for mv, c in cands if space.config_hash(c) not in tried]
        for _, c in cands:
            tried.add(space.config_hash(c))
        if gate:
            cands = [(mv, c) for mv, c in cands if not _known_quality_fail(c, inc, campaign, log)]
        if not cands:
            return False
        r_bar = r_inc * (1 + threshold)
        passers, jobs = [], []
        for mv, c in cands:
            desc = ",".join(f"{k}={c[k]!r}" for k in mv)
            known, att, ln = _t2_known(c, r_bar, campaign, families)
            if known is True:
                log(f"  {desc}: already sustained >= {r_bar:.3f} req/s (reused)")
                passers.append((att, mv, c, None, ln))
            elif known is False:
                log(f"  {desc}: already missed the SLO at <= {r_bar:.3f} req/s (reused)")
            else:
                jobs.append((mv, c))
        if jobs:
            res = runner.run_many([(c, probe(r_bar), {"move": {k: [inc[k], c[k]] for k in mv}, "notes": "screen"})
                                   for mv, c in jobs], campaign, lanes=lanes, log=log)
            used += len(jobs)
            for (mv, c), (tid, st, m) in zip(jobs, res):
                att = _probe_att(m, r_bar) if st == "done" else None
                desc = ",".join(f"{k}={c[k]!r}" for k in mv)
                log(f"  {desc}: attainment {'n/a' if att is None else f'{att:.3f}'} at {r_bar:.3f} req/s on {_lane(tid)}")
                if att is not None and len(mv) == 1:
                    single[(mv[0], repr(c[mv[0]]))] = att - plans.ATTAINMENT
                if att is not None and att >= plans.ATTAINMENT:
                    passers.append((att, mv, c, tid, _lane(tid)))
                elif att is not None and len(mv) == 1 and att >= plans.ATTAINMENT - 0.15:
                    near.append((att, mv[0], c[mv[0]]))
        if not passers:
            return False
        att, mv, c, tid, ln = max(passers, key=lambda x: x[0])
        ln = ln if ln in lanes else lanes[0]
        move = {k: [inc[k], c[k]] for k in mv}
        (ctid, cst, cm), (itid, ist, im_) = runner.run_many(
            [(c, probe(r_bar), {"notes": "confirm-challenger", "move": move}),
             (inc, probe(r_bar), {"notes": "confirm-incumbent"})], campaign, lanes=(ln,), log=log)
        used += 2
        c_att = _probe_att(cm, r_bar) if cst == "done" else None
        i_att = _probe_att(im_, r_bar) if ist == "done" else None
        if cst == "done" and ist == "done":
            _clock_check(campaign, label, cm, im_, log)
        c_pass = c_att is not None and c_att >= plans.ATTAINMENT
        i_fail = i_att is not None and i_att < plans.ATTAINMENT
        detail = {"knob": label, "move": move, "bar": r_bar, "screen_att": att, "fresh_att": c_att,
                  "incumbent_att": i_att, "lane": ln, "trial": ctid, "incumbent_trial": itid, "screen_trial": tid}
        if c_pass and i_att is not None and not i_fail:
            log(f"bar too low: the incumbent also sustains {r_bar:.3f} req/s; raising its capacity estimate "
                f"({move} is retried next pass)")
            ledger.event(campaign, "bar-raised", {**detail, "before": r_inc, "after": r_bar})
            r_inc = r_bar
            tried.discard(space.config_hash(c))
            raised[0] += 1
            return False
        if not (c_pass and i_fail):
            log(f"reject after confirm {move}: fresh attainment {c_att} at {r_bar:.3f} req/s (incumbent {i_att})")
            ledger.event(campaign, "reject-confirm", detail)
            return False
        if gate and space.numerics_key(c) != space.numerics_key(inc):
            ok, qd, q_used = quality_gate(c, campaign, ln, log)
            used += q_used
            if not ok:
                log(f"quality gate FAILED for {move}: {qd}")
                ledger.event(campaign, "quality-reject", {**detail, **qd})
                return False
            ledger.event(campaign, "quality-pass", {"move": move, **qd})
        ltid, lst, lm = runner.run_many([(c, f"{fine_prefix}@{r_bar:.4f}", {"notes": "new-incumbent-capacity",
                                                                             "move": move})],
                                        campaign, lanes=(ln,), log=log)[0]
        used += 1
        r_new = max((_capacity_of(lm) if lst == "done" else None) or r_bar, r_bar)
        slo = (_val(lm, metric) if lst == "done" else None) or r_bar * c_att * osl
        log(f"ACCEPT {move}  capacity {r_inc:.3f} -> {r_new:.3f} req/s ({r_new / r_inc - 1:+.1%}); {metric} {slo:.0f}")
        ledger.event(campaign, "accept", {**detail, "before": r_inc, "after": r_new, "slo_tps": slo,
                                          "capacity_trial": ltid})
        inc, r_inc = c, r_new
        inc_resolved.clear()
        single.clear()
        for mm in (cm, lm):
            inc_resolved.update((mm or {}).get("resolved") or {})
        _fill_resolved(inc_resolved, inc, campaign)
        _save_best(campaign, track, metric, slo, inc, capacity_req_s=r_new)
        return True

    passes, kicks_done, stop = 0, 0, "budget"
    while used < max_trials:
        accepted, raised[0] = 0, 0
        for knob in order:
            if used >= max_trials:
                break
            cands = []
            for c in space.neighbors(inc, knob):
                if _noop(inc_resolved, knob, c[knob]):
                    log(f"  {knob}={c[knob]!r}: no-op (the incumbent already resolves to it)")
                    tried.add(space.config_hash(c))
                    continue
                cands.append(((knob,), c))
            accepted += evaluate(cands, knob)
        passes += 1
        ledger.event(campaign, "pass-done", {"pass": passes, "accepted": accepted, "used": used, "capacity": r_inc,
                                             "bar_raises": raised[0]})
        if accepted or raised[0]:
            continue  # a raised bar leaves its challenger untried: re-screen it at the new bar
        if used >= max_trials:
            break  # the pass was cut short by the budget: not a plateau
        if coupled and used < max_trials:
            cc = _coupled_candidates(inc, track, tried, single)
            if cc:
                log(f"coupled pass: {len(cc)} paired moves")
                if evaluate([(pair, c) for pair, c in cc], "coupled") or raised[0]:
                    continue
        if kicks and kicks_done < MAX_KICKS and used < max_trials:
            picks, seen = [], set()
            for a, knob, value in sorted(near, reverse=True):
                if knob not in seen and inc.get(knob) != value:
                    picks.append((knob, value))
                    seen.add(knob)
            c = _combine(inc, picks[:3]) if len(picks) >= 2 else None
            kicks_done += 1
            if c is not None:
                log(f"guided kick {kicks_done}: {picks[:3]}")
                ledger.event(campaign, "kick", {"moves": picks[:3]})
                near.clear()
                if evaluate([(tuple(k for k, _ in picks[:3]), c)], "kick"):
                    continue
        if used >= max_trials:
            break  # coupled moves or kicks were cut short by the budget
        log(f"PLATEAU after {passes} passes, {used} trials. Capacity {r_inc:.3f} req/s")
        ledger.event(campaign, "plateau", {"passes": passes, "used": used, "config": inc, "capacity": r_inc})
        stop = "plateau"
        break
    if stop == "budget":
        _end(campaign, used, max_trials, log, config=inc, capacity=r_inc)
    return inc
