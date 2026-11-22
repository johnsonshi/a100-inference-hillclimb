"""Run a campaign: an ordered list of stages from a JSON file. Resumable: a stage recorded as done in the
ledger is skipped after a controller restart; a climb restarts from its best-so-far config and reuses
recent identical results. Usage: python -m harness.campaign campaigns/main-v3.json

Stage types:
- batch:    run `configs` with `plan` (with "reuse": true, configs with a recent identical result are skipped)
- climb:    climber.climb(track, start_from best of an earlier stage, threshold, max_trials, calibrate, ...)
- verify:   re-measure climb winners (`from_climbs`) and `configs` with `plan`; a plan ending in "@capacity"
            is filled with each config's measured sustainable rate; `repeat` runs per config. Optional
            `graft` {"climb": id, "knobs": [...]} copies those knobs from another climb's winner, then `override`
            sets knobs (e.g. the best-capacity config with the best single-user config's speculation and a
            batch-size cutoff). Identical configs are measured once; `reuse` adopts recent identical results.
- ablation: revert each knob a climb changed, one at a time, and measure what it was worth (`start_climb` takes
            the start config from another climb, e.g. the one a later climb stage continued)"""
import json
import sys
import time
from pathlib import Path

from . import climber, ledger, plans, runner, score, space


def done_stages(name):
    with ledger.conn() as c:
        rows = c.execute("SELECT detail FROM events WHERE campaign=? AND kind='stage-done'", (name,)).fetchall()
    return {json.loads(r["detail"])["id"] for r in rows}


START_TIE_TOL = 0.02  # screen results within 2% of the best count as a tie (T2 rungs are >=2.6% apart)


def best_config(campaign_prefix, metric, engine=None, plan=None, log=print, gate_campaign=None, lane="n0",
                tie_tol=START_TIE_TOL):
    """Best done trial (by metric, re-scored with the current code) among campaigns starting with the prefix,
    including trials a reuse-batch stage adopted instead of re-running. Configs whose numerics class fails
    the quality gate are skipped; with `gate_campaign`, a class without quality2 results is gated first
    (one quality2 run, recorded under that campaign)."""
    where = "campaign LIKE ? AND state='done' AND metrics_json IS NOT NULL"
    args = [campaign_prefix + "%"]
    if engine:
        where += " AND engine=?"
        args.append(engine)
    if plan:
        where += " AND (plan=? OR plan LIKE ?)"
        args += [plan, plan + "@%"]
    rows = list(ledger.rows(where, tuple(args)))
    with ledger.conn() as c:
        reused = [json.loads(r["detail"])["trial"] for r in
                  c.execute("SELECT detail FROM events WHERE campaign LIKE ? AND kind='reuse'", (campaign_prefix + "%",))]
    for tid in reused:
        rows += [r for r in ledger.rows("trial_id=? AND state='done'", (tid,))
                 if not engine or r["engine"] == engine]
    top = campaign_prefix.split(":", 1)[0]
    ranked = []
    for r in rows:
        try:
            v = score.scores(json.loads(r["metrics_json"])).get(metric)
        except (ValueError, TypeError):
            v = None
        if v is not None:
            ranked.append((v, r["trial_id"], json.loads(r["config_json"])))
    verdicts = {}

    def verdict(cfg, run_gate):
        """True/False from quality2 results (running the gate first if allowed and needed), None if unknown."""
        h = space.config_hash(cfg)
        if h in verdicts and (verdicts[h] is not None or not run_gate):
            return verdicts[h]
        ok, detail = climber.quality_verdict(cfg, top)
        if ok is None and run_gate and gate_campaign:
            ok, detail, _ = climber.quality_gate(cfg, gate_campaign, lane, log)
            ledger.event(gate_campaign, "quality-pass" if ok else "quality-reject", {"start_candidate": True, **detail})
        if ok is False:
            log(f"  start candidate {space.canonical(cfg)} skipped: quality gate {detail.get('fails')}")
        verdicts[h] = ok
        return ok

    # Within the tie tolerance of the best value, prefer the config with the fewest changes from the defaults:
    # a change the screen did not show to help (e.g. SGLang FP8 KV landing on the same T2 rung as BF16 KV) must
    # not ride into the climb's start; the climb can still add it if it beats the climb threshold. The tie is
    # broken *before* any quality-gate run, so only the chosen config is gated; if it fails, the next is chosen.
    nk = lambda c: len(space.canonical(c))
    cands = [x for x in ranked if verdict(x[2], False) is not False]
    while cands:
        cands.sort(key=lambda x: x[0], reverse=True)
        tie = [x for x in cands if x[0] >= cands[0][0] * (1 - tie_tol)]
        pick = min(tie, key=lambda x: (nk(x[2]), -x[0]))
        if verdict(pick[2], True) is not False:
            if space.config_hash(pick[2]) != space.config_hash(cands[0][2]):
                log(f"  start: {space.canonical(pick[2])} ({metric} {pick[0]:.4g}) ties {space.canonical(cands[0][2])} "
                    f"({cands[0][0]:.4g}) within {tie_tol:.0%} with fewer changes")
            return pick[2]
        h = space.config_hash(pick[2])
        cands = [x for x in cands if space.config_hash(x[2]) != h]
    return None


def climb_result(name, climb_id):
    """(final config, measured capacity or None) of a climb: its best-so-far file, else its start config."""
    sub = f"{name}:{climb_id}"
    f = climber.BEST_DIR / f"{sub}.json"
    if f.exists():
        d = json.loads(f.read_text())
        return d["config"], d.get("capacity_req_s")
    with ledger.conn() as c:
        r = c.execute("SELECT detail FROM events WHERE campaign=? AND kind='climb-start' ORDER BY ts DESC LIMIT 1",
                      (sub,)).fetchone()
    return (json.loads(r["detail"])["start"], None) if r else (None, None)


def climb_start(name, climb_id):
    with ledger.conn() as c:
        r = c.execute("SELECT detail FROM events WHERE campaign=? AND kind='climb-start' ORDER BY ts ASC LIMIT 1",
                      (f"{name}:{climb_id}",)).fetchone()
    return json.loads(r["detail"])["start"] if r else None


def capacity(name, cfg, family="t2s"):
    """Latest measured sustainable rate of a config (any ladder of the family in this campaign)."""
    caps = []
    for r in climber._rows_like(cfg, name, max_age_s=10 ** 9):
        if climber._family(r["plan"]) == family:
            s = score.scores(json.loads(r["metrics_json"] or "{}"))
            if s.get("t2_capacity") or s.get("t2_max_rate"):
                caps.append((r["finished_at"], s.get("t2_capacity") or s.get("t2_max_rate")))
    return max(caps)[1] if caps else None


def run_batch(st, sub, name):
    jobs = []
    for cfg in st["configs"]:
        rec = climber._recent(cfg, {climber._family(st["plan"])}, name) if st.get("reuse") else []
        full = [x for x in rec if x[2] == st["plan"]]  # the full plan (a probe or fine ladder is not a screen)
        if full:
            tid = full[-1][0]
            print(f"  reuse: recent identical result {tid} for {space.canonical(cfg)} with {st['plan']}", flush=True)
            ledger.event(sub, "reuse", {"trial": tid, "config": space.canonical(cfg), "plan": st["plan"]})
            continue
        jobs.append((cfg, st["plan"], {"notes": st["id"]}))
    runner.run_many(jobs, sub, lanes=tuple(st.get("lanes", ["n0", "n1"])))


def _modify(st, name, cfg):
    """Apply a verify stage's `graft` (knobs from another climb's winner) and `override` (explicit knobs)."""
    g, ov = st.get("graft"), st.get("override")
    if not g and not ov:
        return cfg, False
    c = space.full(cfg)
    if g:
        donor = climb_result(name, g["climb"])[0]
        if not donor:
            return None, True
        donor = space.full(donor)
        c.update({k: donor[k] for k in g["knobs"]})
    c.update(ov or {})
    return c, True


def run_verify(st, sub, name):
    cfgs = []
    for cid in st.get("from_climbs", []):
        cfg, cap = climb_result(name, cid)
        if cfg:
            cfg, changed = _modify(st, name, cfg)
            if cfg:
                cfgs.append((cfg, None if changed else cap, cid))
    cfgs += [(c, None, "extra") for c in st.get("configs", [])]
    seen, uniq = set(), []
    for cfg, cap, src in cfgs:  # e.g. a T3 winner that equals its T2 start is measured once
        h = space.config_hash(cfg)
        if h in seen:
            print(f"  verify: {src} is identical to an earlier config in this stage; measured once", flush=True)
            continue
        seen.add(h)
        uniq.append((cfg, cap, src))
    cfgs = uniq
    jobs = []
    for cfg, cap, src in cfgs:
        if space.validity(cfg):
            print(f"  verify: {src} {space.canonical(cfg)} is invalid ({space.validity(cfg)}); skipped", flush=True)
            continue
        if st.get("reuse") and not st.get("repeat") and not st["plan"].endswith("@capacity"):
            rec = [x for x in climber._recent(cfg, {climber._family(st["plan"])}, name) if x[2] == st["plan"]]
            if rec:
                ledger.event(sub, "reuse", {"trial": rec[-1][0], "config": space.canonical(cfg), "plan": st["plan"]})
                print(f"  reuse: recent identical result {rec[-1][0]} for {src}", flush=True)
                continue
        plan = st["plan"]
        if plan.endswith("@capacity"):
            base = plan.split("@", 1)[0]
            fam = plans.STEADY[base][0] if base in plans.STEADY else next(
                (f for f, (_, _, v) in plans.CAPACITY_FAMILIES.items() if v == base), "t2s" if base == "t2mv" else None)
            cap = cap or capacity(name, space.full(cfg), fam)
            if not cap:
                print(f"  verify: no measured capacity for {space.canonical(cfg)}; skipped", flush=True)
                continue
            plan = f"{base}@{cap:.4f}"
        jobs += [(cfg, plan, {"notes": f"{st['id']}:{src}"}) for _ in range(st.get("repeat", 1))]
    runner.run_many(jobs, sub, lanes=tuple(st.get("lanes", ["n0", "n1"])))


def run_ablation(st, sub, name):
    """Leave-one-out: for a climb's final config, revert each changed knob alone and measure it."""
    final, cap = climb_result(name, st["climb"])
    start = climb_start(name, st.get("start_climb", st["climb"]))
    if not final or not start:
        print(f"  ablation: no result for {st['climb']}; skipped", flush=True)
        return
    final, start = space.full(final), space.full(start)
    changed = [k for k in final if k != "engine" and final[k] != start[k]]
    track = climber.TRACKS[st["track"]]
    jobs, labels = [], []
    for k in changed:
        c = dict(final, **{k: start[k]})
        if space.validity(c):
            continue
        if track["kind"] == "capacity":
            fam = track["family"]
            r0 = 0.7 * (cap or capacity(name, final, fam) or 1.0)
            plan = f"{plans.CAPACITY_FAMILIES[fam][1]}@{r0:.4f}"
        else:
            plan = track["plan"]
        jobs.append((c, plan, {"notes": f"ablation:{k}", "move": {k: [final[k], start[k]]}}))
        labels.append(k)
    res = runner.run_many(jobs, sub, lanes=tuple(st.get("lanes", ["n0", "n1"])))
    metric = track["metric"]
    out = {k: (m or {}).get("scores", {}).get(metric) for k, (tid, stt, m) in zip(labels, res)}
    ledger.event(name, "ablation", {"climb": st["climb"], "metric": metric, "final": final, "start": start,
                                    "reverted_scores": out})


def abort_orphans(log=print):
    """A starting controller owns no trials yet: ledger rows still 'running' belong to a killed predecessor
    (their pods are deleted by the next trial's lane cleanup)."""
    for r in ledger.rows("state='running'"):
        ledger.finish(r["trial_id"], "aborted", "controller restarted while the trial ran", None)
        log(f"marked orphaned trial {r['trial_id']} ({r['campaign']}, {r['plan']}) aborted")


def run(path):
    camp = json.loads(Path(path).read_text())
    name = camp["name"]
    abort_orphans(lambda *a: print(*a, flush=True))
    if camp.get("reuse_from"):
        rf = camp["reuse_from"]
        climber.REUSE_FROM[name] = (rf["campaigns"], set(rf["plan_families"]))
    done = done_stages(name)
    print(f"campaign {name}: {len(camp['stages'])} stages, done={sorted(done)}", flush=True)
    for st in camp["stages"]:
        sid = st["id"]
        if sid in done:
            continue
        sub = f"{name}:{sid}"
        ledger.event(name, "stage-start", st)
        print(f"== stage {sid} ({st['type']})", flush=True)
        if st["type"] == "batch":
            run_batch(st, sub, name)
        elif st["type"] == "verify":
            run_verify(st, sub, name)
        elif st["type"] == "ablation":
            run_ablation(st, sub, name)
        elif st["type"] == "climb":
            best_file = climber.BEST_DIR / f"{sub}.json"
            if best_file.exists():
                start = json.loads(best_file.read_text())["config"]
                print(f"resuming climb from best-so-far {best_file}", flush=True)
            elif "start_from_climb" in st:
                start = climb_result(name, st["start_from_climb"])[0] or st["start"]
            elif "start_from" in st:
                sf = st["start_from"]
                start = best_config(sf["campaign_prefix"], sf["metric"], sf.get("engine"), sf.get("plan"),
                                    gate_campaign=sub if st.get("gate", True) else None) or st["start"]
            else:
                start = st["start"]
            climber.climb(start, st["track"], sub, threshold=st.get("threshold", 0.03),
                          max_trials=st.get("max_trials", 80), knob_order=st.get("knobs"),
                          lanes=tuple(st.get("lanes", ["n0", "n1"])), calibrate=st.get("calibrate", False),
                          coupled=st.get("coupled", True), kicks=st.get("kicks", True), gate=st.get("gate", True))
        ledger.event(name, "stage-done", {"id": sid})
    print(f"campaign {name} complete", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
    while True:  # keep the controller pod around for inspection
        time.sleep(3600)
