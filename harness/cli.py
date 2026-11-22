"""CLI: python -m harness.cli <command> ...

  run    --engine vllm --set variant=fp8 --set max_num_seqs=128 --plan smoke --lane n0 [--campaign X]
  batch  --file configs.json --plan screen [--campaign X]       (list of config dicts, both lanes)
  climb  --track t1|t2 --start start.json --campaign X [--threshold 0.03] [--max-trials 80]
  report [--campaign X]
  status [--campaign main-v3] [--db results-cluster/ledger.sqlite]   one-shot dashboard
  watch  [--campaign main-v3] [--interval 120]                        pull the in-cluster ledger + dashboard, repeat
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from . import climber, ledger, runner, space


def parse_set(pairs):
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        if v.lower() in ("none", "null"):
            v = None
        elif v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        out[k] = v
    return out


def cmd_report(args):
    where, params = ("campaign=?", (args.campaign,)) if args.campaign else ("1=1", ())
    rows = ledger.rows(where, params)
    print(f"{'trial':32} {'plan':6} {'state':10} {'eng':6} {'var':5} {'kv':>7} {'start':>6} "
          f"{'t1_tps':>7} {'ttft':>7} {'c32_tps':>8} {'t2_gp':>8}  move/reason")
    for r in rows:
        f = lambda v, fmt: (fmt.format(v) if v is not None else "-")
        extra = r["move"] or ""
        if r["state"] not in ("done",):
            extra = (r["reason"] or "")[:120]
        print(f"{r['trial_id']:32} {r['plan']:6} {r['state']:10} {r['engine']:6} {str(r['variant']):5} "
              f"{f(r['kv_tokens'], '{:d}'):>7} {f(r['startup_s'], '{:.0f}'):>6} {f(r['t1_decode_tps'], '{:.1f}'):>7} "
              f"{f(r['t1_ttft_ms'], '{:.0f}'):>7} {f(r['thr_c32_tps'], '{:.0f}'):>8} {f(r['t2_goodput_tps'], '{:.0f}'):>8}  {extra}")


def _fmt_age(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def cmd_status(args, clear=False):
    """Campaign dashboard from the ledger: stages, running trials, per-climb progress and best, recent events."""
    if args.db:
        ledger.DB = Path(args.db)
    camp_file = Path(__file__).resolve().parents[1] / "campaigns" / f"{args.campaign}.json"
    stages = json.loads(camp_file.read_text())["stages"] if camp_file.exists() else []
    with ledger.conn() as c:
        ev = [dict(r) for r in c.execute("SELECT * FROM events WHERE campaign=? OR campaign LIKE ? ORDER BY ts",
                                         (args.campaign, args.campaign + ":%"))]
        trials = [dict(r) for r in c.execute("SELECT * FROM trials WHERE campaign LIKE ? ORDER BY started_at",
                                             (args.campaign + ":%",))]
    now = time.time()
    done = {json.loads(e["detail"])["id"]: e["ts"] for e in ev if e["kind"] == "stage-done"}
    started = {json.loads(e["detail"])["id"]: e["ts"] for e in ev if e["kind"] == "stage-start"}
    out = []
    out.append(f"\033[1m{args.campaign}\033[0m  ledger {ledger.DB}  @ {time.strftime('%Y-%m-%d %H:%M:%S')}  "
               f"trials {len(trials)} (running {sum(t['state'] == 'running' for t in trials)})")
    durs = [t["finished_at"] - t["started_at"] for t in trials if t["finished_at"] and t["state"] == "done"]
    out.append("")
    out.append(f"{'stage':22} {'type':8} {'state':9} {'trials':>6} {'accepts':>7}  best / note")
    for st in stages:
        sid = st["id"]
        sub = f"{args.campaign}:{sid}"
        tr = [t for t in trials if t["campaign"] == sub]
        sev = [e for e in ev if e["campaign"] == sub]
        acc = [json.loads(e["detail"]) for e in sev if e["kind"] == "accept"]
        state = "done" if sid in done else ("RUNNING" if sid in started else "pending")
        note = ""
        if st["type"] == "climb":
            cap = [json.loads(e["detail"]) for e in sev if e["kind"] == "capacity"]
            if acc:
                a = acc[-1]
                note = f"last accept {a.get('move') or a.get('knob')}: {a.get('before', 0):.3g} -> {a.get('after', 0):.3g}"
            elif cap:
                note = f"incumbent capacity {cap[-1]['incumbent']:.3f} req/s"
            if any(e["kind"] == "plateau" for e in sev):
                note += "  [plateau]"
            elif any(e["kind"] == "budget-exhausted" for e in sev):
                note += "  [budget]"
        elif tr:
            ok = [t for t in tr if t["state"] == "done"]
            note = f"{len(ok)} done, {len(tr) - len(ok)} other"
        if state == "RUNNING" and sid in started:
            note = f"({_fmt_age(now - started[sid])}) " + note
        color = {"done": "\033[32m", "RUNNING": "\033[33m"}.get(state, "\033[90m")
        out.append(f"{sid:22} {st['type']:8} {color}{state:9}\033[0m {len(tr):>6} {len(acc):>7}  {note[:110]}")
    run = [t for t in trials if t["state"] == "running"]
    if run:
        out.append("")
        out.append("running trials:")
        for t in run:
            mv = t["move"] or t["notes"] or ""
            out.append(f"  {t['trial_id']} {t['plan']:16} {t['engine']}/{t['variant']} {mv[:70]}  "
                       f"({_fmt_age(now - t['started_at'])})")
    if durs:
        other = {}
        for t in trials:
            if t["state"] not in ("done", "running"):
                other[t["state"]] = other.get(t["state"], 0) + 1
        out.append(f"median trial {_fmt_age(statistics.median(durs))}; not done: "
                   + (", ".join(f"{k} {v}" for k, v in sorted(other.items())) or "none"))
    out.append("")
    out.append("recent events:")
    for e in ev[-8:]:
        d = e["detail"]
        out.append(f"  {time.strftime('%H:%M', time.localtime(e['ts']))} {e['campaign'].split(':')[-1]:18} "
                   f"{e['kind']:15} {d[:100]}")
    text = "\n".join(out)
    if clear:
        print("\033[2J\033[H", end="")
    print(text, flush=True)


def cmd_watch(args):
    repo = Path(__file__).resolve().parents[1]
    while True:
        subprocess.run(["bash", str(repo / "tools" / "pull.sh")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env={**os.environ, "LEDGER_ONLY": "1"})
        args.db = args.db or str(repo / "results-cluster" / "ledger.sqlite")
        cmd_status(args, clear=True)
        time.sleep(args.interval)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    r.add_argument("--set", action="append")
    r.add_argument("--plan", default="smoke")
    r.add_argument("--lane", default="n0")
    r.add_argument("--campaign", default="adhoc")
    r.add_argument("--notes")
    b = sub.add_parser("batch")
    b.add_argument("--file", required=True)
    b.add_argument("--plan", default="screen")
    b.add_argument("--campaign", default="batch")
    b.add_argument("--lanes", default="n0,n1")
    c = sub.add_parser("climb")
    c.add_argument("--track", required=True, choices=sorted(climber.TRACKS))
    c.add_argument("--start", required=True)
    c.add_argument("--campaign", required=True)
    c.add_argument("--threshold", type=float, default=0.03)
    c.add_argument("--max-trials", type=int, default=80)
    c.add_argument("--knobs", help="comma-separated knob order override")
    c.add_argument("--lanes", default="n0,n1")
    p = sub.add_parser("report")
    p.add_argument("--campaign")
    for nm in ("status", "watch"):
        w = sub.add_parser(nm)
        w.add_argument("--campaign", default="main-v3")
        w.add_argument("--db", help="ledger file (default: results-cluster/ledger.sqlite if present)")
        w.add_argument("--interval", type=int, default=120)
    args = ap.parse_args(argv)
    if args.cmd in ("status", "watch") and not args.db:
        default = Path(__file__).resolve().parents[1] / "results-cluster" / "ledger.sqlite"
        args.db = str(default) if default.exists() else None

    if args.cmd == "run":
        cfg = {"engine": args.engine, **parse_set(args.set)}
        print(" ".join(space.server_command(cfg)[0]))
        runner.run_trial(cfg, args.plan, args.lane, args.campaign, notes=args.notes)
    elif args.cmd == "batch":
        cfgs = json.load(open(args.file))
        jobs = [(cfg, args.plan, {"notes": cfg.pop("_notes", None)}) for cfg in cfgs]
        runner.run_many(jobs, args.campaign, lanes=tuple(args.lanes.split(",")))
    elif args.cmd == "climb":
        start = json.load(open(args.start))
        climber.climb(start, args.track, args.campaign, threshold=args.threshold, max_trials=args.max_trials,
                      knob_order=args.knobs.split(",") if args.knobs else None,
                      lanes=tuple(args.lanes.split(",")))
    elif args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "watch":
        cmd_watch(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
