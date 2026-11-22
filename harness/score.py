"""Turn a trial's raw artifacts (vllm bench JSON per step, server metric samples, GPU CSV,
server log) into metrics and track scores."""
import csv
import datetime
import json
import re
from pathlib import Path

from . import plans

# SGLang decode log line (every --decode-log-interval steps), e.g.
# "[2026-11-19 13:21:07] Decode batch, #running-req: 1, ..., accept len: 1.70, accept rate: 0.10, ..."
# accept len = tokens produced per verify step including the bonus token, so accepted drafts = accept len - 1
# (the same quantity as vLLM's accepted tokens per draft step). Container clocks are UTC.
SG_DECODE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\] Decode batch.*?accept len: ([\d.]+)", re.M)
KV_PATTERNS = [
    (re.compile(r"GPU KV cache size: ([\d,]+) tokens"), "vllm"),
    (re.compile(r"max_total_num_tokens=(\d+)"), "sglang"),
]
STEP_FIELDS = ["completed", "failed", "duration", "request_throughput", "request_goodput",
               "output_throughput", "total_token_throughput", "max_concurrency",
               "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
               "mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms",
               "mean_itl_ms", "median_itl_ms", "p90_itl_ms", "p99_itl_ms",
               "mean_e2el_ms", "median_e2el_ms", "p90_e2el_ms", "p99_e2el_ms",
               "total_output_tokens", "total_input_tokens"]
A100_PCIE_HBM_BPS = 1.935e12
# nvidia-smi clocks_throttle_reasons bits. "GPU idle" (0x1) is set whenever no kernel runs, so it is
# excluded; the SW power cap (0x4) binds on this 300 W PCIe card for most busy samples.
THROTTLE_IDLE, THROTTLE_SW_POWER = 0x1, 0x4
THROTTLE_SLOWDOWN = 0x8 | 0x20 | 0x40 | 0x80  # HW slowdown, SW/HW thermal, HW power brake


def _gpu_rows(path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for r in csv.reader(f):
            try:
                ts = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(
                    tzinfo=datetime.timezone.utc).timestamp()  # the server container runs in UTC
            except (ValueError, IndexError):
                ts = None
            try:
                rows.append({"ts": ts, "util": float(r[1].split()[0]), "mem_util": float(r[2].split()[0]),
                             "mem_used_mib": float(r[3].split()[0]), "power_w": float(r[4].split()[0]),
                             "sm_mhz": float(r[5].split()[0]), "temp_c": float(r[7].split()[0]),
                             "throttle": int(r[8].strip(), 16) if len(r) > 8 and r[8].strip() else 0})
            except (ValueError, IndexError):
                continue
    return rows


def _gpu_summary(rows, prefix="gpu_"):
    if not rows:
        return {}
    busy = [r for r in rows if r["util"] > 5] or rows
    n = len(busy)
    return {f"{prefix}util_mean_busy": sum(r["util"] for r in busy) / n,
            f"{prefix}mem_bw_util_mean_busy": sum(r["mem_util"] for r in busy) / n,
            f"{prefix}mem_used_max_mib": max(r["mem_used_mib"] for r in rows),
            f"{prefix}power_max_w": max(r["power_w"] for r in rows),
            f"{prefix}power_mean_busy_w": sum(r["power_w"] for r in busy) / n,
            f"{prefix}sm_mhz_mean_busy": sum(r["sm_mhz"] for r in busy) / n,
            f"{prefix}temp_max_c": max(r["temp_c"] for r in rows),
            f"{prefix}power_capped_frac_busy": sum(1 for r in busy if r["throttle"] & THROTTLE_SW_POWER) / n,
            f"{prefix}slowdown_frac_busy": sum(1 for r in busy if r["throttle"] & THROTTLE_SLOWDOWN) / n,
            f"{prefix}throttle_seen": sorted({hex(r["throttle"] & ~THROTTLE_IDLE) for r in rows
                                              if r["throttle"] & ~THROTTLE_IDLE})}


def _gpu(path):
    return _gpu_summary(_gpu_rows(path))


def _resolved(text):
    """What the engine actually ran, from its startup log (engines silently fall back to other paths)."""
    r = {}
    ab = (re.findall(r"Using (\w+) attention backend", text) or re.findall(r"Use (\w+) backend by default", text)
          or re.findall(r"'attention_backend': '(\w+)'", text))
    if ab:
        r["attention_backend"] = ab[-1]
    k = sorted(set(re.findall(r"Using (\w+Kernel) for", text)))
    if k:
        r["linear_kernels"] = k
    g = sorted(set(re.findall(r"Capturing CUDA graphs \((\w+)\)", text)))
    if g:
        r["cudagraph_captured"] = g
    if "Directly load AOT compilation" in text:
        r["compile_cache_hit"] = True
    t = re.findall(r"torch\.compile took ([\d.]+) s", text)
    if t:
        r["torch_compile_s"] = float(t[-1])
    p = re.findall(r"Capture target prefill CUDA graph end\. elapsed=([\d.]+) s", text)
    if p:
        r["sglang_prefill_graph_s"] = float(p[-1])
    d = re.findall(r"Decode batch.*?cuda graph: (True|False)", text)
    if d:
        r["sglang_decode_graph_frac"] = round(d.count("True") / len(d), 4)
    return r


def _sglang_accept(text):
    """[(epoch seconds, accept len)] from SGLang's decode log lines (empty without speculative decoding)."""
    out = []
    for ts, al in SG_DECODE.findall(text):
        try:
            t = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
            out.append((t, float(al)))
        except ValueError:
            continue
    return out


def extract(trial_dir: Path):
    trial_dir = Path(trial_dir)
    out = {"steps": []}
    summary = {}
    log = trial_dir / "server.log"
    log_text = log.read_text(errors="replace") if log.exists() else ""
    sg_acc = _sglang_accept(log_text)
    if (trial_dir / "summary.json").exists():
        summary = json.loads((trial_dir / "summary.json").read_text())
        out["startup_s"] = summary.get("startup_s")
    srv_steps = {s["step"]: s for s in summary.get("steps", [])}
    gpu_rows = _gpu_rows(trial_dir / "gpu.csv")
    cpu_snaps = _cpu_snaps(trial_dir / "cpu.txt")
    for p in sorted(trial_dir.glob("step[0-9][0-9].json")):
        d = json.loads(p.read_text())
        name = p.stem
        srv = srv_steps.get(name, {})
        spec = srv.get("spec", {})
        rec = {"step": name, "isl": spec.get("isl"), "osl": spec.get("osl"), "c": spec.get("c"),
               "n": spec.get("n"), "dataset": spec.get("dataset", "random"), "primary": bool(spec.get("primary")),
               "request_rate": spec.get("request_rate")}
        rec.update({k: d.get(k) for k in STEP_FIELDS if k in d})
        rec.update({f"srv_{k}": v for k, v in srv.get("server", {}).items()})
        if srv.get("t_start") and srv.get("t_end"):  # per-step clocks/power (power state is a covariate)
            window = [r for r in gpu_rows if r["ts"] and srv["t_start"] <= r["ts"] <= srv["t_end"]]
            g = _gpu_summary(window, prefix="step_gpu_")
            rec.update({k: v for k, v in g.items() if not k.endswith("throttle_seen")})
            rec.update(_cpu_summary(cpu_snaps, srv["t_start"], srv["t_end"], prefix="step_cpu_"))
        rec.update(_slo_split(d, spec))
        rec.update(_window_split(d, spec))
        rec.update(_server_derived(srv.get("server", {}), srv.get("wall_s")))
        if sg_acc and srv.get("t_start") and srv.get("t_end") and rec.get("spec_mean_accepted") is None:
            # log timestamps have 1-s resolution: allow 1 s of slack at both ends of the step window
            acc = [a for t, a in sg_acc if srv["t_start"] - 1 <= t <= srv["t_end"] + 1]
            if acc:
                rec["sg_accept_len"] = sum(acc) / len(acc)
                rec["spec_mean_accepted"] = rec["sg_accept_len"] - 1
        out["steps"].append(rec)
    if log_text:
        text = log_text
        for pat, _ in KV_PATTERNS:
            m = pat.findall(text)
            if m:
                out["kv_tokens"] = int(m[-1].replace(",", ""))
                break
        out["resolved"] = _resolved(text)
    head = trial_dir / "server.head.log"
    if head.exists():
        head_text = head.read_text(errors="replace")
        for k, v in _resolved(head_text).items():
            out.setdefault("resolved", {}).setdefault(k, v)
        if "kv_tokens" not in out:
            for pat, _ in KV_PATTERNS:
                m = pat.findall(head_text)
                if m:
                    out["kv_tokens"] = int(m[-1].replace(",", ""))
                    break
    out.update(_gpu_summary(gpu_rows))
    q = trial_dir / "quality.json"
    if q.exists():
        qd = json.loads(q.read_text())
        out["quality"] = {k: v for k, v in qd.items() if k not in ("samples", "refs")}
    if summary.get("t3"):
        out["t3"] = summary["t3"]
    pr = trial_dir / "profile.json"
    if pr.exists():
        out["profile"] = json.loads(pr.read_text())
    if cpu_snaps:
        out["cpu_top_procs"] = _cpu_top(cpu_snaps)
    out["scores"] = scores(out)
    return out


def _window_split(d, spec):
    """Steady-state steps (spec "measure"): SLO attainment and TTFT/TPOT pass fractions of the requests that arrived
    inside the measurement window, the same rule as bench_driver.window_attainment."""
    meas = spec.get("measure")
    st, tt, lat, ol = (d.get(k) for k in ("start_times", "ttfts", "latencies", "output_lens"))
    if not meas or not (st and tt and lat and ol) or not len(st) == len(tt) == len(lat) == len(ol):
        return {}
    err = d.get("errors") or []
    slo = plans.SLO
    t0, lo, hi = min(st), meas.get("from_s", 0.0), meas.get("until_s", float("inf"))
    n = ok_t = ok_p = good = 0
    for i, s in enumerate(st):
        if not lo <= s - t0 < hi:
            continue
        n += 1
        if (i < len(err) and err[i]) or tt[i] is None:
            continue
        tpot = (lat[i] - tt[i]) / (ol[i] - 1) if ol[i] and ol[i] > 1 else 0.0
        a, b = tt[i] * 1000 <= slo["ttft_ms"], tpot * 1000 <= slo["tpot_ms"]
        ok_t, ok_p, good = ok_t + a, ok_p + b, good + (a and b)
    if not n:
        return {}
    return {"attainment_window": good / n, "window_ttft_pass_frac": ok_t / n, "window_tpot_pass_frac": ok_p / n,
            "window_n": n, "window_s": [lo, hi]}


def _slo_split(d, spec):
    """From --save-detailed per-request data: which SLO binds (fraction meeting TTFT only / TPOT only)."""
    ttfts, lats, outs = d.get("ttfts"), d.get("latencies"), d.get("output_lens")
    if not (ttfts and lats and outs) or len(ttfts) != len(lats) or len(outs) != len(lats):
        return {}
    slo = plans.SLO
    out_speed = sum(o for o in outs if o) / sum(l for l in lats if l) if any(lats) else None
    ok_t = [t is not None and t * 1000 <= slo["ttft_ms"] for t in ttfts]
    tpots = [((l - t) / (o - 1)) * 1000 if (t is not None and o and o > 1) else None for t, l, o in zip(ttfts, lats, outs)]
    ok_p = [x is not None and x <= slo["tpot_ms"] for x in tpots]
    n = len(ttfts)
    ft, fp = sum(ok_t) / n, sum(ok_p) / n
    binding = None
    if ft < 1 or fp < 1:
        binding = "both" if abs(ft - fp) < 0.02 else ("ttft" if ft < fp else "tpot")
    return {"ttft_pass_frac": ft, "tpot_pass_frac": fp, "binding_slo": binding,
            "mean_output_tokens": sum(o for o in outs if o) / n, "user_tokens_per_s": out_speed}


def _server_derived(srv, wall):
    """Per-request queue/prefill/decode time, speculative acceptance per position, achieved FLOP/s and HBM BW."""
    out = {}
    for k in ("queue", "prefill", "decode"):
        cnt = srv.get(f"delta_{k}_s_count")
        if cnt:
            out[f"mean_{k}_ms"] = 1000 * srv.get(f"delta_{k}_s_sum", 0) / cnt
    drafts = srv.get("delta_spec_drafts")
    if drafts:
        out["spec_mean_accepted"] = srv.get("delta_spec_accepted", 0) / drafts
        if srv.get("delta_spec_pos"):
            out["spec_accept_by_pos"] = [round(x / drafts, 4) for x in srv["delta_spec_pos"]]
    if wall:
        if srv.get("delta_est_flops"):
            out["achieved_tflops"] = srv["delta_est_flops"] / wall / 1e12
        if srv.get("delta_est_read_bytes"):
            out["achieved_read_tbps"] = srv["delta_est_read_bytes"] / wall / 1e12
            out["mbu"] = srv["delta_est_read_bytes"] / wall / A100_PCIE_HBM_BPS
    return out


def _cpu_snaps(path):
    """cpu.txt -> [(t, {pid: (comm, cpu_seconds)})] (utime+stime in clock ticks / 100)."""
    if not path.exists():
        return []
    snaps, cur = [], None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("T "):
            try:
                cur = (float(line[2:]), {})
                snaps.append(cur)
            except ValueError:
                cur = None
        elif cur is not None:
            parts = line.split("|")
            if len(parts) == 4:
                try:
                    cur[1][parts[0]] = (parts[1], (int(parts[2]) + int(parts[3])) / 100.0)
                except ValueError:
                    pass
    return snaps


def _cpu_rates(snaps, t0=None, t1=None):
    """Mean cores used per process between the first and last snapshot inside [t0, t1]."""
    win = [s for s in snaps if (t0 is None or s[0] >= t0) and (t1 is None or s[0] <= t1)]
    if len(win) < 2:
        return {}
    a, b = win[0], win[-1]
    dt = b[0] - a[0]
    rates = {}
    for pid, (comm, cpu) in b[1].items():
        if pid in a[1] and dt > 0:
            rates[(pid, comm)] = (cpu - a[1][pid][1]) / dt
    return rates


def _cpu_summary(snaps, t0, t1, prefix="cpu_"):
    rates = _cpu_rates(snaps, t0, t1)
    if not rates:
        return {}
    top = max(rates.items(), key=lambda kv: kv[1])
    return {f"{prefix}top_proc": top[0][1], f"{prefix}top_proc_cores": top[1],
            f"{prefix}total_cores": sum(v for v in rates.values() if v > 0)}


def _cpu_top(snaps):
    """Mean cores per process over the run, each process measured from its first snapshot (the engine core
    starts after the sampler)."""
    first, last = {}, {}
    for t, procs in snaps:
        for pid, (comm, cpu) in procs.items():
            first.setdefault(pid, (t, cpu))
            last[pid] = (t, cpu, comm)
    rates = {(pid, last[pid][2]): (last[pid][1] - first[pid][1]) / (last[pid][0] - first[pid][0])
             for pid in last if last[pid][0] > first[pid][0]}
    return [[comm, round(v, 3)] for (pid, comm), v in sorted(rates.items(), key=lambda kv: -kv[1])[:5]]


def _accounted(x):
    """Every request of the step is accounted for, as completed or failed. A failed request (e.g. one dropped
    connection in a ~1,700-request steady-state window) counts as an SLO miss rather than discarding the step."""
    c = x.get("completed")
    return bool(c) and x.get("n") is not None and c + (x.get("failed") or 0) == x["n"]


def _attainment(x):
    """Share of a step's requests that met the SLO, failed requests counting as misses (vLLM's goodput and
    throughput count completed requests only). Steady-state steps: of the requests inside the measurement window."""
    if x.get("attainment_window") is not None:
        return x["attainment_window"]
    c, f = x["completed"], x.get("failed") or 0
    return x["request_goodput"] / x["request_throughput"] * (c / (c + f))


def t2_points(m):
    """(offered req/s, SLO attainment, mean output tokens per request) for every Poisson step whose requests are all
    accounted for. With natural stopping (T2 chat) the mean output length comes from the run, not the max_tokens cap.
    Steady-state steps count only the requests inside their measurement window."""
    pts = []
    for x in m.get("steps", []):
        if not _accounted(x):
            continue
        thr, gp, rate = x.get("request_throughput"), x.get("request_goodput"), x.get("request_rate")
        if gp is None or not thr or not rate or not x.get("osl"):
            continue
        out_tok = (x["total_output_tokens"] / x["completed"]) if x.get("total_output_tokens") else x["osl"]
        pts.append((rate, _attainment(x), out_tok))
    return pts


def t2_capacity(pts, target=None):
    """Sustainable rate interpolated where attainment crosses the target, between the highest passing rate
    and the lowest failing rate above it (finer than the ladder's ~2.6% steps)."""
    target = target if target is not None else plans.ATTAINMENT
    passing = [p for p in pts if p[1] >= target]
    if not passing:
        return None
    rp, ap, _ = max(passing)
    above = [p for p in pts if p[0] > rp and p[1] < target]
    if not above:
        return rp
    rf, af, _ = min(above)
    return rp + (ap - target) / (ap - af) * (rf - rp) if ap > af else rp


def scores(m):
    s = {}
    if m.get("quality", {}).get("gsm8k_acc") is not None:
        s["gsm8k_acc"] = m["quality"]["gsm8k_acc"]
    steps = m.get("steps", [])
    ok = [x for x in steps if x.get("completed") and x.get("completed") == x.get("n")]
    prim = [x for x in ok if x.get("primary") and x.get("median_tpot_ms")]
    c1 = [x for x in ok if x.get("c") == 1 and x.get("isl") == 1024 and x.get("median_tpot_ms")]
    if c1:
        s["t1_rand_decode_tps"] = 1000.0 / c1[0]["median_tpot_ms"]
    if prim:
        c1 = prim
    if c1:
        x = c1[0]
        s["t1_decode_tps"] = 1000.0 / x["median_tpot_ms"]
        s["t1_ttft_ms"] = x.get("median_ttft_ms")
        if x.get("user_tokens_per_s"):  # actual output tokens / end-to-end time (natural stopping safe)
            s["aa_speed_c1"] = x["user_tokens_per_s"]
        elif x.get("median_e2el_ms"):
            s["aa_speed_c1"] = x["osl"] / (x["median_e2el_ms"] / 1000.0)
    c1_8k = [x for x in ok if x.get("c") == 1 and x.get("isl") == 8192]
    if c1_8k:
        s["t1_ttft8k_ms"] = c1_8k[0].get("median_ttft_ms")
    gps, rates = [], []
    for x in steps:
        if not _accounted(x):
            continue
        thr, gp = x.get("request_throughput"), x.get("request_goodput")
        if gp is None or not thr or not x.get("osl"):
            continue
        if _attainment(x) >= 0.9:  # only rates where >=90% of requests (steady-state: of its window) met the SLO
            gps.append(gp * x["osl"])
            if x.get("request_rate"):
                rates.append(x["request_rate"])
    if gps:
        s["t2_goodput_tps"] = max(gps)  # vLLM goodput: divides by duration incl. drain (kept for continuity)
    if rates:
        s["t2_max_rate"] = max(rates)
    pts = t2_points(m)
    if pts:
        s["t2_points"] = [[round(r, 4), round(a, 4)] for r, a, _ in pts]
        passing = [p for p in pts if p[1] >= plans.ATTAINMENT]
        if passing:
            rate, att, osl = max(passing)
            s["t2_slo_tps"] = rate * att * osl  # offered rate x attainment x output tokens: no drain bias
            s["t2_capacity"] = t2_capacity(pts)
        # MLPerf-style: highest rate whose p99 TTFT and p99 TPOT both meet the SLO (reported, not climbed)
        p99_ok = [x["request_rate"] for x in steps if x.get("request_rate") and x.get("completed") == x.get("n")
                  and x.get("p99_ttft_ms") is not None and x["p99_ttft_ms"] <= plans.SLO["ttft_ms"]
                  and x.get("p99_tpot_ms") is not None and x["p99_tpot_ms"] <= plans.SLO["tpot_ms"]]
        if p99_ok:
            s["t2_max_rate_p99"] = max(p99_ok)
    q = m.get("quality") or {}
    for k in ("passkey_acc", "nll_delta", "top1_agree"):
        if q.get(k) is not None:
            s[k] = q[k]
    if q.get("thinking") is not None:
        s["gsm8k_thinking"] = q["thinking"]
    t3 = m.get("t3")
    if t3 and t3.get("ttft_p90_ms"):
        s["t3_ttft_p90_ms"] = t3["ttft_p90_ms"]
        s["t3_ttft_p50_ms"] = t3.get("ttft_p50_ms")
        s["t3_errors"] = t3.get("errors")
        s["t3_turns_done"] = t3.get("turns_done")
        # a run that dropped turns must not look fast: only score complete runs
        if not t3.get("errors") and t3.get("turns_done") == t3.get("turns_planned"):
            s["t3_score"] = 1000.0 / t3["ttft_p90_ms"]
    pr = m.get("profile")
    if pr and pr.get("span_per_token_ms"):
        for k in ("span_per_token_ms", "busy_per_token_ms", "idle_per_token_ms", "gpu_idle_frac", "kernels_per_token",
                  "decode_span_per_token_ms", "measured_tpot_ms"):
            s[f"prof_{k}"] = pr.get(k)
    thr = [x for x in ok if x.get("c") == 32 and x.get("isl") == 1024]
    if thr:
        s["thr_c32_tps"] = thr[0].get("output_throughput")
        s["ttft_p99_c32_ms"] = thr[0].get("p99_ttft_ms")
        s["tpot_p99_c32_ms"] = thr[0].get("p99_tpot_ms")
    return s
