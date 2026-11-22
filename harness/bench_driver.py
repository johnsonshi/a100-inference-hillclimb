"""Runs inside the trial pod's `bench` sidecar (vLLM image). Waits for the server, runs the
benchmark plan with `vllm bench serve`, samples server /metrics during each step, and writes
results to /results/<trial>/. Stays alive afterwards so the controller can copy results out.
"""
import glob
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.request

PLAN = json.loads(os.environ["PLAN_JSON"])
TRIAL = os.environ["TRIAL_ID"]
OUT = os.environ.get("HC_OUT") or f"/results/{TRIAL}"  # HC_OUT: offline tests only
BASE = "http://127.0.0.1:8000"
SPEC_BENCH = "/models/datasets/spec_bench/question.jsonl"
SPEC_BENCH_WARMUP_OFFSET = 448  # prompts 448-479 of the fixed permutation are reserved for warmups
SHAREGPT = "/models/datasets/sharegpt/sharegpt_v3_30k.json"  # tools/stage_sharegpt.sh
os.makedirs(OUT, exist_ok=True)

METRIC_PATTERNS = {
    "kv_usage": re.compile(r"^(?:vllm:kv_cache_usage_perc|sglang:token_usage)\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "running": re.compile(r"^(?:vllm:num_requests_running|sglang:num_running_reqs)\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "waiting": re.compile(r"^(?:vllm:num_requests_waiting|sglang:num_queue_reqs)\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
}
COUNTER_PATTERNS = {
    "preemptions": re.compile(r"^vllm:num_preemptions_total\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "prefix_hits": re.compile(r"^vllm:prefix_cache_hits_total\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "prefix_queries": re.compile(r"^vllm:prefix_cache_queries_total\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "ext_prefix_hits": re.compile(r"^vllm:external_prefix_cache_hits_total\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "ext_prefix_queries": re.compile(r"^vllm:external_prefix_cache_queries_total\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "spec_accepted": re.compile(r"^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "spec_drafted": re.compile(r"^vllm:spec_decode_num_draft_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "spec_drafts": re.compile(r"^vllm:spec_decode_num_drafts_total\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    # where TTFT goes: queueing vs prefill compute, and decode time, per request (histogram sums/counts)
    "queue_s_sum": re.compile(r"^vllm:request_queue_time_seconds_sum\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "queue_s_count": re.compile(r"^vllm:request_queue_time_seconds_count\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "prefill_s_sum": re.compile(r"^vllm:request_prefill_time_seconds_sum\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "prefill_s_count": re.compile(r"^vllm:request_prefill_time_seconds_count\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "decode_s_sum": re.compile(r"^vllm:request_decode_time_seconds_sum\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "decode_s_count": re.compile(r"^vllm:request_decode_time_seconds_count\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    # vLLM's own FLOP / byte accounting -> achieved FLOP/s and memory bandwidth (MFU / MBU)
    "est_flops": re.compile(r"^vllm:estimated_flops_per_gpu(?:_total)?\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "est_read_bytes": re.compile(r"^vllm:estimated_read_bytes_per_gpu(?:_total)?\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "sg_cached_tokens": re.compile(r"^sglang:cached_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
    "sg_prompt_tokens": re.compile(r"^sglang:prompt_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)", re.M),
}
SPEC_POS = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([0-9.eE+-]+)',
                      re.M)


def http_get(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, r.read().decode(errors="replace")


def window_attainment(res, measure, slo):
    """SLO attainment, TTFT and TPOT pass fractions, and count of the requests that arrived inside the measurement
    window (`measure`: seconds after the first arrival, [from_s, until_s)), from --save-detailed per-request data.
    A request passes if it completed without error, TTFT <= slo and TPOT <= slo (vLLM's goodput rule). Returns
    None if the per-request data is missing. Mirrored by score._window_split (tests.test_bench_driver checks both)."""
    st, tt, lat, ol = (res.get(k) for k in ("start_times", "ttfts", "latencies", "output_lens"))
    err = res.get("errors") or []
    if not (st and tt and lat and ol) or not len(st) == len(tt) == len(lat) == len(ol):
        return None
    t0, lo, hi = min(st), measure.get("from_s", 0.0), measure.get("until_s", float("inf"))
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
        return None
    return {"attainment": good / n, "ttft_pass": ok_t / n, "tpot_pass": ok_p / n, "n": n}


def write_json(name, obj):
    tmp = f"{OUT}/{name}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, f"{OUT}/{name}")


def scrape():
    try:
        _, text = http_get("/metrics", 5)
    except Exception:
        return None
    snap = {}
    for k, pat in {**METRIC_PATTERNS, **COUNTER_PATTERNS}.items():
        vals = [float(v) for v in pat.findall(text)]
        if vals:
            snap[k] = sum(vals)
    pos = {}
    for p, v in SPEC_POS.findall(text):
        pos[int(p)] = pos.get(int(p), 0.0) + float(v)
    if pos:
        snap["spec_pos"] = [pos[i] for i in sorted(pos)]
    return snap


class Sampler(threading.Thread):
    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop_flag = period, [], threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            s = scrape()
            if s is not None:
                s["t"] = time.time()
                self.samples.append(s)
            self.stop_flag.wait(self.period)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=10)
        return self.samples


def http_post(path, body=None, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode(errors="replace")


KERNEL_CATEGORIES = [("gemm", ("gemm", "gemv", "marlin", "cutlass", "xmma", "matmul", "_mm_", "scaled_mm", "splitk")),
                     ("attention", ("flash", "attn", "attention", "fmha", "paged", "reshape_and_cache")),
                     ("norm_act", ("rms", "norm", "silu", "act_and_mul", "gelu", "rotary")),
                     ("sampling", ("sampl", "softmax", "topk", "top_k", "argmax", "gumbel", "penalt")),
                     ("memcpy", ("memcpy", "memset", "copy"))]


def summarize_trace(path, tokens):
    """Where single-user decode time goes, from a torch-profiler Chrome trace: GPU busy vs idle span,
    kernels per token, time by kernel category, top kernels."""
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        tr = json.load(f)
    ev = tr.get("traceEvents", []) if isinstance(tr, dict) else tr
    gpu = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                 key=lambda e: e["ts"])
    if not gpu:
        return {"error": "no GPU events in trace", "trace": path}
    span = max(e["ts"] + e.get("dur", 0) for e in gpu) - gpu[0]["ts"]
    busy, cs, ce = 0.0, None, None
    for e in gpu:
        s, d = e["ts"], e["ts"] + e.get("dur", 0)
        if ce is None or s > ce:
            if ce is not None:
                busy += ce - cs
            cs, ce = s, d
        else:
            ce = max(ce, d)
    busy += ce - cs
    by_name, by_cat = {}, {}
    for e in gpu:
        n = e.get("name", "?")
        by_name[n] = by_name.get(n, 0.0) + e.get("dur", 0)
        low = n.lower()
        cat = next((c for c, keys in KERNEL_CATEGORIES if any(k in low for k in keys)), "other")
        by_cat[cat] = by_cat.get(cat, 0.0) + e.get("dur", 0)
    cpu_launch = sum(e.get("dur", 0) for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime")
    return {"trace": os.path.basename(path), "tokens": tokens, "kernels": len(gpu),
            "kernels_per_token": len(gpu) / tokens, "span_ms": span / 1e3, "busy_ms": busy / 1e3,
            "gpu_idle_frac": 1 - busy / span if span else None, "span_per_token_ms": span / 1e3 / tokens,
            "busy_per_token_ms": busy / 1e3 / tokens, "idle_per_token_ms": (span - busy) / 1e3 / tokens,
            "cuda_runtime_cpu_ms": cpu_launch / 1e3,
            "by_category_ms": {k: round(v / 1e3, 3) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
            "top_kernels_ms": [[n[:120], round(d / 1e3, 3)] for n, d in sorted(by_name.items(), key=lambda kv: -kv[1])[:15]]}


def profile_run(prof, bench_args, status):
    """Profile single-user decode with the server's torch profiler (vLLM --profiler-config)."""
    step = dict(isl=prof["isl"], osl=prof["osl"], c=1)
    subprocess.run(bench_args(step, 2, 1, 8100, None), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                   timeout=900)  # warm-up, unprofiled
    pdir = f"{OUT}/prof"
    before = set(glob.glob(f"{pdir}/**/*.json*", recursive=True))
    http_post("/start_profile")
    t = time.time()
    rc = subprocess.run(bench_args(step, prof["n"], 1, 8101, "profile_bench.json"), stdout=open(f"{OUT}/profile.log", "w"),
                        stderr=subprocess.STDOUT, timeout=1800).returncode
    wall = time.time() - t
    http_post("/stop_profile")
    trace = None
    for _ in range(120):  # the trace is written asynchronously after stop
        new = [f for f in glob.glob(f"{pdir}/**/*.json*", recursive=True) if f not in before]
        if new:
            trace = max(new, key=os.path.getsize)
            size = os.path.getsize(trace)
            time.sleep(5)
            if os.path.getsize(trace) == size:
                break
        time.sleep(5)
    out = {"rc": rc, "wall_s": wall, "requests": prof["n"], "osl": prof["osl"]}
    if trace:
        try:
            out.update(summarize_trace(trace, prof["n"] * prof["osl"]))
            # decode only: subtract each request's prefill (its measured TTFT) from the GPU span
            b = json.load(open(f"{OUT}/profile_bench.json"))
            if b.get("mean_ttft_ms") and out.get("span_ms"):
                pre = prof["n"] * b["mean_ttft_ms"]
                out.update(prefill_ms=pre, decode_span_per_token_ms=(out["span_ms"] - pre) / (prof["n"] * (prof["osl"] - 1)),
                           measured_tpot_ms=b.get("median_tpot_ms"))
        except Exception as e:
            out["error"] = f"trace parse failed: {e!r}"
    else:
        out["error"] = "no trace file written"
    write_json("profile.json", out)


def summarize_samples(samples, before, after):
    out = {}
    for k in METRIC_PATTERNS:
        vals = [s[k] for s in samples if k in s]
        if vals:
            out[f"max_{k}"] = max(vals)
            out[f"mean_{k}"] = sum(vals) / len(vals)
    if before and after:
        for k in COUNTER_PATTERNS:
            if k in before and k in after:
                out[f"delta_{k}"] = after[k] - before[k]
        if before.get("spec_pos") and after.get("spec_pos") and len(before["spec_pos"]) == len(after["spec_pos"]):
            out["delta_spec_pos"] = [a - b for a, b in zip(after["spec_pos"], before["spec_pos"])]
    return out


def main():
    status = {"state": "waiting_for_server", "trial": TRIAL}
    write_json("status.json", status)
    steps_all = PLAN["steps"] + ([PLAN["rate_search"]["step"]] if PLAN.get("rate_search") else [])
    if any(s.get("dataset") for s in steps_all):
        # dataset loaders in `vllm bench` need pandas (vllm[bench] extra), absent from the serving image
        subprocess.run(["pip", "install", "--quiet", "--disable-pip-version-check", "pandas"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=600)
    t0 = time.time()
    deadline = t0 + PLAN.get("startup_timeout_s", 1800)
    while True:
        try:
            code, _ = http_get("/health", 5)
            if code == 200:
                code, body = http_get("/v1/models", 10)
                if code == 200:
                    break
        except Exception:
            pass
        if time.time() > deadline:
            write_json("status.json", {**status, "state": "failed", "reason": "startup_timeout",
                                       "waited_s": time.time() - t0})
            return
        time.sleep(2)
    startup_s = time.time() - t0
    status.update(state="running", startup_s=startup_s, steps_done=0)
    write_json("status.json", status)

    slo = PLAN.get("slo", {"ttft_ms": 2000, "tpot_ms": 100})
    step_results = []

    def spec_bench_subset(offset, n):
        """SPEC-Bench prompts [offset, offset+n) of a fixed permutation, as a file for --disable-shuffle.
        vLLM's SpecBench loader ignores --seed (always shuffles with seed 0), so steps that must not share
        prompts (warmups vs measured runs, held-out verification) select explicit, disjoint slices."""
        rows = open(SPEC_BENCH).read().splitlines()
        random.Random(0).shuffle(rows)
        sel = [rows[(offset + i) % len(rows)] for i in range(n)]
        path = f"/tmp/spec_bench_{offset}_{n}.jsonl"
        with open(path, "w") as f:
            f.write("\n".join(sel) + "\n")
        return path

    def bench_args(step, n, conc, sd, save_as, warmup=False):
        """`vllm bench serve` arguments for one (sub)step; save_as = result file name or None."""
        ds_name = step.get("dataset")
        if ds_name in ("spec_bench", "sharegpt"):
            ds = ["--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
                  "--temperature", str(step.get("temperature", 0.6)), "--top-p", str(step.get("top_p", 0.95))]
            if ds_name == "spec_bench":
                ds += ["--dataset-name", "spec_bench", "--spec-bench-output-len", str(step["osl"])]
                if "prompt_offset" in step:  # explicit slice; warmups use the reserved tail of the pool
                    off = SPEC_BENCH_WARMUP_OFFSET if warmup else step["prompt_offset"]
                    ds += ["--dataset-path", spec_bench_subset(off, n), "--disable-shuffle"]
                else:  # legacy (v1/v2 t1s): vLLM's own seed-0 order
                    ds += ["--dataset-path", SPEC_BENCH]
            else:  # ShareGPT honors --seed: every rung (and warmup) draws a different shuffle of 30k chats
                ds += ["--dataset-name", "sharegpt", "--dataset-path", SHAREGPT, "--sharegpt-output-len", str(step["osl"])]
            if step.get("ignore_eos", True):
                ds.append("--ignore-eos")
            if "thinking" in step:
                ds += ["--extra-body", json.dumps({"chat_template_kwargs": {"enable_thinking": bool(step["thinking"])}})]
        else:
            ds = ["--backend", step.get("backend", "openai"),
                  "--endpoint", step.get("endpoint", "/v1/completions"),
                  "--dataset-name", "random",
                  "--random-input-len", str(step["isl"]),
                  "--random-output-len", str(step["osl"]),
                  "--random-range-ratio", "0", "--ignore-eos"]
        out = ["vllm", "bench", "serve"] + ds + [
            "--base-url", BASE, "--model", PLAN["tokenizer"], "--served-model-name", PLAN["served_name"],
            "--tokenizer", PLAN["tokenizer"], "--num-prompts", str(n), "--max-concurrency", str(conc),
            "--num-warmups", "0", "--seed", str(sd), "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--metric-percentiles", "50,90,99", "--goodput", f"ttft:{slo['ttft_ms']}", f"tpot:{slo['tpot_ms']}"]
        if save_as:
            # --save-detailed keeps per-request TTFT/ITL/lengths, so the scorer can tell which SLO binds
            out += ["--save-result", "--save-detailed", "--result-dir", OUT, "--result-filename", save_as]
        if step.get("request_rate"):
            out += ["--request-rate", f"{step['request_rate']:.4f}", "--burstiness", "1.0"]
        return out

    def merge_parts(parts):
        """Combine concurrent sub-benchmarks (a mixed-length Poisson stream) into one result."""
        m = {"completed": sum(p.get("completed", 0) for p in parts), "failed": sum(p.get("failed", 0) for p in parts),
             "duration": max(p.get("duration", 0) for p in parts)}
        for k in ("request_throughput", "request_goodput", "output_throughput", "total_token_throughput",
                  "total_output_tokens", "total_input_tokens"):
            m[k] = sum(float(p.get(k) or 0) for p in parts)
        w = [p.get("completed", 0) for p in parts]
        for k in ("mean_ttft_ms", "median_ttft_ms", "mean_tpot_ms", "median_tpot_ms", "mean_itl_ms", "median_itl_ms",
                  "mean_e2el_ms", "median_e2el_ms"):
            if all(k in p for p in parts) and sum(w):
                m[k] = sum(p[k] * wi for p, wi in zip(parts, w)) / sum(w)
        for k in ("p90_ttft_ms", "p99_ttft_ms", "p90_tpot_ms", "p99_tpot_ms", "p90_itl_ms", "p99_itl_ms",
                  "p90_e2el_ms", "p99_e2el_ms"):
            vals = [p[k] for p in parts if k in p]
            if vals:
                m[k] = max(vals)  # conservative: the worse sub-stream
        m["parts"] = [{k: p.get(k) for k in ("completed", "request_throughput", "request_goodput", "p90_ttft_ms",
                                             "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms")} for p in parts]
        return m

    def run_step(i, step):
        name = f"step{i:02d}"
        seed = step.get("seed", 1000 + i)
        parts = step.get("mix")
        first = parts[0] if parts else step
        warm = step.get("warmups", min(4, first.get("c", 1)))
        if warm:
            # Warm up with *different* prompts (seed offset) so measured requests never hit the
            # prefix cache populated by the warmup (vllm bench's own warmup replays prompt #1).
            wstep = {k: v for k, v in first.items() if k != "request_rate"}
            wcmd = bench_args(wstep, warm, min(warm, first.get("c", 1)), seed + 7919, None, warmup=True)
            with open(f"{OUT}/{name}.warmup.log", "w") as log:
                try:
                    subprocess.run(wcmd, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                except subprocess.TimeoutExpired:
                    pass
        before = scrape()
        sampler = Sampler()
        sampler.start()
        ts = time.time()
        if parts:
            procs = []
            for j, sp in enumerate(parts):
                log = open(f"{OUT}/{name}.part{j}.log", "w")
                cmd = bench_args(sp, sp["n"], sp.get("c", 1024), sp.get("seed", seed + j), f"{name}.part{j}.json")
                procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            rc = 0
            for pr, log in procs:
                try:
                    r_ = pr.wait(timeout=step.get("timeout_s", 2400))
                except subprocess.TimeoutExpired:
                    pr.kill()
                    r_ = "timeout"
                log.close()
                rc = rc if r_ == 0 else r_
            if rc == 0:
                try:
                    merged = merge_parts([json.load(open(f"{OUT}/{name}.part{j}.json")) for j in range(len(parts))])
                    with open(f"{OUT}/{name}.json", "w") as f:
                        json.dump(merged, f)
                except Exception as e:
                    rc = f"merge failed: {e!r}"
        else:
            cmd = bench_args(step, step["n"], step["c"], seed, f"{name}.json")
            with open(f"{OUT}/{name}.log", "w") as log:
                try:
                    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=step.get("timeout_s", 1800)).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
        samples = sampler.stop()
        after = scrape()
        t_end = time.time()
        rec = {"step": name, "spec": step, "rc": rc, "wall_s": t_end - ts, "t_start": ts, "t_end": t_end,
               "server": summarize_samples(samples, before, after)}
        res = {}
        if rc == 0:
            try:
                with open(f"{OUT}/{name}.json") as f:
                    res = json.load(f)
                thr = float(res.get("request_throughput") or 0.0)
                gp = float(res.get("request_goodput") or 0.0)
                rec["goodput"] = gp
                # vLLM's goodput and throughput count completed requests only: a failed request is a miss
                c, f = int(res.get("completed") or 0), int(res.get("failed") or 0)
                rec["attainment"] = gp / thr * (c / (c + f)) if thr and c + f else 0.0
                if step.get("measure"):  # steady-state plans: judge only the requests inside the window
                    w = window_attainment(res, step["measure"], slo)
                    rec["attainment_all"] = rec["attainment"]
                    if w:
                        rec["attainment"], rec["window"] = w["attainment"], w
                    else:
                        rec["window_error"] = "no per-request data in the window"
            except Exception as e:
                rec["parse_error"] = repr(e)
        step_results.append(rec)
        write_json(f"{name}.server.json", {"samples": samples, "before": before, "after": after})
        status.update(steps_done=len(step_results))
        write_json("status.json", status)
        return rec

    prof = PLAN.get("profile")
    if prof:
        profile_run(prof, bench_args, status)
        write_json("summary.json", {"startup_s": startup_s, "steps": []})
        status.update(state="done", total_s=time.time() - t0)
        write_json("status.json", status)
        return

    rs = PLAN.get("rate_search")
    if rs:
        # MLPerf-style peak search: geometric ladder until SLO attainment < target, then bisect.
        att_target, base = rs.get("attainment", 0.9), dict(rs["step"])

        def probe(rate):
            st = dict(base, request_rate=rate, n=max(rs.get("min_n", 60), int(rate * rs.get("duration_s", 120))),
                      seed=rs.get("seed", 3000) + len(step_results), stop_on_error=False)
            if base.get("mix_fractions"):  # concurrent Poisson sub-streams whose rates sum to `rate`
                st["mix"] = [dict(sub, request_rate=f * rate, n=max(1, int(round(f * st["n"]))), seed=st["seed"] * 10 + j)
                             for j, (f, sub) in enumerate(base["mix_fractions"])]
            rec = run_step(len(step_results), st)
            return rec.get("rc") == 0 and rec.get("attainment", 0.0) >= att_target

        lo, hi, r = None, None, rs.get("start", 0.5)
        while r <= rs.get("max_rate", 64):
            if probe(r):
                lo, r = r, r * rs.get("factor", 1.5)
            else:
                hi = r
                break
        # a search that starts just below the expected capacity (steady-state plans) steps down if its first rung
        # fails (by down_factor), until a rung passes or the next would fall below min_rate
        down = rs.get("down_factor", rs.get("factor", 1.5))
        while lo is None and hi is not None and hi / down >= rs.get("min_rate", float("inf")):
            r = hi / down
            if probe(r):
                lo = r
            else:
                hi = r
        if lo is not None and hi is not None:
            # a fixed number of bisections, or (with "resolution") as many as it takes to bring hi/lo within it
            res, n = rs.get("resolution"), 0
            while (hi / lo > 1 + res and n < rs.get("max_bisect", 6)) if res else n < rs.get("bisect_iters", 4):
                mid = (lo * hi) ** 0.5
                if probe(mid):
                    lo = mid
                else:
                    hi = mid
                n += 1
        status.update(max_rate_at_slo=lo, first_failing_rate=hi)
    else:
        for i, step in enumerate(PLAN["steps"]):
            rec = run_step(i, step)
            rc = rec["rc"]
            drop = PLAN.get("stop_when_goodput_drops")
            if drop and rc == 0 and "goodput" in rec:
                best_gp = max([rec["goodput"]] + [r.get("goodput", 0.0) for r in step_results[:-1]])
                if i >= 2 and rec["goodput"] < drop * best_gp:
                    status.update(early_stop=f"{rec['step']} goodput {rec['goodput']:.3f} < {drop}x best {best_gp:.3f}")
                    write_json("status.json", status)
                    break
            if rc != 0 and step.get("stop_on_error", True):
                status.update(state="failed", reason=f"{rec['step']} rc={rc}")
                write_json("summary.json", {"startup_s": startup_s, "steps": step_results})
                write_json("status.json", status)
                return
    write_json("summary.json", {"startup_s": startup_s, "steps": step_results})
    status.update(state="done", total_s=time.time() - t0)
    write_json("status.json", status)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never die silently: controller reads status.json
        write_json("status.json", {"state": "failed", "reason": f"driver exception: {e!r}"})
    while True:  # keep the container alive for result collection
        time.sleep(3600)
