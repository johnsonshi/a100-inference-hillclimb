"""Offline test of bench_driver's control flow (rate search over mixed-length Poisson sub-streams, merge,
summary) with a fake server and a fake `vllm bench serve` that writes result files.
Run: python3 -m tests.test_bench_driver"""
import importlib
import json
import os
import subprocess
import sys
import tempfile

from harness import plans

CAP = 1.3  # fake: attainment starts falling above this total rate (req/s)
SLOPE = 3.0
CAP_SS = 1.9  # fake steady state: requests after the first 120 s meet the SLO only at or below this rate
FAKE = {"total_rate": 0.0, "mode": "mix"}


def args_of(cmd):
    return {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}


def write_steady(a):
    """Per-request results like --save-detailed: the ramp-up (first 120 s) always passes; later requests pass only
    if the rate is sustainable (TPOT 50 vs 150 ms); the tail after the measurement window always fails, so a passing
    rung's all-request attainment (~0.86) is below the target while its window's is 1.0. Like the cluster's long
    windows, every rung's last request fails with a client error (completed = n - 1, failed = 1)."""
    n, rate = int(a["--num-prompts"]), float(a["--request-rate"])
    st = [1000.0 + i / rate for i in range(n)]
    ok = [(s - st[0] < 120 or rate <= CAP_SS) and s - st[0] < 360 for s in st]
    ok[-1] = False
    errs = [""] * (n - 1) + ["aiohttp.client_exceptions.ClientOSError: [Errno 104] Connection reset by peer"]
    dur = n / rate + 60
    res = {"completed": n - 1, "failed": 1, "duration": dur, "request_throughput": (n - 1) / dur,
           "request_goodput": sum(ok) / dur, "total_output_tokens": (n - 1) * 512, "total_input_tokens": n * 1024,
           "median_ttft_ms": 200.0, "median_tpot_ms": 50.0, "start_times": st, "ttfts": [0.2] * (n - 1) + [0.0],
           "latencies": [0.2 + 511 * (0.05 if k else 0.15) for k in ok[:-1]] + [0.0],
           "output_lens": [512] * (n - 1) + [0], "errors": errs}
    with open(os.path.join(a["--result-dir"], a["--result-filename"]), "w") as f:
        json.dump(res, f)


def write_result(cmd):
    a = args_of(cmd)
    if FAKE["mode"] == "steady":
        if "--result-filename" in a:
            write_steady(a)
        return
    if a.get("--random-input-len") == "1024" and "--request-rate" in a:  # the 85% stream sets the total
        FAKE["total_rate"] = float(a["--request-rate"]) / 0.85
    if "--result-filename" not in a:
        return
    n, total = int(a["--num-prompts"]), FAKE["total_rate"]
    att = 0.97 if total <= CAP else max(0.0, 0.97 - SLOPE * (total - CAP) / CAP)
    thr = float(a["--request-rate"]) * 0.9
    res = {"completed": n, "failed": 0, "duration": n / thr, "request_throughput": thr, "request_goodput": thr * att,
           "output_throughput": thr * 512, "total_output_tokens": n * 512, "total_input_tokens": n * 1024,
           "median_ttft_ms": 100.0, "median_tpot_ms": 30.0, "p99_ttft_ms": 500.0, "p99_tpot_ms": 40.0}
    with open(os.path.join(a["--result-dir"], a["--result-filename"]), "w") as f:
        json.dump(res, f)


class FakePopen:
    def __init__(self, cmd, stdout=None, stderr=None):
        write_result(cmd)

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def fake_run(cmd, stdout=None, stderr=None, timeout=None, **kw):
    if cmd[:3] == ["vllm", "bench", "serve"]:
        write_result(cmd)
    return subprocess.CompletedProcess(cmd, 0)


def main():
    out = tempfile.mkdtemp(prefix="hc-drv-")
    plan = dict(plans.get("t2m"), tokenizer="/models/x", served_name="nanbeige", slo=plans.SLO, engine="vllm")
    os.environ.update(PLAN_JSON=json.dumps(plan), TRIAL_ID="t-test", HC_OUT=out)
    sys.path.insert(0, "harness")
    bd = importlib.import_module("bench_driver")
    bd.http_get = lambda path, timeout=5: (200, "{}")
    bd.scrape = lambda: {}
    bd.subprocess.run = fake_run
    bd.subprocess.Popen = FakePopen
    bd.main()
    summ = json.load(open(os.path.join(out, "summary.json")))
    status = json.load(open(os.path.join(out, "status.json")))
    rates = [s["spec"]["request_rate"] for s in summ["steps"]]
    print("rungs (total req/s):", [round(r, 3) for r in rates])
    print("attainment:", [round(s.get("attainment", -1), 3) for s in summ["steps"]])
    print("status:", {k: status.get(k) for k in ("state", "max_rate_at_slo", "first_failing_rate")})
    for s in summ["steps"]:
        mix = s["spec"]["mix"]
        assert abs(sum(m["request_rate"] for m in mix) - s["spec"]["request_rate"]) < 1e-9, "sub-stream rates must sum"
        assert sorted(m["isl"] for m in mix) == [1024, 8192]
        assert len({m["seed"] for m in mix}) == 2, "sub-streams need distinct seeds"
        assert os.path.exists(os.path.join(out, f"{s['step']}.json")), "merged result missing"
    crossing = CAP * (1 + (0.97 - plans.ATTAINMENT) / SLOPE)
    assert rates[0] == plan["rate_search"]["start"] and status["state"] == "done"
    assert status["max_rate_at_slo"] <= crossing < status["first_failing_rate"], "bracket must contain the crossing"
    steady(bd)
    print("BENCH DRIVER TEST OK", out)


def steady(bd):
    """Steady-state search (t2ss): rungs are judged on requests inside the measurement window only, the scorer's
    mirror agrees with the driver, and a first rung that fails makes the search step down."""
    from pathlib import Path

    from harness import score
    FAKE["mode"] = "steady"
    for cap_guess in (2.0, 2.2, 4.0):  # first rung 1.8 passes / 1.98 fails / 3.6 fails, five 15% steps above 1.9
        out = tempfile.mkdtemp(prefix="hc-drv-ss-")
        bd.OUT = out
        bd.PLAN = dict(plans.get(f"t2ss@{cap_guess}"), tokenizer="/models/x", served_name="nanbeige", slo=plans.SLO,
                       engine="vllm")
        bd.main()
        summ = json.load(open(os.path.join(out, "summary.json")))
        status = json.load(open(os.path.join(out, "status.json")))
        rates = [s["spec"]["request_rate"] for s in summ["steps"]]
        print(f"steady t2ss@{cap_guess}: rungs {[round(r, 3) for r in rates]} window attainment "
              f"{[round(s['attainment'], 2) for s in summ['steps']]} all-request "
              f"{[round(s['attainment_all'], 2) for s in summ['steps']]}")
        assert abs(rates[0] - 0.9 * cap_guess) < 1e-9 and status["state"] == "done"
        assert status["max_rate_at_slo"] <= CAP_SS < status["first_failing_rate"], status
        assert status["first_failing_rate"] / status["max_rate_at_slo"] <= 1.01 + 1e-9, "bisect to 1%"
        for s in summ["steps"]:
            want = 1.0 if s["spec"]["request_rate"] <= CAP_SS else 0.0
            assert s["attainment"] == want and s["window"]["n"] > 0, s
            assert s["attainment_all"] < 0.9, "neither the ramp-up nor the tail may count in the window"
            mirror = score._window_split(json.load(open(os.path.join(out, f"{s['step']}.json"))), s["spec"])
            assert mirror["attainment_window"] == s["attainment"] and mirror["window_n"] == s["window"]["n"], mirror
        m = score.extract(Path(out))
        sc = score.scores(m)
        assert len(sc["t2_points"]) == len(rates), "a rung with one failed request must still be scored"
        assert status["max_rate_at_slo"] <= sc["t2_capacity"] <= status["first_failing_rate"], (sc, status)
        assert sc["t2_max_rate"] == status["max_rate_at_slo"], "t2_max_rate must judge steady steps by their window"
    assert any(r < 0.5 * 4.0 for r in rates), "a failing first rung must make the search step down, by 15% per rung"
    # outside a measurement window, a failed request is a miss: 99 of 100 requests completed and met the SLO
    one_failed = {"steps": [{"request_rate": 1.0, "n": 100, "completed": 99, "failed": 1, "request_throughput": 0.99,
                             "request_goodput": 0.99, "osl": 512, "total_output_tokens": 99 * 512}]}
    assert score.t2_points(one_failed) == [(1.0, 0.99, 512.0)], score.t2_points(one_failed)
    FAKE["mode"] = "mix"


if __name__ == "__main__":
    main()
