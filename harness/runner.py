"""Runs trials on the AKS cluster: renders a pod (server + bench sidecar) pinned to a lane's
node, waits for the bench driver to finish, copies results, records the ledger row.
Never touches the cluster or node pools themselves - only pods in the default namespace."""
import datetime
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path

from . import ledger, plans, score, settings, space

settings.load()
# Locally: the kubeconfig context from .env (HILLCLIMB_KUBE_CONTEXT). Inside the cluster the controller image sets
# HILLCLIMB_KUBE_CONTEXT="" to use the pod's service account.
CONTEXT = os.environ.get("HILLCLIMB_KUBE_CONTEXT")
NS = "default"
LANES = settings.lanes()
REPO = Path(__file__).resolve().parents[1]
TRIALS = REPO / "results" / "trials"
LABEL = "app.kubernetes.io/part-of=a100-hillclimb,app=hillclimb-trial"
DRIVERS = ("bench_driver.py", "quality_driver.py", "t3_driver.py")
REFS_CM = "hillclimb-quality-refs"
_driver_lock = threading.Lock()
_driver_synced = False


def kubectl(*args, input=None, timeout=120, check=True):
    if CONTEXT is None:
        raise RuntimeError("HILLCLIMB_KUBE_CONTEXT is not set: copy .env.example to .env and fill it in")
    cmd = ["kubectl", *(["--context", CONTEXT] if CONTEXT else []), "-n", NS, *args]
    try:
        p = subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if check:
            raise
        return ""
    if check and p.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args[:3])} failed: {p.stderr.strip()[:400]}")
    return p.stdout


# Per-process CPU time sampler for the server container (utime+stime from /proc, every 2 s): shows whether
# the engine's scheduler/API processes are CPU-bound (the per-token fixed overhead question).
CPU_SAMPLER = r"""import os, sys, time
f = open(sys.argv[1], "w")
while True:
    rows = []
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                s = open("/proc/%s/stat" % p).read()
            except Exception:
                continue
            r = s.rfind(")")
            x = s[r + 2:].split()
            rows.append("%s|%s|%s|%s" % (p, s[s.find("(") + 1:r], x[11], x[12]))
    f.write("T %f\n" % time.time() + "\n".join(rows) + "\n")
    f.flush()
    time.sleep(2)
"""


def publish_refs(local, trial_id, log=print):
    """Make a BF16 quality run's teacher-forcing references available to later quality2 trials."""
    q = local / "quality.json"
    if not q.exists():
        return
    refs = json.loads(q.read_text()).get("refs")
    if not refs:
        return
    cm = {"apiVersion": "v1", "kind": "ConfigMap",
          "metadata": {"name": REFS_CM, "namespace": NS,
                       "labels": {"app.kubernetes.io/part-of": "a100-hillclimb"},
                       "annotations": {"hillclimb/source-trial": trial_id}},
          "data": {"refs.json": json.dumps(refs)}}
    kubectl("apply", "-f", "-", input=json.dumps(cm))
    log(f"published {len(refs)} quality references from {trial_id} to configmap/{REFS_CM}")


def sync_driver():
    global _driver_synced
    with _driver_lock:
        if _driver_synced:
            return
        cm = {"apiVersion": "v1", "kind": "ConfigMap",
              "metadata": {"name": "hillclimb-bench-driver", "namespace": NS,
                           "labels": {"app.kubernetes.io/part-of": "a100-hillclimb"}},
              "data": {f: (REPO / "harness" / f).read_text() for f in DRIVERS}}
        kubectl("apply", "-f", "-", input=json.dumps(cm))
        _driver_synced = True


def render_pod(trial_id, cfg, plan_name, node):
    cfg = space.full(cfg)
    eng = cfg["engine"]
    plan = dict(plans.get(plan_name))
    profile_dir = f"/results/{trial_id}/prof" if plan.get("profile") and eng == "vllm" else None
    # vLLM's MFU estimator costs some Python per scheduler step: enable it only on load plans (rate searches,
    # Poisson steps, T3), where steps are long and GPU-bound, never on single-user latency plans
    load = (bool(plan.get("rate_search")) or plan.get("driver") == "t3_driver.py"
            or any(s.get("request_rate") or s.get("mix") for s in plan.get("steps", [])))
    args, extra_env = space.server_command(cfg, profile_dir=profile_dir, mfu=load)
    plan.update(tokenizer=space.TOKENIZER, served_name=space.SERVED_NAME, slo=plans.SLO, engine=eng,
                startup_timeout_s=plan.get("startup_timeout_s", 1800))
    sampler = ("nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,"
               "clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader "
               f"-lms 2000 > /results/{trial_id}/gpu.csv 2>&1 &")
    cpu_sampler = f"python3 -c {shlex.quote(CPU_SAMPLER)} /results/{trial_id}/cpu.txt >/dev/null 2>&1 &"
    script = f"mkdir -p /results/{trial_id} && {sampler} {cpu_sampler} exec {shlex.join(args)}"
    env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": "/cache/hf-home",
           "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HOME": f"/cache/home-{eng}",
           "TRITON_CACHE_DIR": f"/cache/triton-{eng}", "VLLM_CACHE_ROOT": "/cache/vllm",
           "XDG_CACHE_HOME": f"/cache/xdg-{eng}", "PYTHONUNBUFFERED": "1", **extra_env}
    if eng == "sglang":  # SGLang's torch.compile cache otherwise lands in /tmp and is lost with the pod
        env["TORCHINDUCTOR_CACHE_DIR"] = "/cache/inductor-sglang"
    vols = [("models", "/mnt/nvme/hf", "/models", True), ("cache", "/mnt/nvme/cache", "/cache", False),
            ("results", "/mnt/nvme/results", "/results", False)]
    # vLLM's CPU KV offload region lives in /dev/shm (it refuses to start unless the tmpfs has room for it), so the
    # tmpfs gets the region on top of the usual 32 GiB; SGLang's HiCache doesn't use /dev/shm
    shm_gib = 32 + (cfg.get("kv_offloading_gib") or 0)
    name = f"hc-{trial_id}"
    return name, {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": name, "namespace": NS,
                     "labels": {"app": "hillclimb-trial", "app.kubernetes.io/part-of": "a100-hillclimb",
                                "hillclimb/trial": trial_id, "hillclimb/engine": eng}},
        "spec": {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": plan.get("timeout_s", 3600) + 2400,
            "terminationGracePeriodSeconds": 5,
            "nodeSelector": {"kubernetes.io/hostname": node},
            "containers": [
                {"name": "server", "image": space.IMAGES[eng], "command": ["/bin/bash", "-c"], "args": [script],
                 "env": [{"name": k, "value": v} for k, v in env.items()],
                 "ports": [{"containerPort": 8000}],
                 "resources": {"requests": {"cpu": "8", "memory": "64Gi", "nvidia.com/gpu": "1"},
                               "limits": {"memory": "180Gi", "nvidia.com/gpu": "1"}},
                 "volumeMounts": [{"name": v[0], "mountPath": v[2], "readOnly": v[3]} for v in vols]
                 + [{"name": "dshm", "mountPath": "/dev/shm"}]},
                {"name": "bench", "image": space.BENCH_IMAGE,
                 "command": ["python3", f"/driver/{plan.get('driver', 'bench_driver.py')}"],
                 "env": [{"name": "TRIAL_ID", "value": trial_id}, {"name": "PLAN_JSON", "value": json.dumps(plan)},
                         {"name": "HF_HUB_OFFLINE", "value": "1"}, {"name": "VLLM_NO_USAGE_STATS", "value": "1"},
                         {"name": "HOME", "value": "/tmp"}],
                 "resources": {"requests": {"cpu": "4", "memory": "8Gi"}, "limits": {"memory": "16Gi"}},
                 "volumeMounts": [{"name": "models", "mountPath": "/models", "readOnly": True},
                                  {"name": "results", "mountPath": "/results"},
                                  {"name": "driver", "mountPath": "/driver"},
                                  {"name": "refs", "mountPath": "/refs", "readOnly": True}]},
            ],
            "volumes": [{"name": v[0], "hostPath": {"path": v[1], "type": "Directory"}} for v in vols]
            + [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": f"{shm_gib}Gi"}},
               {"name": "driver", "configMap": {"name": "hillclimb-bench-driver"}},
               {"name": "refs", "configMap": {"name": REFS_CM, "optional": True}}],
        },
    }


def cleanup_lane(node):
    out = kubectl("get", "pods", "-l", LABEL, "-o", "json", check=False)
    for p in json.loads(out or '{"items": []}')["items"]:
        if p["spec"].get("nodeName") == node or p["spec"].get("nodeSelector", {}).get("kubernetes.io/hostname") == node:
            kubectl("delete", "pod", p["metadata"]["name"], "--wait=true", "--timeout=120s", check=False, timeout=180)


def run_trial(cfg, plan_name, lane, campaign, parent_id=None, move=None, notes=None, log=print):
    cfg = space.full(cfg)
    node = LANES.get(lane)
    if not node:
        raise RuntimeError(f"no node for lane {lane}: set HILLCLIMB_LANES in .env (or the controller's env)")
    chash = space.config_hash(cfg)
    trial_id = f"t{datetime.datetime.utcnow():%y%m%d%H%M%S}-{chash[:6]}-{lane}"
    ledger.start(trial_id, campaign, plan_name, lane, node, cfg, chash, parent_id, move, notes)
    reason = space.validity(cfg)
    if reason:
        ledger.finish(trial_id, "invalid", reason, None)
        return trial_id, "invalid", None
    sync_driver()
    cleanup_lane(node)
    name, pod = render_pod(trial_id, cfg, plan_name, node)
    kubectl("apply", "-f", "-", input=json.dumps(pod))
    log(f"[{lane}] {trial_id} started {cfg['engine']}/{cfg['variant']} move={move}")
    t0 = time.time()
    plan_timeout = plans.get(plan_name).get("timeout_s", 3600) + 2100
    state, why = "error", "unknown"
    while True:
        time.sleep(15)
        raw = kubectl("get", "pod", name, "-o", "json", check=False)
        if not raw:
            state, why = "error", "pod disappeared"
            break
        pod_s = json.loads(raw).get("status", {})
        cs = {c["name"]: c for c in pod_s.get("containerStatuses", [])}
        srv = cs.get("server", {}).get("state", {})
        if "terminated" in srv:
            t = srv["terminated"]
            state, why = "infeasible", f"server exited: {t.get('reason')} code={t.get('exitCode')}"
            break
        waiting = srv.get("waiting", {})
        if waiting.get("reason") in ("ErrImagePull", "ImagePullBackOff", "CreateContainerConfigError"):
            if time.time() - t0 > 900:
                state, why = "error", waiting["reason"]
                break
        if pod_s.get("phase") == "Pending" and time.time() - t0 > 1200:
            state, why = "error", "pending too long"
            break
        if cs.get("bench", {}).get("state", {}).get("running"):
            st = kubectl("exec", name, "-c", "bench", "--", "cat", f"/results/{trial_id}/status.json",
                         check=False, timeout=60)
            try:
                st = json.loads(st)
            except (ValueError, TypeError):
                st = {}
            if st.get("state") == "done":
                state, why = "done", st.get("early_stop")
                break
            if st.get("state") == "failed":
                state, why = "failed", st.get("reason")
                break
        if time.time() - t0 > plan_timeout:
            state, why = "error", "trial timeout"
            break
    local = TRIALS / trial_id
    local.mkdir(parents=True, exist_ok=True)
    kubectl("cp", f"{name}:/results/{trial_id}", str(local), "-c", "bench", check=False, timeout=600)
    (local / "server.log").write_text(kubectl("logs", name, "-c", "server", "--tail", "6000", check=False, timeout=120))
    # The tail above can lose the startup lines (backend, kernels, graph capture) on a long, busy run; keep them.
    (local / "server.head.log").write_text(kubectl("logs", name, "-c", "server", "--limit-bytes", "262144",
                                                   check=False, timeout=120))
    (local / "pod.json").write_text(json.dumps(pod, indent=1))
    kubectl("delete", "pod", name, "--wait=false", check=False)
    metrics = score.extract(local)
    if state == "done" and plans.get(plan_name).get("make_refs"):
        publish_refs(local, trial_id, log)
    if state == "infeasible":
        tail = (local / "server.log").read_text(errors="replace").strip().splitlines()[-12:]
        why = (why or "") + " | " + " / ".join(x[-200:] for x in tail)
    ledger.finish(trial_id, state, why, metrics)
    log(f"[{lane}] {trial_id} {state} {('(' + why[:160] + ')') if why else ''} scores={metrics.get('scores')} "
        f"kv={metrics.get('kv_tokens')} startup={metrics.get('startup_s')} in {time.time() - t0:.0f}s")
    return trial_id, state, metrics


def run_many(jobs, campaign, lanes=("n0", "n1"), log=print):
    """jobs: list of (cfg, plan_name, meta dict). Runs across lanes in parallel; returns results in order."""
    q = queue.Queue()
    for i, j in enumerate(jobs):
        q.put((i, j))
    results = [None] * len(jobs)

    def worker(lane):
        while True:
            try:
                i, (cfg, plan_name, meta) = q.get_nowait()
            except queue.Empty:
                return
            try:
                results[i] = run_trial(cfg, plan_name, lane, campaign, log=log, **(meta or {}))
            except Exception as e:  # keep the lane alive
                log(f"[{lane}] job {i} crashed: {e!r}")
                results[i] = (None, "error", None)

    threads = [threading.Thread(target=worker, args=(ln,), daemon=True) for ln in lanes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results
