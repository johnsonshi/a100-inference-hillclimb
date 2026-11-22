"""T3 driver (runs in the trial pod's sidecar instead of bench_driver): agentic multi-turn sessions.

Sessions arrive as a Poisson process. Each session shares one ~3k-token system prompt (tool docs; reused
across sessions through the prefix cache), then runs N turns. Every turn adds a document (plan: ~1.5k tokens)
plus a question, so context grows to ~11k tokens by the last of 6 turns, and there is tool think time between
turns, so a session's KV must survive idle periods (or be recomputed). Replies are streamed (thinking off,
max_tokens capped) to measure time to first token per turn. Score: TTFT p90 over all turns (lower is better)."""
import json
import os
import random
import threading
import time
import urllib.request

PLAN = json.loads(os.environ["PLAN_JSON"])
TRIAL = os.environ["TRIAL_ID"]
OUT = f"/results/{TRIAL}"
BASE = "http://127.0.0.1:8000"
os.makedirs(OUT, exist_ok=True)


def write_json(name, obj):
    with open(f"{OUT}/{name}.tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(f"{OUT}/{name}.tmp", f"{OUT}/{name}")


def metrics():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=5) as r:
            text = r.read().decode(errors="replace")
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        for k, name in (("prefix_hits", "vllm:prefix_cache_hits_total"), ("prefix_queries", "vllm:prefix_cache_queries_total"),
                        ("ext_hits", "vllm:external_prefix_cache_hits_total"),
                        ("ext_queries", "vllm:external_prefix_cache_queries_total"),
                        ("preemptions", "vllm:num_preemptions_total"), ("sg_cached", "sglang:cached_tokens_total"),
                        ("sg_prompt", "sglang:prompt_tokens_total")):
            if line.startswith(name + "{") or line.startswith(name + " "):
                try:
                    out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def stream_turn(messages, max_tokens, temperature):
    """One streamed chat turn: (ttft_s, e2e_s, text, completion_tokens)."""
    body = {"model": PLAN["served_name"], "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, parts, ntok = None, [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except ValueError:
                continue
            if d.get("usage"):
                ntok = d["usage"].get("completion_tokens")
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    parts.append(piece)
    return ttft, time.time() - t0, "".join(parts), ntok


def build_texts(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PLAN["tokenizer"])
    rng = random.Random(cfg["seed"])
    qs = [json.loads(line)["turns"][0] for line in open("/models/datasets/spec_bench/question.jsonl")]
    gsm = [json.loads(line)["question"] for line in open("/models/datasets/gsm8k/test.jsonl")]

    def fill(pool, n_tokens, header):
        text, items = header, list(pool)
        rng.shuffle(items)
        for it in items:
            if len(tok(text, add_special_tokens=False)["input_ids"]) >= n_tokens:
                break
            text += "\n- " + it
        return text

    system = fill(qs, cfg["system_tokens"], "You are an assistant for an internal analytics team. Tool and "
                  "policy notes follow; consult them when relevant.\n")
    docs = [fill(gsm, cfg["doc_tokens"], f"Attached document #{i}:") for i in range(cfg["turns"] * 4)]
    return system, docs, qs, rng


def main():
    write_json("status.json", {"state": "waiting_for_server"})
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        if time.time() - t0 > PLAN.get("startup_timeout_s", 1800):
            write_json("status.json", {"state": "failed", "reason": "startup_timeout"})
            return
        time.sleep(2)
    startup_s = time.time() - t0
    write_json("status.json", {"state": "running", "startup_s": startup_s})
    cfg = PLAN["t3"]
    system, docs, qs, rng = build_texts(cfg)
    # one short warm-up turn on a different system prompt, so measured sessions start with a cold prefix cache
    try:
        stream_turn([{"role": "system", "content": "Warm-up."}, {"role": "user", "content": "Say hi."}], 8, 0.0)
    except Exception:
        pass
    arrivals, t = [], 0.0
    for _ in range(cfg["sessions"]):
        t += rng.expovariate(cfg["session_rate"])
        arrivals.append(t)
    plans = [[(rng.randrange(len(docs)), rng.randrange(len(qs)), rng.expovariate(1 / cfg["think_mean_s"]))
              for _ in range(cfg["turns"])] for _ in arrivals]
    turns, errors, lock = [], [], threading.Lock()
    before = metrics()
    start = time.time()

    def session(i):
        time.sleep(max(0.0, start + arrivals[i] - time.time()))
        hist = [{"role": "system", "content": system}]
        for j, (d, q, think) in enumerate(plans[i]):
            hist.append({"role": "user", "content": docs[d] + "\n\nQuestion: " + qs[q]})
            try:
                ttft, e2e, text, ntok = stream_turn(hist, cfg["max_tokens"], 0.6)
            except Exception as e:
                with lock:
                    errors.append({"session": i, "turn": j, "error": repr(e)[:200]})
                return
            with lock:
                turns.append({"session": i, "turn": j, "ttft_s": ttft, "e2e_s": e2e, "tokens": ntok})
            hist.append({"role": "assistant", "content": text})
            time.sleep(think)

    threads = [threading.Thread(target=session, args=(i,), daemon=True) for i in range(len(arrivals))]
    for th in threads:
        th.start()
    deadline = time.time() + PLAN.get("timeout_s", 5400) - 300
    for th in threads:
        th.join(timeout=max(1.0, deadline - time.time()))
    wall = time.time() - start
    after = metrics()
    ttfts = [x["ttft_s"] * 1000 for x in turns if x["ttft_s"] is not None]
    by_turn = {}
    for x in turns:
        if x["ttft_s"] is not None:
            by_turn.setdefault(x["turn"], []).append(x["ttft_s"] * 1000)
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    hit = (delta.get("prefix_hits", 0) / delta["prefix_queries"]) if delta.get("prefix_queries") else None
    if hit is None and delta.get("sg_prompt"):
        hit = delta.get("sg_cached", 0) / delta["sg_prompt"]
    res = {"sessions": len(arrivals), "turns_planned": len(arrivals) * cfg["turns"], "turns_done": len(turns),
           "errors": len(errors), "error_samples": errors[:5], "wall_s": wall,
           "ttft_p50_ms": pct(ttfts, 50), "ttft_p90_ms": pct(ttfts, 90), "ttft_p99_ms": pct(ttfts, 99),
           "e2e_p90_s": pct([x["e2e_s"] for x in turns], 90),
           "ttft_p90_by_turn_ms": {str(k): pct(v, 90) for k, v in sorted(by_turn.items())},
           "output_tokens": sum(x["tokens"] or 0 for x in turns), "prefix_hit_ratio": hit,
           "ext_hit_ratio": (delta.get("ext_hits", 0) / delta["ext_queries"]) if delta.get("ext_queries") else None,
           "preemptions": delta.get("preemptions"), "config": cfg}
    write_json("summary.json", {"startup_s": startup_s, "steps": [], "t3": res})
    write_json("status.json", {"state": "done", "startup_s": startup_s})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_json("status.json", {"state": "failed", "reason": f"t3 driver exception: {e!r}"})
    while True:
        time.sleep(3600)
