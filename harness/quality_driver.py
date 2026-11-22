"""Quality driver (runs in the trial pod's sidecar instead of bench_driver): GSM8K accuracy at
temperature 0 (thinking disabled unless the plan sets "thinking": true), greedy 128-token outputs (thinking
off) for speculative-equivalence checks, and with the v2 gate (quality2): passkey retrieval at 8k-28k
context (thinking off) and teacher-forced NLL / top-1 agreement on BF16's greedy continuations.
Writes /results/<trial>/quality.json and status.json."""
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PLAN = json.loads(os.environ["PLAN_JSON"])
TRIAL = os.environ["TRIAL_ID"]
OUT = f"/results/{TRIAL}"
BASE = "http://127.0.0.1:8000"
os.makedirs(OUT, exist_ok=True)


def write_json(name, obj):
    with open(f"{OUT}/{name}.tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(f"{OUT}/{name}.tmp", f"{OUT}/{name}")


def chat(content, max_tokens, temperature=0.0, thinking=None):
    """Greedy chat turn; `thinking` defaults to the plan's setting (GSM8K), short probes pass False."""
    thinking = bool(PLAN.get("thinking", False)) if thinking is None else thinking
    body = {"model": PLAN["served_name"], "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": temperature, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return (m.get("content") or "") + ("" if m.get("content") else (m.get("reasoning_content") or m.get("reasoning") or ""))


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(PLAN["tokenizer"])
    return _TOK


def passkey_check():
    """Retrieve a 5-digit key hidden at 10/50/90% depth of 8k/16k/28k-token filler (thinking off, greedy).
    Long-context attention / KV-cache numerics (e.g. FP8 KV) fail here before they show up in GSM8K."""
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    per = len(tok()(filler, add_special_tokens=False)["input_ids"])
    rng = random.Random(0)
    res = []
    for target in (8000, 16000, 28000):
        for depth in (0.1, 0.5, 0.9):
            key = rng.randint(10000, 99999)
            reps = target // per
            k = int(reps * depth)
            text = (filler * k + f"The pass key is {key}. Remember it. {key} is the pass key. " + filler * (reps - k)
                    + "\nWhat is the pass key? Reply with the number only.")
            try:
                out = chat(text, 32, thinking=False)
                ok = str(key) in out
            except Exception as e:
                out, ok = f"<error {e!r}>", False
            res.append({"tokens": target, "depth": depth, "ok": ok, "out": out[-80:]})
    return {"passkey_acc": sum(r["ok"] for r in res) / len(res), "passkey": res}


def teacher_forced(prompt_ids, cont_ids):
    """Log-probability of each continuation token and the model's own top-1 token at that position,
    with the continuation fed as part of the prompt (teacher forcing)."""
    ids = prompt_ids + cont_ids
    if PLAN.get("engine") == "sglang":
        d = post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                               "return_logprob": True, "logprob_start_len": len(prompt_ids), "top_logprobs_num": 1})
        mi = d.get("meta_info", {})
        lps = [x[0] for x in mi.get("input_token_logprobs", [])][-len(cont_ids):]
        tops = [(x[0][1] if x else None) for x in mi.get("input_top_logprobs", [])][-len(cont_ids):]
        return lps, tops
    d = post("/v1/completions", {"model": PLAN["served_name"], "prompt": ids, "max_tokens": 1, "temperature": 0,
                                 "prompt_logprobs": 1})
    pl = d["choices"][0].get("prompt_logprobs") or []
    lps, tops = [], []
    for j in range(len(prompt_ids), len(ids)):
        e = pl[j] or {}
        a = e.get(str(ids[j])) or {}
        lps.append(a.get("logprob"))
        top = [int(t) for t, v in e.items() if v.get("rank") == 1]
        tops.append(top[0] if top else None)
    return lps, tops


def make_refs(rows):
    """BF16 references: greedy continuations (128 tokens, thinking off) and their own teacher-forced
    log-probabilities. Candidates are scored on exactly these token sequences."""
    refs = []
    for r in rows:
        prompt_ids = tok().apply_chat_template([{"role": "user", "content": r["turns"][0]}], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False)
        # transformers returns a BatchEncoding (a UserDict, not a dict) for tokenize=True
        prompt_ids = [int(x) for x in (prompt_ids["input_ids"] if hasattr(prompt_ids, "keys") else prompt_ids)]
        d = post("/v1/completions", {"model": PLAN["served_name"], "prompt": prompt_ids, "max_tokens": 128,
                                     "temperature": 0, "return_token_ids": True})
        # the exact generated ids (re-tokenizing the text can split differently)
        cont_ids = [int(x) for x in (d["choices"][0].get("token_ids") or
                                     tok()(d["choices"][0]["text"], add_special_tokens=False)["input_ids"])][:128]
        if not cont_ids:
            continue
        lps, _ = teacher_forced(prompt_ids, cont_ids)
        refs.append({"prompt_ids": prompt_ids, "cont_ids": cont_ids, "ref_lp": lps})
    return refs


def score_refs(refs):
    """Teacher-forced NLL of BF16's continuations under this config, and how often this config's own
    top-1 token equals BF16's greedy token (token agreement)."""
    dn, agree, n = [], 0, 0
    for ref in refs:
        try:
            lps, tops = teacher_forced(ref["prompt_ids"], ref["cont_ids"])
        except Exception:
            continue
        for lp, rl, top, tid in zip(lps, ref["ref_lp"], tops, ref["cont_ids"]):
            if lp is None or rl is None:
                continue
            dn.append(-lp + rl)
            agree += int(top == tid)
            n += 1
    if not n:
        return {}
    return {"nll_delta": sum(dn) / n, "top1_agree": agree / n, "ref_tokens": n}


def num(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k_answer(text):
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"\\boxed\{\s*\$?(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", text)
    return num(m[-1]) if m else None


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

    rows = [json.loads(l) for l in open("/models/datasets/gsm8k/test.jsonl")][: PLAN.get("gsm8k_n", 500)]
    prompt = ("Solve the following math problem. Show brief reasoning, then give the final answer as a "
              "single number on the last line in the form '#### <number>'.\n\n{q}")

    def one(r):
        try:
            out = chat(prompt.format(q=r["question"]), PLAN.get("gsm8k_max_tokens", 1024))
        except Exception as e:
            return {"ok": False, "error": repr(e)}
        gold = num(r["answer"].split("####")[-1].strip())
        pred = gsm8k_answer(out)
        return {"ok": pred is not None and gold is not None and abs(pred - gold) < 1e-6, "pred": pred, "gold": gold,
                "out_tail": out[-200:]}

    t = time.time()
    with ThreadPoolExecutor(max_workers=PLAN.get("concurrency", 32)) as ex:
        res = list(ex.map(one, rows))
    acc = sum(r["ok"] for r in res) / len(res)
    errors = sum(1 for r in res if "error" in r)

    refs = [json.loads(l) for l in open("/models/datasets/spec_bench/question.jsonl")][:: max(1, 480 // 32)][:32]

    def gen(r):
        try:
            return chat(r["turns"][0], 128, thinking=False)
        except Exception as e:
            return f"<error {e!r}>"

    with ThreadPoolExecutor(max_workers=8) as ex:
        greedy = list(ex.map(gen, refs))
    q = {"gsm8k_n": len(res), "gsm8k_acc": acc, "gsm8k_errors": errors, "gsm8k_s": time.time() - t,
         "thinking": bool(PLAN.get("thinking", False)), "greedy_outputs": greedy, "samples": res[:20]}
    # optional sections fail soft: an error is recorded instead of failing the GSM8K result with it
    if PLAN.get("passkey"):
        try:
            q.update(passkey_check())
        except Exception as e:
            q["passkey_error"] = repr(e)[:300]
    try:
        if PLAN.get("make_refs"):
            q["refs"] = make_refs(refs)
            q.update(score_refs(q["refs"]))  # BF16 against itself: nll_delta ~0, top1_agree ~1 (sanity check)
        elif PLAN.get("use_refs") and os.path.exists("/refs/refs.json"):
            q.update(score_refs(json.load(open("/refs/refs.json"))))
    except Exception as e:
        q["refs_error"] = repr(e)[:300]
    write_json("quality.json", q)
    write_json("summary.json", {"startup_s": startup_s, "steps": []})
    write_json("status.json", {"state": "done", "startup_s": startup_s, "gsm8k_acc": acc})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_json("status.json", {"state": "failed", "reason": f"quality driver exception: {e!r}"})
    while True:
        time.sleep(3600)
