#!/usr/bin/env python3
"""Charts for the study (README.md), drawn from the ledger by trial ID.

Needs matplotlib, which the harness itself doesn't use:
    python tools/post_charts.py [ledger.sqlite] [output dir]
Defaults: results/ledger.sqlite and assets/."""
import json
import sqlite3
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import score  # noqa: E402

LEDGER = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "ledger.sqlite"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "assets"
DB = sqlite3.connect(LEDGER)

GRAY, BLUE, LIGHT_BLUE, AMBER, RED, GREEN, INK = "#9AA0A6", "#1A73E8", "#8AB4F8", "#E37400", "#D93025", "#188038", "#202124"
FOOT = "Nanbeige4.3-33B on one A100 80GB PCIe · vLLM 0.34.0, SGLang 0.5.23 · each value is a measured trial"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#E8EAED", "axes.axisbelow": True,
                     "axes.edgecolor": "#5F6368", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def metrics(tid):
    row = DB.execute("SELECT metrics_json FROM trials WHERE trial_id=?", (tid,)).fetchone()
    if not row:
        sys.exit(f"trial {tid} not in {LEDGER}")
    return json.loads(row[0])


def sc(tid, key):
    return score.scores(metrics(tid))[key]


def p90(tid):
    return metrics(tid)["t3"]["ttft_p90_ms"] / 1000.0


def mean(xs):
    return statistics.mean(xs)


def pct(v, digits=0):
    """Signed percentage with a typographic minus."""
    return f"{v * 100:+.{digits}f}%".replace("-", "−")


def finish(fig, name, foot=FOOT):
    fig.text(0.01, 0.01, foot, fontsize=8.5, color="#5F6368")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", OUT / name)


# ---- values (trial IDs as in the post) ----
T1_BF16 = sc("t261119124531-4af508-n0", "t1_decode_tps")
T1_INT4 = sc("t261119124531-696455-n1", "t1_decode_tps")
T1_SUFFIX6 = sc("t261119130117-4c6b3a-n0", "t1_decode_tps")
T1_BEST = sc("t261119184031-c00f32-n0", "t1_decode_tps")
T1_K6_ABLATION = sc("t261121024032-4c6b3a-n0", "t1_decode_tps")
SG_T1 = mean([sc(t, "t1_decode_tps") for t in ("t261119132328-86d189-n0", "t261119213213-86d189-n1",
                                                 "t261119214037-86d189-n0", "t261119214037-86d189-n1")])
SG_T1_N0 = mean([sc(t, "t1_decode_tps") for t in ("t261119132328-86d189-n0", "t261119214037-86d189-n0")])

T2_BF16 = sc("t261119050906-d249d0-n0", "t2_capacity")  # harness-v2 run reused by main-v3
T2_W8A8 = sc("t261119091730-3e5ff4-n0", "t2_capacity")  # harness-v2 run reused by main-v3
T2_BEST = sc("t261119133142-5b5362-n1", "t2_capacity")
T2_STEADY = sc("t261120202050-5b5362-n0", "t2_capacity")
SG_T2 = sc("t261120021649-a2d675-n1", "t2_capacity")  # climb-t2-sglang incumbent on n1, the node its sustained search ran on
SG_T2_FP8 = sc("t261119135542-885ff4-n0", "t2_capacity")
SG_T2_BF16KV = sc("t261119062807-7528f9-n0", "t2_capacity")
SG_T2_STEADY = sc("t261120202050-a2d675-n1", "t2_capacity")

C_BF16 = sc("t261121013833-4af508-n0", "t2_capacity")
C_INT4 = sc("t261119142416-696455-n0", "t2_capacity")
C_W8A8 = sc("t261119142416-c0689e-n1", "t2_capacity")
C_FP8KV = sc("t261119150503-5b5362-n0", "t2_capacity")
C_BEST = sc("t261120054853-4ea7a5-n1", "t2_capacity")
C_CAP256 = sc("t261121024700-5b5362-n0", "t2_capacity")
C_STEADY = sc("t261120211411-4ea7a5-n0", "t2_capacity")
C_STEADY256 = sc("t261121035457-5b5362-n0", "t2_capacity")
C_BOTH = sc("t261121001216-95035c-n0", "t2_capacity")
C_INT4_SUFFIX = sc("t261120233300-c00f32-n0", "t2_capacity")
C_INT4_CUTOFF = sc("t261121001216-356690-n1", "t2_capacity")
SG_C_BF16 = sc("t261121013833-f8592d-n1", "t2_capacity")
SG_C_W8A8 = sc("t261119163301-a2d675-n1", "t2_capacity")
SG_C_BEST = sc("t261120100142-6a6ccb-n0", "t2_capacity")
SG_C_NOMIXED = sc("t261121031607-a2d675-n0", "t2_capacity")
SG_C_STEADY = sc("t261120211411-6a6ccb-n1", "t2_capacity")

T3_BF16 = p90("t261121022208-4af508-n0")
T3_BEST = mean([p90(t) for t in ("t261120120206-5b5362-n0", "t261120120206-5b5362-n1",
                                  "t261120121006-5b5362-n0", "t261120121006-5b5362-n1")])
T3_NOPREFIX = p90("t261120121806-e90512-n0")
T3_OFFLOAD = p90("t261120173053-c1d8a4-n0")
SG_T3_NOHI = mean([p90(t) for t in ("t261120140240-a2d675-n0", "t261120140240-a2d675-n1", "t261120141140-a2d675-n0",
                                     "t261120141140-a2d675-n1", "t261120145347-a2d675-n0", "t261120153144-a2d675-n0",
                                     "t261120155000-a2d675-n0", "t261121034558-a2d675-n0")])
SG_T3_HI = mean([p90(t) for t in ("t261120150247-fb7108-n0", "t261120151235-fb7108-n0",
                                   "t261120154029-fb7108-n0", "t261120155917-fb7108-n1")])


def followup(stage, key="t2_capacity", required=True):
    """Values of the two runs (one per node) of a followup-v3 stage; exits if required and not finished."""
    rows = DB.execute("SELECT trial_id, metrics_json FROM trials WHERE campaign=? AND state='done' ORDER BY trial_id",
                      (f"followup-v3:{stage}",)).fetchall()
    vals = [(tid, score.scores(json.loads(mj)).get(key)) for tid, mj in rows]
    vals = [(t, v) for t, v in vals if v]
    if len(vals) < 2 and required:
        sys.exit(f"followup-v3:{stage} has {len(vals)} finished runs; wait for it to finish")
    return vals


def out_tokens(tid):
    st = [x for x in metrics(tid)["steps"] if x.get("completed")]
    return sum(x["total_output_tokens"] for x in st) / sum(x["completed"] for x in st)


def hero():
    d_chat = followup("steady-defaults-chat")
    d_syn = followup("steady-defaults-synthetic", required=False)
    dc = mean([v for _, v in d_chat])
    ds = mean([v for _, v in d_syn]) if len(d_syn) >= 2 else T2_BF16
    syn_best = T2_STEADY if len(d_syn) >= 2 else T2_BEST
    syn_ylabel = "requests/s sustained" if len(d_syn) >= 2 else "requests/s (2-min test)"
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.4))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.78, bottom=0.14, wspace=0.34)
    fig.suptitle("Baseline vLLM (BF16 defaults) vs. tuned configs on one A100 80GB", fontsize=16,
                 fontweight="bold", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.885, "Nanbeige4.3-33B. Defaults = vLLM, BF16 weights, default settings. Capacity = highest Poisson "
             "rate with ≥90% of requests at TTFT ≤ 2 s and TPOT ≤ 100 ms.", fontsize=10.5,
             color="#5F6368")
    panels = [
        ("One user", "decode tokens/s", [("Defaults", T1_BF16, GRAY, None), ("Best", T1_BEST, BLUE, None)], False,
         f"{T1_BEST / T1_BF16:.1f}× faster", "{:.1f}"),
        ("Synthetic load · 1,024 in / 512 out", syn_ylabel,
         [("Defaults", ds, GRAY, [v for _, v in d_syn] if len(d_syn) >= 2 else None), ("Best", syn_best, BLUE, None)], False,
         f"{syn_best / ds:.1f}× capacity", "{:.2f}"),
        ("Chat · ~140 in / ~870 out", "requests/s sustained",
         [("Defaults", dc, GRAY, [v for _, v in d_chat]), ("Best", C_STEADY256, BLUE, None)], False,
         f"{C_STEADY256 / dc:.1f}× capacity", "{:.2f}"),
        ("Agentic sessions", "p90 time to first token (s, log)",
         [("Defaults", T3_BF16, GRAY, None), ("Best (vLLM)", T3_BEST, BLUE, None)], True,
         f"~{round(T3_BF16 / T3_BEST, -1):.0f}× lower", "{:.2f}"),
    ]
    for ax, (title, ylabel, bars, log, gain, fmt) in zip(axes, panels):
        xs = range(len(bars))
        ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
        for x, (_, v, _, runs) in zip(xs, bars):
            if runs:
                ax.errorbar(x, v, yerr=[[v - min(runs)], [max(runs) - v]], color=INK, capsize=4, lw=1)
            ax.annotate(fmt.format(v) if v < 100 else f"{v:.0f}", (x, max(runs) if runs else v), ha="center",
                        va="bottom", fontsize=11, fontweight="bold", xytext=(0, 3), textcoords="offset points")
        ax.set_xticks(list(xs), [b[0] for b in bars], fontsize=10)
        ax.set_title(f"{title}\n{gain}", loc="left", fontsize=12.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="x", visible=False)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(0.5, 400)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_ylim(0, max(max(b[3]) if b[3] else b[1] for b in bars) * 1.2)
    finish(fig, "hero.png", FOOT + " · defaults' sustained capacity: mean of two runs (bars show both)")


def waterfall(ax, steps, color_up=BLUE, color_down=RED, fmt="{:.1f}", labels=None):
    """steps: (label, value, kind) with kind in {"base", "delta", "total"}; a delta's value is the new level.
    labels: optional per-step annotation overriding the default."""
    level = 0.0
    for i, (label, value, kind) in enumerate(steps):
        custom = labels[i] if labels else None
        if kind in ("base", "total"):
            ax.bar(i, value, color=GRAY if kind == "base" else color_up, width=0.6)
            ax.annotate(custom or fmt.format(value), (i, value), ha="center", va="bottom", xytext=(0, 3),
                        textcoords="offset points", fontweight="bold")
        else:
            lo, hi = sorted((level, value))
            up = value >= level
            ax.bar(i, hi - lo, bottom=lo, color=LIGHT_BLUE if up else color_down, width=0.6)
            ax.annotate(custom or ("+" if up else "−") + fmt.format(abs(value - level)), (i, hi), ha="center",
                        va="bottom", xytext=(0, 3), textcoords="offset points", color=INK if up else color_down,
                        fontweight="bold")
        if i:
            ax.plot([i - 1.3, i - 0.3], [level, level], color="#BDC1C6", lw=1, ls=":")
        level = value
    ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=10)
    ax.grid(axis="x", visible=False)


def single_user():
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.17)
    waterfall(ax, [("BF16\ndefaults", T1_BF16, "base"), ("INT4 weights\n(W4A16)", T1_INT4, "delta"),
                   ("+ suffix decoding\nk=6", T1_SUFFIX6, "delta"), ("draft length\nk=4 (climb)", T1_BEST, "delta"),
                   ("Best: INT4 +\nsuffix k=4", T1_BEST, "total")],
              labels=[None, None, None, f"+{T1_BEST - T1_SUFFIX6:.1f} ({pct(T1_BEST / T1_SUFFIX6 - 1, 1)})", None])
    for i, (a, b) in enumerate([(T1_BF16, T1_INT4), (T1_INT4, T1_SUFFIX6)], start=1):  # the k=4 step is too thin
        ax.text(i, (a + b) / 2, f"{b / a:.1f}×" if b / a > 2 else pct(b / a - 1), ha="center", va="center",
                fontsize=10.5, color=INK, fontweight="bold")
    ax.axhline(SG_T1, color=AMBER, ls="--", lw=1.5)
    ax.text(-0.45, SG_T1 + 1.5, f"SGLang's best: INT4 + n-gram, {SG_T1:.1f} tok/s", color=AMBER, fontsize=10)
    ax.set_ylabel("decode tokens/s, one user")
    ax.set_ylim(0, T1_BEST * 1.15)
    fig.suptitle(f"Single-user decode throughput on vLLM ({T1_BEST / T1_BF16:.1f}× over BF16 defaults)",
                 x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.885, "32 SPEC-Bench chat prompts, natural stopping (≤768 tokens, thinking on), temperature 0.6.",
             fontsize=10.5, color="#5F6368")
    finish(fig, "single_user_steps.png")


def chat():
    d_chat = followup("steady-defaults-chat")
    dc = mean([v for _, v in d_chat])
    steps = [("BF16\ndefaults", C_BF16, dc), ("INT8 W8A8\nweights", C_W8A8, None), ("+ FP8\nKV cache", C_FP8KV, C_STEADY256),
             ("+ max_num_seqs 512\n(the climb's accept)", C_BEST, C_STEADY)]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.17)
    w = 0.36
    for i, (label, short, long_) in enumerate(steps):
        ax.bar(i - w / 2, short, w, color=GRAY if i == 0 else BLUE)
        ax.annotate(f"{short:.2f}", (i - w / 2, short), ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
        if long_ is None:
            ax.text(i + w / 2, 0.15, "not\nmeasured", ha="center", va="bottom", fontsize=9, color="#5F6368")
            continue
        ax.bar(i + w / 2, long_, w, color="#DADCE0" if i == 0 else LIGHT_BLUE, hatch="//", edgecolor="white")
        top = long_
        if i == 0:  # the defaults' two sustained runs, one per node
            runs = [v for _, v in d_chat]
            ax.errorbar(i + w / 2, long_, yerr=[[long_ - min(runs)], [max(runs) - long_]], color=INK, capsize=3, lw=1)
            top = max(runs)
        ax.annotate(f"{long_:.2f}", (i + w / 2, top), ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="2-minute test"),
                       Patch(facecolor=LIGHT_BLUE, hatch="//", edgecolor="white", label="sustained (9-minute steps)")],
              frameon=False, loc="upper left")
    ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=10)
    ax.set_ylabel("chat capacity, requests/s at the SLO")
    ax.set_ylim(0, C_BEST * 1.2)
    ax.grid(axis="x", visible=False)
    fig.suptitle(f"vLLM chat capacity by configuration: 2-minute vs. 9-minute windows ({C_STEADY256 / dc:.1f}× sustained)",
                 x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.885, "ShareGPT first turns, ~140 tokens in and ~870 out, Poisson arrivals. Each step adds one change "
             "to the previous config.", fontsize=10.5, color="#5F6368")
    finish(fig, "chat_steps.png", FOOT + " · defaults' sustained capacity: mean of two runs, one per node (whiskers: both)")


def steady():
    rows_syn = DB.execute("SELECT trial_id, metrics_json FROM trials WHERE campaign='followup-v3:steady-defaults-synthetic' "
                          "AND state='done' ORDER BY trial_id").fetchall()
    d_syn = [score.scores(json.loads(mj)).get("t2_capacity") for _, mj in rows_syn]
    d_syn = [v for v in d_syn if v]
    d_chat = [v for _, v in followup("steady-defaults-chat")]
    groups = [
        ("vLLM defaults\nsynthetic load", T2_BF16, d_syn if len(d_syn) >= 2 else None, "defaults"),
        ("vLLM tuned\nsynthetic load", T2_BEST, [T2_STEADY], "vllm"),
        ("SGLang tuned\nsynthetic load", SG_T2, [SG_T2_STEADY], "sglang"),
        ("vLLM defaults\nchat", C_BF16, d_chat, "defaults"),
        ("vLLM tuned\nchat", C_FP8KV, [C_STEADY256], "vllm"),
        ("SGLang tuned\nchat", SG_C_BEST, [SG_C_STEADY], "sglang"),
        ("vLLM chat, 512 cap\n(the climb's accept)", C_BEST, [C_STEADY], "vllm"),
    ]
    colors = {"defaults": (GRAY, "#DADCE0"), "vllm": (BLUE, LIGHT_BLUE), "sglang": (AMBER, "#FDD663")}
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.16)
    w = 0.36
    for i, (label, short, runs, pal) in enumerate(groups):
        ax.bar(i - w / 2, short, w, color=colors[pal][0])
        ax.annotate(f"{short:.2f}", (i - w / 2, short), ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
        if not runs:
            ax.text(i + w / 2, 0.15, "2-min\nonly*", ha="center", va="bottom", fontsize=9, color="#5F6368")
            continue
        long_ = mean(runs)
        ax.bar(i + w / 2, long_, w, color=colors[pal][1], hatch="//", edgecolor="white")
        if len(runs) > 1:
            ax.errorbar(i + w / 2, long_, yerr=[[long_ - min(runs)], [max(runs) - long_]], color=INK, capsize=3, lw=1)
        ax.annotate(f"{long_:.2f}\n({pct(long_ / short - 1)})", (i + w / 2, max(runs)), ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", fontweight="bold")
    ax.axvline(2.5, color="#DADCE0", lw=1)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GRAY, label="2-minute test"),
                       Patch(facecolor="#DADCE0", hatch="//", edgecolor="white", label="sustained (7–9-minute steps)")],
              frameon=False, loc="upper left", fontsize=10)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=9.5)
    ax.set_ylabel("capacity, requests/s at the SLO")
    ax.set_ylim(0, 5.6)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Request capacity in 2-minute vs. sustained (7–9 minute) load tests",
                 x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.885, "Same configs, judged after the initial 2–3 minute ramp-up. Blue = vLLM tuned, "
             "amber = SGLang tuned, gray = vLLM defaults.", fontsize=10.5, color="#5F6368")
    finish(fig, "steady_vs_2min.png", FOOT + " · *synthetic defaults measured in 2-minute test only; chat defaults: mean of two sustained runs")


def followup_one(stage, cfg_hash, key, default=None):
    """The value of one config (by hash prefix) in a followup-v3 stage; returns default if provided and not finished."""
    for tid, mj in DB.execute("SELECT trial_id, metrics_json FROM trials WHERE campaign=? AND state='done'",
                              (f"followup-v3:{stage}",)):
        if f"-{cfg_hash}-" in tid:
            v = score.scores(json.loads(mj)).get(key)
            if v:
                return v
    if default is not None:
        return default
    sys.exit(f"followup-v3:{stage} has no finished {cfg_hash} run; wait for it to finish")


def pareto():
    both_t1 = followup_one("one-user-cap256", "061b9a", "t1_decode_tps")
    cut_t1 = followup_one("one-user-cap256", "356690", "t1_decode_tps")
    both_c = followup_one("chat-2min", "061b9a", "t2_capacity", default=C_FP8KV)
    both_s_runs = followup("steady-both-cap256", required=False)
    both_s = mean([v for _, v in both_s_runs]) if len(both_s_runs) >= 2 else None
    d_chat = mean([v for _, v in followup("steady-defaults-chat")])
    pts = [  # (label, chat req/s in 2-minute tests, one-user tok/s, engine, sustained chat req/s, label offset)
        ("vLLM INT4 + suffix k=4, always on\n(the one-user winner)", C_INT4_SUFFIX, T1_BEST, "vLLM", None, (12, 2)),
        ("vLLM INT4 + suffix,\ndrafting only at small batches", C_INT4_CUTOFF, cut_t1, "vLLM", None, (12, 4)),
        ("vLLM INT4", C_INT4, T1_INT4, "vLLM", None, (10, -14)),
        ("vLLM \"both\": chat config + suffix,\ndrafting only at small batches", both_c, both_t1, "vLLM", both_s,
         (12, 6)),
        ("vLLM chat config\n(W8A8 + FP8 KV)", C_FP8KV, sc("t261120230054-5b5362-n0", "t1_decode_tps"), "vLLM",
         C_STEADY256, (14, -26)),
        ("vLLM W8A8 + suffix k=6, always on", sc("t261119155134-bea9c6-n0", "t2_capacity"),
         sc("t261119130944-bea9c6-n1", "t1_decode_tps"), "vLLM", None, (10, -4)),
        ("vLLM BF16 defaults", C_BF16, T1_BF16, "vLLM", d_chat, (10, -16)),
        ("SGLang INT4 + n-gram\n(one-user best)",
         mean([sc("t261121010330-b2ed29-n0", "t2_capacity"), sc("t261121010330-b2ed29-n1", "t2_capacity")]),
         SG_T1, "SGLang", None, (12, -10)),
        ("SGLang chat config\n(W8A8 + mixed_chunk)", SG_C_BEST, sc("t261120231125-6a6ccb-n1", "t1_decode_tps"),
         "SGLang", SG_C_STEADY, (12, -8)),
    ]
    front = [p for p in pts if not any(q is not p and q[1] >= p[1] and q[2] >= p[2] for q in pts)]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.12)
    fr = sorted(front, key=lambda p: p[1])
    ax.plot([p[1] for p in fr], [p[2] for p in fr], color=BLUE, lw=2, alpha=0.3, zorder=1)
    for p in pts:
        label, x, y, eng, sus, off = p
        col = BLUE if eng == "vLLM" else AMBER
        on = p in front
        ax.scatter(x, y, s=95 if on else 55, color=col, zorder=3, edgecolor=INK if on else "white", linewidth=1.3)
        if sus:
            ax.annotate("", (sus, y), (x, y), arrowprops=dict(arrowstyle="->", color=col, lw=1, alpha=0.6,
                                                               shrinkA=6, shrinkB=5))
            ax.scatter(sus, y, s=60, facecolor="white", edgecolor=col, linewidth=1.6, zorder=3)
        far = abs(off[0]) > 60 or abs(off[1]) > 20
        ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color="#9AA0A6", lw=0.8) if far else None)
    ax.scatter([], [], color=BLUE, s=55, label="vLLM, 2-minute capacity")
    ax.scatter([], [], color=AMBER, s=55, label="SGLang, 2-minute capacity")
    ax.scatter([], [], facecolor="white", edgecolor="#5F6368", linewidth=1.6, s=60, label="the same config, sustained")
    ax.scatter([], [], color=BLUE, s=95, edgecolor=INK, linewidth=1.3, label="on the 2-minute frontier")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    ax.set_xlabel("chat capacity, requests/s")
    ax.set_ylabel("one user, decode tokens/s")
    ax.set_xlim(0, 5.25)
    ax.set_ylim(15, 105)
    fig.suptitle("Single-user decode speed vs. chat request capacity", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.02, 0.905, "Always-on speculation reaches 95 tok/s for a single user but drops chat capacity to 0.8 req/s. "
             "Disabling speculation above 16 running requests preserves both.", fontsize=10.5,
             color="#5F6368")
    finish(fig, "pareto.png", FOOT + " · SGLang INT4 + n-gram: speed at memory fraction 0.91, capacity at 0.88")


def cost():
    def per_m(cap, out):
        return 1152.78 / (cap * out)

    d_chat = followup("steady-defaults-chat")
    d_syn = followup("steady-defaults-synthetic", required=False)
    syn_def_cap = mean([v for _, v in d_syn]) if len(d_syn) >= 2 else T2_BF16
    syn_def_label = "vLLM defaults (BF16)" if len(d_syn) >= 2 else "vLLM defaults (BF16, 2-min*)"
    groups = [
        ("Synthetic load · 1,024 in / 512 out",
         [(syn_def_label, per_m(syn_def_cap, 512), GRAY),
          ("vLLM W8A8 + FP8 KV", per_m(T2_STEADY, 512), BLUE),
          ("SGLang W8A8", per_m(SG_T2_STEADY, 512), AMBER)]),
        ("Chat · ~140 in / ~870 out",
         [("vLLM defaults (BF16)", per_m(mean([v for _, v in d_chat]), mean([out_tokens(t) for t, _ in d_chat])), GRAY),
          ("vLLM W8A8 + FP8 KV", per_m(C_STEADY256, out_tokens("t261121035457-5b5362-n0")), BLUE),
          ("SGLang W8A8 + mixed_chunk", per_m(SG_C_STEADY, out_tokens("t261120211411-6a6ccb-n1")), AMBER)]),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.12)
    w = 0.26
    for g, (gname, bars) in enumerate(groups):
        has_base = bars[0][0].startswith("vLLM defaults")
        base = bars[0][1]
        for j, (lab, v, col) in enumerate(bars):
            x = g + (j - (len(bars) - 1) / 2) * w
            ax.bar(x, v, w * 0.92, color=col, label=lab if g == len(groups) - 1 else None)
            txt = f"${v:.2f}" if (not has_base or j == 0) else f"${v:.2f}\n({pct(v / base - 1)})"
            ax.annotate(txt, (x, v), ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                        fontweight="bold" if (not has_base or j) else None, fontsize=10)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=10.5)
    ax.set_ylabel("$ per million output tokens, on demand")
    ax.set_ylim(0, max(b[1] for g in groups for b in g[1]) * 1.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("On-demand A100 cost per million output tokens at sustained capacity",
                 x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.885, "On-demand A100 80GB VM list price, evaluated at each config's sustained capacity (7–9-minute steps).",
             fontsize=10.5, color="#5F6368")
    finish(fig, "cost.png", FOOT + " · VM list price only, at 100% of sustained capacity")


def flashinfer_control():
    """2-minute chat capacity of W8A8 + BF16 KV forced onto FlashInfer (followup-v3), or None until it has run."""
    row = DB.execute("SELECT metrics_json FROM trials WHERE campaign='followup-v3:chat-2min' AND state='done' "
                     "AND trial_id LIKE '%-6b80f0-%'").fetchone()
    return score.scores(json.loads(row[0])).get("t2_capacity") if row else None


def knobs():
    """Log-scale '× effect' of single settings, in two panels: changes the search tried, and defaults worth keeping."""
    t1_fp8kv = sc("t261119200442-83da1f-n0", "t1_decode_tps")
    t1_async = sc("t261119192320-7d3457-n0", "t1_decode_tps")
    t1_nograph = sc("t261119191233-907589-n0", "t1_decode_tps")
    t1_compile = sc("t261119225019-e6efad-n0", "t1_decode_tps")
    tried = [  # (setting, workload, factor on the workload's score, note); agentic score = 1 / p90 TTFT
        ("INT4 weights instead of BF16", "one user", T1_INT4 / T1_BF16, f"{T1_BF16:.1f} → {T1_INT4:.1f} tok/s"),
        ("INT8 W8A8 weights instead of BF16", "chat", C_W8A8 / C_BF16, f"{C_BF16:.2f} → {C_W8A8:.2f} req/s"),
        ("64 GiB HiCache host tier (SGLang)", "agentic", SG_T3_NOHI / SG_T3_HI,
         f"p90 TTFT {SG_T3_NOHI:.2f} → {SG_T3_HI:.2f} s"),
        ("suffix decoding, k=6", "one user", T1_SUFFIX6 / T1_INT4, f"{T1_INT4:.1f} → {T1_SUFFIX6:.1f} tok/s"),
        ("FP8 KV cache (vLLM)", "chat", C_FP8KV / C_W8A8, f"{C_W8A8:.2f} → {C_FP8KV:.2f} req/s"),
        ("mixed_chunk (SGLang), 2-minute test", "chat", SG_C_BEST / SG_C_NOMIXED,
         f"{SG_C_NOMIXED:.2f} → {SG_C_BEST:.2f} req/s"),
        ("draft length 4 instead of 6", "one user", T1_BEST / T1_SUFFIX6, f"{T1_SUFFIX6:.1f} → {T1_BEST:.1f} tok/s"),
        ("max_num_seqs 512 (vLLM), sustained", "chat", C_STEADY / C_STEADY256,
         f"{C_STEADY256:.2f} → {C_STEADY:.2f} req/s (outline: 2-minute test, ×{C_BEST / C_CAP256:.2f})"),
        ("FP8 KV cache (SGLang)*", "synthetic load", SG_T2_FP8 / SG_T2_BF16KV, f"{SG_T2_BF16KV:.2f} → {SG_T2_FP8:.2f} req/s"),
        ("64 GiB CPU KV offload (vLLM)", "agentic", T3_BEST / T3_OFFLOAD, "within the 9–13% noise; no CPU-tier hits"),
        ("FP8 KV cache for one user", "one user", t1_fp8kv / T1_BEST, f"{T1_BEST:.1f} → {t1_fp8kv:.1f} tok/s"),
        ("speculation always on, under load", "chat", C_INT4_SUFFIX / C_INT4, f"{C_INT4:.2f} → {C_INT4_SUFFIX:.2f} req/s"),
        ("torch_compile (SGLang, not profiled)", "one user", t1_compile / SG_T1_N0, f"{SG_T1_N0:.1f} → {t1_compile:.1f} tok/s"),
    ]
    fi = flashinfer_control()
    if fi:
        tried.append(("FlashInfer attention alone (vLLM W8A8, BF16 KV)", "chat", fi / C_W8A8,
                      f"{C_W8A8:.2f} → {fi:.2f} req/s"))
    noise = {"64 GiB CPU KV offload (vLLM)"}
    keep = [  # defaults: factor of keeping them on, against turning them off
        ("async scheduling", "one user", T1_BEST / t1_async, f"{t1_async:.1f} → {T1_BEST:.1f} tok/s"),
        ("CUDA graphs", "one user", T1_BEST / t1_nograph, f"{t1_nograph:.1f} → {T1_BEST:.1f} tok/s"),
        ("prefix caching", "agentic", T3_NOPREFIX / T3_BEST, f"p90 TTFT {T3_NOPREFIX:.1f} → {T3_BEST:.2f} s"),
    ]
    tried.sort(key=lambda r: r[2])
    keep.sort(key=lambda r: r[2])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9.8), gridspec_kw={"height_ratios": [len(tried), len(keep) + 0.6]})
    fig.subplots_adjust(left=0.31, right=0.72, top=0.89, bottom=0.08, hspace=0.32)
    for ax, rows, title in ((a1, tried, "Changes the search tried, one at a time (× the workload's score)"),
                            (a2, keep, "Defaults worth keeping (× the score they add, against turning them off)")):
        ys = range(len(rows))
        vals = [r[2] for r in rows]
        ax.barh(ys, [v - 1 if v >= 1 else -(1 - v) for v in vals], left=[1] * len(rows), height=0.6,
                color=[GRAY if r[0] in noise or 0.99 <= r[2] <= 1.01 else (GREEN if r[2] > 1 else RED) for r in rows])
        ax.set_xscale("log")
        ax.set_xlim(0.03, 40 if ax is a2 else 4)
        ax.axvline(1, color="#5F6368", lw=1)
        ax.set_yticks(list(ys), [f"{r[0]}  ({r[1]})" for r in rows], fontsize=10)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.set_xticks([0.03, 0.1, 0.3, 1, 3, 10, 30] if ax is a2 else [0.03, 0.1, 0.3, 1, 2, 3])
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=12)
        for y, r in zip(ys, rows):
            v = r[2]
            ax.annotate(f"×{v:.2f}   {r[3]}", (max(v, 1, C_BEST / C_CAP256 if "max_num_seqs" in r[0] else 0), y),
                        xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    y_cap = [r[0] for r in tried].index("max_num_seqs 512 (vLLM), sustained")
    a1.barh([y_cap], [C_BEST / C_CAP256 - 1], left=[1], height=0.6, fill=False, edgecolor=GRAY, ls="--", lw=1.2)
    fig.suptitle("Relative effect of individual configuration changes", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.02, 0.925, "Each bar changes one setting and nothing else. Agentic score = 1 / p90 TTFT.", fontsize=10.5,
             color="#5F6368")
    finish(fig, "what_moved.png", FOOT + " · *starts from a previous-harness run")


def search():
    """Every challenger each climb tried, against the bar it had to clear."""
    def parse(r):
        return json.loads(r["config_json"]), json.loads(r["move"]) if r["move"] else None

    def base_of(cfg, move):
        b = dict(cfg)
        for k, (frm, _to) in move.items():
            b[k] = frm
        return space_hash(b)

    score_climbs = [("vLLM, one user", "climb-t1n-vllm", "t1_decode_tps", 0.01),
                    ("SGLang, one user", "climb-t1n-sglang", "t1_decode_tps", 0.0625),
                    ("vLLM, agentic", "climb-t3-vllm", "t3_score", 0.2825),
                    ("SGLang, agentic", "climb-t3-sglang", "t3_score", 0.4003)]
    cap_climbs = [("vLLM, synthetic load", "climb-t2-vllm"), ("SGLang, synthetic load", "climb-t2-sglang"),
                  ("vLLM, chat", "climb-t2c-vllm"), ("SGLang, chat", "climb-t2c-sglang")]
    accepted = {"climb-t1n-vllm": {"spec_k": [6, 4]}, "climb-t2c-vllm": {"max_num_seqs": [256, 512]},
                "climb-t2c-sglang": {"mixed_chunk": [False, True]}, "climb-t3-sglang": {"hicache_gib": [0, 64]}}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.9))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.82, bottom=0.08, hspace=0.42)
    n_shown = n_failed = 0
    for i, (label, camp, key, thr) in enumerate(score_climbs):
        rows = DB.execute("SELECT * FROM trials WHERE campaign=? ORDER BY trial_id", (f"main-v3:{camp}",)).fetchall()
        rows = [dict(zip([d[0] for d in DB.execute("SELECT * FROM trials LIMIT 0").description], r)) for r in rows]
        ref = {}
        for r in rows:
            if not r["move"] and r["state"] == "done":
                v0 = score.scores(json.loads(r["metrics_json"])).get(key)
                if v0:
                    ref.setdefault(space_hash(json.loads(r["config_json"])), []).append(v0)
        for r in rows:
            if not r["move"]:
                continue
            cfg, move = parse(r)
            if r["state"] != "done":
                n_failed += 1
                continue
            b = base_of(cfg, move)
            if b not in ref:
                continue
            v = score.scores(json.loads(r["metrics_json"])).get(key)
            if not v:
                n_failed += 1
                continue
            ch = v / mean(ref[b]) - 1
            acc = move == accepted.get(camp)
            a1.scatter(max(ch, -0.99) * 100, i + (0.12 if acc else 0), s=46 if acc else 26,
                       color=GREEN if acc else "#80868B", alpha=1 if acc else 0.55, zorder=3 if acc else 2)
            n_shown += 1
        a1.plot([thr * 100] * 2, [i - 0.32, i + 0.32], color=BLUE, lw=2.2)
    a1.set_yticks(range(len(score_climbs)), [c[0] for c in score_climbs])
    a1.set_ylim(-0.6, len(score_climbs) - 0.4)
    a1.invert_yaxis()
    a1.axvline(0, color="#5F6368", lw=1)
    a1.set_xlim(-100, 75)
    a1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    a1.set_title("Score climbs: each challenger against the config it changed (blue tick = acceptance threshold)",
                 loc="left", fontsize=11.5)
    a1.grid(axis="y", visible=False)
    for i, (label, camp) in enumerate(cap_climbs):
        rows = DB.execute("SELECT * FROM trials WHERE campaign=? ORDER BY trial_id", (f"main-v3:{camp}",)).fetchall()
        rows = [dict(zip([d[0] for d in DB.execute("SELECT * FROM trials LIMIT 0").description], r)) for r in rows]
        for r in rows:
            if not r["move"] or not (r["plan"] or "").startswith(("t2probe@", "t2cprobe@")):
                continue
            if r["state"] != "done":
                n_failed += 1
                continue
            bar = float(r["plan"].split("@")[1])
            pts = score.t2_points(json.loads(r["metrics_json"]))
            if not pts:
                n_failed += 1
                continue
            att = min(pts, key=lambda p: abs(p[0] - bar))[1]
            acc = json.loads(r["move"]) == accepted.get(camp)
            a2.scatter(att * 100, i + (0.12 if acc else 0), s=46 if acc else 26, color=GREEN if acc else "#80868B",
                       alpha=1 if acc else 0.55, zorder=3 if acc else 2)
            n_shown += 1
    a2.axvline(90, color=BLUE, lw=2.2)
    a2.set_yticks(range(len(cap_climbs)), [c[0] for c in cap_climbs])
    a2.set_ylim(-0.6, len(cap_climbs) - 0.4)
    a2.invert_yaxis()
    a2.set_xlim(-3, 102)
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    a2.set_title("Capacity climbs: share of requests within the SLO at 1.05× the incumbent's capacity "
                 "(blue line = 90% to pass)", loc="left", fontsize=11.5)
    a2.grid(axis="y", visible=False)
    a2.scatter([], [], color=GREEN, s=46, label="accepted change (screen and confirming re-runs)")
    a2.scatter([], [], color="#80868B", alpha=0.55, s=26, label="rejected")
    fig.legend(*a2.get_legend_handles_labels(), frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.905),
               ncol=2, fontsize=10)
    fig.suptitle(f"Hill-climb challenger runs vs. acceptance thresholds ({n_shown} runs across 8 climbs)", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.02, 0.915, "Each point compares a challenger with the mean of all its incumbent's runs. The climber "
             "compared same-node runs,\nso the accepted draft length plots at +1.8% here and +2.0% in the text.",
             fontsize=10.5, color="#5F6368", va="top")
    finish(fig, "search.png", FOOT + f"\nAgentic score = 1 / p90 TTFT; SGLang agentic's tick is its final 40% bar "
           f"(earlier challengers faced 30–38%) · not shown: {n_failed} runs that failed, incumbent re-runs, and the "
           "one-trial CPU-offload continuation")


def space_hash(cfg):
    from harness import space
    return space.config_hash(space.full(cfg))


def agentic():
    """p90 TTFT on the agentic workload against KV pool size: first the pool has to fit, then compute speed matters."""
    def kv(tid):
        return DB.execute("SELECT kv_tokens FROM trials WHERE trial_id=?", (tid,)).fetchone()[0]

    def t3(tid, key):
        return metrics(tid)["t3"].get(key)

    rows = [  # (label, [trials], color)
        ("vLLM BF16 defaults", ["t261121022208-4af508-n0"], GRAY),
        ("vLLM INT4, BF16 KV", ["t261121022208-696455-n1"], LIGHT_BLUE),
        ("vLLM INT4, FP8 KV", ["t261120133725-e9a8ac-n0"], LIGHT_BLUE),
        ("vLLM W8A8, BF16 KV", ["t261120123242-c0689e-n0", "t261121023133-c0689e-n1"], LIGHT_BLUE),
        ("vLLM W8A8, FP8 KV (vLLM's best)", ["t261120120206-5b5362-n0", "t261120120206-5b5362-n1",
                                            "t261120121006-5b5362-n0", "t261120121006-5b5362-n1"], BLUE),
        ("SGLang W8A8, BF16 KV", ["t261120140240-a2d675-n0", "t261120140240-a2d675-n1", "t261120141140-a2d675-n0",
                                  "t261120141140-a2d675-n1", "t261120145347-a2d675-n0", "t261120153144-a2d675-n0",
                                  "t261120155000-a2d675-n0", "t261121034558-a2d675-n0"], "#FDD663"),
        ("SGLang W8A8 + 64 GiB HiCache (SGLang's best)", ["t261120150247-fb7108-n0", "t261120151235-fb7108-n0",
                                                          "t261120154029-fb7108-n0", "t261120155917-fb7108-n1"], AMBER),
    ]
    data = []
    for lab, ts, col in rows:
        pre = [t3(t, "preemptions") for t in ts]
        data.append((lab, mean([p90(t) for t in ts]), min(p90(t) for t in ts), max(p90(t) for t in ts), kv(ts[0]),
                     mean([t3(t, "prefix_hit_ratio") for t in ts]), col, len(ts),
                     None if any(x is None for x in pre) else max(pre)))
    data.sort(key=lambda d: -d[1])
    fig, ax = plt.subplots(figsize=(13, 6.2))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.84, bottom=0.12)
    for y, (lab, m_, lo, hi, kvt, h, col, n, pre) in enumerate(data):
        ax.barh(y, m_, color=col, height=0.6)
        if n > 1:
            ax.errorbar(m_, y, xerr=[[m_ - lo], [hi - m_]], color=INK, capsize=4, lw=1)
        rng = f"{lo:.2f}–{hi:.2f} s ({n} runs)" if n > 1 else (f"{m_:.0f} s" if m_ > 10 else f"{m_:.2f} s")
        txt = (f"{rng}   ·   KV pool {kvt:,} tokens{' + 64 GiB host tier' if 'HiCache' in lab else ''}   ·   "
               f"prefix hits {h * 100:.0f}%" + (f"   ·   {pre:.0f} preemptions" if pre is not None else ""))
        if m_ > 20:  # long bar: write inside it
            ax.annotate(txt, (m_, y), xytext=(-8, 0), textcoords="offset points", va="center", ha="right",
                        fontsize=9.5, color="white", fontweight="bold")
        else:
            ax.annotate(txt, (max(hi, m_), y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5)
    ax.axvline(2, color=RED, ls="--", lw=1)
    ax.text(2.06, len(data) - 0.45, "2 s: the TTFT target in the capacity tests", color=RED, fontsize=9, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.5, 200)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    ax.set_yticks(range(len(data)), [d[0] for d in data], fontsize=10)
    ax.set_ylim(-0.6, len(data) - 0.2)
    ax.set_xlabel("p90 time to first token over 288 turns (log scale)")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Multi-turn agentic workload: p90 time to first token and KV cache size", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.02, 0.89, "48 sessions × 6 turns arriving at 0.2 sessions/s, thinking off; a shared 3,000-token system "
             "prompt and a new 1,200-token document every turn; ≤256 tokens out per turn.", fontsize=10.5,
             color="#5F6368")
    finish(fig, "agentic.png", FOOT + " · SGLang doesn't report preemptions")


def pipeline():
    """The harness's stages as a two-row diagram, with each stage's trials and trial-hours from the ledger."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    groups = {"screen": ("quality2-refs", "quality2", "t1-screen", "t2-screen", "t2c-screen"), "climb": ("climb-",),
              "verify": ("verify-",), "steady": ("steady-",), "ablate": ("ablation-",),
              "cross": ("profile", "cross-", "both-", "baseline-")}
    n, h = {g: 0 for g in groups}, {g: 0.0 for g in groups}
    quality = aborted = 0
    for camp, state, s_, f in DB.execute("SELECT campaign, state, started_at, finished_at FROM trials "
                                         "WHERE campaign LIKE 'main-v3:%'"):
        sid = camp.split(":", 1)[1]
        g = next(k for k, pre in groups.items() if sid.startswith(pre))
        n[g] += 1
        h[g] += ((f or 0) - (s_ or 0)) / 3600 if s_ and f else 0
        quality += sid.startswith("quality")
        aborted += g == "steady" and state == "aborted"
    stages = [  # (title, body, group, extra count note, column, row)
        ("1. Screen", "each big lever once:\nweights, KV cache,\nspeculation, engine", "screen",
         f" ({quality} quality gates)", 0, 0),
        ("2. Climb", "one setting at a time;\nkept only if a fresh\nre-run confirms the gain", "climb", "", 1, 0),
        ("3. Verify", "held-out prompts;\nquality gate on\nevery winner", "verify", "", 2, 0),
        ("4. Sustained", "7–9-minute windows,\njudged after\nthe ramp-up", "steady", f" ({aborted} aborted)", 2, 1),
        ("5. Ablate", "revert each accepted\nchange on the\nfinal config", "ablate", "", 1, 1),
        ("6. Cross-check", "speed vs capacity for\neach winner; the defaults'\nbaselines; a GPU profile", "cross", "",
         0, 1),
    ]
    W, H, GX, GY = 3.0, 2.2, 1.0, 1.25
    fig, ax = plt.subplots(figsize=(12, 7.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.07)
    ax.set_xlim(-0.2, 3 * W + 2 * GX + 0.2)
    ax.set_ylim(-0.1, 2 * H + GY + 0.75)
    ax.axis("off")

    def box_xy(col, row):
        return col * (W + GX), (1 - row) * (H + GY) + 0.5

    for i, (title, body, g, note, col, row) in enumerate(stages):
        x, y = box_xy(col, row)
        color = BLUE if g == "climb" else "#3C4043"
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="#E8F0FE" if g == "climb" else "#F1F3F4", edgecolor=color, lw=1.6))
        ax.text(x + W / 2, y + H - 0.35, title, ha="center", va="center", fontsize=13, fontweight="bold", color=color)
        ax.text(x + W / 2, y + H / 2 - 0.1, body, ha="center", va="center", fontsize=10, color=INK, linespacing=1.35)
        ax.text(x + W / 2, y - 0.25, f"{n[g]} trials{note} · {h[g]:.1f} trial-hours", ha="center", va="center",
                fontsize=10.5, color=GREEN, fontweight="bold")
        if i < len(stages) - 1:
            ncol, nrow = stages[i + 1][4], stages[i + 1][5]
            nx, ny = box_xy(ncol, nrow)
            if nrow == row:  # horizontal arrow, left-to-right on top, right-to-left below
                a, b = ((x + W + 0.03, y + H / 2), (nx - 0.03, ny + H / 2)) if ncol > col else \
                       ((x - 0.03, y + H / 2), (nx + W + 0.03, ny + H / 2))
            else:  # down from the end of the first row
                a, b = (x + W / 2, y - 0.5), (nx + W / 2, ny + H + 0.05)
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color="#5F6368", lw=1.4))
    fig.suptitle(f"Hill-climb harness pipeline ({sum(n.values())} trials across 6 stages)", x=0.01, ha="left", fontsize=15,
                 fontweight="bold")
    fig.text(0.01, 0.905, "One trial = one Kubernetes pod pinned to a GPU node: the inference server plus a benchmark "
             "sidecar.\nAn in-cluster controller ran the stages unattended, one trial per node at a time.",
             fontsize=10.5, color="#5F6368", va="top")
    finish(fig, "pipeline.png", f"Trial-hours ({sum(h.values()):.1f}) are summed over the two nodes, which ran in "
           "parallel. Not shown: the earlier campaign whose results this one reused (83 trials, 7.8 hours), and the "
           "follow-up runs for this write-up.")


if __name__ == "__main__":
    only = set(sys.argv[3:])
    for chart in (hero, single_user, chat, steady, pareto, cost, knobs, pipeline, search, agentic):
        if not only or chart.__name__ in only:
            chart()
