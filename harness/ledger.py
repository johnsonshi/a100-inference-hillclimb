"""SQLite ledger: one row per trial, every accepted move traceable to its evidence."""
import json
import sqlite3
import time
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "results" / "ledger.sqlite"
SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
  trial_id TEXT PRIMARY KEY, campaign TEXT, plan TEXT, lane TEXT, node TEXT,
  engine TEXT, variant TEXT, config_hash TEXT, config_json TEXT,
  parent_id TEXT, move TEXT, state TEXT, reason TEXT,
  started_at REAL, finished_at REAL, startup_s REAL, kv_tokens INTEGER,
  t1_decode_tps REAL, t1_ttft_ms REAL, t2_goodput_tps REAL, thr_c32_tps REAL,
  scores_json TEXT, metrics_json TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS events (
  ts REAL, campaign TEXT, kind TEXT, detail TEXT
);
"""


def conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=30)
    c.executescript(SCHEMA)
    c.row_factory = sqlite3.Row
    return c


def start(trial_id, campaign, plan, lane, node, cfg, chash, parent_id=None, move=None, notes=None):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO trials (trial_id,campaign,plan,lane,node,engine,variant,config_hash,"
                  "config_json,parent_id,move,state,started_at,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (trial_id, campaign, plan, lane, node, cfg["engine"], cfg.get("variant"), chash,
                   json.dumps(cfg, sort_keys=True), parent_id, json.dumps(move) if move else None,
                   "running", time.time(), notes))


def finish(trial_id, state, reason, metrics):
    s = (metrics or {}).get("scores", {})
    with conn() as c:
        c.execute("UPDATE trials SET state=?, reason=?, finished_at=?, startup_s=?, kv_tokens=?, t1_decode_tps=?,"
                  " t1_ttft_ms=?, t2_goodput_tps=?, thr_c32_tps=?, scores_json=?, metrics_json=? WHERE trial_id=?",
                  (state, reason, time.time(), (metrics or {}).get("startup_s"), (metrics or {}).get("kv_tokens"),
                   s.get("t1_decode_tps"), s.get("t1_ttft_ms"), s.get("t2_goodput_tps"), s.get("thr_c32_tps"),
                   json.dumps(s), json.dumps(metrics or {}), trial_id))


def event(campaign, kind, detail):
    with conn() as c:
        c.execute("INSERT INTO events VALUES (?,?,?,?)", (time.time(), campaign, kind, json.dumps(detail)))


def rows(where="1=1", args=()):
    with conn() as c:
        return [dict(r) for r in c.execute(f"SELECT * FROM trials WHERE {where} ORDER BY started_at", args)]
