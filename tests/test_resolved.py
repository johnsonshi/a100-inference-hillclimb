"""What a trial actually ran: recovered from the server log's head when a long run's tail lost the startup lines,
and from the incumbent's other trials when one trial's record lacks it. Run: python3 -m tests.test_resolved"""
import json
import tempfile
from pathlib import Path

from harness import climber, ledger, score, space

STARTUP = ("[2026-11-20 08:00:00] Attention backend not specified. Use flashinfer backend by default.\n"
           "[2026-11-20 08:00:05] max_total_num_tokens=191772, chunked_prefill_size=8192\n"
           "[2026-11-20 08:00:47] Capture target prefill CUDA graph end. elapsed=41.9 s, mem usage=2.41 GB\n")
DECODE = "[2026-11-20 08:10:00] Decode batch, #running-req: 120, cuda graph: {}, gen throughput (token/s): 900\n"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="hc-resolved-"))

    trial = tmp / "t-long"
    trial.mkdir()
    (trial / "server.log").write_text(DECODE.format("True") * 3 + DECODE.format("False"))
    (trial / "server.head.log").write_text(STARTUP + DECODE.format("False") * 5)
    m = score.extract(trial)
    assert m["resolved"]["attention_backend"] == "flashinfer", m["resolved"]
    assert m["resolved"]["sglang_prefill_graph_s"] == 41.9, m["resolved"]
    assert m["resolved"]["sglang_decode_graph_frac"] == 0.75, "decode coverage must come from the tail only"
    assert m["kv_tokens"] == 191772, m.get("kv_tokens")

    ledger.DB = tmp / "ledger.sqlite"
    cfg = space.full({"engine": "sglang", "variant": "int8w8a8", "mem_fraction_static": 0.91, "mixed_chunk": True})
    chash = space.config_hash(cfg)
    runs = [("t-rate", "t2c@4.3982", {"resolved": {"sglang_decode_graph_frac": 0.87}}),
            ("t-probe", "t2cprobe@4.3982", {"resolved": {"attention_backend": "flashinfer",
                                                          "sglang_decode_graph_frac": 0.88}})]
    for tid, plan, metrics in runs:
        ledger.start(tid, "sim:climb-t2c-sglang", plan, "n0", "node", cfg, chash)
        ledger.finish(tid, "done", None, metrics)
    inc_resolved = {"sglang_decode_graph_frac": 0.86}
    climber._fill_resolved(inc_resolved, cfg, "sim:climb-t2c-sglang")
    assert inc_resolved == {"sglang_decode_graph_frac": 0.86, "attention_backend": "flashinfer"}, inc_resolved
    flashinfer = {**cfg, "attention_backend": "flashinfer"}
    assert climber._noop(inc_resolved, "attention_backend", flashinfer["attention_backend"])
    print("RESOLVED TEST OK", json.dumps(inc_resolved))


if __name__ == "__main__":
    main()
