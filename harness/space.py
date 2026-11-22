"""Search space: knobs per engine (grouped by dimension L0-L6), model variants, images, and the
rendering of a config dict into a server command line. A config is a plain dict:
{"engine": "vllm"|"sglang", <knob>: <value>, ...}; unspecified knobs take their defaults.
"""
import hashlib
import json
import os

from . import settings

settings.load()
# Engine images come from the registry mirror in HILLCLIMB_REGISTRY (.env locally, the controller's env in the
# cluster); without one, from Docker Hub.
REGISTRY = os.environ.get("HILLCLIMB_REGISTRY", "").rstrip("/")


def _image(ref):
    return f"{REGISTRY}/{ref}" if REGISTRY else ref


IMAGES = {
    "vllm": _image("vllm/vllm-openai:v0.34.0"),
    "sglang": _image("lmsysorg/sglang:v0.5.23"),
}
BENCH_IMAGE = IMAGES["vllm"]
SERVED_NAME = "nanbeige"
MAX_MODEL_LEN = 32768
VARIANTS = {
    "bf16": "/models/Nanbeige4.3-33B",
    "fp8": "/models/Nanbeige4.3-33B-FP8",
    "int8": "/models/Nanbeige4.3-33B-INT8",
    "int4": "/models/Nanbeige4.3-33B-INT4",
    # custom: data-free INT8 W8A8 (llm-compressor), uses Ampere INT8 tensor cores for prefill
    "int8w8a8": "/models/Nanbeige4.3-33B-W8A8-INT8",
}
DRAFTS = {"dflash2": "/models/Nanbeige4.3-33B-DFlash2"}
TOKENIZER = VARIANTS["bf16"]

# Dimensions (the blog series): L0 engine, L1 weights, L2 KV cache, L3 speculative decoding,
# L4 scheduler & batching, L5 kernels & compilation, L6 topology (multi-replica, outside a trial).
# Each knob: layer, kind ("ord" = neighbors are adjacent values, "cat" = all other values), values, default.
COMMON = {
    "variant": ("L1", "cat", ["bf16", "fp8", "int8", "int4", "int8w8a8"], "bf16"),
}
KNOBS = {
    "vllm": {
        **COMMON,
        "kv_cache_dtype": ("L2", "cat", ["auto", "fp8"], "auto"),
        "prefix_caching": ("L2", "cat", [True, False], True),
        "block_size": ("L2", "cat", [None, 16, 32, 128], None),           # None = 64 in v0.34
        # CPU KV tier: a write-through copy only helps if it is larger than the GPU pool (~213k BF16 tokens ~= 41 GiB)
        "kv_offloading_gib": ("L2", "ord", [0, 64, 128], 0),
        "extensible_kv": ("L2", "cat", [True, False], True),
        "gpu_memory_utilization": ("L2", "ord", [0.90, 0.93, 0.95], 0.90),  # 0.90 = vLLM default
        "spec": ("L3", "cat", [None, "ngram", "ngram_gpu", "suffix", "dflash2"], None),
        "spec_k": ("L3", "ord", [1, 2, 3, 4, 6, 8], 4),
        # draft length by running batch size (num_speculative_tokens_per_batch_size): "taper" keeps k for
        # batches <=4, halves it up to 16, then drafts 1 token; "cutoff" does the same but drafts nothing
        # above 16 (speculation off under load). Verification is only nearly free while decode is
        # memory-bound: INT4 + suffix k=6 collapsed to 0.64 req/s on T2 chat (2.38 without speculation).
        "spec_schedule": ("L3", "cat", [None, "taper", "cutoff"], None),
        "max_num_seqs": ("L4", "ord", [32, 64, 128, 256, 512], 256),
        "max_num_batched_tokens": ("L4", "ord", [256, 512, 1024, 2048, 4096, 8192, 16384], 2048),
        "long_prefill_threshold": ("L4", "ord", [0, 256, 512, 1024, 2048], 0),  # 0 = off
        "long_prefill_adaptive": ("L4", "cat", [False, True], False),
        "async_scheduling": ("L4", "cat", [None, False], None),          # None = engine decides (on)
        "api_server_count": ("L4", "cat", [None, 2, 4], None),
        "performance_mode": ("L5", "cat", [None, "interactivity", "throughput"], None),  # None = balanced
        "attention_backend": ("L5", "cat", [None, "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION"], None),
        "optimization_level": ("L5", "ord", [0, 1, 2], 2),              # -O3 == -O2 in v0.34
        "cudagraph_mode": ("L5", "cat", [None, "PIECEWISE", "FULL_DECODE_ONLY", "FULL", "NONE"], None),
        "compile_sizes": ("L5", "cat", [None, "1", "1,2,4,8"], None),   # Inductor max-autotune for these sizes
        "flashinfer_sampler": ("L5", "cat", [None, "0", "1"], None),
    },
    "sglang": {
        **COMMON,
        "kv_cache_dtype": ("L2", "cat", ["auto", "fp8_e4m3"], "auto"),
        "radix_cache": ("L2", "cat", [True, False], True),
        "page_size": ("L2", "cat", [None, 16, 64], None),
        "mem_fraction_static": ("L2", "ord", [None, 0.85, 0.88, 0.91, 0.94], None),
        # hierarchical (CPU) KV cache; SGLang warns when the host pool is smaller than the device pool (32 GiB is)
        "hicache_gib": ("L2", "ord", [0, 64, 128], 0),
        "alloc_expandable": ("L2", "cat", [False, True], False),        # PYTORCH_CUDA_ALLOC_CONF expandable
        "spec": ("L3", "cat", [None, "ngram", "dflash2"], None),
        "spec_k": ("L3", "ord", [1, 2, 3, 4, 6, 8], 4),
        "max_running_requests": ("L4", "ord", [None, 32, 64, 128, 256, 512], None),
        "chunked_prefill_size": ("L4", "ord", [None, 512, 1024, 2048, 4096, 8192, 16384], None),
        "schedule_policy": ("L4", "cat", ["fcfs", "lpm"], "fcfs"),
        "schedule_conservativeness": ("L4", "ord", [0.5, 1.0, 1.5], 1.0),
        "overlap_schedule": ("L4", "cat", [True, False], True),
        "num_continuous_decode_steps": ("L4", "ord", [1, 2, 4], 1),
        "mixed_chunk": ("L4", "cat", [False, True], False),
        "attention_backend": ("L5", "cat", [None, "flashinfer", "triton", "fa3"], None),
        "cuda_graph_max_bs_decode": ("L5", "ord", [None, 64, 160, 256, 512], None),
        "prefill_cuda_graph": ("L5", "cat", [True, False], True),       # False = --disable-prefill-cuda-graph
        "torch_compile": ("L5", "cat", [False, True], False),
        "torch_compile_max_bs": ("L5", "ord", [None, 1, 8, 32], None),
    },
}
SPEC_SCHEDULES = {"taper": lambda k: [[1, 4, k], [5, 16, max(1, k // 2)], [17, 4096, 1]],
                  "cutoff": lambda k: [[1, 4, k], [5, 16, max(1, k // 2)], [17, 4096, 0]]}


def full(cfg):
    """Fill defaults so equal configs compare equal."""
    eng = cfg["engine"]
    out = {"engine": eng}
    for k, (_, _, _, default) in KNOBS[eng].items():
        out[k] = cfg.get(k, default)
    unknown = set(cfg) - set(out)
    if unknown:
        raise ValueError(f"unknown knobs for {eng}: {sorted(unknown)}")
    return out


def canonical(cfg):
    """Engine plus every knob that differs from its default. Adding a knob (with a default that keeps the
    old behavior) leaves existing configs' identity unchanged, so older ledger results stay reusable."""
    c = full(cfg)
    knobs = KNOBS[c["engine"]]
    return {"engine": c["engine"], **{k: v for k, v in c.items() if k != "engine" and v != knobs[k][3]}}


def config_hash(cfg):
    return hashlib.sha1(json.dumps(canonical(cfg), sort_keys=True).encode()).hexdigest()[:10]


def numerics_key(cfg):
    """What can change model outputs (the quality-gate class): engine, weights, KV dtype, speculation."""
    c = full(cfg)
    return f"{c['engine']}/{c['variant']}/{c['kv_cache_dtype']}/{c['spec'] or 'nospec'}"


def diff(a, b):
    fa, fb = full(a), full(b)
    return {k: (fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}


def validity(cfg):
    """Return None if the config is expected to be valid, else a reason string."""
    c = full(cfg)
    if c["engine"] == "vllm":
        if c["kv_cache_dtype"] == "fp8" and c["attention_backend"] == "FLASH_ATTN":
            return "vLLM FlashAttention-2 does not support FP8 KV cache on Ampere"
        if c["kv_cache_dtype"] == "fp8" and c["attention_backend"] in ("TRITON_ATTN", "FLEX_ATTENTION"):
            return ("vLLM v0.34 FP8 KV cache on A100: Triton attention needs SM89+ and FlexAttention has no FP8 KV "
                    "- measured infeasible")
        if c["spec"] is None and c["spec_k"] != KNOBS["vllm"]["spec_k"][3]:
            return "spec_k without speculative decoding is a no-op"
        if c["spec"] is None and c["spec_schedule"]:
            return "spec_schedule without speculative decoding is a no-op"
        if c["kv_offloading_gib"] and not c["prefix_caching"]:
            return "KV offloading without prefix caching is pointless"
        if c["max_num_batched_tokens"] < c["max_num_seqs"]:
            return "max_num_batched_tokens < max_num_seqs: a step could not give every running sequence a token"
        if c["long_prefill_adaptive"] and not c["long_prefill_threshold"]:
            return "long_prefill_adaptive only applies when long_prefill_threshold > 0"
        if c["long_prefill_threshold"] and c["long_prefill_threshold"] >= c["max_num_batched_tokens"]:
            return "long_prefill_threshold >= max_num_batched_tokens is a no-op"
        if c["compile_sizes"] and c["optimization_level"] == 0:
            return "compile_sizes needs torch.compile (optimization_level > 0)"
    else:
        if c["variant"] == "fp8":
            return ("SGLang v0.5.23 on Ampere: compressed-tensors FP8 block-scaled weights unsupported "
                    "(only channel/tensor strategies) - measured infeasible")
        if c["spec"] is None and c["spec_k"] != KNOBS["sglang"]["spec_k"][3]:
            return "spec_k without speculative decoding is a no-op"
        if c["attention_backend"] == "fa3":
            return "FlashAttention-3 needs Hopper (SM90); A100 is SM80"
        if c["torch_compile_max_bs"] and not c["torch_compile"]:
            return "torch_compile_max_bs without torch_compile is a no-op"
        if c["hicache_gib"] and not c["radix_cache"]:
            return "hierarchical cache needs the radix cache"
        if c["mem_fraction_static"] == 0.94 and c["prefill_cuda_graph"] and not c["chunked_prefill_size"]:
            return ("SGLang v0.5.23 on A100 80GB: mem_fraction_static 0.94 leaves ~4.3 GB after the KV pool; the 8192-token "
                    "prefill CUDA graph takes ~3.9 GB and the next capture runs out of memory - measured infeasible")
    return None


def neighbor_values(cfg, knob):
    """A knob's candidate values from `cfg` (ord: adjacent values; cat: every other value), not yet
    checked for validity: a value may only be valid together with another knob's change."""
    c = full(cfg)
    _, kind, values, _ = KNOBS[c["engine"]][knob]
    cur = c[knob]
    if kind == "ord" and cur in values:
        i = values.index(cur)
        return [values[j] for j in (i - 1, i + 1) if 0 <= j < len(values)]
    return [v for v in values if v != cur]


def neighbors(cfg, knob):
    c = full(cfg)
    cand = neighbor_values(c, knob)
    out = []
    for v in cand:
        n = dict(c)
        n[knob] = v
        if validity(n) is None:
            out.append(n)
    return out


def _vllm_args(c, profile_dir=None, mfu=False):
    a = ["vllm", "serve", VARIANTS[c["variant"]],
         "--served-model-name", SERVED_NAME, "--host", "0.0.0.0", "--port", "8000",
         "--max-model-len", str(MAX_MODEL_LEN),
         "--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
         "--kv-cache-dtype", c["kv_cache_dtype"],
         "--max-num-seqs", str(c["max_num_seqs"]),
         "--max-num-batched-tokens", str(c["max_num_batched_tokens"]),
         "--optimization-level", str(c["optimization_level"]),
         # load-time only (weights are on local NVMe): prefetch reads files ahead of the GPU copy
         "--safetensors-load-strategy", "prefetch"]
    env = {}
    if not c["prefix_caching"]:
        a.append("--no-enable-prefix-caching")
    if c["block_size"]:
        a += ["--block-size", str(c["block_size"])]
    if c["kv_offloading_gib"]:
        a += ["--kv-offloading-size", str(c["kv_offloading_gib"]), "--kv-offloading-backend", "native"]
    if not c["extensible_kv"]:
        a.append("--no-enable-extensible-kv-cache")
    if c["gpu_memory_utilization"] != KNOBS["vllm"]["gpu_memory_utilization"][3]:
        a += ["--gpu-memory-utilization", str(c["gpu_memory_utilization"])]
    if c["spec"]:
        spec = {"num_speculative_tokens": c["spec_k"]}
        if c["spec"] in ("ngram", "ngram_gpu"):
            spec.update(method=c["spec"], prompt_lookup_min=2, prompt_lookup_max=5)
        elif c["spec"] == "suffix":
            spec.update(method="suffix")
        elif c["spec"] == "dflash2":
            spec.update(method="dflash", model=DRAFTS["dflash2"])
        if c["spec_schedule"]:
            spec["num_speculative_tokens_per_batch_size"] = SPEC_SCHEDULES[c["spec_schedule"]](c["spec_k"])
        a += ["--speculative-config", json.dumps(spec)]
    if c["long_prefill_threshold"]:
        a += ["--long-prefill-token-threshold", str(c["long_prefill_threshold"])]
        if c["long_prefill_adaptive"]:
            a.append("--long-prefill-token-threshold-adaptive")
    if c["async_scheduling"] is False:
        a.append("--no-async-scheduling")
    if c["api_server_count"]:
        a += ["--api-server-count", str(c["api_server_count"])]
    if c["performance_mode"]:
        a += ["--performance-mode", c["performance_mode"]]
    if c["attention_backend"]:
        a += ["--attention-backend", c["attention_backend"]]
    comp = {}
    if c["cudagraph_mode"]:
        comp["cudagraph_mode"] = c["cudagraph_mode"]
    if c["compile_sizes"]:
        comp["compile_sizes"] = [int(x) for x in c["compile_sizes"].split(",")]
    if comp:
        a += ["--compilation-config", json.dumps(comp)]
    if c["flashinfer_sampler"] is not None:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = c["flashinfer_sampler"]
    if profile_dir:
        a += ["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": profile_dir})]
    if mfu:  # per-step FLOP / byte estimates (vllm:estimated_*_per_gpu) -> achieved TFLOP/s and MBU
        a.append("--enable-mfu-metrics")
    return a, env


def _sglang_args(c):
    a = ["python3", "-m", "sglang.launch_server", "--model-path", VARIANTS[c["variant"]],
         "--served-model-name", SERVED_NAME, "--host", "0.0.0.0", "--port", "8000",
         "--context-length", str(MAX_MODEL_LEN), "--enable-metrics",
         "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
         "--kv-cache-dtype", c["kv_cache_dtype"], "--schedule-policy", c["schedule_policy"],
         "--num-continuous-decode-steps", str(c["num_continuous_decode_steps"])]
    if not c["radix_cache"]:
        a.append("--disable-radix-cache")
    if c["page_size"]:
        a += ["--page-size", str(c["page_size"])]
    if c["mem_fraction_static"]:
        a += ["--mem-fraction-static", str(c["mem_fraction_static"])]
    if c["spec"] == "ngram":
        a += ["--speculative-algorithm", "NGRAM", "--speculative-num-draft-tokens", str(c["spec_k"])]
    elif c["spec"] == "dflash2":
        a += ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", DRAFTS["dflash2"],
              "--speculative-num-draft-tokens", str(c["spec_k"]),
              "--speculative-draft-attention-backend", "flashinfer"]
    if c["max_running_requests"]:
        a += ["--max-running-requests", str(c["max_running_requests"])]
    if c["chunked_prefill_size"]:
        a += ["--chunked-prefill-size", str(c["chunked_prefill_size"])]
    if not c["overlap_schedule"]:
        a.append("--disable-overlap-schedule")
    if c["mixed_chunk"]:
        a.append("--enable-mixed-chunk")
    if c["attention_backend"]:
        a += ["--attention-backend", c["attention_backend"]]
    if c["cuda_graph_max_bs_decode"]:
        a += ["--cuda-graph-max-bs-decode", str(c["cuda_graph_max_bs_decode"])]
    if not c["prefill_cuda_graph"]:
        a.append("--disable-prefill-cuda-graph")
    if c["schedule_conservativeness"] != KNOBS["sglang"]["schedule_conservativeness"][3]:
        a += ["--schedule-conservativeness", str(c["schedule_conservativeness"])]
    if c["hicache_gib"]:
        a += ["--enable-hierarchical-cache", "--hicache-size", str(c["hicache_gib"]),
              "--hicache-write-policy", "write_through"]
    if c["torch_compile"]:
        a.append("--enable-torch-compile")
        if c["torch_compile_max_bs"]:
            a += ["--torch-compile-max-bs", str(c["torch_compile_max_bs"])]
    env = {}
    if c["alloc_expandable"]:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return a, env


def server_command(cfg, profile_dir=None, mfu=False):
    """Measurement flags (profiler, MFU estimates) are not part of a config's identity."""
    c = full(cfg)
    return _vllm_args(c, profile_dir, mfu) if c["engine"] == "vllm" else _sglang_args(c)
