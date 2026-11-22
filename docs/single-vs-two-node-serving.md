# One node or two? Deploying vLLM on your A100 nodes over Ethernet

> Companion to `a100-inference-engineering-report.md` · 18 Nov 2026 · Research only; nothing was changed.
>
> Cluster facts come from read-only `az` and `kubectl` calls. Framework claims are quoted from llm-d v1.0.0, vLLM v0.34.0, and Dynamo v1.6.2 docs and source.

---

## TL;DR

- **You're right: Ethernet only.** Verified on `<cluster>`:
  - **No InfiniBand or RDMA.** Azure reports `RdmaEnabled: False` for `Standard_NC24ads_A100_v4`.
  - **No NVLink.** Each VM has one A100 PCIe card, so there's nothing to link, and Azure's docs list no NVLink for this series.
  - **Accelerated networking (SR-IOV) is on, capped at 20 Gb/s (≈ 2.5 GB/s) per VM.**
  - The two VMs have **no proximity placement group and no zone pinning**, so they aren't guaranteed to sit close together.
- **Start with one model, one GPU, and a plain `vllm serve` per node. No Dynamo, no llm-d.** Dynamo's own README: *"If you're running a single model on a single GPU, your inference engine alone is probably sufficient."*
- **Use the second node for A/B experiments first** (for example vLLM vs SGLang). Later, run it as a second replica behind routing that sends repeat prompts to the replica that already has them cached (llm-d's "Optimized Baseline"). Only HTTP requests cross the network in that design, so Ethernet is fine, and this routing is where multi-node TTFT wins come from.
- **Don't split prefill and decode across these two nodes, except as a lab to measure why not.**
  - llm-d: *"Disaggregated Serving requires high performance (RDMA) interconnects… Without RDMA, NIXL falls back to TCP… should only be used for testing and development."*
  - vLLM: *"Disaggregated prefill DOES NOT improve throughput."*
  - Every request would ship its KV cache over 2.5 GB/s: ≈0.3 s for an 8k-token prompt and ≈1.3 s at 32k (Qwen3-Coder-30B-A3B, BF16).
  - With two GPUs, one prefill GPU plus one decode GPU leaves the prefill GPU mostly idle on chat-like traffic. That's roughly half the throughput of two plain replicas.
- **Never split one model across these nodes with tensor parallelism**; it needs an all-reduce every layer, for every token, over Ethernet. If a model doesn't fit on one GPU, quantize it to fit or use a VM with two or more GPUs.
- **The single-node knob worth adding:** vLLM's CPU KV offload (`--kv-offloading-size`). Each node has 220 GiB of RAM. Reloading a cached 8k-token prefix over PCIe takes ≈30 ms; recomputing it takes ≈650 ms.

---

## Contents

1. [What your network actually is](#1-what-your-network-actually-is)
2. [Single node: ways to run vLLM without Dynamo](#2-single-node-ways-to-run-vllm-without-dynamo)
3. [Two nodes: options ranked](#3-two-nodes-options-ranked)
4. [Why not prefill/decode disaggregation here](#4-why-not-prefilldecode-disaggregation-here)
5. [Measure the network yourself (example, not applied)](#5-measure-the-network-yourself-example-not-applied)
6. [Suggested plan for your two nodes](#6-suggested-plan-for-your-two-nodes)
7. [Sources](#sources)

---

## 1. What your network actually is

| Item | Value | How verified |
|---|---|---|
| GPUs per node | 1× A100 80GB PCIe | `az vm list-skus` capability `GPUs: 1` |
| GPU-to-GPU link | None within a node (one GPU). Azure's docs list no NVLink for NC A100 v4. | Azure NC A100 v4 docs |
| RDMA / InfiniBand | **None** | `RdmaEnabled: False` (both NC24ads and NC48ads) |
| Node network | Accelerated networking on; **20,000 Mb/s max per VM**; up to 2 NICs | VMSS NIC config `enableAcceleratedNetworking: true`; Azure docs; `MaxNetworkInterfaces: 2` |
| Placement | No availability zones, no proximity placement group, `singlePlacementGroup: false` | `az vmss list` |
| Pod networking | Azure CNI **Overlay**, Azure dataplane, no network policy, pod CIDR 10.244.0.0/16 | `az aks show` |
| Local disks | 64 GiB temp disk plus a **960 GiB local NVMe disk** per VM | Azure NC A100 v4 docs |

**Bandwidth ladder.** This is the number that decides architecture:

| Link | Bandwidth | Compared with your Ethernet |
|---|---|---|
| A100 HBM2e (on the GPU) | 1,935 GB/s | ≈775× |
| NVLink 3 (A100 SXM; inside ND A100 v4 VMs) | 600 GB/s | ≈240× |
| PCIe Gen4 x16 (GPU ↔ host RAM on your VM) | ≈32 GB/s each way | ≈13× |
| HDR InfiniBand per GPU (ND A100 v4) | 25 GB/s (200 Gb/s) | 10× |
| **Your node-to-node Ethernet** | **≈2.5 GB/s per VM** (20 Gb/s, best case) | 1× |

Rule: designs that move **model state** between GPUs every token (tensor parallelism) or move **KV cache** every request (prefill/decode disaggregation) pay this tax. Designs that move only **requests** (independent replicas plus routing) don't.

---

## 2. Single node: ways to run vLLM without Dynamo

| Option | What you get | Extra moving parts | Use it when |
|---|---|---|---|
| **Plain Deployment + Service running `vllm serve`** (main report §10) | Every engine flag is explicit; `/metrics` for Prometheus | None beyond Kubernetes | **Start here.** Best for benchmarking. |
| vLLM **production-stack** Helm chart (`vllm-stack-0.1.14`, Nov 2026) | Router (round-robin, session-ID sticky; prefix-aware listed as work in progress), Prometheus/Grafana dashboards, KV offload | Router and observability pods | You want dashboards, or a simple router once you add a second replica |
| **KAITO** (AKS add-on, v0.13.0) | One `Workspace` YAML; serves with vLLM by default | Operator | Quick "give me an endpoint" demos. For benchmarking, a plain Deployment keeps every flag visible. |
| **llm-d** Optimized Baseline (v1.0.0) | Gateway plus endpoint picker (EPP) with prefix-cache- and load-aware routing | Gateway, EPP, Gateway API Inference Extension CRDs | Multiple replicas ([§3](#3-two-nodes-options-ranked)) |
| **NVIDIA Dynamo** (v1.6.2) | Frontend, KV-aware router, SLA planner, disaggregation | Several components | Multi-GPU or multi-node fleets. Its README says one GPU doesn't need it. |

**Single-node knobs for the TTFT and tokens/s game, beyond the main report §8.3:**

| Knob | Flag | Why it helps | Notes |
|---|---|---|---|
| **CPU KV offload** | `--kv-offloading-size 64 --kv-offloading-backend native` (size in GiB) | Evicted prefixes are reloaded over PCIe instead of recomputed. An 8k-token prefix (0.8 GB in BF16) reloads in ≈30 ms at ≈25 GB/s versus ≈650 ms to recompute on the A100. Big TTFT wins for multi-turn and agentic traffic. | Raise the pod's memory request and limit by the offload size (220 GiB RAM per node). Watch `vllm:external_prefix_cache_hits` and `vllm:external_prefix_cache_queries`. |
| More HTTP front-ends | `--api-server-count N` | Adds API-server processes if the Python HTTP layer becomes the bottleneck at high concurrency | Only after you see API-server CPU saturate |
| Long-prompt interference | `--max-num-batched-tokens` (chunked-prefill budget) | Smaller chunks keep decode steps flowing while long prompts prefill: steadier inter-token latency, at some cost to TTFT | This is the single-GPU answer to what disaggregation promises ([§4](#4-why-not-prefilldecode-disaggregation-here)) |
| Faster cold starts | Model weights on the 960 GiB local NVMe instead of a network PVC | Faster model loads after a pod restart | Needs the NVMe exposed to pods (for example via Azure Container Storage local NVMe); verify on your node image first |

---

## 3. Two nodes: options ranked

| Option | What crosses the network | OK on Ethernet? | Verdict |
|---|---|---|---|
| **A. Two independent experiments** (vLLM on node 0, SGLang on node 1) | Benchmark traffic only | ✅ | **Best use while learning**: doubles your experiment speed |
| **B. Two replicas, round-robin Kubernetes Service** | Requests only | ✅ | Baseline. Spreads one conversation's turns across replicas, wasting prefix cache. |
| **C. Two replicas plus cache- and load-aware routing** (llm-d Optimized Baseline; or production-stack session-sticky routing) | Requests only | ✅ | **Recommended multi-node design** |
| D. Prefill/decode disaggregation (1 prefill + 1 decode) via llm-d or Dynamo | KV cache, on every request | ❌ TCP only; llm-d says test/dev only | **Lab exercise only** ([§4](#4-why-not-prefilldecode-disaggregation-here)) |
| E. Tensor parallelism across nodes | Activations, every layer, every token | ❌ | Don't |
| F. Pipeline parallelism across nodes | Activations at the stage boundary, every step | ⚠️ Works, but adds latency | Only if a model can't fit on one GPU *and* quantizing isn't acceptable |

**Why C beats B.**

- In multi-turn and agentic traffic, every turn resends the whole conversation. If turn 2 lands on the other replica, its prefix cache is cold and the full prompt is recomputed: ≈0.65 s for an 8k-token history instead of near zero.
- llm-d's endpoint picker scores replicas on prefix-cache overlap and load. Dynamo's README credits KV-aware routing with *"2x faster time to first token"*, citing a Baseten benchmark on Qwen3-Coder 480B.
- **Measure it yourself:**
  1. Run AIPerf multi-turn load against B, then C: `aiperf profile … --conversation-num 50 --conversation-turn-mean 6`.
  2. Compare TTFT p50/p99.
  3. Compute prefix-cache hit rate from `vllm:prefix_cache_hits` ÷ `vllm:prefix_cache_queries`.
- **llm-d prerequisites:** it needs the Gateway API Inference Extension CRDs and a compatible gateway; check the Optimized Baseline guide for AKS specifics.

**If a model doesn't fit on one A100 80GB,** prefer, in order:
1. A quantized checkpoint that fits on one GPU (for example a 70B model in 4-bit is ≈35–40 GB).
2. A VM with two GPUs on the same host, such as `NC48ads_A100_v4` (PCIe between the GPUs, still RDMA-less), using tensor parallelism inside that one VM.
3. Pipeline parallelism across your two nodes, as a last resort.

---

## 4. Why not prefill/decode disaggregation here

**What prefill/decode disaggregation (P/D) is for, per llm-d and vLLM:**
- Steadier inter-token latency: long prompts stop interrupting ongoing decodes.
- Tuning TTFT and inter-token latency separately, with different parallelism per role.
- Large-scale MoE with wide expert parallelism.

llm-d suggests exploring it for *"medium-large models (e.g. gpt-oss-120b)"* and *"longer input sequence lengths (e.g 10k ISL | 1k OSL, not 200 ISL | 200 OSL)."* Its default P/D topology is 5 GPUs (1 prefill at TP=1, 1 decode at TP=4) on H200, and its tuning example uses 16 GPUs.

**Cost 1: KV transfer on every request.** Qwen3-Coder-30B-A3B in BF16 has 96 KiB of KV cache per token, so:

| Prompt length | KV to ship | Over your Ethernet (2.5 GB/s) | Over InfiniBand (25 GB/s) | Prefill compute it saves (est., A100) |
|---|---|---|---|---|
| 1k tokens | 0.10 GB | ≈40 ms | ≈4 ms | ≈57 ms |
| 8k | 0.81 GB | ≈320 ms | ≈32 ms | ≈650 ms |
| 32k | 3.2 GB | ≈1.3 s | ≈130 ms | ≈5.1 s |
| 100k | 9.8 GB | ≈3.9 s | ≈390 ms | ≈37 s |

- Transfer times assume line rate. TCP through UCX/NIXL, staged through host memory with no GPUDirect RDMA, will be slower.
- Prefill estimates assume 40 % of A100 peak compute, from the main report §6.2.
- In P/D the prefill still has to run on the prefill GPU, so the transfer is *added* to TTFT. On your network that's roughly **+50–70 %** TTFT for 1k–8k prompts; FP8 KV cache halves the transfer.

**Cost 2: half your hardware stops decoding.**
- With two GPUs, P/D means one prefills and one decodes. At 1k in / 1k out, a request spends ≈0.06 s in prefill and tens of seconds in decode. The prefill GPU would sit idle almost all the time while the decode GPU does nearly all the work, and holds all the KV cache.
- Two ordinary replicas both decode, for roughly **2× the throughput** on chat-like traffic.
- vLLM's docs say it plainly: *"Disaggregated prefill DOES NOT improve throughput."*

**What you can do instead on one GPU.**
- Tune the chunked-prefill budget (`--max-num-batched-tokens`) so long prompts are split into small chunks interleaved with decode steps.
- vLLM's docs acknowledge that chunked prefill "can achieve the same goal" (controlling tail inter-token latency), though the right chunk size takes tuning. That's exactly the kind of sweep this project is for.

**If you run it anyway as a lab** (worthwhile for the write-up):
1. Use llm-d's `guides/pd-disaggregation` with NIXL over TCP, adapted to one prefill and one decode pod at TP=1.
2. Predict TTFT from the table above, then measure it.
3. The gap between your prediction and your measurement is your "why P/D needs RDMA" evidence.

**What would make P/D worthwhile:**
- An RDMA fabric, such as ND A100 v4 / NDm A100 v4 (8× A100 SXM with NVLink plus 200 Gb/s InfiniBand per GPU) or the ND H100/H200 series.
- Enough GPUs to tune the prefill:decode ratio.
- Long-prompt workloads on medium-to-large models.

---

## 5. Measure the network yourself (example, not applied)

This needs pods, so it's a change. Your call. No GPU is required. The cluster's allowed-images policy currently *warns* on non-ACR images, so import the image to your ACR first (`az acr import`).

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: iperf-server
  namespace: default
  labels: {app: iperf-server}
spec:
  nodeName: <node-0>
  containers:
    - name: iperf3
      image: <your-acr>.azurecr.io/networkstatic/iperf3:latest   # entrypoint is iperf3
      args: ["-s"]
      ports:
        - containerPort: 5201
---
apiVersion: v1
kind: Service
metadata:
  name: iperf-server
  namespace: default
spec:
  selector: {app: iperf-server}
  ports:
    - port: 5201
      targetPort: 5201
---
apiVersion: v1
kind: Pod
metadata:
  name: iperf-client
  namespace: default
spec:
  nodeName: <node-1>
  restartPolicy: Never
  containers:
    - name: iperf3
      image: <your-acr>.azurecr.io/networkstatic/iperf3:latest
      args: ["-c", "iperf-server", "-P", "8", "-t", "30"]   # add "-R" for the reverse direction
```

- Read the results with `kubectl logs iperf-client`. Expect at most ≈20 Gb/s in aggregate.
- For latency, run `ping` between the two pod IPs.
- Once the GPU is exposed, run `nvidia-smi topo -m` in a GPU pod to confirm the PCIe-only topology.
- Clean up afterward with `kubectl delete pod iperf-server iperf-client` and `kubectl delete svc iperf-server`.

---

## 6. Suggested plan for your two nodes

| Step | Setup | What you learn |
|---|---|---|
| 1 | Node 0: vLLM. Node 1: SGLang. Same model, same client, same settings. | Engine head-to-head (main report Phase 5), at twice the experiment speed |
| 2 | Single node: CPU KV offload off vs on, with a multi-turn AIPerf workload whose total context exceeds GPU KV capacity | How much TTFT a second memory tier buys on A100 |
| 3 | Both nodes vLLM: round-robin Service (B) vs llm-d Optimized Baseline (C), with multi-turn load | Cache-aware routing's TTFT gain: the scale-out story that works on Ethernet |
| 4 (optional lab) | P/D over TCP with llm-d's guide, 1 prefill + 1 decode | Measured KV-transfer tax vs your §4 prediction; evidence that P/D needs RDMA |

**The enterprise narrative this supports:** "On RDMA-less A100 PCIe capacity, the right architecture is one replica per GPU plus cache-aware routing. Disaggregation and wide expert parallelism need RDMA fabrics (ND-series). Here's the measured difference."

---

## Sources

**Azure and your cluster** (read-only, 18 Nov 2026)
- `az vm list-skus -l centralus`: `RdmaEnabled`, `AcceleratedNetworkingEnabled`, `GPUs`, `MaxNetworkInterfaces`
- `az aks show` (network profile) and `az vmss list` (NIC acceleration, zones, proximity placement group)
- Azure NC A100 v4 series (network bandwidth, local NVMe, accelerated networking): <https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series>
- Azure ND A100 v4 series (NVLink and InfiniBand, for contrast): <https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/ndasra100v4-series>

**Frameworks**
- llm-d well-lit paths: <https://github.com/llm-d/llm-d/blob/main/guides/README.md>
- llm-d P/D guide: <https://github.com/llm-d/llm-d/blob/main/guides/pd-disaggregation/README.md>
- llm-d P/D architecture and when to use it: <https://github.com/llm-d/llm-d/blob/main/docs/architecture/disaggregation/pd-disaggregation.md>
- vLLM disaggregated prefilling docs (v0.34.0): <https://github.com/vllm-project/vllm/blob/v0.34.0/docs/features/disagg_prefill.md>
- vLLM KV offloading config (v0.34.0): `vllm/config/cache.py` (`kv_offloading_size`, `kv_offloading_backend`)
- vLLM metrics (v0.34.0): `vllm/v1/metrics/loggers.py` (`vllm:prefix_cache_hits`, `vllm:prefix_cache_queries`, `vllm:external_prefix_cache_*`)
- NVIDIA Dynamo README ("When to use Dynamo"): <https://github.com/ai-dynamo/dynamo>
- vLLM production-stack: <https://github.com/vllm-project/production-stack>
- KAITO: <https://github.com/kaito-project/kaito>
