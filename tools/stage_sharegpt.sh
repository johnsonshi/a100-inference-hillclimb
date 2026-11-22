#!/usr/bin/env bash
# Stage a fixed 30k-conversation ShareGPT subset on every GPU node's NVMe (/mnt/nvme/hf/datasets/sharegpt),
# read by the T2 chat plan (t2c) through `vllm bench serve --dataset-name sharegpt`.
# Why ShareGPT: SPEC-Bench has only 480 prompts (a ~1000-request rate search would mostly re-send prompts,
# flattering prefix caching and suffix decoding), and vLLM's SpecBench loader ignores --seed.
# Idempotent: skips a node whose subset already exists. Usage: tools/stage_sharegpt.sh (lanes and registry from .env)
set -euo pipefail
cd "$(dirname "$0")/.."
. tools/env.sh HILLCLIMB_KUBE_CONTEXT HILLCLIMB_REGISTRY HILLCLIMB_LANES
CTX=$HILLCLIMB_KUBE_CONTEXT
IMG=$HILLCLIMB_REGISTRY/vllm/vllm-openai:v0.34.0
URL=https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
read -r -d '' PY <<'EOF' || true
import json, os, random, sys, urllib.request
out = "/hf/datasets/sharegpt/sharegpt_v3_30k.json"
if os.path.exists(out):
    print("exists", out); sys.exit(0)
os.makedirs(os.path.dirname(out), exist_ok=True)
raw = "/hf/datasets/sharegpt/raw.json"
urllib.request.urlretrieve(os.environ["URL"], raw)
data = json.load(open(raw))
ok = [d for d in data if len(d.get("conversations", [])) >= 2
      and d["conversations"][0].get("from") == "human" and d["conversations"][1].get("from") == "gpt"
      and d["conversations"][0].get("value", "").strip()]
random.Random(2026).shuffle(ok)
sub = [{"id": d.get("id"), "conversations": d["conversations"][:2]} for d in ok[:30000]]
json.dump(sub, open(out + ".tmp", "w"))
os.replace(out + ".tmp", out)
os.remove(raw)
print("wrote", out, len(sub), "of", len(ok), "valid of", len(data))
EOF
for lane_node in ${HILLCLIMB_LANES//,/ }; do
  node=${lane_node#*=}
  name="hc-stage-sharegpt-${lane_node%%=*}"
  kubectl --context "$CTX" -n default delete pod "$name" --ignore-not-found >/dev/null
  kubectl --context "$CTX" -n default apply -f - <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: $name
  labels: {app: hillclimb-stage, app.kubernetes.io/part-of: a100-hillclimb}
spec:
  restartPolicy: Never
  nodeSelector: {kubernetes.io/hostname: $node}
  containers:
  - name: stage
    image: $IMG
    command: ["python3", "-c", $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$PY")]
    env: [{name: URL, value: "$URL"}]
    resources: {requests: {cpu: "1", memory: 6Gi}, limits: {cpu: "2", memory: 10Gi}}
    volumeMounts: [{name: hf, mountPath: /hf}]
  volumes: [{name: hf, hostPath: {path: /mnt/nvme/hf, type: Directory}}]
YAML
done
echo "staging pods started; check: kubectl --context $CTX -n default logs -l app=hillclimb-stage"
