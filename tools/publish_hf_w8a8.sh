#!/usr/bin/env bash
# Safely publish /mnt/nvme/hf/Nanbeige4.3-33B-W8A8-INT8 from the AKS cluster node directly to Hugging Face Hub.
# Reads the HF token from ~/.cache/huggingface/token (created via `hf auth login`) or HF_TOKEN in .env.
# Never prints or commits the token; uses a temporary K8s Secret that is deleted on exit.
# Usage: tools/publish_hf_w8a8.sh [repo_name]   (default repo_name: Nanbeige4.3-33B-W8A8-INT8)
set -euo pipefail
cd "$(dirname "$0")/.."
. tools/env.sh HILLCLIMB_KUBE_CONTEXT HILLCLIMB_REGISTRY HILLCLIMB_LANES

REPO_NAME="${1:-Nanbeige4.3-33B-W8A8-INT8}"
TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$TOKEN_FILE" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "No Hugging Face token found."
  echo "Run 'hf auth login' first (or set HF_TOKEN in .env)."
  exit 1
fi

K="kubectl --context $HILLCLIMB_KUBE_CONTEXT -n default"
node0=$(echo "${HILLCLIMB_LANES%%,*}" | cut -d= -f2)
IMG="$HILLCLIMB_REGISTRY/vllm/vllm-openai:v0.34.0"
SEC="hc-hf-upload-secret"
CM="hc-hf-model-card"
POD="hc-hf-upload-w8a8"

cleanup() {
  $K delete secret "$SEC" --ignore-not-found >/dev/null 2>&1 || true
  $K delete configmap "$CM" --ignore-not-found >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup
$K delete pod "$POD" --ignore-not-found >/dev/null 2>&1 || true
$K create secret generic "$SEC" --from-literal=HF_TOKEN="$HF_TOKEN" >/dev/null
$K create configmap "$CM" --from-file=README.md=docs/hf-model-card.md >/dev/null

$K apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: $POD
  labels:
    app.kubernetes.io/part-of: a100-hillclimb
spec:
  restartPolicy: Never
  nodeName: $node0
  tolerations:
  - operator: Exists
  containers:
  - name: upload
    image: $IMG
    env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: $SEC
          key: HF_TOKEN
    - name: REPO_NAME
      value: "$REPO_NAME"
    command:
    - python3
    - -c
    - |
      import os
      from pathlib import Path
      from huggingface_hub import HfApi

      api = HfApi(token=os.environ["HF_TOKEN"])
      user = api.whoami()["name"]
      repo_name = os.environ["REPO_NAME"]
      repo_id = repo_name if "/" in repo_name else f"{user}/{repo_name}"
      print(f"Authenticated as: {user} -> target repo: {repo_id}", flush=True)

      api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

      card = Path("/card/README.md").read_text().replace("__REPO_ID__", repo_id)
      api.upload_file(
          path_or_fileobj=card.encode("utf-8"),
          path_in_repo="README.md",
          repo_id=repo_id,
          repo_type="model",
          commit_message="Add model card with A100 benchmarks, quality evaluations, and vLLM/SGLang recipes",
      )
      print("Uploaded README.md; starting folder upload of /models/Nanbeige4.3-33B-W8A8-INT8 ...", flush=True)

      api.upload_folder(
          folder_path="/models/Nanbeige4.3-33B-W8A8-INT8",
          repo_id=repo_id,
          repo_type="model",
          ignore_patterns=["DONE", ".*"],
          commit_message="Upload Nanbeige4.3-33B INT8 W8A8 (compressed-tensors) checkpoint",
      )
      print(f"Done! Published at: https://huggingface.co/{repo_id}", flush=True)
    volumeMounts:
    - name: models
      mountPath: /models
      readOnly: true
    - name: card
      mountPath: /card
      readOnly: true
  volumes:
  - name: models
    hostPath:
      path: /mnt/nvme/hf
  - name: card
    configMap:
      name: $CM
YAML

echo "Started upload pod $POD on cluster; streaming progress..."
$K wait --for=condition=Ready pod/"$POD" --timeout=60s >/dev/null 2>&1 || true
$K logs -f "$POD"
$K delete pod "$POD" --ignore-not-found >/dev/null 2>&1 || true
