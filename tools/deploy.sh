#!/usr/bin/env bash
# Build the controller image from the current commit, publish campaign files, and point the controller at a
# campaign. Usage: tools/deploy.sh [campaign-name]   (default main-v3; the working tree must be committed)
# DRY_RUN=1 prints the image and the controller patch without building or changing anything.
# Touches only objects labeled app.kubernetes.io/part-of=a100-hillclimb; never the cluster or its node pools.
# Environment names (context, registry, subscription, lane nodes) come from .env; see .env.example.
set -euo pipefail
cd "$(dirname "$0")/.."
. tools/env.sh HILLCLIMB_KUBE_CONTEXT HILLCLIMB_REGISTRY HILLCLIMB_LANES HILLCLIMB_ACR_NAME HILLCLIMB_AZ_SUBSCRIPTION
SUB=$HILLCLIMB_AZ_SUBSCRIPTION
ACR=$HILLCLIMB_ACR_NAME
CAMPAIGN=${1:-main-v3}
K="kubectl --context $HILLCLIMB_KUBE_CONTEXT -n default"
[ -f "campaigns/$CAMPAIGN.json" ] || { echo "no campaigns/$CAMPAIGN.json"; exit 1; }
if [ -n "$(git status --porcelain -- harness campaigns tools Dockerfile 2>/dev/null)" ]; then
  echo "commit harness/campaigns/tools changes first"; exit 1
fi
SHA=$(git rev-parse --short HEAD)
IMG=$HILLCLIMB_REGISTRY/hillclimb/controller:$SHA
# Image, campaign, and the controller's environment (engine image registry and lane nodes, from .env).
PATCH=$(python3 - "$IMG" "$CAMPAIGN" <<'EOF'
import json, os, sys
img, campaign = sys.argv[1:]
env = [{"name": k, "value": os.environ[k]} for k in ("HILLCLIMB_REGISTRY", "HILLCLIMB_LANES")]
print(json.dumps([
    {"op": "replace", "path": "/spec/template/spec/containers/0/image", "value": img},
    {"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": [f"/campaigns/{campaign}.json"]},
    {"op": "add", "path": "/spec/template/spec/containers/0/env", "value": env},
]))
EOF
)
if [ -n "${DRY_RUN:-}" ]; then echo "would deploy $IMG"; echo "$PATCH"; exit 0; fi
if ! az acr repository show-tags --subscription $SUB -n $ACR --repository hillclimb/controller -o tsv 2>/dev/null | grep -qx "$SHA"; then
  az acr build --subscription $SUB -r $ACR -t hillclimb/controller:$SHA . --no-logs -o none
fi
python3 - "$CAMPAIGN" <<'EOF' | $K apply -f -
import json, pathlib, sys
data = {p.name: p.read_text() for p in sorted(pathlib.Path("campaigns").glob("*.json"))}
print(json.dumps({"apiVersion": "v1", "kind": "ConfigMap",
                  "metadata": {"name": "hillclimb-campaigns", "namespace": "default",
                               "labels": {"app.kubernetes.io/part-of": "a100-hillclimb"}},
                  "data": data}))
EOF
$K patch deploy hillclimb-controller --type=json -p "$PATCH"
$K scale deploy/hillclimb-controller --replicas=1
$K rollout status deploy/hillclimb-controller --timeout=300s
echo "deployed $IMG running campaign $CAMPAIGN"
