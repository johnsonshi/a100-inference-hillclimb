#!/usr/bin/env bash
# Copy the in-cluster controller's ledger + trial artifacts to ./results-cluster (read-only on the cluster).
# LEDGER_ONLY=1 copies just the ledger and best-so-far files (what `harness.cli watch` needs). The full copy skips
# server logs, metric samples, per-process CPU samples and profiler traces (fetch those per trial when needed).
set -euo pipefail
cd "$(dirname "$0")/.."
. tools/env.sh HILLCLIMB_KUBE_CONTEXT
CTX=$HILLCLIMB_KUBE_CONTEXT
POD=$(kubectl --context "$CTX" -n default get pods -l app=hillclimb-controller -o jsonpath='{.items[0].metadata.name}')
mkdir -p results-cluster
kubectl --context "$CTX" -n default cp "$POD:/app/results/ledger.sqlite" results-cluster/ledger.sqlite -c controller >/dev/null
kubectl --context "$CTX" -n default cp "$POD:/app/results/best" results-cluster/best -c controller >/dev/null 2>&1 || true
if [ -z "${LEDGER_ONLY:-}" ]; then
  kubectl --context "$CTX" -n default exec "$POD" -c controller -- sh -c 'cd /app/results && tar czf - --exclude="*/server.log" --exclude="*/server.head.log" --exclude="*.server.json" --exclude="*/cpu.txt" --exclude="*/prof" trials' | tar xzf - -C results-cluster
fi
echo "pulled $(sqlite3 results-cluster/ledger.sqlite 'select count(*) from trials') trials from $POD"
