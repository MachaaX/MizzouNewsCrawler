#!/usr/bin/env bash
# Create Kubernetes secret for Decodo unblock proxy credentials
set -euo pipefail

NAMESPACE="${1:-production}"

echo "Creating decodo-unblock-credentials secret in namespace: $NAMESPACE"

if [[ -z "${UNBLOCK_PROXY_USER:-}" || -z "${UNBLOCK_PROXY_PASS:-}" ]]; then
  echo "UNBLOCK_PROXY_USER/UNBLOCK_PROXY_PASS env vars must be set before running this script." >&2
  echo "You can also use scripts/sync-decodo-credentials.sh to create GitHub/GCP secrets from the values." >&2
  exit 1
fi

kubectl create secret generic decodo-unblock-credentials \
  --from-literal=username="$UNBLOCK_PROXY_USER" \
  --from-literal=password="$UNBLOCK_PROXY_PASS" \
  --namespace="$NAMESPACE" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Secret created/updated successfully"
