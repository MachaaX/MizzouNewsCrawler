#!/usr/bin/env bash
# Create Kubernetes secret for Decodo ISP proxy credentials
set -euo pipefail

NAMESPACE="${1:-production}"

echo "Creating decodo-proxy-creds secret in namespace: $NAMESPACE"

if [[ -z "${DECODO_USERNAME:-}" || -z "${DECODO_PASSWORD:-}" ]]; then
  echo "DECODO_USERNAME/DECODO_PASSWORD env vars must be set before running this script." >&2
  echo "You can also use scripts/sync-decodo-credentials.sh to create GitHub/GCP secrets from the values." >&2
  exit 1
fi

kubectl create secret generic decodo-proxy-creds \
  --from-literal=username="$DECODO_USERNAME" \
  --from-literal=password="$DECODO_PASSWORD" \
  --namespace="$NAMESPACE" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Secret created/updated successfully"
