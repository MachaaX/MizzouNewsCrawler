#!/bin/bash
# Create Kubernetes secret for Decodo unblock proxy credentials

set -e

NAMESPACE="${1:-production}"

echo "Creating decodo-unblock-credentials secret in namespace: $NAMESPACE"

kubectl create secret generic decodo-unblock-credentials \
  --from-literal=username=U0000332559 \
  --from-literal=password=PW_1b20cd078bbfbf554faa89e9af56f7ea8 \
  --namespace=$NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Secret created/updated successfully"
