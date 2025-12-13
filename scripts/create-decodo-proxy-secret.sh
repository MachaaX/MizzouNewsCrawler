#!/usr/bin/env bash
# Create Kubernetes secret for Decodo ISP proxy credentials (decodo-proxy-creds)
set -euo pipefail

print_usage() {
  cat <<EOF
Usage: $0 [--username USER] [--password PASS] [--full-url URL] [NAMESPACE]

Creates/updates a Kubernetes secret named decodo-proxy-creds with the provided credentials.
If --full-url is provided, the secret will contain key full_proxy_url; otherwise it will contain username/password.

Examples:
  export DECODO_USERNAME=user
  export DECODO_PASSWORD=pass
  ./scripts/create-decodo-proxy-secret.sh production

  # Or use full URL
  ./scripts/create-decodo-proxy-secret.sh --full-url "https://user:pass@isp.decodo.com:10000" production

EOF
}

ISP_USERNAME="${DECODO_USERNAME:-}"
ISP_PASSWORD="${DECODO_PASSWORD:-}"
ISP_FULL_URL=""
NAMESPACE="${1:-production}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --username)
      ISP_USERNAME="$2"; shift 2;;
    --password)
      ISP_PASSWORD="$2"; shift 2;;
    --full-url)
      ISP_FULL_URL="$2"; shift 2;;
    -h|--help)
      print_usage; exit 0;;
    *)
      # Assume namespace if not provided
      if [[ -z "$NAMESPACE" ]]; then
        NAMESPACE="$1"; shift 1
      else
        echo "Unknown option: $1"; print_usage; exit 1
      fi;;
  esac
done

if [[ -z "$ISP_FULL_URL" && ( -z "$ISP_USERNAME" || -z "$ISP_PASSWORD" ) ]]; then
  echo "Provide either --full-url or both --username and --password" >&2
  exit 1
fi

SECRET_NAME="decodo-proxy-creds"
if [[ -n "$ISP_FULL_URL" ]]; then
  kubectl create secret generic "$SECRET_NAME" --namespace "$NAMESPACE" --from-literal=full_proxy_url="$ISP_FULL_URL" --dry-run=client -o yaml | kubectl apply -f -
else
  kubectl create secret generic "$SECRET_NAME" --namespace "$NAMESPACE" --from-literal=username="$ISP_USERNAME" --from-literal=password="$ISP_PASSWORD" --dry-run=client -o yaml | kubectl apply -f -
fi

echo "✅ Secret $SECRET_NAME created/updated in namespace $NAMESPACE"
