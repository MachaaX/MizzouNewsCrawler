#!/usr/bin/env bash
set -euo pipefail

# Sync Decodo Unblock credentials across GCP Secret Manager, GitHub Repo Secrets,
# and optional Kubernetes secret. This helps ensure all environments (GitHub Actions,
# production K8s, and GCP secret store) have the same credentials.

print_usage() {
  cat <<EOF
Usage: $0 [--username USER] [--password PASS] [--gcp-project PROJECT] [--github-repo OWNER/REPO] [--namespace NAMESPACE] [--no-k8s] [--no-gh] [--no-gcp]

By default, this script will:
  - Create/update GCP Secret Manager secret: decodo-unblock-credentials
  - Create/update GitHub repo secrets: UNBLOCK_PROXY_USER, UNBLOCK_PROXY_PASS
  - Create/update Kubernetes secret: decodo-unblock-credentials (username/password keys)

Flags:
  --username USER        Decodo unblock username. Overrides env UNBLOCK_PROXY_USER.
  --password PASS        Decodo unblock password. Overrides env UNBLOCK_PROXY_PASS.
  --gcp-project PROJECT  GCP project id to use. Defaults to GOOGLE_CLOUD_PROJECT env var or gcloud configured project.
  --github-repo REPO     GitHub repo in owner/repo format. Defaults to git remote origin.
  --namespace NAMESPACE  K8s namespace to apply K8s secret. Defaults to production.
  --no-k8s               Skip creating/updating the Kubernetes secret.
  --no-gh                Skip creating/updating GitHub repo secrets.
  --no-gcp               Skip creating/updating GCP Secret Manager secret.
  -h, --help             Show this help and exit.

EOF
}

USERNAME=${UNBLOCK_PROXY_USER:-}
PASSWORD=${UNBLOCK_PROXY_PASS:-}
GCP_PROJECT=${GOOGLE_CLOUD_PROJECT:-}
GH_REPO=""
NAMESPACE=${1:-production}
UPDATE_K8S=true
UPDATE_GH=true
UPDATE_GCP=true

READ_FROM_GCP=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --username)
      USERNAME="$2"; shift 2;;
    --password)
      PASSWORD="$2"; shift 2;;
    --gcp-project)
      GCP_PROJECT="$2"; shift 2;;
    --github-repo)
      GH_REPO="$2"; shift 2;;
    --namespace)
      NAMESPACE="$2"; shift 2;;
    --from-gcp)
      READ_FROM_GCP=true; shift 1;;
    --no-k8s)
      UPDATE_K8S=false; shift 1;;
    --no-gh)
      UPDATE_GH=false; shift 1;;
    --no-gcp)
      UPDATE_GCP=false; shift 1;;
    -h|--help)
      print_usage; exit 0;;
    *)
      # Unknown flag; assume it's namespace if not already set
      if [[ -z "$USERNAME" && -z "$PASSWORD" && -z "$GCP_PROJECT" && -z "$GH_REPO" ]]; then
        NAMESPACE="$1"; shift 1
      else
        echo "Unknown option: $1"; print_usage; exit 1
      fi;;
  esac
done

# Resolve defaults
if [[ -z "$USERNAME" && -z "$PASSWORD" ]]; then
  echo "UNBLOCK_PROXY_USER and UNBLOCK_PROXY_PASS must be provided either via env vars or with --username/--password flags." >&2
  exit 1
fi
if [[ -z "$USERNAME" || -z "$PASSWORD" ]]; then
  echo "Both --username and --password must be provided together." >&2
  exit 1
fi

if [[ -z "$GCP_PROJECT" && "$UPDATE_GCP" == true ]]; then
  # Try to read from gcloud configured project
  if command -v gcloud >/dev/null 2>&1; then
    GCP_PROJECT=$(gcloud config get-value project 2>/dev/null || true)
  fi
fi

# Determine GH repo from git if not provided
if [[ -z "$GH_REPO" && "$UPDATE_GH" == true ]]; then
  if command -v git >/dev/null 2>&1; then
    ORIGIN_URL=$(git config --get remote.origin.url || true)
    if [[ -n "$ORIGIN_URL" ]]; then
      # Convert git@github.com:owner/repo.git or https://github.com/owner/repo.git
      if [[ "$ORIGIN_URL" =~ ^git@github.com:(.+)/(.+).git$ ]]; then
        GH_REPO="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
      else
        # Use https case
        REPO_PATH=${ORIGIN_URL#*github.com[:/]}
        REPO_PATH=${REPO_PATH%.git}
        GH_REPO="$REPO_PATH"
      fi
    fi
  fi
fi

if [[ "$UPDATE_GH" == true && -z "$GH_REPO" ]]; then
  echo "GitHub repo not determined. Use --github-repo owner/repo or run within a cloned repository." >&2
  exit 1
fi

if [[ "$UPDATE_GCP" == true && -z "$GCP_PROJECT" ]]; then
  echo "GCP project not configured and gcloud not authenticated. Set GOOGLE_CLOUD_PROJECT or pass --gcp-project." >&2
  exit 1
fi

SECRET_NAME="decodo-unblock-credentials"

set +u
LOCAL_USERNAME=$USERNAME
LOCAL_PASSWORD=$PASSWORD
set -u

if [[ "$UPDATE_GCP" == true ]]; then
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud not found. Skipping GCP Secret Manager update." >&2
  else
    echo "Synchronizing GCP secret $SECRET_NAME (project: $GCP_PROJECT) ..."
    SECRET_JSON=$(printf '{"username":"%s","password":"%s"}' "$LOCAL_USERNAME" "$LOCAL_PASSWORD")
    if gcloud secrets describe "$SECRET_NAME" --project "$GCP_PROJECT" >/dev/null 2>&1; then
      printf '%s' "$SECRET_JSON" | gcloud secrets versions add "$SECRET_NAME" --data-file=- --project "$GCP_PROJECT"
      echo "Updated secret $SECRET_NAME in GCP Secret Manager"
    else
      printf '%s' "$SECRET_JSON" | gcloud secrets create "$SECRET_NAME" --data-file=- --replication-policy="automatic" --project "$GCP_PROJECT"
      echo "Created secret $SECRET_NAME in GCP Secret Manager"
    fi
  fi
fi

if [[ "$READ_FROM_GCP" == true ]]; then
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud not available; cannot read secret from GCP" >&2
    exit 1
  fi
  if [[ -z "$GCP_PROJECT" ]]; then
    echo "GCP project required to read secret from GCP" >&2
    exit 1
  fi
  echo "Reading secret $SECRET_NAME from GCP Secret Manager (project: $GCP_PROJECT) ..."
  SECRET_JSON=$(gcloud secrets versions access latest --secret="$SECRET_NAME" --project="$GCP_PROJECT" 2>/dev/null || true)
  if [[ -z "$SECRET_JSON" ]]; then
    echo "Secret $SECRET_NAME not found in GCP project $GCP_PROJECT" >&2
    exit 1
  fi
  # Extract JSON keys
  if command -v jq >/dev/null 2>&1; then
    LOCAL_USERNAME=$(printf '%s' "$SECRET_JSON" | jq -r '.username')
    LOCAL_PASSWORD=$(printf '%s' "$SECRET_JSON" | jq -r '.password')
  else
    # Fallback to python parsing if jq unavailable
    LOCAL_USERNAME=$(python - <<'PY'
import json,sys
data=sys.stdin.read()
obj=json.loads(data)
print(obj.get('username',''))
PY
    )
    LOCAL_PASSWORD=$(python - <<'PY'
import json,sys
data=sys.stdin.read()
obj=json.loads(data)
print(obj.get('password',''))
PY
    )
  fi
fi

if [[ "$UPDATE_GH" == true ]]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI not found. Skipping GitHub secret update." >&2
  else
    echo "Setting GitHub repo secrets for $GH_REPO..."
    # Set GitHub repository-level secrets
    echo -n "$LOCAL_USERNAME" | gh secret set UNBLOCK_PROXY_USER -R "$GH_REPO" --body - || true
    echo -n "$LOCAL_PASSWORD" | gh secret set UNBLOCK_PROXY_PASS -R "$GH_REPO" --body - || true
    echo "✅ GitHub secrets UNBLOCK_PROXY_USER/UNBLOCK_PROXY_PASS set in $GH_REPO"
  fi
fi

if [[ "$UPDATE_K8S" == true ]]; then
  if ! command -v kubectl >/dev/null 2>&1; then
    echo "kubectl not installed; skipping K8s secret creation." >&2
  else
    echo "Creating/updating k8s secret $SECRET_NAME in namespace $NAMESPACE..."
    kubectl create secret generic "$SECRET_NAME" --namespace "$NAMESPACE" \
      --from-literal=username="$LOCAL_USERNAME" \
      --from-literal=password="$LOCAL_PASSWORD" \
      --dry-run=client -o yaml | kubectl apply -f -
    echo "✅ Kubernetes secret $SECRET_NAME created/updated in namespace $NAMESPACE"
  fi
fi

echo "Done."
