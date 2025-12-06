#!/bin/bash
# Grant Cloud Build permissions to the GitHub Actions service account

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-mizzou-news-crawler}"
SERVICE_ACCOUNT=""
REQUIRED_ROLES=(
  "roles/cloudbuild.builds.editor"
  "roles/cloudbuild.builds.viewer"
)

usage() {
  cat <<'EOF'
Usage: ./scripts/grant-github-actions-permissions.sh [--project PROJECT_ID] [--service-account EMAIL]

Ensures the GitHub Actions service account can trigger Cloud Build jobs.
If no service account is provided, defaults to github-actions@PROJECT_ID.iam.gserviceaccount.com.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT_ID="$2"
      shift 2
      ;;
    --service-account)
      SERVICE_ACCOUNT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SERVICE_ACCOUNT" ]]; then
  SERVICE_ACCOUNT="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI not found. Install it before running this script." >&2
  exit 1
fi

echo "Target project: $PROJECT_ID"
echo "Service account: $SERVICE_ACCOUNT"

gcloud iam service-accounts describe "$SERVICE_ACCOUNT" \
  --project="$PROJECT_ID" >/dev/null 2>&1 || {
  echo "Service account not found. Confirm the email and project." >&2
  exit 1
}

for role in "${REQUIRED_ROLES[@]}"; do
  echo "Granting $role..."
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="$role" \
    --condition=None >/dev/null 2>&1
done

echo "Permissions verified. GitHub Actions can trigger Cloud Build jobs now."
