#!/usr/bin/env bash
set -euo pipefail

# Create or update the MLflow App Platform app from infra/digitalocean/app.yaml.
#
# Renders the spec with the target image tag, validates it, creates the app
# on the first run and updates + redeploys it afterwards, then waits for the
# deployment to become ACTIVE (skip with --no-wait).
#
# The app deploys pre-built images from the DigitalOcean Container Registry;
# this script does not build or push them, it only deploys what is already
# in the registry.
#
# Usage: scripts/deploy_do_app.sh [tag] [--no-wait]
#   tag    registry tag to deploy (default: latest)
#
# The image owner defaults to the GHCR owner (all lowercase, as GHCR
# requires); override with the OWNER environment variable.
#
# Requires doctl, authenticated either via `doctl auth init` or by having
# DIGITALOCEAN_ACCESS_TOKEN set in the environment.

APP_NAME="${APP_NAME:-mlops-itu-jtk}"
SPEC="infra/digitalocean/app.yaml"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TAG="latest"
OWNER="${OWNER:-jeppe-t-k}"
WAIT=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-wait) WAIT=0 ;;
        *) TAG="$1" ;;
    esac
    shift
done

if ! command -v doctl >/dev/null 2>&1; then
    echo "doctl not found. Install it: https://docs.digitalocean.com/reference/doctl/"
    exit 1
fi

if [[ -z "${DIGITALOCEAN_ACCESS_TOKEN:-}" ]] && ! doctl auth list 2>/dev/null | grep -q '(current)'; then
    echo "doctl is not authenticated. Run: doctl auth init"
    exit 1
fi

spec_file="$(mktemp)"
trap 'rm -f "$spec_file"' EXIT
sed -e "s/__TAG__/$TAG/" -e "s/__OWNER__/$OWNER/" \
    "$SCRIPT_DIR/$SPEC" > "$spec_file"

echo "Validating app spec (owner: $OWNER, tag: $TAG)..."
doctl apps spec validate "$spec_file" > /dev/null

app_id="$(doctl apps list --format ID,Spec.Name --no-header \
    | awk -v name="$APP_NAME" '$2 == name { print $1; exit }')"

if [[ -z "$app_id" ]]; then
    echo "Creating $APP_NAME (the first deployment starts with it)..."
    app_id="$(doctl apps create --spec "$spec_file" --format ID --no-header)"
else
    echo "Updating $APP_NAME ($app_id)..."
    doctl apps update "$app_id" --spec "$spec_file" > /dev/null
    # A spec update alone does not pick up new images behind the same tag.
    doctl apps create-deployment "$app_id" > /dev/null
fi
echo "App ID: $app_id"

if [[ "$WAIT" -ne 1 ]]; then
    exit 0
fi

# Phases: PENDING_BUILD, BUILDING, PENDING_DEPLOY, DEPLOYING, then
# ACTIVE, or ERROR / CANCELED / SUPERSEDED.
phase=""
for _ in $(seq 1 100); do
    phase="$(doctl apps list-deployments "$app_id" --format Phase --no-header | head -n 1)"
    echo "deployment phase: $phase"
    case "$phase" in
        ACTIVE) break ;;
        ERROR|CANCELED|SUPERSEDED)
            echo "Deployment ended in $phase. Recent logs:"
            doctl apps logs "$app_id" --type build --tail 50 || true
            doctl apps logs "$app_id" --type deploy --tail 50 || true
            exit 1 ;;
    esac
    sleep 15
done

if [[ "$phase" != "ACTIVE" ]]; then
    echo "Deployment did not finish within 25 minutes."
    exit 1
fi

ingress="$(doctl apps get "$app_id" --format DefaultIngress --no-header)"
echo ""
echo "Deployed:"
echo "  MLflow UI:       $ingress/"
echo "  Inference:       $ingress/invocations"
