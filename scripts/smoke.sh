#!/usr/bin/env bash
# Docker smoke check for ContextCore (Validation & Lead lane).
#
# Builds the compose stack, waits for the backend health route, posts a BM25
# compression request, and reports the execution mode. Exits non-zero if the
# stack never becomes healthy, /v1/compress does not return 200, or the
# execution-mode header is missing. A fallback response is a warning, not a
# failure, so the stack is still verified even while the engine deps are broken
# (see docs/evaluation.md defect E3).
#
# Usage:
#   bash scripts/smoke.sh
#   CONTEXTCORE_URL=http://localhost:8000 bash scripts/smoke.sh
set -euo pipefail

BASE_URL="${CONTEXTCORE_URL:-http://localhost:8000}"
TIMEOUT="${CONTEXTCORE_SMOKE_TIMEOUT:-240}"
COMPOSE="${COMPOSE_CMD:-docker compose}"

cleanup() { $COMPOSE down --remove-orphans >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "Building and starting the stack..."
$COMPOSE up -d --build

echo "Waiting for ${BASE_URL}/health (timeout ${TIMEOUT}s)..."
deadline=$(( $(date +%s) + TIMEOUT ))
until curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "FAIL: backend did not become healthy within ${TIMEOUT}s" >&2
    $COMPOSE logs backend || true
    exit 1
  fi
  sleep 3
done
echo "health: $(curl -fsS "${BASE_URL}/health")"

payload='{"system_prompt":"You are a test assistant.","history":[{"role":"user","content":"What is the limit?"}],"context_blocks":[{"id":"api","content":"The limit is 1000 requests per minute."}],"query":"What is the limit?","token_budget":40,"scorer":"bm25"}'

headers=$(mktemp)
body=$(mktemp)
code=$(curl -sS -o "$body" -D "$headers" -w '%{http_code}' \
  -H 'Content-Type: application/json' -X POST "${BASE_URL}/v1/compress" -d "$payload")
mode=$(grep -i '^x-contextcore-execution:' "$headers" | tr -d '\r' | awk '{print $2}')

echo "POST /v1/compress -> HTTP ${code}, X-ContextCore-Execution: ${mode:-<missing>}"
cat "$body"; echo

rm -f "$headers" "$body"

if [ "$code" != "200" ]; then
  echo "FAIL: /v1/compress returned HTTP ${code}" >&2
  exit 1
fi
if [ -z "${mode:-}" ]; then
  echo "FAIL: missing X-ContextCore-Execution header" >&2
  exit 1
fi
if [ "$mode" = "fallback" ]; then
  echo "WARN: backend served the offline fallback; engine import failed (see E3)." >&2
fi
echo "PASS: docker smoke check (execution mode: ${mode})"
