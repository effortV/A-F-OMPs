#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE=(
  "${PROJECT_ROOT}/scripts/server_compose.sh"
  --project-directory "${PROJECT_ROOT}"
  -f "${PROJECT_ROOT}/docker-compose.yml"
  -f "${PROJECT_ROOT}/docker-compose.server.yml"
  --profile docker-ui
)

"${COMPOSE[@]}" ps
echo
curl --fail --silent --show-error http://127.0.0.1:8000/api/health
echo
echo "Latest backup:"
find "${PROJECT_ROOT}/backups" -mindepth 1 -maxdepth 1 -type d -name '20??-??-??T??????Z' \
  -printf '%f\n' 2>/dev/null | sort | tail -n 1 || true
