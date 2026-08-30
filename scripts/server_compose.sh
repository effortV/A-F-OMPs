#!/usr/bin/env bash
set -Eeuo pipefail

if docker compose version >/dev/null 2>&1; then
  exec docker compose "$@"
fi
if command -v docker-compose >/dev/null 2>&1; then
  exec docker-compose "$@"
fi
echo "Docker Compose is not installed." >&2
exit 127
