#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: NF_ATLAS_ALLOW_RESTORE=YES $0 /absolute/path/to/snapshot" >&2
  exit 2
fi
if [[ "${NF_ATLAS_ALLOW_RESTORE:-}" != "YES" ]]; then
  echo "Restore refused. Set NF_ATLAS_ALLOW_RESTORE=YES after verifying the target server." >&2
  exit 3
fi

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT="$(realpath -- "$1")"
ARCHIVE_IMAGE="${BACKUP_ARCHIVE_IMAGE:-docker.1ms.run/library/postgres:16-alpine}"
case "${SNAPSHOT}" in
  "${PROJECT_ROOT}/backups/"*|"${PROJECT_ROOT}/migration/"*) ;;
  *) echo "Snapshot must be inside ${PROJECT_ROOT}/backups or ${PROJECT_ROOT}/migration" >&2; exit 4 ;;
esac

required=(postgres.dump neo4j-data.tar.gz chroma.tar.gz minio.tar.gz SHA256SUMS)
for name in "${required[@]}"; do
  [[ -f "${SNAPSHOT}/${name}" ]] || { echo "Missing ${name}" >&2; exit 5; }
done
(cd "${SNAPSHOT}" && sha256sum -c SHA256SUMS)

COMPOSE=(
  "${PROJECT_ROOT}/scripts/server_compose.sh"
  --project-directory "${PROJECT_ROOT}"
  -f "${PROJECT_ROOT}/docker-compose.yml"
  -f "${PROJECT_ROOT}/docker-compose.server.yml"
  --profile docker-ui
)

echo "Stopping NF-Atlas on this server..."
"${COMPOSE[@]}" down --remove-orphans

for volume in nf_postgres nf_neo4j_data nf_minio nf_chroma; do
  docker volume create "nf-atlas_${volume}" >/dev/null
done
if [[ -f "${SNAPSHOT}/models.tar.gz" ]]; then
  docker volume create nf-atlas_nf_models >/dev/null
fi

restore_archive() {
  local volume="$1"
  local archive="$2"
  docker run --rm \
    --mount "type=volume,src=${volume},dst=/target" \
    --mount "type=bind,src=${SNAPSHOT},dst=/snapshot,readonly" \
    "${ARCHIVE_IMAGE}" sh -ec \
    'find /target -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; tar -xzf "/snapshot/'"${archive}"'" -C /target'
}

echo "Restoring Neo4j, Chroma and MinIO volumes..."
restore_archive nf-atlas_nf_neo4j_data neo4j-data.tar.gz
restore_archive nf-atlas_nf_chroma chroma.tar.gz
restore_archive nf-atlas_nf_minio minio.tar.gz
if [[ -f "${SNAPSHOT}/models.tar.gz" ]]; then
  echo "Restoring the optional embedding-model cache..."
  restore_archive nf-atlas_nf_models models.tar.gz
fi

echo "Starting PostgreSQL and restoring the logical dump..."
"${COMPOSE[@]}" up -d postgres
until "${COMPOSE[@]}" exec -T postgres pg_isready -U nfagent -d nfagent >/dev/null 2>&1; do sleep 2; done
"${COMPOSE[@]}" exec -T postgres pg_restore -U nfagent -d nfagent --clean --if-exists < "${SNAPSHOT}/postgres.dump"

echo "Starting the complete server stack..."
"${COMPOSE[@]}" up -d postgres redis neo4j minio chroma grobid api worker priority-worker ui
echo "Restore submitted. Run scripts/server_status.sh until all health checks pass."
