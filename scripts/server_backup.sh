#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_ROOT="${BACKUP_ROOT:-${PROJECT_ROOT}/backups}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
RESTART_SERVICES="${BACKUP_RESTART_SERVICES:-true}"
ARCHIVE_IMAGE="${BACKUP_ARCHIVE_IMAGE:-docker.1ms.run/library/postgres:16-alpine}"
STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
PARTIAL_DIR="${BACKUP_ROOT}/.partial-${STAMP}"
FINAL_DIR="${BACKUP_ROOT}/${STAMP}"
LOCK_FILE="${BACKUP_ROOT}/.backup.lock"

case "${BACKUP_ROOT}" in
  "${PROJECT_ROOT}/backups"|"${PROJECT_ROOT}/backups/"*) ;;
  *) echo "BACKUP_ROOT must stay inside ${PROJECT_ROOT}/backups" >&2; exit 2 ;;
esac
case "${RESTART_SERVICES}" in
  true|false) ;;
  *) echo "BACKUP_RESTART_SERVICES must be true or false" >&2; exit 2 ;;
esac

mkdir -p -- "${BACKUP_ROOT}"
chmod 700 "${BACKUP_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another NF-Atlas backup is already running." >&2
  exit 3
fi

if [[ -e "${PARTIAL_DIR}" || -e "${FINAL_DIR}" ]]; then
  echo "Backup target already exists: ${STAMP}" >&2
  exit 4
fi
mkdir -p -- "${PARTIAL_DIR}"
chmod 700 "${PARTIAL_DIR}"

COMPOSE=(
  "${PROJECT_ROOT}/scripts/server_compose.sh"
  --project-directory "${PROJECT_ROOT}"
  -f "${PROJECT_ROOT}/docker-compose.yml"
  -f "${PROJECT_ROOT}/docker-compose.server.yml"
  --profile docker-ui
)

restore_services() {
  "${COMPOSE[@]}" up -d postgres redis neo4j minio chroma grobid api worker priority-worker ui >/dev/null || true
}
trap restore_services EXIT

echo "[1/7] Quiescing API and workers..."
"${COMPOSE[@]}" stop -t 120 ui priority-worker worker api >/dev/null

echo "[2/7] Dumping PostgreSQL..."
"${COMPOSE[@]}" exec -T postgres pg_dump -U nfagent -d nfagent -Fc > "${PARTIAL_DIR}/postgres.dump"
"${COMPOSE[@]}" exec -T postgres psql -U nfagent -d nfagent -At -c \
  "SELECT 'chunks='||count(*) FROM chunks UNION ALL SELECT 'conversations='||count(*) FROM conversations UNION ALL SELECT 'discovery_candidates='||count(*) FROM discovery_candidates UNION ALL SELECT 'documents='||count(*) FROM documents UNION ALL SELECT 'extracted_facts='||count(*) FROM extracted_facts UNION ALL SELECT 'import_jobs='||count(*) FROM import_jobs UNION ALL SELECT 'knowledge_bases='||count(*) FROM knowledge_bases UNION ALL SELECT 'knowledge_insights='||count(*) FROM knowledge_insights UNION ALL SELECT 'literature_automations='||count(*) FROM literature_automations UNION ALL SELECT 'messages='||count(*) FROM messages UNION ALL SELECT 'task_controls='||count(*) FROM task_controls UNION ALL SELECT 'training_traces='||count(*) FROM training_traces ORDER BY 1;" \
  > "${PARTIAL_DIR}/database-counts.txt"

echo "[3/7] Stopping mutable data services for consistent volume archives..."
"${COMPOSE[@]}" stop -t 120 neo4j chroma minio >/dev/null

archive_volume() {
  local service="$1"
  local source_path="$2"
  local output_name="$3"
  local container_id
  container_id="$("${COMPOSE[@]}" ps -a -q "${service}")"
  if [[ -z "${container_id}" ]]; then
    echo "Missing container for service ${service}" >&2
    exit 5
  fi
  docker run --rm \
    --volumes-from "${container_id}" \
    --mount "type=bind,src=${PARTIAL_DIR},dst=/backup" \
    "${ARCHIVE_IMAGE}" \
    tar -C "${source_path}" -czf "/backup/${output_name}" .
}

echo "[4/7] Archiving Neo4j graph data..."
archive_volume neo4j /data neo4j-data.tar.gz

echo "[5/7] Archiving Chroma vectors and MinIO document objects..."
archive_volume chroma /chroma/chroma chroma.tar.gz
archive_volume minio /data minio.tar.gz

echo "[6/7] Saving deployment configuration and checksums..."
cp -- "${PROJECT_ROOT}/docker-compose.yml" "${PARTIAL_DIR}/"
cp -- "${PROJECT_ROOT}/docker-compose.server.yml" "${PARTIAL_DIR}/"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  cp -- "${PROJECT_ROOT}/.env" "${PARTIAL_DIR}/server.env"
  chmod 600 "${PARTIAL_DIR}/server.env"
fi
(
  cd "${PARTIAL_DIR}"
  sha256sum postgres.dump neo4j-data.tar.gz chroma.tar.gz minio.tar.gz \
    docker-compose.yml docker-compose.server.yml database-counts.txt > SHA256SUMS
)
printf 'created_utc=%s\nproject_root=%s\nredis_restore=worker_recovery_from_postgresql\n' \
  "${STAMP}" "${PROJECT_ROOT}" > "${PARTIAL_DIR}/MANIFEST.txt"
sync
mv -- "${PARTIAL_DIR}" "${FINAL_DIR}"

echo "[7/7] Finalizing service state and applying retention..."
if [[ "${RESTART_SERVICES}" == "true" ]]; then
  restore_services
else
  echo "Backup complete; API and workers remain stopped for migration cutover."
fi
trap - EXIT

if [[ "${RETENTION_DAYS}" =~ ^[0-9]+$ ]] && (( RETENTION_DAYS > 0 )); then
  find "${BACKUP_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name '20??-??-??T??????Z' -mtime "+${RETENTION_DAYS}" -exec rm -rf -- {} +
fi

echo "Backup complete: ${FINAL_DIR}"
