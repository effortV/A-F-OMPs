#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MARKER="# NF_ATLAS_DAILY_BACKUP"
printf -v BACKUP_SCRIPT_Q '%q' "${PROJECT_ROOT}/scripts/server_backup.sh"
printf -v BACKUP_LOG_Q '%q' "${PROJECT_ROOT}/backups/backup.log"
BACKUP_COMMAND="20 3 * * * ${BACKUP_SCRIPT_Q} >> ${BACKUP_LOG_Q} 2>&1 ${MARKER}"
CURRENT="$(crontab -l 2>/dev/null || true)"
FILTERED="$(
  printf '%s\n' "${CURRENT}" \
    | grep -v -F "${MARKER}" \
    | grep -v -F "${PROJECT_ROOT}/scripts/server_backup.sh" \
    || true
)"
{
  printf '%s\n' "${FILTERED}"
  printf '%s\n' "${BACKUP_COMMAND}"
} | sed '/^[[:space:]]*$/d' | crontab -
echo "Installed daily NF-Atlas backup at 03:20 for user $(id -un)."
