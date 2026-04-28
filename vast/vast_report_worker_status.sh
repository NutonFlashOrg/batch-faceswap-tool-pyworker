# Shared Vast serverless worker_status reporting.
# Source from /app/vast/start_vast.sh and /app/vast/start_server.sh.
# Keep error_msg fixed/literal — interpolating arbitrary log text into JSON is unsafe without escaping.

vast_report_worker_status_error() {
  local error_msg="$1"
  local report_addr="${REPORT_ADDR:-https://run.vast.ai}"
  local mtoken="${MASTER_TOKEN:-}"
  local version="${PYWORKER_VERSION:-0}"

  IFS=',' read -r -a REPORT_ADDRS <<< "${report_addr}"
  for addr in "${REPORT_ADDRS[@]}"; do
    curl -sS --connect-timeout 3 --max-time 10 \
      -X POST -H 'Content-Type: application/json' \
      -d "$(cat <<JSON
{
  "id": ${CONTAINER_ID:-0},
  "mtoken": "${mtoken}",
  "version": "${version}",
  "error_msg": "${error_msg}",
  "url": "${URL:-}"
}
JSON
)" "${addr%/}/worker_status/" || true
  done
  return 0
}

vast_report_worker_status_error_and_exit() {
  local error_msg="$1"
  echo "ERROR: $error_msg"
  vast_report_worker_status_error "$error_msg"
  exit 1
}
