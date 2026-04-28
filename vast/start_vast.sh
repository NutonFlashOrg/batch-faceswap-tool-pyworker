#!/bin/bash
set -euo pipefail

# shellcheck source=vast_report_worker_status.sh
source /app/vast/vast_report_worker_status.sh

export JOB_PREFIX="${JOB_PREFIX:-vast}"
export SERVERLESS=true
export BACKEND=comfyui-json
export COMFYUI_API_BASE="http://127.0.0.1:8189"
export MODEL_LOG="${MODEL_LOG:-/app/logs/backend.log}"
export PYTHONPATH="${COMFY_ROOT:-/app}"
export COMFYUI_URL="${COMFYUI_URL:-http://127.0.0.1:8188}"

# Path layout (default /app; override if base uses different layout)
export COMFY_ROOT="${COMFY_ROOT:-/app}"
export COMFY_OUTPUT_DIR="${COMFY_OUTPUT_DIR:-${COMFY_ROOT}/output}"
export COMFY_INPUT_DIR="${COMFY_INPUT_DIR:-${COMFY_ROOT}/input}"
export MODELS_DIR="${MODELS_DIR:-${COMFY_ROOT}/models}"

# Stuck-worker detection thresholds
export VAST_HEALTH_STUCK_TOTAL_SEC="${VAST_HEALTH_STUCK_TOTAL_SEC:-900}"
export VAST_HEALTH_STUCK_PROGRESS_SEC="${VAST_HEALTH_STUCK_PROGRESS_SEC:-300}"
export VAST_BUSY_WATCHDOG_SEC="${VAST_BUSY_WATCHDOG_SEC:-30}"
export VAST_BUSY_WATCHDOG_TOTAL_SEC="${VAST_BUSY_WATCHDOG_TOTAL_SEC:-900}"
export VAST_BUSY_WATCHDOG_PROGRESS_SEC="${VAST_BUSY_WATCHDOG_PROGRESS_SEC:-300}"

# Port fallback if Vast misses injection
export WORKER_PORT="${WORKER_PORT:-3000}"

mkdir -p "$(dirname "${MODEL_LOG}")" /app/logs
exec > >(tee -a /app/logs/combined.log) 2>&1

# Serverless mode: strip ComfyUI-Manager so it doesn't register nodes,
# scan the custom_nodes dir, or attempt dependency pulls at ComfyUI startup.
# SSH/dev templates bypass this entrypoint via their own onstart script and
# keep Manager active for interactive workflow development.
if [ -d /app/custom_nodes/ComfyUI-Manager ] || [ -d /app/custom_nodes/comfyui-manager ]; then
  echo "[vast] Serverless boot: removing ComfyUI-Manager for zero node-registry overhead"
  rm -rf /app/custom_nodes/ComfyUI-Manager /app/custom_nodes/comfyui-manager
fi

cleanup() {
  echo "[vast] === Worker exited. Last 80 lines of backend.log ==="
  tail -80 /app/logs/backend.log 2>/dev/null || true
  echo "[vast] === End backend.log dump ==="
}
trap cleanup EXIT

echo "[vast] Running backend bootstrapper sequentially..."
if /app/vast/entrypoint.sh; then
  :
else
  rc=$?
  msg="backend bootstrap failed before PyWorker startup (exit ${rc})"
  echo "[vast] ${msg}"
  vast_report_worker_status_error "${msg}"
  exit 1
fi

echo "[vast] Backend bootstrap complete. Starting pinned start_server.sh..."
exec /app/vast/start_server.sh
