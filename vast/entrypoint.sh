#!/bin/bash
set -euo pipefail

# Backend bootstrapper: GPU preflight, model bootstrap, provisioning gate, backend start.
# Does NOT run PyWorker; Vast's start_server.sh owns that.
# Writes readiness to MODEL_LOG so PyWorker (from start_server.sh) can detect it.

# Source Vast/prepared runtime env (matches official template; entrypoint may start before start_server.sh)
set -a
[ -f /etc/environment ] && . /etc/environment || true
[ -f /workspace/.env ] && . /workspace/.env || true
set +a

export MODEL_LOG="${MODEL_LOG:-/app/logs/backend.log}"
COMFY_ROOT="${COMFY_ROOT:-/app}"
export MODELS_DIR="${MODELS_DIR:-${COMFY_ROOT}/models}"
mkdir -p /app/logs "$(dirname "$MODEL_LOG")"

cleanup() {
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[entrypoint] bootstrap failed rc=$rc"
    tail -100 "$MODEL_LOG" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Provisioning gate: remove only after backend is healthy
touch /.provisioning

echo "[entrypoint] Starting backend bootstrapper..."

# --- GPU preflight ---
echo "[gpu-preflight] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[gpu-preflight] NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"

for i in 1 2 3 4 5; do
  echo "[gpu-preflight] attempt $i"

  if nvidia-smi -L && python3 - <<'PY'
import os, sys
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES"))
print("NVIDIA_VISIBLE_DEVICES", os.getenv("NVIDIA_VISIBLE_DEVICES"))
print("is_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.device_count() < 1:
    sys.exit(1)
print("capability", torch.cuda.get_device_capability(0))
print("device0", torch.cuda.get_device_name(0))
x = torch.tensor([1.0], device="cuda")
print("cuda_ok", x.item())
PY
  then
    echo "[gpu-preflight] success"
    break
  fi

  if [ "$i" -eq 5 ]; then
    echo "[gpu-preflight] failed permanently; exiting so worker gets replaced"
    exit 1
  fi

  sleep 3
done

# --- Validate baked models exist (no runtime HF download) ---
MODELS_DIR="${MODELS_DIR:-/app/models}"
export MODELS_DIR

echo "[entrypoint] Validating baked models..."
for d in diffusion_models loras text_encoders vae; do
  if [ ! -d "${MODELS_DIR}/${d}" ] || [ -z "$(ls -A "${MODELS_DIR}/${d}" 2>/dev/null)" ]; then
    echo "[entrypoint] ERROR: Required model dir ${d} missing or empty"
    exit 1
  fi
done
echo "[entrypoint] Model validation passed (baked)"

# --- Env presence check (no secrets printed) ---
echo "[entrypoint] env presence check:"
python3 - <<'PY'
import os
keys = [
    "S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_REGION", "S3_BUCKET",
    "BENCHMARK_IMAGE_BUCKET", "BENCHMARK_IMAGE_KEY",
    "BENCHMARK_VIDEO_BUCKET", "BENCHMARK_VIDEO_KEY",
]
for k in keys:
    v = os.getenv(k)
    if v:
        if "KEY" in k or "TOKEN" in k or "SECRET" in k:
            print(f"  {k}=SET(len={len(v)})")
        else:
            print(f"  {k}={v}")
    else:
        print(f"  {k}=<missing>")
PY

export PYTHONPATH="${COMFY_ROOT}"
export COMFYUI_URL="${COMFYUI_URL:-http://127.0.0.1:8188}"

# --- Start backend server (owns ComfyUI, writes "Backend ready" to MODEL_LOG) ---
echo "[entrypoint] Starting backend server -> $MODEL_LOG"
python3 "${COMFY_ROOT}/vast/backend_server.py" &
BACKEND_PID=$!
echo "[entrypoint] Backend PID=$BACKEND_PID"

# Wait for HTTP readiness on /health (200 when ComfyUI ready); log line alone is not sufficient
echo "[entrypoint] Waiting for backend HTTP readiness..."
MAX_WAIT=300
ELAPSED=0
until curl -fsS "http://127.0.0.1:8189/health" >/dev/null 2>&1; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "[entrypoint] Backend died during startup"
    tail -100 "$MODEL_LOG" 2>/dev/null || true
    exit 1
  fi
  sleep 2
  ELAPSED=$((ELAPSED + 2))
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "[entrypoint] Timeout waiting for backend HTTP readiness after ${MAX_WAIT}s"
    tail -100 "$MODEL_LOG" 2>/dev/null || true
    exit 1
  fi
done

rm -f /.provisioning
echo "[entrypoint] Backend HTTP ready. Provisioning complete."
