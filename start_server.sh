#!/bin/bash

set -e -o pipefail

# Check for force update flag
FORCE_UPDATE=false
if [ -f "/.force_update" ]; then
    echo "Force update flag detected at /.force_update"
    FORCE_UPDATE=true
fi

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

SERVER_DIR="$WORKSPACE_DIR/vast-pyworker"
ENV_PATH="${ENV_PATH:-$WORKSPACE_DIR/worker-env}"
DEBUG_LOG="$WORKSPACE_DIR/debug.log"
PYWORKER_LOG="$WORKSPACE_DIR/pyworker.log"

REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"
USE_SSL="${USE_SSL:-true}"
WORKER_PORT="${WORKER_PORT:-3000}"
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

exec &> >(tee -a "$DEBUG_LOG")

function echo_var(){
    echo "$1: ${!1}"
}

function report_error_and_exit(){
    local error_msg="$1"
    echo "ERROR: $error_msg"

    MTOKEN="${MASTER_TOKEN:-}"
    VERSION="${PYWORKER_VERSION:-0}"

    IFS=',' read -r -a REPORT_ADDRS <<< "${REPORT_ADDR}"
    for addr in "${REPORT_ADDRS[@]}"; do
        curl -sS -X POST -H 'Content-Type: application/json' \
            -d "$(cat <<JSON
{
  "id": ${CONTAINER_ID:-0},
  "mtoken": "${MTOKEN}",
  "version": "${VERSION}",
  "error_msg": "${error_msg}",
  "url": "${URL:-}"
}
JSON
)" "${addr%/}/worker_status/" || true
    done

    exit 1
}

function install_vastai_sdk() {
    local uv_flags=()
    if [ "${USE_SYSTEM_PYTHON:-}" = "true" ]; then
        uv_flags+=(--system --break-system-packages)
    fi
    if [ "$FORCE_UPDATE" = true ]; then
        uv_flags+=(--force-reinstall)
        echo "Force reinstalling vastai-sdk"
    fi

    # If SDK_BRANCH is set, install vastai-sdk from the vast-sdk repo at that branch/tag/commit.
    if [ -n "${SDK_BRANCH:-}" ]; then
        if [ -n "${SDK_VERSION:-}" ]; then
            echo "WARNING: Both SDK_BRANCH and SDK_VERSION are set; using SDK_BRANCH=${SDK_BRANCH}"
        fi
        echo "Installing vastai-sdk from https://github.com/vast-ai/vast-sdk/ @ ${SDK_BRANCH}"
        if ! uv pip install "${uv_flags[@]}" "vastai-sdk @ git+https://github.com/vast-ai/vast-sdk.git@${SDK_BRANCH}"; then
            report_error_and_exit "Failed to install vastai-sdk from vast-ai/vast-sdk@${SDK_BRANCH}"
        fi
        return 0
    fi

    if [ -n "${SDK_VERSION:-}" ]; then
        echo "Installing vastai-sdk version ${SDK_VERSION}"
        if ! uv pip install "${uv_flags[@]}" "vastai-sdk==${SDK_VERSION}"; then
            report_error_and_exit "Failed to install vastai-sdk==${SDK_VERSION}"
        fi
        return 0
    fi

    echo "Installing default vastai-sdk"
    if ! uv pip install "${uv_flags[@]}" vastai-sdk; then
        report_error_and_exit "Failed to install vastai-sdk"
    fi
}

[ -n "$BACKEND" ] && [ -z "$HF_TOKEN" ] && report_error_and_exit "HF_TOKEN must be set when BACKEND is set!"
[ -z "$CONTAINER_ID" ] && report_error_and_exit "CONTAINER_ID must be set!"
[ "$BACKEND" = "comfyui" ] && [ -z "$COMFY_MODEL" ] && report_error_and_exit "For comfyui backends, COMFY_MODEL must be set!"

echo "start_server.sh"
date

echo_var BACKEND
echo_var REPORT_ADDR
echo_var WORKER_PORT
echo_var WORKSPACE_DIR
echo_var SERVER_DIR
echo_var ENV_PATH
echo_var DEBUG_LOG
echo_var PYWORKER_LOG
echo_var MODEL_LOG

ROTATE_MODEL_LOG="${ROTATE_MODEL_LOG:-false}"
if [ "$ROTATE_MODEL_LOG" = "true" ] && [ -e "$MODEL_LOG" ]; then
    echo "Rotating model log at $MODEL_LOG to $MODEL_LOG.old"
    if ! cat "$MODEL_LOG" >> "$MODEL_LOG.old"; then
        report_error_and_exit "Failed to rotate model log"
    fi
    if ! : > "$MODEL_LOG"; then
        report_error_and_exit "Failed to truncate model log"
    fi
fi

# Populate /etc/environment with quoted values
if ! grep -q "VAST" /etc/environment; then
    if ! env -0 | grep -zEv "^(HOME=|SHLVL=)|CONDA" | while IFS= read -r -d '' line; do
            name=${line%%=*}
            value=${line#*=}
            printf '%s="%s"\n' "$name" "$value"
        done > /etc/environment; then
        echo "WARNING: Failed to populate /etc/environment, continuing anyway"
    fi
fi

if [ "${USE_SYSTEM_PYTHON:-}" = "true" ]; then
    echo "Using system Python: $(which python3)"
    if ! which uv > /dev/null 2>&1; then
        if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
            report_error_and_exit "Failed to install uv package manager"
        fi
        if [[ -f ~/.local/bin/env ]]; then
            if ! source ~/.local/bin/env; then
                report_error_and_exit "Failed to source uv environment"
            fi
        fi
    fi
    install_vastai_sdk
    touch ~/.no_auto_tmux
elif [ ! -d "$ENV_PATH" ]; then
    echo "setting up venv"
    if ! which uv; then
        if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
            report_error_and_exit "Failed to install uv package manager"
        fi
        if [[ -f ~/.local/bin/env ]]; then
            if ! source ~/.local/bin/env; then
                report_error_and_exit "Failed to source uv environment"
            fi
        else
            echo "WARNING: ~/.local/bin/env not found after uv installation"
        fi
    fi

    if [[ ! -d $SERVER_DIR ]]; then
        if ! git clone "${PYWORKER_REPO:-https://github.com/vast-ai/pyworker}" "$SERVER_DIR"; then
            report_error_and_exit "Failed to clone pyworker repository"
        fi
    elif [ "$FORCE_UPDATE" = true ]; then
        echo "Force updating pyworker repository"
        if ! (cd "$SERVER_DIR" && git fetch --all); then
            report_error_and_exit "Failed to fetch pyworker repository updates"
        fi
    fi
    if [[ -n ${PYWORKER_REF:-} ]]; then
        if [ "$FORCE_UPDATE" = true ]; then
            echo "Force updating to pyworker reference: $PYWORKER_REF"
            if ! (cd "$SERVER_DIR" && git checkout "$PYWORKER_REF" && git pull); then
                report_error_and_exit "Failed to force update pyworker reference: $PYWORKER_REF"
            fi
        else
            if ! (cd "$SERVER_DIR" && git checkout "$PYWORKER_REF"); then
                report_error_and_exit "Failed to checkout pyworker reference: $PYWORKER_REF"
            fi
        fi
    elif [ "$FORCE_UPDATE" = true ]; then
        echo "Force updating pyworker to latest"
        if ! (cd "$SERVER_DIR" && git pull); then
            report_error_and_exit "Failed to pull latest pyworker changes"
        fi
    fi

    if ! uv venv --python-preference only-managed "$ENV_PATH" -p 3.10; then
        report_error_and_exit "Failed to create virtual environment"
    fi
    
    if ! source "$ENV_PATH/bin/activate"; then
        report_error_and_exit "Failed to activate virtual environment"
    fi

    if ! uv pip install -r "${SERVER_DIR}/requirements.txt"; then
        report_error_and_exit "Failed to install Python requirements"
    fi

    install_vastai_sdk

    if ! touch ~/.no_auto_tmux; then
        report_error_and_exit "Failed to create ~/.no_auto_tmux"
    fi
else
    if [[ -f ~/.local/bin/env ]]; then
        if ! source ~/.local/bin/env; then
            report_error_and_exit "Failed to source uv environment"
        fi
    fi
    if ! source "$ENV_PATH/bin/activate"; then
        report_error_and_exit "Failed to activate existing virtual environment"
    fi
    echo "environment activated"
    echo "venv: $VIRTUAL_ENV"

    # Handle force update for existing environment
    if [ "$FORCE_UPDATE" = true ]; then
        echo "Performing force update on existing environment"

        if [[ -d $SERVER_DIR ]]; then
            echo "Force updating pyworker repository"
            if ! (cd "$SERVER_DIR" && git fetch --all); then
                report_error_and_exit "Failed to fetch pyworker repository updates"
            fi

            if [[ -n ${PYWORKER_REF:-} ]]; then
                echo "Force updating to pyworker reference: $PYWORKER_REF"
                if ! (cd "$SERVER_DIR" && git checkout "$PYWORKER_REF" && git pull); then
                    report_error_and_exit "Failed to force update pyworker reference: $PYWORKER_REF"
                fi
            else
                echo "Force updating pyworker to latest"
                if ! (cd "$SERVER_DIR" && git pull); then
                    report_error_and_exit "Failed to pull latest pyworker changes"
                fi
            fi
        fi

        install_vastai_sdk
    fi
fi

# Remove force update flag after successful update
if [ "$FORCE_UPDATE" = true ]; then
    echo "Removing force update flag"
    rm -f "/.force_update"
    echo "Force update completed successfully"
fi

if [ "$USE_SSL" = true ]; then

    if ! cat << EOF > /etc/openssl-san.cnf
    [req]
    default_bits       = 2048
    distinguished_name = req_distinguished_name
    req_extensions     = v3_req

    [req_distinguished_name]
    countryName         = US
    stateOrProvinceName = CA
    organizationName    = Vast.ai Inc.
    commonName          = vast.ai

    [v3_req]
    basicConstraints = CA:FALSE
    keyUsage         = nonRepudiation, digitalSignature, keyEncipherment
    subjectAltName   = @alt_names

    [alt_names]
    IP.1   = 0.0.0.0
EOF
    then
        report_error_and_exit "Failed to write OpenSSL config"
    fi

    if ! openssl req -newkey rsa:2048 -subj "/C=US/ST=CA/CN=pyworker.vast.ai/" \
        -nodes \
        -sha256 \
        -keyout /etc/instance.key \
        -out /etc/instance.csr \
        -config /etc/openssl-san.cnf; then
        report_error_and_exit "Failed to generate SSL certificate request"
    fi

    max_retries=5
    retry_delay=2
    for attempt in $(seq 1 "$max_retries"); do
        http_code=$(curl -sS -o /etc/instance.crt -w '%{http_code}' \
            --header 'Content-Type: application/octet-stream' \
            --data-binary @/etc/instance.csr \
            -X POST "https://console.vast.ai/api/v0/sign_cert/?instance_id=$CONTAINER_ID")
        if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
            break
        fi
        echo "SSL cert signing attempt $attempt/$max_retries failed (HTTP $http_code)"
        if [ "$attempt" -eq "$max_retries" ]; then
            report_error_and_exit "Failed to sign SSL certificate after $max_retries attempts (HTTP $http_code)"
        fi
        sleep "$retry_delay"
        retry_delay=$((retry_delay * 2))
    done
fi

export REPORT_ADDR WORKER_PORT USE_SSL UNSECURED

# ─── SDK Deployment Mode ───────────────────────────────────────────────
if [ "$IS_DEPLOYMENT" = "true" ]; then
    echo "=== SDK Deployment Mode ==="
    echo "DEPLOYMENT_ID: $DEPLOYMENT_ID"

    DEPLOY_DIR="/workspace/deployment"
    mkdir -p "$DEPLOY_DIR"

    VAST_API_BASE="${VAST_API_BASE:-https://console.vast.ai}"

    # Download deployment code, retrying until the blob is available on S3.
    # The s3_key exists in the DB as soon as the deployment is created, but the
    # actual upload may still be in flight from the client side.

    # Install SDK (uses the install_vastai_sdk function which supports SDK_BRANCH/SDK_VERSION)
    install_vastai_sdk
    # Run deployment in serve mode
    export VAST_DEPLOYMENT_MODE=serve
    echo "Starting deployment: python3 $DEPLOY_DIR/deployment.py"
    serve-vast-deployment
    exit $?
fi
# ─── End SDK Deployment Mode ───────────────────────────────────────────

if ! cd "$SERVER_DIR"; then
    report_error_and_exit "Failed to cd into SERVER_DIR: $SERVER_DIR"
fi

# Optional: light benchmark (CALIBRATION_RUNS) + matching prod JSON from manifest (CALIBRATION_PROD_RUNS).
# Template BENCHMARK_GENERATION_LANE selects a single manifest row; set CALIBRATION_ALL_MANIFEST_LANES=1 to time every manifest row.
# Requires PYWORKER_REPO with scripts/calibrate_vast_workload_multi_lane.py or calibrate_workload_timing.py.
# See comfy-vast-serverless/docs/VAST_BENCHMARK_LANES_AND_BOT_COST.md
if [ "${RUN_WORKLOAD_CALIBRATION:-}" = "1" ] || [ "${RUN_WORKLOAD_CALIBRATION:-}" = "true" ]; then
    CAL_URL="${CALIBRATION_BACKEND_URL:-http://127.0.0.1:8189/generate/sync}"
    CAL_RUNS="${CALIBRATION_RUNS:-30}"
    CAL_PROD_RUNS="${CALIBRATION_PROD_RUNS:-5}"
    CAL_WARM="${CALIBRATION_WARMUP:-1}"
    CAL_PROD_WARM="${CALIBRATION_PROD_WARMUP:-0}"
    CAL_BASE="${CALIBRATION_BASELINE:-100}"

    if [ -n "${CALIBRATION_MANIFEST:-}" ]; then
        if [ ! -f "${CALIBRATION_MANIFEST}" ]; then
            report_error_and_exit "RUN_WORKLOAD_CALIBRATION: CALIBRATION_MANIFEST is set but not a file: ${CALIBRATION_MANIFEST}"
        fi
        ML_SCRIPT="${SERVER_DIR}/scripts/calibrate_vast_workload_multi_lane.py"
        if [ ! -f "$ML_SCRIPT" ]; then
            report_error_and_exit "RUN_WORKLOAD_CALIBRATION: missing ${ML_SCRIPT} (set PYWORKER_REF to a repo that includes scripts/)"
        fi
        echo "[calibration] RUN_WORKLOAD_CALIBRATION=1 manifest=${CALIBRATION_MANIFEST} bench_runs=${CAL_RUNS} prod_runs=${CAL_PROD_RUNS}"
        ML_ARGS=(
            python3 "$ML_SCRIPT"
            --manifest "${CALIBRATION_MANIFEST}"
            --backend-url "$CAL_URL"
            --bench-runs "$CAL_RUNS"
            --warmup "$CAL_WARM"
            --prod-runs "$CAL_PROD_RUNS"
            --prod-warmup "$CAL_PROD_WARM"
            --baseline "$CAL_BASE"
        )
        if [ "${CALIBRATION_ALL_MANIFEST_LANES:-}" = "1" ] || [ "${CALIBRATION_ALL_MANIFEST_LANES:-}" = "true" ]; then
            ML_ARGS+=(--all-manifest-lanes)
        fi
        set +e
        "${ML_ARGS[@]}" 2>&1 | tee -a "$PYWORKER_LOG"
        CAL_EC=${PIPESTATUS[0]}
        set -e
        if [ "$CAL_EC" -ne 0 ]; then
            report_error_and_exit "RUN_WORKLOAD_CALIBRATION: calibrate_vast_workload_multi_lane.py failed (exit ${CAL_EC})"
        fi
        echo "[calibration] finished OK (manifest)"
    else
        CAL_SCRIPT="${SERVER_DIR}/scripts/calibrate_workload_timing.py"
        if [ -f "$CAL_SCRIPT" ]; then
            echo "[calibration] RUN_WORKLOAD_CALIBRATION=1 ${CAL_SCRIPT} (set CALIBRATION_MANIFEST for prod JSON manifest mode)"
            CAL_ARGS=(python3 "$CAL_SCRIPT" --backend-url "$CAL_URL" --runs "$CAL_RUNS" --warmup "$CAL_WARM" --baseline "$CAL_BASE")
            if [ -n "${CALIBRATION_PROD_PAYLOAD:-}" ] && [ -f "${CALIBRATION_PROD_PAYLOAD}" ]; then
                CAL_ARGS+=(--prod-payload "${CALIBRATION_PROD_PAYLOAD}")
            fi
            if [ -n "${CALIBRATION_PROD_P50_SECONDS:-}" ]; then
                CAL_ARGS+=(--prod-p50-seconds "${CALIBRATION_PROD_P50_SECONDS}")
            fi
            set +e
            "${CAL_ARGS[@]}" 2>&1 | tee -a "$PYWORKER_LOG"
            CAL_EC=${PIPESTATUS[0]}
            set -e
            if [ "$CAL_EC" -eq 0 ]; then
                echo "[calibration] finished OK"
            else
                echo "[calibration] WARNING: script exited ${CAL_EC}; continuing to PyWorker"
            fi
        else
            echo "[calibration] SKIP: $CAL_SCRIPT missing (set PYWORKER_REF to a repo that includes scripts/)"
        fi
    fi
fi

echo "launching PyWorker server"

set +e

PY_STATUS=1

if [ -f "$SERVER_DIR/worker.py" ]; then
    echo "trying worker.py"
    python3 -m "worker" |& tee -a "$PYWORKER_LOG"
    PY_STATUS=${PIPESTATUS[0]}
fi

if [ "${PY_STATUS}" -ne 0 ] && [ -f "$SERVER_DIR/workers/$BACKEND/worker.py" ]; then
    echo "trying workers.${BACKEND}.worker"
    python3 -m "workers.${BACKEND}.worker" |& tee -a "$PYWORKER_LOG"
    PY_STATUS=${PIPESTATUS[0]}
fi

if [ "${PY_STATUS}" -ne 0 ] && [ -f "$SERVER_DIR/workers/$BACKEND/server.py" ]; then
    echo "trying workers.${BACKEND}.server"
    python3 -m "workers.${BACKEND}.server" |& tee -a "$PYWORKER_LOG"
    PY_STATUS=${PIPESTATUS[0]}
fi

set -e

if [ "${PY_STATUS}" -ne 0 ]; then
    if [ ! -f "$SERVER_DIR/worker.py" ] && [ ! -f "$SERVER_DIR/workers/$BACKEND/worker.py" ] && [ ! -f "$SERVER_DIR/workers/$BACKEND/server.py" ]; then
        report_error_and_exit "Failed to find PyWorker"
    fi
    report_error_and_exit "PyWorker exited with status ${PY_STATUS}"
fi

echo "launching PyWorker server done"
