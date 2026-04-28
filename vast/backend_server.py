#!/usr/bin/env python3
"""
Backend HTTP server for Vast PyWorker: exposes POST /generate/sync, calls process_generation.

Owns ComfyUI (starts it at startup). Runs on port 8189. At startup: assert_cuda_ready,
start ComfyUI, truncate model_log_file and write "Backend ready" so PyWorker (LogActionConfig.on_load)
can detect readiness.

Fatal infra failures (ComfyUI unreachable/dead mid-job, CUDA init, bind failure) write
VAST_WORKER_FATAL_MARKER (default /tmp/worker-fatal), append "Error:" to model_log_file,
return 503 from /health while the process lasts, then os._exit(1) so the platform replaces
the worker. Normal request failures (bad workflow, missing input, timeout, S3 config)
return JSON error only and must NOT poison the worker log.
"""

import asyncio
import dataclasses
import json
import logging
import os
import pathlib
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Optional

import requests

COMFY_ROOT = os.getenv("COMFY_ROOT", "/app")
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

from comfy_backend import (  # noqa: E402
    WORKER_FATAL_MARKER,
    _force_worker_replacement,
    assert_cuda_ready,
    comfy,
    is_fatal_infra_error_message,
    is_fatal_worker_error,
    process_generation,
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("vast-backend")
for _name in ("botocore", "boto3", "s3transfer", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

MODEL_LOG_FILE = os.getenv("MODEL_LOG") or os.getenv(
    "VAST_MODEL_LOG_FILE", f"{COMFY_ROOT}/logs/backend.log"
)
BACKEND_PORT = int(os.getenv("VAST_BACKEND_PORT", "8189"))
MODELS_DIR = pathlib.Path(os.getenv("MODELS_DIR", f"{COMFY_ROOT}/models"))


@dataclasses.dataclass
class WorkerState:
    """Thread-safe mutable state for the single active generation slot."""

    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    active: bool = False
    prompt_id: Optional[str] = None
    job_id: Optional[str] = None
    started_at: Optional[float] = None
    last_progress_at: Optional[float] = None

    def begin(self, job_id: str) -> None:
        with self._lock:
            self.active = True
            self.job_id = job_id
            self.prompt_id = None
            now = time.time()
            self.started_at = now
            self.last_progress_at = now

    def set_prompt_id(self, prompt_id: str) -> None:
        with self._lock:
            self.prompt_id = prompt_id

    def touch_progress(self) -> None:
        with self._lock:
            self.last_progress_at = time.time()

    def end(self) -> None:
        with self._lock:
            self.active = False
            self.prompt_id = None
            self.job_id = None
            self.started_at = None
            self.last_progress_at = None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active": self.active,
                "prompt_id": self.prompt_id,
                "job_id": self.job_id,
                "started_at": self.started_at,
                "last_progress_at": self.last_progress_at,
            }


STATE = WorkerState()
_GENERATION_SEM = threading.BoundedSemaphore(1)


def _verify_readiness() -> None:
    """Verify model dirs and optional benchmark env before declaring ready.
    Raises if required dirs missing. Logs warning if benchmark S3 not configured.
    """
    required_dirs = ("diffusion_models", "loras", "text_encoders", "vae")
    for d in required_dirs:
        p = MODELS_DIR / d
        if not p.is_dir():
            raise RuntimeError(f"Required model directory not found: {p}")
    # Benchmark uses S3 input when BENCHMARK_IMAGE_BUCKET + BENCHMARK_IMAGE_KEY set
    bench_bucket = os.getenv("BENCHMARK_IMAGE_BUCKET") or os.getenv("S3_BUCKET")
    bench_key = (os.getenv("BENCHMARK_IMAGE_KEY") or "").strip()
    if not bench_bucket or not bench_key:
        logger.warning(
            "Benchmark S3 not configured (BENCHMARK_IMAGE_BUCKET/S3_BUCKET and "
            "BENCHMARK_IMAGE_KEY); benchmark may fail or use fallback"
        )


def write_ready_log() -> None:
    """Truncate model log file and write Backend ready so PyWorker on_load matches."""
    log_path = pathlib.Path(MODEL_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("Backend ready\n")
        f.flush()
    logger.info("Wrote readiness to %s", MODEL_LOG_FILE)


def append_fatal_error_log(message: str, exc: BaseException | None = None) -> None:
    """Append 'Error: ...' to model log for FATAL infra failures only.
    PyWorker on_error will trigger worker replacement. Use only for:
    - ComfyUI process died
    - CUDA init failure
    - Backend could not bind port
    - Required model dirs missing at startup
    """
    tb_lines = (
        "".join(
            traceback.format_exception(
                type(exc), exc, getattr(exc, "__traceback__", None)
            )
        )
        if exc
        else ""
    )
    err_blob = f"Error: {message}\n{tb_lines}"

    # Emit to stderr first (Vast captures this); ensures error is visible before reboot
    try:
        sys.stderr.write(
            "\n"
            + "=" * 60
            + "\nVAST_BACKEND_ERROR (process_generation failed):\n"
            + err_blob
            + "=" * 60
            + "\n"
        )
        sys.stderr.flush()
    except Exception:
        pass

    try:
        with open(MODEL_LOG_FILE, "a") as f:
            f.write(err_blob)
            f.flush()
    except Exception as e:
        logger.warning("Failed to write error to model log: %s", e)


async def handle_generate_sync(request):
    """POST /generate/sync: body = {"input": {...}}; call process_generation; return JSON or 500."""
    from aiohttp import web

    try:
        body = await request.read()
        if not body:
            return web.json_response(
                {"success": False, "error": "Empty body"},
                status=400,
            )
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        return web.json_response(
            {"success": False, "error": f"Invalid JSON: {e}"},
            status=400,
        )

    server_received_at = time.time()

    if "input" in data:
        payload = data
        job_id = (
            payload.get("id")
            or (payload.get("input") or {}).get("request_id")
            or uuid.uuid4().hex
        )
        if "id" not in payload:
            payload = {**payload, "id": job_id}
    else:
        job_id = data.get("request_id") or uuid.uuid4().hex
        payload = {"input": data, "id": job_id}

    payload["_timing"] = {"server_received_at": server_received_at}

    if not isinstance(payload.get("input"), dict):
        return web.json_response(
            {"success": False, "error": "input must be an object"},
            status=400,
        )

    acquired = _GENERATION_SEM.acquire(blocking=True, timeout=5.0)
    if not acquired:
        return web.json_response(
            {"success": False, "error": "Worker busy, generation slot occupied"},
            status=503,
        )
    try:
        result = await asyncio.to_thread(
            process_generation, payload, worker_state=STATE
        )
        if result.get("success") is True:
            return web.json_response(result, status=200)
        err = str(result.get("error") or "generation failed")
        if is_fatal_infra_error_message(err):
            logger.error("Request failure (fatal infra): %s", err)
            return web.json_response(result, status=503)
        # Non-fatal client errors (bad workflow, no output, validation failure) return HTTP 200.
        #
        # Returning 500 here causes two compounding retry loops:
        #   1. Vast.ai platform rescue: sees 500 from the worker proxy, re-queues the
        #      same frozen request payload indefinitely (poison loop).
        #   2. Vast SDK client (_retryable): treats any 5xx as retryable and retries
        #      at the SDK level with exponential backoff before giving up.
        #
        # The bot's VastGenerationClient._map_response already reads success:False from
        # the JSON body regardless of HTTP status, so 200 is fully transparent to the bot.
        # Fatal infra errors still use 503 to trigger worker replacement.
        logger.error("Request failure (not fatal): %s", err)
        return web.json_response(result, status=200)
    except Exception as e:
        logger.exception("process_generation raised: %s", e)
        msg = str(e)
        if is_fatal_worker_error(e):
            append_fatal_error_log(msg, e)
            _force_worker_replacement(msg, e)
        return web.json_response(
            {"success": False, "error": msg},
            status=503 if is_fatal_infra_error_message(msg) else 500,
        )
    finally:
        _GENERATION_SEM.release()


HEALTH_STUCK_TOTAL_SEC = float(os.getenv("VAST_HEALTH_STUCK_TOTAL_SEC", "900"))
HEALTH_STUCK_PROGRESS_SEC = float(os.getenv("VAST_HEALTH_STUCK_PROGRESS_SEC", "300"))
# Read timeout for the inner ComfyUI /system_stats probe. Was 2s, which
# routinely tripped during sampling and post-sampling memory cleanup —
# Vast saw the resulting 503 and recycled the worker mid-generation. 30s
# accommodates expected backpressure; the stuck-detection thresholds above
# remain the real escape hatch when a generation actually hangs.
HEALTH_COMFY_PROBE_TIMEOUT_SEC = float(
    os.getenv("VAST_HEALTH_COMFY_PROBE_TIMEOUT_SEC", "30")
)

# Tracks the last 503 reason emitted at ERROR level so repeat 10s probes don't
# spam the diagnostic log. Logs again whenever the reason string changes
# (including when health flips back to OK and then re-fails differently).
_last_503_log_reason: Optional[str] = None


def _signal_name(rc: int) -> str:
    """Map a negative subprocess returncode to its Unix signal name (best-effort)."""
    if rc >= 0:
        return ""
    sig = -rc
    names = {
        9: "SIGKILL (OOM-killer or kill -9)",
        11: "SIGSEGV (segmentation fault — likely C extension crash)",
        15: "SIGTERM (clean shutdown request)",
        6: "SIGABRT (abort)",
        7: "SIGBUS (bus error / mmap failure)",
        4: "SIGILL (illegal instruction)",
        1: "SIGHUP",
        2: "SIGINT",
    }
    return names.get(sig, f"signal {sig}")


def _log_503(reason: str, **details: Any) -> None:
    """Log a 503 reason at ERROR level once per distinct reason string."""
    global _last_503_log_reason
    rendered = reason if not details else f"{reason} ({', '.join(f'{k}={v}' for k,v in details.items())})"
    if rendered == _last_503_log_reason:
        return
    _last_503_log_reason = rendered
    logger.error("Health 503: %s", rendered)


async def handle_health(request):
    """GET /health: PyWorker healthcheck. 200 only when ComfyUI child alive, system_stats OK,
    and no stuck generation detected."""
    from aiohttp import web

    try:
        if WORKER_FATAL_MARKER.exists():
            _log_503("worker_fatal_marker")
            return web.json_response(
                {
                    "ok": False,
                    "ready": False,
                    "reason": "worker_fatal_marker",
                    "error": "worker_fatal_marker",
                },
                status=503,
            )
        proc = comfy.comfyui_process
        if proc is None:
            _log_503("comfy_not_started")
            return web.json_response(
                {"ok": False, "ready": False, "reason": "comfy_not_started"},
                status=503,
            )
        rc = proc.poll()
        if rc is not None:
            sig = _signal_name(rc)
            _log_503("comfy_dead", returncode=rc, signal=sig or "(clean exit)")
            return web.json_response(
                {
                    "ok": False,
                    "ready": False,
                    "reason": "comfy_dead",
                    "returncode": rc,
                    "signal": sig,
                },
                status=503,
            )
        if not getattr(comfy, "is_ready", False):
            _log_503("not_ready")
            return web.json_response(
                {"ok": False, "ready": False, "reason": "not_ready"},
                status=503,
            )
        # Probe ComfyUI. Under load (sampling, VAE encode/decode, post-sample
        # CPU offload, large-tensor finally blocks) /system_stats can take
        # tens of seconds to answer — that's expected backpressure, not a
        # death signal. We capture the probe outcome and decide what to do
        # with it AFTER consulting STATE: if a generation is in flight and
        # the stuck-detection thresholds aren't tripped, the worker is alive
        # and busy and we MUST return 200 so Vast doesn't recycle it.
        probe_failed = False
        probe_response: dict[str, Any] = {}
        try:
            r = requests.get(
                f"{comfy.comfyui_url}/system_stats",
                timeout=HEALTH_COMFY_PROBE_TIMEOUT_SEC,
            )
            if r.status_code != 200:
                probe_failed = True
                probe_response = {
                    "reason": "comfy_unhealthy",
                    "status": r.status_code,
                }
        except Exception as e:
            probe_failed = True
            probe_response = {
                "reason": "comfy_unreachable",
                "error": repr(e)[:200],
            }

        snap = STATE.snapshot()

        if probe_failed and not snap["active"]:
            # No in-flight generation to alibi the probe failure → real outage.
            log_kwargs = {k: v for k, v in probe_response.items() if k != "reason"}
            _log_503(probe_response["reason"], **log_kwargs)
            return web.json_response(
                {"ok": False, "ready": False, **probe_response},
                status=503,
            )

        # Stuck-detection — runs whether the probe succeeded or failed. When
        # it trips, _force_worker_replacement writes WORKER_FATAL_MARKER,
        # which the *next* /health probe sees and returns 503 for. That keeps
        # this handler's 503 surface narrow: only fatal-marker, dead process,
        # not-ready, or probe-failure-without-active-generation.
        if snap["active"]:
            now = time.time()
            elapsed = now - (snap["started_at"] or now)
            since_progress = now - (snap["last_progress_at"] or now)

            if elapsed > HEALTH_STUCK_TOTAL_SEC:
                reason = (
                    f"generation stuck: total {elapsed:.0f}s > "
                    f"{HEALTH_STUCK_TOTAL_SEC:.0f}s limit"
                )
                logger.error("Health check failing: %s", reason)
                _force_worker_replacement(reason)

            if since_progress > HEALTH_STUCK_PROGRESS_SEC:
                reason = (
                    f"generation stuck: no history progress for {since_progress:.0f}s > "
                    f"{HEALTH_STUCK_PROGRESS_SEC:.0f}s limit"
                )
                logger.error("Health check failing: %s", reason)
                _force_worker_replacement(reason)

        # Clear the 503 log-suppression so the next failure re-logs even if
        # it has the same reason as a previous one (worker died, recovered,
        # died again is more interesting than silent re-failure).
        global _last_503_log_reason
        _last_503_log_reason = None
        body: dict[str, Any] = {"ok": True, "ready": True, "state": snap}
        if probe_failed:
            # Annotate but don't surface as failure: Vast sees 200, we see
            # in our diagnostic log that the probe was busy. INFO-level
            # because this is the EXPECTED path when sampling/cleanup is
            # spiking; not worth ERROR or even WARN.
            body["probe"] = "busy"
            body["probe_detail"] = probe_response
            logger.info(
                "Health probe absorbed (active job, within stuck thresholds): %s",
                probe_response,
            )
        return web.json_response(body, status=200)
    except Exception as e:
        logger.exception("Health check internal error")
        return web.json_response({"ok": False, "error": repr(e)}, status=500)


def create_app():
    from aiohttp import web

    # Base64-encoded images in workflow_json can exceed 1MB default; allow up to 50MB
    client_max_size = int(
        os.getenv("VAST_BACKEND_CLIENT_MAX_SIZE", str(50 * 1024 * 1024))
    )
    app = web.Application(client_max_size=client_max_size)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/generate/sync", handle_generate_sync)
    return app


async def _run_server():
    """Start aiohttp server, write readiness only after socket is bound."""
    from aiohttp import web

    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", BACKEND_PORT)
    await site.start()
    logger.info("Backend server listening on 0.0.0.0:%s", BACKEND_PORT)
    write_ready_log()
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def _start_comfyui_watchdog() -> None:
    interval = float(os.getenv("VAST_COMFY_WATCHDOG_SEC", "5"))

    def loop() -> None:
        while True:
            time.sleep(interval)
            proc = comfy.comfyui_process
            if proc is not None and proc.poll() is not None:
                _force_worker_replacement(
                    f"watchdog: ComfyUI exited rc={proc.returncode}"
                )

    threading.Thread(target=loop, daemon=True, name="comfyui-watchdog").start()


def _start_busy_watchdog() -> None:
    """Kill worker if a generation is stuck beyond thresholds, independent of health checks."""
    interval = float(os.getenv("VAST_BUSY_WATCHDOG_SEC", "30"))
    max_total = float(os.getenv("VAST_BUSY_WATCHDOG_TOTAL_SEC", "900"))
    max_no_progress = float(os.getenv("VAST_BUSY_WATCHDOG_PROGRESS_SEC", "300"))

    def loop() -> None:
        while True:
            time.sleep(interval)
            snap = STATE.snapshot()
            if not snap["active"]:
                continue
            now = time.time()
            elapsed = now - (snap["started_at"] or now)
            since_progress = now - (snap["last_progress_at"] or now)

            if elapsed > max_total:
                _force_worker_replacement(
                    f"busy-watchdog: generation total {elapsed:.0f}s > {max_total:.0f}s"
                )
            if since_progress > max_no_progress:
                _force_worker_replacement(
                    f"busy-watchdog: no progress for {since_progress:.0f}s > {max_no_progress:.0f}s"
                )

    threading.Thread(target=loop, daemon=True, name="busy-watchdog").start()


def run_backend_server():
    """Assert CUDA, start ComfyUI, verify readiness, then run aiohttp server. Writes readiness after socket bind."""
    assert_cuda_ready()
    comfy.start_comfyui()
    _verify_readiness()
    _start_comfyui_watchdog()
    _start_busy_watchdog()

    asyncio.run(_run_server())


if __name__ == "__main__":
    run_backend_server()
