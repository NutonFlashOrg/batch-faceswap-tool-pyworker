#!/usr/bin/env python3
"""
Shared ComfyUI backend: S3, ComfyUI manager, generation orchestration.

Used by vast/backend_server.py. PyWorker transforms requests and forwards
workflow_json (pre-patched with base64 images). Backend runs, uploads to S3.
"""

import json
import logging
import mimetypes
import os
import pathlib
import re
import shlex
import sys
import traceback
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import boto3
import requests

import s3_boto_resilience

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("comfy-backend")

_COMFY_ROOT = os.getenv("COMFY_ROOT", "/app")
WORKER_FATAL_MARKER = pathlib.Path(
    os.getenv("VAST_WORKER_FATAL_MARKER", "/tmp/worker-fatal")
)

# Tolerance for the GET /history poll loop in wait_for_completion. ComfyUI's
# main loop blocks under cleanup nodes (easy cleanGpuUsed, VRAMCleanup) while
# unload_all_models() moves a 14B-param transformer GPU->CPU and gc.collect()
# walks a heap full of large tensor refs — both hold the GIL, so /history is
# unresponsive for tens of seconds. Defaults: per-call timeout 30s,
# 6 consecutive failures = 180s tolerance window before declaring the worker
# dead. Long enough to absorb any realistic chained-workflow cleanup pause,
# short enough that a real ComfyUI crash still trips well inside the
# workflow_timeout (typically 7200s).
COMFYUI_HISTORY_POLL_TIMEOUT_SEC = float(
    os.getenv("COMFYUI_HISTORY_POLL_TIMEOUT_SEC", "30")
)
COMFYUI_HISTORY_POLL_MAX_CONSECUTIVE_ERRORS = int(
    os.getenv("COMFYUI_HISTORY_POLL_MAX_CONSECUTIVE_ERRORS", "6")
)


def _comfy_input_directory() -> pathlib.Path:
    """ComfyUI input tree (staged S3 audio, etc.); align with pyworker ``COMFY_INPUT_*`` env."""
    raw = (os.getenv("COMFY_INPUT_ROOT") or os.getenv("COMFY_INPUT_DIR") or "").strip()
    if raw:
        return pathlib.Path(raw).resolve()
    return (pathlib.Path(_COMFY_ROOT) / "input").resolve()


def _model_log_file_path() -> pathlib.Path:
    return pathlib.Path(
        os.getenv("MODEL_LOG")
        or os.getenv("VAST_MODEL_LOG_FILE", f"{_COMFY_ROOT}/logs/backend.log")
    )


def _append_fatal_model_log(message: str, exc: BaseException | None = None) -> None:
    try:
        tb = (
            "".join(
                traceback.format_exception(
                    type(exc), exc, getattr(exc, "__traceback__", None)
                )
            )
            if exc
            else ""
        )
        blob = f"Error: {message}\n{tb}"
        p = _model_log_file_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(blob)
            f.flush()
    except Exception:
        pass


def _force_worker_replacement(reason: str, exc: BaseException | None = None) -> None:
    """Irrecoverable worker state: marker + model log, brief sleep for /health, hard exit."""
    logger.error(
        "Fatal worker failure; exiting so platform replaces worker: %s", reason
    )
    try:
        WORKER_FATAL_MARKER.write_text(reason[:4096], encoding="utf-8")
    except Exception:
        pass
    _append_fatal_model_log(reason, exc)
    try:
        sys.stderr.write(
            "\n" + "=" * 60 + "\nVAST_BACKEND_FATAL: " + reason + "\n" + "=" * 60 + "\n"
        )
        sys.stderr.flush()
    except Exception:
        pass
    time.sleep(0.5)
    os._exit(1)


# Substrings (matched case-insensitively) for infra failures: wrong HTTP status + worker exit.
_FATAL_INFRA_SUBSTRINGS = (
    "comfyui process is not running",
    "comfyui process died during workflow submission",
    "comfyui died during active workflow wait",
    "comfyui died while polling history",
    "comfyui became unreachable during workflow execution",
    "watchdog: comfyui exited",
    "comfyui died (code",
    "comfyui died during startup",
    "comfyui process is not alive",
    "comfyui health bad status",
    "comfyui health probe failed",
    "failed to establish a new connection",
    "connection refused",
    "max retries exceeded",
    "comfyui startup failure",
    "because comfyui died",
    "comfyui died",
    "workflow timed out after",
)


def is_fatal_infra_error_message(msg: str) -> bool:
    """
    True for ComfyUI/CUDA/transport failures that must not be HTTP 200 and should
    fail health / replace the worker (when raised as exceptions).
    """
    if not msg:
        return False
    if re.search(r"return\s*code:\s*-9\b", msg, re.I):
        return True
    if re.search(r"\(\s*code\s*-9\s*\)", msg, re.I):
        return True
    lower = msg.lower()
    for s in _FATAL_INFRA_SUBSTRINGS:
        if s in lower:
            return True
    if "127.0.0.1" in msg and "connection refused" in lower:
        return True
    return False


def is_fatal_worker_error(exc: BaseException) -> bool:
    """True when the worker should exit and be replaced (ComfyUI infra death, CUDA init, SIGKILL/-9)."""
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    msg = str(exc)
    if (
        "CUDA preflight failed" in msg
        or "torch._C._cuda_init" in msg
        or "CUDA unknown error" in msg
    ):
        return True
    return is_fatal_infra_error_message(msg)


class S3Client:
    """S3 client for Hetzner: download input images, upload output videos/images."""

    def __init__(self):
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL")
        self.access_key = os.getenv("S3_ACCESS_KEY_ID")
        self.secret_key = os.getenv("S3_SECRET_ACCESS_KEY")
        self.region = os.getenv("S3_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET")
        self.prefix = (os.getenv("S3_PREFIX", "users") or "users").strip("/")

        addressing_style = os.getenv("S3_ADDRESSING_STYLE", "path")
        verify_ssl = os.getenv("S3_VERIFY_SSL", "true").lower() in (
            "1",
            "true",
            "yes",
            "y",
        )

        if not all([self.endpoint_url, self.access_key, self.secret_key, self.bucket]):
            raise RuntimeError(
                "Missing S3 configuration env vars (S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, "
                "S3_SECRET_ACCESS_KEY, S3_BUCKET)"
            )

        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            verify=verify_ssl,
            config=s3_boto_resilience.build_s3_boto_config(
                signature_version="s3v4",
                s3={"addressing_style": addressing_style},
            ),
        )

    def build_key(self, user_id: str, generation_id: str, filename: str) -> str:
        """Build S3 key for generation output"""
        user_id = str(user_id)
        generation_id = str(generation_id)
        return f"{self.prefix}/{user_id}/generations/{generation_id}/{filename}"

    def download_file(self, bucket: str, key: str, local_path: str) -> None:
        """Download file from S3 to local path"""
        with s3_boto_resilience.S3_IO_SEM:
            s3_boto_resilience.download_file_with_retry(
                self.client, bucket, key, local_path
            )

    def upload_file(
        self, local_path: str, key: str, content_type: str | None = None,
        bucket: str | None = None,
    ) -> dict:
        """Upload file to S3 and return metadata.

        Args:
            bucket: Target bucket. Caller must provide explicitly (per-request routing).
        """
        target_bucket = bucket or self.bucket
        if not content_type:
            content_type = (
                mimetypes.guess_type(local_path)[0] or "application/octet-stream"
            )

        extra_args = {"ContentType": content_type}
        with s3_boto_resilience.S3_IO_SEM:
            s3_boto_resilience.upload_file_with_retry(
                self.client, local_path, target_bucket, key, extra_args=extra_args
            )
            head = s3_boto_resilience.head_object_with_retry(
                self.client, target_bucket, key
            )
        return {
            "bucket": target_bucket,
            "key": key,
            "etag": head.get("ETag"),
            "size_bytes": head.get("ContentLength"),
            "content_type": content_type,
        }


_s3_client: Optional["S3Client"] = None


def get_s3_client() -> Optional["S3Client"]:
    """Lazy S3 client: init on first use to avoid import-time failures."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    try:
        _s3_client = S3Client()
        logger.info(
            f"S3 client initialized: bucket={_s3_client.bucket}, prefix={_s3_client.prefix}"
        )
        return _s3_client
    except Exception as e:
        logger.warning(f"S3 client not configured: {e}")
        return None


class LogStreamer:
    """Buffer Python log records during one request and push them to S3 for live tail.

    Hooks a logging.Handler onto the root logger so anything emitted during the
    request — including ComfyUI subprocess stdout that ``ComfyUIManager.log_output``
    forwards as ``logger.info("ComfyUI: ...")`` — lands in an in-memory buffer.
    A background daemon thread overwrites ``s3://{bucket}/{prefix}/log.txt`` every
    ``push_interval`` seconds. ``stop()`` flushes once more, removes the handler,
    and writes the empty ``log.done`` marker so the consumer knows the stream is
    closed (vs the worker dying mid-job).

    All S3 operations are best-effort: any error is logged and swallowed so a
    diagnostic-bus failure can never poison the actual workflow.
    """

    _FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    def __init__(
        self,
        *,
        request_id: str,
        s3_client: Any,
        bucket: str,
        prefix: str,
        push_interval: float = 2.0,
        generation_id: Optional[str] = None,
        lane: Optional[str] = None,
    ) -> None:
        self.request_id = request_id
        self.s3 = s3_client
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.push_interval = max(0.5, push_interval)
        self.generation_id = generation_id
        self.lane = lane

        self._buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._handler: Optional[logging.Handler] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started_at = time.time()
        self._last_pushed_at: Optional[float] = None
        self._last_pushed_size = -1

    def start(self) -> None:
        formatter = logging.Formatter(self._FORMAT)
        streamer = self

        class _BufferHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    line = (formatter.format(record) + "\n").encode(
                        "utf-8", errors="replace"
                    )
                    with streamer._buffer_lock:
                        streamer._buffer.extend(line)
                except Exception:
                    pass

        handler = _BufferHandler()
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)
        self._handler = handler

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"logstream-{self.request_id[:8]}",
        )
        self._thread.start()
        logger.info(
            "LogStreamer started: s3://%s/%s/ (push_interval=%.1fs)",
            self.bucket, self.prefix, self.push_interval,
        )

    def _loop(self) -> None:
        probe_every = max(1, int(os.getenv("LOG_STREAM_MEM_PROBE_EVERY_N_PUSHES", "5")))
        push_count = 0
        while not self._stop_event.wait(self.push_interval):
            push_count += 1
            if push_count % probe_every == 0:
                self._emit_memory_probe()
            self._push_safe()

    @staticmethod
    def _read_int_file(path: str) -> Optional[int]:
        try:
            with open(path) as f:
                return int(f.read().strip())
        except Exception:
            return None

    @staticmethod
    def _read_meminfo() -> Dict[str, int]:
        out: Dict[str, int] = {}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    name, _, rest = line.partition(":")
                    parts = rest.split()
                    if len(parts) >= 2 and parts[1] == "kB":
                        try:
                            out[name.strip()] = int(parts[0]) * 1024
                        except ValueError:
                            pass
        except Exception:
            pass
        return out

    @staticmethod
    def _read_status_rss() -> Optional[int]:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except Exception:
            pass
        return None

    @staticmethod
    def _read_cgroup_mem() -> tuple[Optional[int], Optional[int]]:
        """Return (current, max) bytes from cgroup v2, falling back to v1.

        ``max`` is None when the cgroup has no enforced ceiling (``"max"`` in v2
        or a sentinel value in v1).
        """
        # cgroup v2
        cur = LogStreamer._read_int_file("/sys/fs/cgroup/memory.current")
        if cur is not None:
            try:
                with open("/sys/fs/cgroup/memory.max") as f:
                    raw = f.read().strip()
                lim = None if raw == "max" else int(raw)
            except Exception:
                lim = None
            return cur, lim
        # cgroup v1
        cur = LogStreamer._read_int_file(
            "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        )
        lim = LogStreamer._read_int_file(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        )
        # v1 uses a huge sentinel (~9.2 EB) to mean "no limit"
        if lim is not None and lim > (1 << 62):
            lim = None
        return cur, lim

    @staticmethod
    def _read_gpu_mem() -> tuple[Optional[int], Optional[int]]:
        """Return (used_bytes, free_bytes) from torch.cuda when available."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None, None
            free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
            return total - free, free
        except Exception:
            return None, None

    def _emit_memory_probe(self) -> None:
        """Log one compact memory snapshot line. Best-effort: never raises."""
        try:
            mb = lambda b: f"{b // (1024*1024)}MB" if b is not None else "?"
            rss = self._read_status_rss()
            mem = self._read_meminfo()
            sys_avail = mem.get("MemAvailable")
            sys_total = mem.get("MemTotal")
            cg_cur, cg_max = self._read_cgroup_mem()
            gpu_used, gpu_free = self._read_gpu_mem()
            cg_pct = (
                f"({cg_cur * 100 // cg_max}%)"
                if cg_cur is not None and cg_max
                else ""
            )
            cg_str = (
                f"{mb(cg_cur)}/{mb(cg_max) if cg_max else 'unlimited'}{cg_pct}"
            )
            logger.info(
                "mem-probe rss=%s cgroup=%s sys_avail=%s/%s gpu_used=%s gpu_free=%s",
                mb(rss), cg_str, mb(sys_avail), mb(sys_total),
                mb(gpu_used), mb(gpu_free),
            )
        except Exception as exc:
            logger.debug("mem-probe failed: %s", exc)

    def _push_safe(self) -> None:
        try:
            self._push()
        except Exception as exc:
            logger.warning("LogStreamer push failed (request_id=%s): %s",
                           self.request_id, exc)

    def _push(self) -> None:
        with self._buffer_lock:
            size = len(self._buffer)
            if size == self._last_pushed_size:
                return
            snapshot = bytes(self._buffer)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/log.txt",
            Body=snapshot,
            ContentType="text/plain; charset=utf-8",
        )
        self._last_pushed_at = time.time()
        self._last_pushed_size = size
        meta = {
            "request_id": self.request_id,
            "started_at": self._started_at,
            "last_pushed_at": self._last_pushed_at,
            "size_bytes": size,
        }
        if self.generation_id:
            meta["generation_id"] = self.generation_id
        if self.lane:
            meta["lane"] = self.lane
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}/meta.json",
                Body=json.dumps(meta).encode("utf-8"),
                ContentType="application/json",
            )
        except Exception as exc:
            logger.warning("LogStreamer meta.json push failed: %s", exc)

    def stop(self) -> None:
        if self._handler is not None:
            try:
                logging.getLogger().removeHandler(self._handler)
            except Exception:
                pass
            self._handler = None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        # Force a final flush even if nothing changed since the last push, so
        # the consumer sees the very last lines before log.done.
        with self._buffer_lock:
            self._last_pushed_size = -1
        self._push_safe()
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}/log.done",
                Body=b"",
                ContentType="text/plain",
            )
        except Exception as exc:
            logger.warning("LogStreamer log.done write failed: %s", exc)
        logger.info(
            "LogStreamer stopped: s3://%s/%s/ (final %d bytes)",
            self.bucket, self.prefix,
            self._last_pushed_size if self._last_pushed_size >= 0 else 0,
        )


def _maybe_start_log_streamer(
    *,
    s3: Optional["S3Client"],
    request_id: str,
    generation_id: str,
    lane: Optional[str],
) -> Optional[LogStreamer]:
    """Best-effort LogStreamer factory; never raises so process_generation is unaffected."""
    if s3 is None or not request_id:
        return None
    try:
        push_interval = float(os.getenv("LOG_STREAM_PUSH_INTERVAL_SEC", "2.0"))
        streamer = LogStreamer(
            request_id=request_id,
            s3_client=s3.client,
            bucket=s3.bucket,
            prefix=f"diagnostics/{request_id}",
            push_interval=push_interval,
            generation_id=generation_id or None,
            lane=lane or None,
        )
        streamer.start()
        return streamer
    except Exception as exc:
        logger.warning("Failed to start LogStreamer: %s", exc)
        return None


def _safe_join(base: pathlib.Path, subfolder: str, filename: str) -> pathlib.Path:
    """Safely join paths, preventing directory traversal"""
    sub = (subfolder or "").strip("/").replace("\\", "/")
    fn = filename.replace("\\", "/")
    p = (base / sub / fn).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise RuntimeError("Invalid subfolder/filename path traversal detected")
    return p


def resolve_comfy_file(
    filename: str, subfolder: str = "", file_type: str = "output"
) -> pathlib.Path:
    """Resolve ComfyUI file path from output descriptor"""
    base_map = {
        "output": pathlib.Path("/app/output"),
        "input": pathlib.Path("/app/input"),
        "temp": pathlib.Path("/app/temp"),
    }
    base = base_map.get((file_type or "output").lower(), pathlib.Path("/app/output"))
    return _safe_join(base, subfolder, filename)


def _safe_component(s: str) -> str:
    """Sanitize string for use in file paths"""
    s = str(s or "")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s[:80] if s else "x"


def make_job_subdir(
    user_id: str,
    generation_id: str,
    job_id: str | None,
    *,
    platform_prefix: str | None = None,
) -> str:
    """Generate unique subdirectory for job outputs.
    platform_prefix from JOB_PREFIX env (default: jobs). Vast sets vast.
    """
    prefix = platform_prefix or os.getenv("JOB_PREFIX", "jobs")
    rid_src = (job_id or "").strip()
    rid = _safe_component(rid_src[:12]) if rid_src else uuid.uuid4().hex[:12]
    return (
        f"{prefix}/u{_safe_component(user_id)}/g{_safe_component(generation_id)}/{rid}"
    )


def _bool(v, default=None):
    """Convert various types to boolean"""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return default


def cleanup_job_output(run_subdir: str) -> None:
    """Delete the per-job output folder under /app/output"""
    base_output = pathlib.Path("/app/output").resolve()
    job_dir = (base_output / run_subdir).resolve()
    if str(job_dir).startswith(str(base_output)) and job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.info(f"Cleaned up job output directory: {job_dir}")


def cleanup_job_input(run_subdir: str) -> None:
    """Delete per-job staged inputs under Comfy input (e.g. S3 audio copy for LoadAudio)."""
    if not (run_subdir or "").strip():
        return
    base_in = _comfy_input_directory()
    job_dir = (base_in / run_subdir).resolve()
    if str(job_dir).startswith(str(base_in)) and job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.info("Cleaned up job input directory: %s", job_dir)


def assert_cuda_ready() -> None:
    """Verify GPU/CUDA is usable before ComfyUI startup. Raises RuntimeError on failure."""
    import torch

    logger.info("CUDA_VISIBLE_DEVICES=%s", os.getenv("CUDA_VISIBLE_DEVICES"))
    logger.info("NVIDIA_VISIBLE_DEVICES=%s", os.getenv("NVIDIA_VISIBLE_DEVICES"))

    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        logger.info(
            "nvidia-smi rc=%s out=%s err=%s",
            smi.returncode,
            smi.stdout.strip() if smi.stdout else "",
            smi.stderr.strip() if smi.stderr else "",
        )
    except Exception as e:
        logger.warning("nvidia-smi failed: %s", e)

    logger.info(
        "torch=%s torch.version.cuda=%s",
        torch.__version__,
        torch.version.cuda,
    )
    avail = torch.cuda.is_available()
    count = torch.cuda.device_count()
    logger.info("torch.cuda.is_available=%s device_count=%s", avail, count)

    if not avail or count < 1:
        raise RuntimeError("CUDA preflight failed before ComfyUI startup")

    name = torch.cuda.get_device_name(0)
    logger.info("cuda device0=%s", name)
    _ = torch.tensor([1.0], device="cuda")


class ComfyUIManager:
    def __init__(self):
        self.host = os.getenv("COMFYUI_HOST", "0.0.0.0")
        self.port = int(os.getenv("COMFYUI_PORT", "8188"))
        self.comfyui_url = os.getenv("COMFYUI_URL") or f"http://127.0.0.1:{self.port}"
        self.start_timeout = int(os.getenv("COMFYUI_START_TIMEOUT", "180"))
        self.comfyui_process: Optional[subprocess.Popen] = None
        self.is_ready = False
        self._start_lock = threading.Lock()
        self.log_tail: Deque[str] = deque(maxlen=200)

        self.extra_args = os.getenv("COMFYUI_ARGS", "").strip()

    def is_process_alive(self) -> bool:
        return self.comfyui_process is not None and self.comfyui_process.poll() is None

    def interrupt_prompt(self) -> None:
        """POST /interrupt to ComfyUI to cancel the active prompt. Best-effort."""
        try:
            r = requests.post(f"{self.comfyui_url}/interrupt", timeout=3)
            logger.info("ComfyUI interrupt response: %s", r.status_code)
        except Exception as e:
            logger.warning("Failed to interrupt ComfyUI: %s", e)

    def assert_comfy_healthy(self) -> None:
        if not self.is_process_alive():
            rc = (
                None
                if self.comfyui_process is None
                else self.comfyui_process.returncode
            )
            raise RuntimeError(f"ComfyUI process is not alive (code={rc})")
        try:
            r = requests.get(f"{self.comfyui_url}/system_stats", timeout=3)
            if r.status_code != 200:
                raise RuntimeError(f"ComfyUI health bad status: {r.status_code}")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"ComfyUI health probe failed: {e}") from e
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"ComfyUI health probe failed: {e}") from e

    def _terminate_process(self) -> None:
        if self.comfyui_process is None:
            return
        if self.comfyui_process.poll() is None:
            logger.warning("Terminating previous ComfyUI process before restart...")
            self.comfyui_process.terminate()
            try:
                self.comfyui_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.comfyui_process.kill()
                self.comfyui_process.wait()
        self.is_ready = False
        self.comfyui_process = None

    def start_comfyui(self) -> None:
        with self._start_lock:
            if self.is_ready:
                try:
                    r = requests.get(f"{self.comfyui_url}/system_stats", timeout=5)
                    if r.status_code == 200:
                        return
                    self.is_ready = False
                except Exception as e:
                    logger.warning(
                        f"ComfyUI was marked ready but not responding: {e}, restarting..."
                    )
                    self.is_ready = False

            self._terminate_process()

            args = [
                "python3",
                "-u",
                "/app/main.py",
                "--listen",
                self.host,
                "--port",
                str(self.port),
                "--preview-method",
                "none",
            ]

            if self.extra_args:
                args += shlex.split(self.extra_args)

            logger.info(f"Starting ComfyUI: {' '.join(args)}")

            try:
                proc = subprocess.Popen(
                    args,
                    cwd="/app",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                self.comfyui_process = proc

                def log_output(p: subprocess.Popen) -> None:
                    if p.stdout:
                        for line in iter(p.stdout.readline, ""):
                            if line:
                                s = line.rstrip()
                                self.log_tail.append(s)
                                logger.info(f"ComfyUI: {s}")

                threading.Thread(target=log_output, args=(proc,), daemon=True).start()

            except Exception as e:
                raise RuntimeError(f"Failed to launch ComfyUI process: {e}") from e

            deadline = time.time() + self.start_timeout
            last_err = None
            check_interval = 1
            while time.time() < deadline:
                if self.comfyui_process.poll() is not None:
                    returncode = self.comfyui_process.returncode
                    tail = (
                        "\n".join(list(self.log_tail)[-60:])
                        or "(no ComfyUI output captured)"
                    )
                    logger.error(
                        f"ComfyUI process died unexpectedly with return code {returncode}. "
                        "Last logs: %s",
                        tail,
                    )
                    raise RuntimeError(
                        f"ComfyUI died (code {returncode}). Last logs:\n{tail}"
                    )

                try:
                    r = requests.get(f"{self.comfyui_url}/system_stats", timeout=2)
                    if r.status_code == 200:
                        self.is_ready = True
                        logger.info("ComfyUI is ready.")
                        return
                except requests.exceptions.ConnectionError as e:
                    last_err = e
                except Exception as e:
                    last_err = e

                time.sleep(check_interval)

            if self.comfyui_process.poll() is not None:
                returncode = self.comfyui_process.returncode
                self._terminate_process()
                self.is_ready = False
                tail = (
                    "\n".join(list(self.log_tail)[-60:])
                    or "(no ComfyUI output captured)"
                )
                logger.error(
                    "ComfyUI died during startup (return code %s). Last logs: %s",
                    returncode,
                    tail,
                )
                raise RuntimeError(
                    f"ComfyUI died during startup (code {returncode}). Last logs:\n{tail}"
                )

            self._terminate_process()
            self.is_ready = False
            tail = (
                "\n".join(list(self.log_tail)[-60:]) or "(no ComfyUI output captured)"
            )
            raise RuntimeError(
                f"ComfyUI did not become ready within {self.start_timeout}s. "
                f"Last error: {last_err}. Hung process was terminated. Last logs:\n{tail}"
            )

    def submit_workflow(self, workflow: Dict[str, Any]) -> str:
        if self.comfyui_process and self.comfyui_process.poll() is not None:
            returncode = self.comfyui_process.returncode
            raise RuntimeError(
                f"ComfyUI process is not running (return code: {returncode}). "
                f"Cannot submit workflow."
            )

        if not self.is_ready:
            logger.warning("ComfyUI not marked as ready, attempting to start...")
            self.start_comfyui()

        def _post_prompt() -> str:
            r = requests.post(
                f"{self.comfyui_url}/prompt", json={"prompt": workflow}, timeout=30
            )
            if r.status_code != 200:
                raise RuntimeError(
                    f"Failed to submit workflow: HTTP {r.status_code}: {r.text}"
                )
            data = r.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"No prompt_id returned: {data}")
            return prompt_id

        try:
            return _post_prompt()
        except requests.exceptions.ConnectionError as e:
            if self.comfyui_process and self.comfyui_process.poll() is not None:
                returncode = self.comfyui_process.returncode
                raise RuntimeError(
                    f"ComfyUI process died during workflow submission (return code: {returncode}). "
                    f"Connection error: {e}"
                ) from e
            logger.warning(
                "ConnectionError while ComfyUI process alive; retrying start + POST once."
            )
            self.is_ready = False
            self.start_comfyui()
            return _post_prompt()

    def wait_for_completion(
        self, prompt_id: str, timeout: int, worker_state=None
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout
        consecutive_conn_errors = 0

        def log_tail_block() -> str:
            return (
                "\n".join(list(self.log_tail)[-60:]) or "(no ComfyUI output captured)"
            )

        while time.time() < deadline:
            if (
                self.comfyui_process is not None
                and self.comfyui_process.poll() is not None
            ):
                rc = self.comfyui_process.returncode
                raise RuntimeError(
                    f"ComfyUI died during active workflow wait (code {rc}). "
                    f"Last logs:\n{log_tail_block()}"
                )

            try:
                r = requests.get(
                    f"{self.comfyui_url}/history/{prompt_id}",
                    timeout=COMFYUI_HISTORY_POLL_TIMEOUT_SEC,
                )
                consecutive_conn_errors = 0
                if worker_state is not None:
                    worker_state.touch_progress()
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                consecutive_conn_errors += 1
                if self.comfyui_process and self.comfyui_process.poll() is not None:
                    rc = self.comfyui_process.returncode
                    raise RuntimeError(
                        f"ComfyUI died while polling history (code {rc}). "
                        f"Last logs:\n{log_tail_block()}"
                    ) from e
                if consecutive_conn_errors >= COMFYUI_HISTORY_POLL_MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(
                        f"ComfyUI became unreachable during workflow execution: {e}"
                    ) from e
                time.sleep(1)
                continue

            if r.status_code != 200:
                time.sleep(2)
                continue
            hist = r.json() or {}
            if not isinstance(hist, dict) or prompt_id not in hist:
                time.sleep(2)
                continue
            item = hist[prompt_id]
            status = (item or {}).get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(f"Workflow failed: {status.get('messages', status)}")
            if item and "outputs" in item and (item.get("outputs") or {}):
                return item
            if status.get("status_str") in ("success", "completed"):
                # ComfyUI may set status before populating outputs; wait and re-poll
                outputs = (item or {}).get("outputs") or {}
                if not outputs:
                    time.sleep(2)
                    try:
                        r2 = requests.get(
                            f"{self.comfyui_url}/history/{prompt_id}",
                            timeout=COMFYUI_HISTORY_POLL_TIMEOUT_SEC,
                        )
                        consecutive_conn_errors = 0
                        if worker_state is not None:
                            worker_state.touch_progress()
                    except (
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                    ) as e:
                        consecutive_conn_errors += 1
                        if (
                            self.comfyui_process
                            and self.comfyui_process.poll() is not None
                        ):
                            rc = self.comfyui_process.returncode
                            raise RuntimeError(
                                f"ComfyUI died while polling history (code {rc}). "
                                f"Last logs:\n{log_tail_block()}"
                            ) from e
                        if consecutive_conn_errors >= COMFYUI_HISTORY_POLL_MAX_CONSECUTIVE_ERRORS:
                            raise RuntimeError(
                                "ComfyUI became unreachable during workflow execution: "
                                f"{e}"
                            ) from e
                        continue
                    if r2.status_code == 200:
                        hist2 = r2.json() or {}
                        item2 = hist2.get(prompt_id, item)
                        if item2 and (item2.get("outputs") or {}):
                            return item2
                return item
            time.sleep(2)

        self.interrupt_prompt()
        raise RuntimeError(f"Workflow timed out after {timeout} seconds")

    def extract_outputs(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract output descriptors (no base64 - files on disk)"""
        outputs = workflow_data.get("outputs", {}) or {}
        videos: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []
        texts: List[Dict[str, Any]] = []

        for node_id, node_output in outputs.items():
            if not isinstance(node_output, dict):
                continue

            for key in ("gifs", "videos"):
                if key in node_output:
                    for v in node_output[key]:
                        videos.append(
                            {
                                "filename": v["filename"],
                                "subfolder": v.get("subfolder", ""),
                                "type": v.get("type", "output"),
                                "format": v.get("format", "video/mp4"),
                                "node_id": node_id,
                            }
                        )

            if "images" in node_output:
                for im in node_output["images"]:
                    images.append(
                        {
                            "filename": im["filename"],
                            "subfolder": im.get("subfolder", ""),
                            "type": im.get("type", "output"),
                            "node_id": node_id,
                        }
                    )

            if "text" in node_output:
                for t in node_output["text"]:
                    texts.append({"text": t, "node_id": node_id})

        logger.info(
            f"Extracted output descriptors: {len(videos)} videos, {len(images)} images, "
            f"{len(texts)} texts. Output node IDs: {list(outputs.keys())}"
        )

        return {
            "videos": videos,
            "images": images,
            "texts": texts,
            "raw_outputs": outputs,
        }


comfy = ComfyUIManager()


def process_generation(payload: Dict[str, Any], worker_state=None) -> Dict[str, Any]:
    """
    Run one generation. Payload from PyWorker: workflow_json (pre-patched), user_id, generation_id.

    payload: {"input": {workflow_json, run_subdir, user_id, generation_id, ...}, "id": optional}.

    Returns: {success, error?, prompt_id?, videos, images, texts, video_count, image_count, run_subdir?, timing?}
    """
    job_input = payload.get("input", {}) or {}
    workflow_json = job_input.get("workflow_json")
    timeout = int(job_input.get("timeout", 600))

    user_id = str(job_input.get("user_id") or "")
    generation_id = str(job_input.get("generation_id") or "")
    job_id = str(payload.get("id") or "")
    s3_bucket = str(job_input.get("s3_bucket") or "").strip()

    if not isinstance(workflow_json, dict):
        return {
            "success": False,
            "error": "input.workflow_json must be ComfyUI API prompt JSON (PyWorker transforms before forwarding).",
        }

    if not s3_bucket:
        return {
            "success": False,
            "error": "input.s3_bucket is required. Caller must specify the target S3 bucket.",
        }

    run_subdir = (job_input.get("run_subdir") or "").strip() or make_job_subdir(
        user_id, generation_id, job_id
    )

    s3 = get_s3_client()
    if not s3:
        return {"success": False, "error": "S3 client not configured."}

    if not user_id or not generation_id:
        return {
            "success": False,
            "error": "input.user_id and input.generation_id are required for S3 output keys.",
        }

    logger.info(f"Job run_subdir: {run_subdir}")

    timing = dict(payload.get("_timing") or {})
    t_client = payload.get("_client_sent_at") or job_input.get("_client_sent_at")
    local_path = None

    streamer = _maybe_start_log_streamer(
        s3=s3,
        request_id=job_id,
        generation_id=generation_id,
        lane=str(job_input.get("generation_lane") or "") or None,
    )

    try:
        timing["job_started_at"] = time.time()
        if worker_state is not None:
            worker_state.begin(job_id)

        p = pathlib.Path("/app/models/diffusion_models")
        logger.info(
            "diffusion_models exists=%s is_symlink=%s real=%s",
            p.exists(),
            p.is_symlink() if p.exists() else False,
            p.resolve() if p.exists() else None,
        )

        prompt_id = comfy.submit_workflow(workflow_json)
        timing["workflow_submitted_at"] = time.time()
        if worker_state is not None:
            worker_state.set_prompt_id(prompt_id)
        logger.info(f"Workflow submitted with prompt_id: {prompt_id}")

        wf_data = comfy.wait_for_completion(
            prompt_id, timeout=timeout, worker_state=worker_state
        )
        timing["workflow_finished_at"] = time.time()
        logger.info(f"Workflow completed for prompt_id: {prompt_id}")

        out = comfy.extract_outputs(wf_data)

        if out["videos"]:
            v = out["videos"][0]
            local_path = resolve_comfy_file(
                v["filename"], v.get("subfolder", ""), v.get("type", "output")
            )
            if not local_path.exists():
                return {
                    "success": False,
                    "error": f"Video missing on disk: {local_path}",
                    "prompt_id": prompt_id,
                }

            ext = pathlib.Path(v["filename"]).suffix or ".mp4"
            s3_filename = f"result{ext}"
            key = s3.build_key(
                user_id=user_id, generation_id=generation_id, filename=s3_filename
            )

            meta = s3.upload_file(str(local_path), key, content_type=v.get("format", "video/mp4"), bucket=s3_bucket)

            logger.info(
                f"Uploaded video to S3: {key} ({meta['size_bytes']} bytes)"
            )

            timing["response_sent_at"] = time.time()
            timing["accept_delay_sec"] = (
                timing["server_received_at"] - t_client
                if t_client is not None
                else None
            )
            timing["queue_inside_worker_sec"] = (
                timing["workflow_submitted_at"] - timing["job_started_at"]
            )
            timing["worker_exec_sec"] = (
                timing["workflow_finished_at"] - timing["job_started_at"]
            )

            return {
                "success": True,
                "prompt_id": prompt_id,
                "videos": [
                    {
                        **meta,
                        "node_id": v["node_id"],
                        "filename": s3_filename,
                    }
                ],
                "images": [],
                "texts": out["texts"],
                "video_count": 1,
                "image_count": 0,
                "run_subdir": run_subdir,
                "timing": timing,
            }

        if out["images"]:
            im = out["images"][0]
            local_path = resolve_comfy_file(
                im["filename"], im.get("subfolder", ""), im.get("type", "output")
            )
            if not local_path.exists():
                return {
                    "success": False,
                    "error": f"Image missing on disk: {local_path}",
                    "prompt_id": prompt_id,
                }

            ext = pathlib.Path(im["filename"]).suffix or ".png"
            s3_filename = f"result{ext}"
            key = s3.build_key(
                user_id=user_id, generation_id=generation_id, filename=s3_filename
            )

            content_type = mimetypes.guess_type(str(local_path))[0] or "image/png"
            meta = s3.upload_file(str(local_path), key, content_type=content_type, bucket=s3_bucket)

            logger.info(
                f"Uploaded image to S3: {key} ({meta['size_bytes']} bytes)"
            )

            timing["response_sent_at"] = time.time()
            timing["accept_delay_sec"] = (
                timing["server_received_at"] - t_client
                if t_client is not None
                else None
            )
            timing["queue_inside_worker_sec"] = (
                timing["workflow_submitted_at"] - timing["job_started_at"]
            )
            timing["worker_exec_sec"] = (
                timing["workflow_finished_at"] - timing["job_started_at"]
            )

            return {
                "success": True,
                "prompt_id": prompt_id,
                "videos": [],
                "images": [
                    {
                        **meta,
                        "node_id": im["node_id"],
                        "filename": s3_filename,
                    }
                ],
                "texts": out["texts"],
                "video_count": 0,
                "image_count": 1,
                "run_subdir": run_subdir,
                "timing": timing,
            }

        logger.error(
            f"Workflow completed but produced no videos or images. "
            f"Outputs: {list(out.get('raw_outputs', {}).keys())}"
        )
        timing["response_sent_at"] = time.time()
        timing["accept_delay_sec"] = (
            timing["server_received_at"] - t_client if t_client is not None else None
        )
        timing["queue_inside_worker_sec"] = (
            timing["workflow_submitted_at"] - timing["job_started_at"]
        )
        timing["worker_exec_sec"] = (
            timing["workflow_finished_at"] - timing["job_started_at"]
        )
        return {
            "success": False,
            "error": "Workflow completed but produced no videos or images. Check workflow output nodes.",
            "prompt_id": prompt_id,
            "videos": [],
            "images": [],
            "texts": out["texts"],
            "video_count": 0,
            "image_count": 0,
            "timing": timing,
        }

    except Exception as e:
        logger.exception(f"Handler error: {e}")
        msg = str(e)
        if is_fatal_worker_error(e):
            _force_worker_replacement(msg, e)
        out_err = {
            "success": False,
            "error": str(e),
            "comfyui_tail": list(comfy.log_tail)[-60:],
            "videos": [],
            "images": [],
            "texts": [],
        }
        if timing.get("job_started_at"):
            timing["response_sent_at"] = time.time()
            timing["accept_delay_sec"] = (
                timing["server_received_at"] - t_client
                if t_client is not None
                else None
            )
            out_err["timing"] = timing
        return out_err

    finally:
        if streamer is not None:
            try:
                streamer.stop()
            except Exception as e:
                logger.warning("LogStreamer stop failed: %s", e)

        try:
            cleanup_job_input(run_subdir)
        except Exception as e:
            logger.warning("Failed to cleanup job input directory: %s", e)

        try:
            cleanup_job_output(run_subdir)
        except Exception as e:
            logger.warning(f"Failed to cleanup job output directory: {e}")

        if worker_state is not None:
            worker_state.end()
