"""S3 boto3/botocore resilience: standard retries + app-level transient retries.

Keep copies in sync:
- comfy-vast-serverless/s3_boto_resilience.py
- vast-serverless-pyworker/workers/comfyui-json/s3_boto_resilience.py
- bot/app/services/s3_boto_resilience.py
- admin-panel/backend/app/services/s3_boto_resilience.py
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    HTTPClientError,
    ReadTimeoutError,
)

try:
    from botocore.exceptions import ConnectionClosedError as _ConnectionClosedError
except ImportError:  # pragma: no cover - older botocore
    _ConnectionClosedError = None  # type: ignore[assignment,misc]

_log = logging.getLogger(__name__)

RETRYABLE_S3_CODES = frozenset(
    {
        "SlowDown",
        "InternalError",
        "RequestTimeout",
        "RequestTimeoutException",
    }
)
RETRYABLE_HTTP_STATUS = frozenset({500, 502, 503, 504})

_RETRYABLE_TRANSPORT: tuple[type[BaseException], ...] = (
    EndpointConnectionError,
    ReadTimeoutError,
    ConnectTimeoutError,
    HTTPClientError,
    ConnectionError,
) + ((_ConnectionClosedError,) if _ConnectionClosedError is not None else ())


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _resolve_max_attempts(explicit: int | None) -> int:
    if explicit is not None:
        return max(1, explicit)
    v = _env_int("S3_BOTO_MAX_ATTEMPTS")
    if v is not None:
        return max(1, v)
    v = _env_int("AWS_MAX_ATTEMPTS")
    if v is not None:
        return max(1, v)
    return 8


def build_s3_boto_config(
    *,
    signature_version: str | None = None,
    s3: dict[str, Any] | None = None,
    max_attempts: int | None = None,
) -> Config:
    """Botocore Config: standard retries, timeouts, tcp_keepalive."""
    attempts = _resolve_max_attempts(max_attempts)
    kw: dict[str, Any] = {
        "retries": {"mode": "standard", "max_attempts": attempts},
        "connect_timeout": 5,
        "read_timeout": 60,
        "tcp_keepalive": True,
    }
    if signature_version:
        kw["signature_version"] = signature_version
    if s3:
        kw["s3"] = s3
    return Config(**kw)


def _io_concurrency() -> int:
    raw = os.getenv("S3_IO_CONCURRENCY", "3")
    try:
        n = int(raw)
        return max(1, min(n, 64))
    except ValueError:
        return 3


S3_IO_SEM = threading.Semaphore(_io_concurrency())


def _client_error_meta(exc: ClientError) -> tuple[str, int | None]:
    err = exc.response.get("Error", {}) or {}
    meta = exc.response.get("ResponseMetadata", {}) or {}
    code = str(err.get("Code") or "")
    status = meta.get("HTTPStatusCode")
    return code, status if isinstance(status, int) else None


def is_retryable_s3_error(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE_TRANSPORT):
        return True
    if isinstance(exc, ClientError):
        code, status = _client_error_meta(exc)
        if status == 404:
            return False
        if code in RETRYABLE_S3_CODES or status in RETRYABLE_HTTP_STATUS:
            return True
    return False


def s3_call_with_retry(
    fn: Callable[[], None],
    *,
    op_name: str,
    attempts: int = 5,
    base_delay: float = 0.6,
    max_delay: float = 12.0,
    verify_success: Callable[[], bool] | None = None,
    log: logging.Logger | None = None,
) -> None:
    """Run fn(); on retryable errors retry with backoff. Optional verify after failure."""
    lg = log or _log
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            fn()
            return
        except Exception as exc:
            if verify_success is not None:
                try:
                    if verify_success():
                        lg.info(
                            "%s ambiguous failure but verify_success=True; treating as ok",
                            op_name,
                        )
                        return
                except Exception:
                    pass

            code = ""
            status: int | None = None
            if isinstance(exc, ClientError):
                code, status = _client_error_meta(exc)

            if not is_retryable_s3_error(exc) or attempt == attempts:
                raise

            last_exc = exc
            sleep_s = min(max_delay, base_delay * (2 ** (attempt - 1)))
            sleep_s *= random.uniform(0.75, 1.25)
            lg.warning(
                "%s retryable error attempt=%s/%s exc=%s s3_code=%s http_status=%s sleep=%.2fs",
                op_name,
                attempt,
                attempts,
                type(exc).__name__,
                code or None,
                status,
                sleep_s,
            )
            time.sleep(sleep_s)

    if last_exc:
        raise last_exc


def upload_file_with_retry(
    client: Any,
    local_path: str,
    bucket: str,
    key: str,
    *,
    extra_args: dict[str, Any] | None = None,
    attempts: int = 5,
) -> None:
    size = os.path.getsize(local_path)
    extra = dict(extra_args or {})

    def do_upload() -> None:
        client.upload_file(local_path, bucket, key, ExtraArgs=extra)

    def verify() -> bool:
        try:
            head = client.head_object(Bucket=bucket, Key=key)
            return head.get("ContentLength") == size
        except Exception:
            return False

    s3_call_with_retry(
        do_upload,
        op_name=f"put_object(upload_file):{bucket}/{key}",
        attempts=attempts,
        verify_success=verify,
    )


def download_file_with_retry(
    client: Any,
    bucket: str,
    key: str,
    local_path: str,
    *,
    attempts: int = 5,
) -> None:
    def do_download() -> None:
        client.download_file(bucket, key, local_path)

    s3_call_with_retry(
        do_download,
        op_name=f"get_object(download_file):{bucket}/{key}",
        attempts=attempts,
    )


def download_fileobj_with_retry(
    client: Any,
    bucket: str,
    key: str,
    fileobj: Any,
    *,
    attempts: int = 5,
) -> None:
    def do_download() -> None:
        try:
            fileobj.seek(0)
            fileobj.truncate(0)
        except Exception:
            pass
        client.download_fileobj(bucket, key, fileobj)

    s3_call_with_retry(
        do_download,
        op_name=f"get_object(download_fileobj):{bucket}/{key}",
        attempts=attempts,
    )


def upload_fileobj_with_retry(
    client: Any,
    fileobj: Any,
    bucket: str,
    key: str,
    *,
    extra_args: dict[str, Any] | None = None,
    expected_size: int,
    attempts: int = 5,
) -> None:
    extra = dict(extra_args or {})

    def do_upload() -> None:
        try:
            fileobj.seek(0)
        except Exception:
            pass
        client.upload_fileobj(fileobj, bucket, key, ExtraArgs=extra)

    def verify() -> bool:
        try:
            head = client.head_object(Bucket=bucket, Key=key)
            return head.get("ContentLength") == expected_size
        except Exception:
            return False

    s3_call_with_retry(
        do_upload,
        op_name=f"put_object(upload_fileobj):{bucket}/{key}",
        attempts=attempts,
        verify_success=verify,
    )


def head_object_with_retry(
    client: Any,
    bucket: str,
    key: str,
    *,
    attempts: int = 5,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def do_head() -> None:
        resp = client.head_object(Bucket=bucket, Key=key)
        out.clear()
        out.update(resp)

    s3_call_with_retry(
        do_head,
        op_name=f"head_object:{bucket}/{key}",
        attempts=attempts,
    )
    return dict(out)
