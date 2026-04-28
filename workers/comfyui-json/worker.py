"""
Vast PyWorker for ComfyUI with app-format request transform.

Accepts app format: {workflow, input_images, user_id, generation_id, ...}
Transforms to Vast format and forwards to API wrapper.

Undresser lanes:
- WAN22_I2V_LONG_5090  (Blackwell, video I2V)
- WAN22_IV2V_FACESWAP_5090  (Blackwell, video IV2V)
- ZIMAGE_TURBO_I2I_5090  (Blackwell, image I2I)
- FLUX_S2_I2I_5090  (Blackwell, image I2I)
"""

import logging
import os
import random
import sys
import uuid
from pathlib import Path

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

for _name in ("botocore", "boto3", "s3transfer", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

_log = logging.getLogger("comfyui-json")

# API wrapper config (our backend on 8189; stock ComfyUI uses 18288)
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1")
MODEL_SERVER_PORT = int(os.getenv("MODEL_SERVER_PORT", "8189"))
MODEL_LOG_FILE = os.getenv("MODEL_LOG_FILE", "/app/logs/backend.log")
MODEL_HEALTHCHECK_ENDPOINT = os.getenv("MODEL_HEALTHCHECK_ENDPOINT", "/health")
BENCHMARK_RUNS = int(os.getenv("BENCHMARK_RUNS", "1"))

# Template / calibration lane (BENCHMARK_GENERATION_LANE on Vast) → default benchmark JSON under misc/.
# Bot/request workload uses generation_lane keys like WAN22_I2V_LONG_5090 (VAST_WORKLOAD_UNITS_<LANE>).
_DEFAULT_BENCHMARK_FILES: dict[str, str] = {
    "WAN22_I2V_LONG_5090": "benchmark_WAN22_I2V_LONG_5090.json",
    "WAN22_IV2V_FACESWAP_5090": "benchmark_WAN22_IV2V_FACESWAP_5090.json",
    "ZIMAGE_TURBO_I2I_5090": "benchmark_ZIMAGE_TURBO_I2I_5090.json",
    "FLUX_S2_I2I_5090": "benchmark_FLUX_S2_I2I_5090.json",
}

# BENCHMARK_GENERATION_LANE (template/calibration) → input.generation_lane for workload_calculator / SDK cost=.
# In undresser, BENCHMARK_GENERATION_LANE IS the generation_lane — no remapping needed.
_BENCHMARK_ENV_LANE_TO_REQUEST_GENERATION_LANE: dict[str, str] = {
    "WAN22_I2V_LONG_5090": "WAN22_I2V_LONG_5090",
    "WAN22_IV2V_FACESWAP_5090": "WAN22_IV2V_FACESWAP_5090",
    "ZIMAGE_TURBO_I2I_5090": "ZIMAGE_TURBO_I2I_5090",
    "FLUX_S2_I2I_5090": "FLUX_S2_I2I_5090",
}
# Lanes that may appear on input.generation_lane (bot traffic). Declared load comes from non-empty
# ``input.vast_workload_units`` when valid (matches bot SDK precedence), else template env
# ``VAST_WORKLOAD_UNITS_<LANE>``.
_KNOWN_WORKLOAD_LANES: frozenset[str] = frozenset(
    _BENCHMARK_ENV_LANE_TO_REQUEST_GENERATION_LANE.values()
)

# Custom backend writes "Backend ready"; stock uses "To see the GUI go to: "
MODEL_LOAD_LOG_MSG = ["Backend ready"]
MODEL_ERROR_LOG_MSGS = [
    "MetadataIncompleteBuffer",
    "Value not in list: ",
    "[ERROR] Provisioning Script failed",
    "Error:",
    "Traceback (most recent call last):",
]


def _normalized_benchmark_lane() -> str:
    return (
        (os.getenv("BENCHMARK_GENERATION_LANE") or "").strip().upper().replace(" ", "_")
    )


def _get_benchmark_workflow_path() -> Path | None:
    """Resolve benchmark JSON: BENCHMARK_WORKFLOW_FILE, then lane default from ``_DEFAULT_BENCHMARK_FILES``."""
    misc_dir = Path(__file__).resolve().parent / "misc"
    override = (os.getenv("BENCHMARK_WORKFLOW_FILE") or "").strip()
    if override:
        p = misc_dir / override
        if p.is_file():
            _log.info("Benchmark workflow: %s (BENCHMARK_WORKFLOW_FILE)", p.name)
            return p
        _log.warning("BENCHMARK_WORKFLOW_FILE=%s not found under misc/", override)

    lane = _normalized_benchmark_lane()
    if lane and lane in _DEFAULT_BENCHMARK_FILES:
        p = misc_dir / _DEFAULT_BENCHMARK_FILES[lane]
        if p.is_file():
            _log.info("Benchmark workflow lane=%s file=%s", lane, p.name)
            return p
        _log.warning("Lane %s expects %s but file missing", lane, p.name)

    _log.warning("No benchmark workflow found under misc/")
    return None


def _get_benchmark_payload() -> dict:
    """Generate benchmark payload. App format is transformed to workflow_json so backend receives
    the same format as production (benchmark bypasses request_parser, so we transform here)."""
    import json

    path = _get_benchmark_workflow_path()
    if path is None:
        return _fallback_benchmark_payload()
    with open(path, encoding="utf-8") as f:
        workflow = json.load(f)

    if "workflow" in workflow:
        workflow = workflow["workflow"]

    # Auto-detect path-mode by scanning the workflow's class_types. If the
    # workflow uses VHS_LoadVideo (Tokyo Sage chained on WAN22_IV2V_FACESWAP_5090),
    # we route image+video through path-mode injection — same pipeline as
    # production. Other benchmark workflows still use ETN_LoadImageBase64 +
    # legacy base64 single-image. This means BENCHMARK_VIDEO_* env vars can be
    # set globally across every lane: worker just ignores them when the lane's
    # workflow doesn't have a video loader.
    class_types = {
        n.get("class_type", "")
        for n in workflow.values()
        if isinstance(n, dict)
    }
    use_path_mode = "VHS_LoadVideo" in class_types

    input_images: list[dict] = []
    img_bucket = (
        os.getenv("BENCHMARK_IMAGE_BUCKET")
        or os.getenv("S3_BUCKET")
        or os.getenv("S3_BUCKET_NAME")
    )
    img_key = (os.getenv("BENCHMARK_IMAGE_KEY") or "").strip()
    vid_bucket = (
        os.getenv("BENCHMARK_VIDEO_BUCKET")
        or os.getenv("S3_BUCKET")
        or os.getenv("S3_BUCKET_NAME")
    )
    vid_key = (os.getenv("BENCHMARK_VIDEO_KEY") or "").strip()

    if use_path_mode:
        if img_bucket and img_key:
            input_images.append({
                "bucket": img_bucket,
                "key": img_key,
                "title": os.getenv("BENCHMARK_IMAGE_TITLE", "reference_image"),
                "content_type": os.getenv(
                    "BENCHMARK_IMAGE_CONTENT_TYPE", "image/jpeg"
                ),
                "inject_mode": "path",
            })
        if vid_bucket and vid_key:
            input_images.append({
                "bucket": vid_bucket,
                "key": vid_key,
                "title": os.getenv("BENCHMARK_VIDEO_TITLE", "reference_video"),
                "content_type": os.getenv(
                    "BENCHMARK_VIDEO_CONTENT_TYPE", "video/mp4"
                ),
                "inject_mode": "path",
            })
    else:
        # Legacy base64 mode (ETN_LoadImageBase64 benchmarks on other lanes).
        if img_bucket and img_key:
            input_images.append({"bucket": img_bucket, "key": img_key})

    lane = _normalized_benchmark_lane()
    bench_input: dict = {
        "workflow": workflow,
        "user_id": "bench",
        "generation_id": f"bench-{uuid.uuid4().hex}",
        "timeout": 300,
        "input_images": input_images,
    }
    req_gl = _BENCHMARK_ENV_LANE_TO_REQUEST_GENERATION_LANE.get(lane)
    if req_gl:
        bench_input["generation_lane"] = req_gl
    elif lane:
        bench_input["generation_lane"] = lane

    # App format; transform to workflow_json so backend receives correct format
    app_payload = {"input": bench_input}
    try:
        from .workflow_transform import transform_app_to_vast
    except ImportError:
        # calibrate_workload_timing.py does ``import worker`` with comfyui-json on sys.path only
        from workflow_transform import transform_app_to_vast

    return transform_app_to_vast(app_payload)


def _fallback_benchmark_payload() -> dict:
    """Fallback when no custom benchmark workflow; use stock Text2Image modifier."""
    inner: dict = {
        "request_id": f"bench-{random.randint(1000, 99999)}",
        "modifier": "Text2Image",
        "modifications": {
            "prompt": "a beautiful sunset over mountains, digital art, highly detailed",
            "width": 512,
            "height": 512,
            "steps": 20,
            "seed": random.randint(0, sys.maxsize),
        },
    }
    lane = _normalized_benchmark_lane()
    req_gl = _BENCHMARK_ENV_LANE_TO_REQUEST_GENERATION_LANE.get(lane)
    if req_gl:
        inner["generation_lane"] = req_gl
    elif lane:
        inner["generation_lane"] = lane
    return {"input": inner}


def request_parser(json_msg: dict) -> dict:
    """Transform app format to Vast format before forwarding to API wrapper."""
    try:
        from .workflow_transform import transform_app_to_vast
    except ImportError:
        from workflow_transform import transform_app_to_vast

    return transform_app_to_vast(json_msg)


def _normalize_lane_token(raw: str) -> str:
    return (raw or "").strip().upper().replace(" ", "_")


def _clamp_dynamic_vast_workload(value: float) -> float:
    lo = float(os.getenv("VAST_WORKLOAD_DYNAMIC_MIN", "1"))
    hi = float(os.getenv("VAST_WORKLOAD_DYNAMIC_MAX", "500000"))
    if value != value:  # NaN
        raise ValueError("vast_workload_units is NaN")
    return max(lo, min(hi, value))


def workload_calculator(payload: dict) -> float:
    """Declared load for Vast routing/scaling — must match bot SDK ``cost=`` for the same request.

    Valid sources (no global ``VAST_WORKLOAD_UNITS`` fallback):

    - Non-empty ``input.vast_workload_units`` (validated float), and/or
    - ``input.generation_lane`` in the known-lane set plus template ``VAST_WORKLOAD_UNITS_<LANE>``
      when the body does not carry ``vast_workload_units``.

    If the request omits both a resolvable lane+env pair and valid ``vast_workload_units``, raises
    ``ValueError`` (client error / invalid payload for this route).
    """
    inp = payload.get("input")
    if not isinstance(inp, dict):
        inp = {}

    lane = _normalize_lane_token(str(inp.get("generation_lane") or ""))
    raw_dyn = inp.get("vast_workload_units")
    dynamic_val: float | None = None
    if raw_dyn is not None and str(raw_dyn).strip() != "":
        try:
            dynamic_val = _clamp_dynamic_vast_workload(float(raw_dyn))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid vast_workload_units={raw_dyn!r}") from e

    if lane:
        if lane not in _KNOWN_WORKLOAD_LANES:
            raise ValueError(
                f"workload_calculator: unknown generation_lane={lane!r}; "
                f"expected one of {sorted(_KNOWN_WORKLOAD_LANES)}"
            )
        if dynamic_val is not None:
            return dynamic_val
        env_key = f"VAST_WORKLOAD_UNITS_{lane}"
        raw = os.getenv(env_key)
        if raw is None or str(raw).strip() == "":
            raise ValueError(
                f"workload_calculator: missing {env_key} for generation_lane={lane!r} "
                f"and no valid input.vast_workload_units; set template env or send vast_workload_units"
            )
        try:
            return float(raw)
        except ValueError as e:
            raise ValueError(f"Invalid {env_key}={raw!r}") from e

    if dynamic_val is not None:
        return dynamic_val
    raise ValueError(
        "workload_calculator: invalid request — input.generation_lane (with template "
        "VAST_WORKLOAD_UNITS_<LANE>) or input.vast_workload_units is required"
    )


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/generate/sync",
            allow_parallel_requests=False,
            max_queue_time=60.0,
            workload_calculator=workload_calculator,
            request_parser=request_parser,
            benchmark_config=BenchmarkConfig(
                generator=_get_benchmark_payload,
                runs=BENCHMARK_RUNS,
                concurrency=1,
            ),
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()
