"""
Transform app request format to Vast /generate/sync format.
Aligns with bot job contract: client sends workflow + S3 refs, worker downloads,
injects base64, patches, forwards to backend for execution, S3 upload.

Accepts: {workflow, input_images, user_id, generation_id, ...}
Produces: {request_id, workflow_json, run_subdir, user_id, generation_id,
          timeout, s3?: {...}} for backend.
"""

import base64
import copy
import logging
import os
import random
import re
import shutil
import uuid
from io import BytesIO
from pathlib import Path

logger = logging.getLogger("workflow_transform")

# Top-level keys the bot sends for backend timing (vastai-sdk replaces the whole body with
# request_parser output; only these are forwarded to avoid leaking arbitrary client fields).
_PASSTHROUGH_KEYS = ("_client_sent_at", "id")


def _merge_passthrough(out: dict, payload: dict) -> dict:
    """Copy timing/routing keys from the original client payload onto the parser output."""
    merged = dict(out)
    for k in _PASSTHROUGH_KEYS:
        if k in payload:
            merged[k] = payload[k]
    return merged


def _random_uint64_seed() -> int:
    """Unsigned 64-bit seed in [0, 2**64-1].

    Use for direct node inputs such as ``RandomNoise.noise_seed`` (Comfy ``min: 0``,
    ``max: 2**64-1``). Negative signed seeds caused ``value_smaller_than_min``.
    """
    return random.getrandbits(64)


def _random_uint32_seed() -> int:
    """Unsigned 32-bit seed in [0, 2**32-1].

    Some custom nodes (e.g. ``SeedVR2VideoUpscaler``) cap ``seed`` at ``2**32-1``;
    uint64 triggers Comfy ``value_bigger_than_max`` and fails prompt validation.
    """
    return random.getrandbits(32)


def _random_reserved_vram_seed() -> int:
    """ComfyUI-ReservedVRAM ``seed`` max is ``2**50`` (see node INPUT_TYPES); uint64 fails validation."""
    return random.randint(0, 2**50 - 1)


# class_type values whose ``seed`` / ``*_seed`` inputs must stay in uint32 range
_UINT32_SEED_CLASS_TYPES = frozenset({"SeedVR2VideoUpscaler"})
# class_type values whose ``seed`` is capped below uint64 (custom node validation)
_FIFTY_BIT_SEED_CLASS_TYPES = frozenset({"ReservedVRAMSetter"})


def _random_primitive_int_seed() -> int:
    """Seed range for ``PrimitiveInt`` / ``PrimitiveFloat`` used as workflow seeds.

    Easy-Use (and similar) primitives validate ``value`` with ``max: 2**63-1``; full
    uint64 triggers ``value_bigger_than_max`` and drops the output subgraph (no video).
    """
    return random.randint(0, 2**63 - 1)


def randomize_workflow_seeds(workflow: dict | None) -> None:
    """In-place: new random seeds/noise each call so repeat runs do not reuse ComfyUI caches.

    Covers integer ``seed`` / ``*_seed`` inputs, widget links ``[node_id, slot]`` into
    primitives, ``RandomNoise.noise_seed``, and ``PrimitiveInt`` nodes titled ``Seed``.
    """
    if not isinstance(workflow, dict):
        return

    linked_primitive_ids: set[str] = set()

    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        cls = node.get("class_type")
        if cls == "RandomNoise" and "noise_seed" in inputs:
            v = inputs["noise_seed"]
            if isinstance(v, int):
                inputs["noise_seed"] = _random_uint64_seed()
            elif isinstance(v, list) and v:
                ref = v[0]
                sid = str(ref) if ref is not None else ""
                if sid and sid in workflow:
                    linked_primitive_ids.add(sid)
            continue
        for key, val in list(inputs.items()):
            if key != "seed" and key != "noise_seed" and not key.endswith("_seed"):
                continue
            if isinstance(val, bool):
                continue
            if cls in _UINT32_SEED_CLASS_TYPES:
                if isinstance(val, int):
                    inputs[key] = _random_uint32_seed()
                elif isinstance(val, float):
                    inputs[key] = float(_random_uint32_seed())
                elif isinstance(val, list) and val:
                    ref = val[0]
                    sid = str(ref) if ref is not None else ""
                    if sid and sid in workflow:
                        linked_primitive_ids.add(sid)
                continue
            if cls in _FIFTY_BIT_SEED_CLASS_TYPES:
                if isinstance(val, int):
                    inputs[key] = _random_reserved_vram_seed()
                elif isinstance(val, float):
                    inputs[key] = float(_random_reserved_vram_seed())
                elif isinstance(val, list) and val:
                    ref = val[0]
                    sid = str(ref) if ref is not None else ""
                    if sid and sid in workflow:
                        linked_primitive_ids.add(sid)
                continue
            if isinstance(val, int):
                inputs[key] = _random_uint64_seed()
            elif isinstance(val, float):
                inputs[key] = float(_random_uint64_seed())
            elif isinstance(val, list) and val:
                ref = val[0]
                sid = str(ref) if ref is not None else ""
                if sid and sid in workflow:
                    linked_primitive_ids.add(sid)

    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        meta = node.get("_meta") or {}
        title = (meta.get("title") or "").strip().lower()
        if title == "seed":
            linked_primitive_ids.add(str(nid))

    for sid in linked_primitive_ids:
        tgt = workflow.get(sid)
        if not isinstance(tgt, dict):
            continue
        cls = tgt.get("class_type")
        tin = tgt.setdefault("inputs", {})
        if cls == "PrimitiveInt" and "value" in tin:
            tin["value"] = _random_primitive_int_seed()
        elif cls == "PrimitiveFloat" and "value" in tin:
            tin["value"] = float(_random_primitive_int_seed())


def _validate_base64_image(b64: str, node_id: str) -> None:
    """Validate that base64 decodes to a loadable image. Raises RuntimeError if invalid."""
    try:
        raw = base64.b64decode(b64)
        if not raw:
            raise RuntimeError(f"ETN node {node_id}: base64 decodes to empty bytes")
        from PIL import Image

        Image.open(BytesIO(raw)).verify()
    except Exception as e:
        raise RuntimeError(
            f"ETN node {node_id}: invalid image data (S3 download or format issue): {e}"
        ) from e


# S3: Vast uses S3_BUCKET_NAME; we also accept S3_BUCKET for compatibility
def _get_s3_config() -> dict | None:
    bucket = os.getenv("S3_BUCKET_NAME") or os.getenv("S3_BUCKET")
    access = os.getenv("S3_ACCESS_KEY_ID")
    secret = os.getenv("S3_SECRET_ACCESS_KEY")
    endpoint = os.getenv("S3_ENDPOINT_URL")
    region = os.getenv("S3_REGION", "us-east-1")
    if not all([bucket, access, secret, endpoint]):
        return None
    return {
        "bucket": bucket,
        "access_key_id": access,
        "secret_access_key": secret,
        "endpoint_url": endpoint,
        "region": region,
    }


def _safe_component(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s[:80] if s else "x"


def _make_job_subdir(user_id: str, generation_id: str, job_id: str | None) -> str:
    prefix = os.getenv("JOB_PREFIX", "vast")
    rid_src = (job_id or "").strip()
    rid = _safe_component(rid_src[:12]) if rid_src else uuid.uuid4().hex[:12]
    return (
        f"{prefix}/u{_safe_component(user_id)}/g{_safe_component(generation_id)}/{rid}"
    )


def _input_entry_kind(entry: dict, index: int) -> str:
    """``image`` (default) or ``audio`` from explicit ``kind`` or file extension."""
    raw = (entry.get("kind") or "").strip().lower()
    if raw in ("audio", "image"):
        return raw
    key = str(entry.get("key") or "")
    ext = Path(key).suffix.lower()
    if ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"):
        return "audio"
    return "image"


def _input_inject_mode(entry: dict) -> str:
    """Return ``"path"`` if entry sets ``inject_mode="path"``, else ``"base64"`` (default).

    ``base64`` is the legacy/production mode: downloaded bytes are base64-encoded into
    ``ETN_LoadImageBase64`` node widgets. ``path`` is used by workflows that keep
    ``LoadImage`` / ``VHS_LoadVideo`` nodes — the file is written to
    ``/app/input/<run_subdir>/`` and the node's path widget is rewritten.
    """
    mode = str(entry.get("inject_mode") or "").strip().lower()
    return "path" if mode == "path" else "base64"


def _input_asset_type(entry: dict) -> str:
    """Return ``"image"`` / ``"video"`` / ``"audio"`` for path-mode entries.

    Priority: explicit ``content_type`` → file extension → ``"image"`` fallback.
    """
    ct = str(entry.get("content_type") or "").strip().lower()
    if ct.startswith("image/"):
        return "image"
    if ct.startswith("video/"):
        return "video"
    if ct.startswith("audio/"):
        return "audio"
    ext = Path(str(entry.get("key") or "")).suffix.lower()
    if ext in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
        return "video"
    if ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"):
        return "audio"
    return "image"


def _download_input_images(
    input_images: list[dict],
    input_dir: Path,
) -> tuple[
    list[tuple[str, Path]],
    list[tuple[str, Path]],
    list[tuple[str, Path, str]],
]:
    """Download S3 objects from ``input_images``. Returns ``(image_rows, audio_rows, path_rows)``.

    - ``image_rows`` / ``audio_rows`` — base64-mode entries, consumed by
      ``ETN_LoadImageBase64`` injection (legacy production behaviour).
    - ``path_rows`` — entries with ``inject_mode="path"``; each row is
      ``(title, local_scratch_path, asset_type)`` where ``asset_type`` is
      ``"image"``, ``"video"``, or ``"audio"``. The caller copies these into
      ``/app/input/<run_subdir>/`` and rewrites ``LoadImage`` / ``VHS_LoadVideo``
      widgets.
    """
    import boto3

    try:
        from .s3_boto_resilience import (
            S3_IO_SEM,
            build_s3_boto_config,
            download_file_with_retry,
        )
    except ImportError:
        from s3_boto_resilience import (
            S3_IO_SEM,
            build_s3_boto_config,
            download_file_with_retry,
        )

    cfg = _get_s3_config()
    if not cfg:
        raise RuntimeError("S3 not configured for input_images download")
    client = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint_url"],
        aws_access_key_id=cfg["access_key_id"],
        aws_secret_access_key=cfg["secret_access_key"],
        region_name=cfg["region"],
        config=build_s3_boto_config(signature_version="s3v4"),
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    images: list[tuple[str, Path]] = []
    audios: list[tuple[str, Path]] = []
    paths: list[tuple[str, Path, str]] = []
    for i, entry in enumerate(input_images):
        bucket = entry.get("bucket")
        key = entry.get("key")
        title = (entry.get("title") or "").strip()
        if not bucket or not key:
            raise RuntimeError(f"input_images[{i}] missing bucket or key")

        if _input_inject_mode(entry) == "path":
            asset_type = _input_asset_type(entry)
            ext = Path(key).suffix or {
                "image": ".png",
                "video": ".mp4",
                "audio": ".wav",
            }.get(asset_type, ".bin")
            safe_name = _safe_component(title) if title else f"input_{i}"
            local_path = (input_dir / f"{safe_name}{ext}").resolve()
            if not str(local_path).startswith(str(input_dir.resolve())):
                raise RuntimeError("Invalid input path traversal")
            with S3_IO_SEM:
                download_file_with_retry(client, bucket, key, str(local_path))
            logger.info(
                "Downloaded %s/%s -> %s (path-mode %s)",
                bucket, key, local_path, asset_type,
            )
            paths.append((title, local_path, asset_type))
            continue

        # base64 mode (legacy / production)
        kind = _input_entry_kind(entry, i)
        ext = Path(key).suffix or (".wav" if kind == "audio" else ".jpg")
        safe_name = _safe_component(title) if title else f"input_{i}"
        local_path = (input_dir / f"{safe_name}{ext}").resolve()
        if not str(local_path).startswith(str(input_dir.resolve())):
            raise RuntimeError("Invalid input path traversal")
        with S3_IO_SEM:
            download_file_with_retry(client, bucket, key, str(local_path))
        logger.info("Downloaded %s/%s -> %s (%s)", bucket, key, local_path, kind)
        row = (title, local_path)
        if kind == "audio":
            audios.append(row)
        else:
            images.append(row)
    return images, audios, paths


def _comfy_input_root() -> Path:
    return Path(
        os.getenv("COMFY_INPUT_ROOT")
        or os.getenv("COMFY_INPUT_DIR")
        or "/app/input"
    )


def _replace_first_quoted_segment(prompt_template: str, spoken_text: str) -> str:
    """Replace text inside first double-quoted segment, keep surrounding template.

    Example:
    ... Exact line:
    "old text"
    ->
    ... Exact line:
    "new text"
    """
    start = prompt_template.find('"')
    if start < 0:
        return spoken_text
    end = prompt_template.find('"', start + 1)
    if end < 0:
        return spoken_text
    return f"{prompt_template[: start + 1]}{spoken_text}{prompt_template[end:]}"


def _patch_workflow(
    workflow: dict,
    run_subdir: str,
    job_input: dict,
    downloaded_images: list[tuple[str, Path]],
    path_assets: list[tuple[str, Path, str]] | None = None,
) -> dict:
    """Patch workflow: sageattn override, VHS_VideoCombine prefix, ETN_LoadImageBase64
    injection (base64 mode), LoadImage / VHS_LoadVideo injection (path mode), prompt.
    """
    wf = copy.deepcopy(workflow)
    # sageattn override: none (runtime patches it per GPU)
    for node in wf.values():
        if (
            isinstance(node, dict)
            and (node.get("inputs") or {}).get("attention_override") == "sageattn"
        ):
            node.setdefault("inputs", {})["attention_override"] = "none"
    # VHS_VideoCombine: set filename_prefix to run_subdir/result so outputs land under job dir
    for node in wf.values():
        if isinstance(node, dict) and node.get("class_type") == "VHS_VideoCombine":
            node.setdefault("inputs", {})["filename_prefix"] = f"{run_subdir}/result"
            node.setdefault("inputs", {})["save_output"] = True
    # ETN_LoadImageBase64: inject downloaded images as base64
    etn_nodes: list[tuple[str, str, dict]] = []
    for nid, node in wf.items():
        if isinstance(node, dict) and node.get("class_type") == "ETN_LoadImageBase64":
            title = (node.get("_meta") or {}).get("title") or ""
            etn_nodes.append((nid, title.strip(), node))
    img_by_title = {t: p for t, p in downloaded_images if t}
    img_no_title = [p for t, p in downloaded_images if not t]
    for nid, ntitle, node in etn_nodes:
        local = img_by_title.get(ntitle) if ntitle else None
        if local is None and img_no_title:
            local = img_no_title.pop(0)
        if local and local.exists():
            b64 = base64.b64encode(local.read_bytes()).decode("utf-8")
            _validate_base64_image(b64, nid)
            node.setdefault("inputs", {})["image"] = b64
    if downloaded_images:
        for nid, _, node in etn_nodes:
            if not (node.get("inputs") or {}).get("image"):
                raise RuntimeError(f"Failed to inject image into ETN node {nid}")

    # Path-mode injection (LoadImage / VHS_LoadVideo).
    # Files are staged under /app/input/<run_subdir>/ so Comfy's loaders can resolve
    # them relative to the configured input root. No effect when path_assets is empty
    # — preserves production base64-only behaviour for lanes that never opt in.
    path_assets = path_assets or []
    if path_assets:
        comfy_input_root = _comfy_input_root().resolve()
        target_dir = (comfy_input_root / run_subdir).resolve()
        if not str(target_dir).startswith(str(comfy_input_root)):
            raise RuntimeError("Invalid comfy input path traversal")
        target_dir.mkdir(parents=True, exist_ok=True)

        staged: list[tuple[str, str, str]] = []  # (title, rel_path, asset_type)
        for title, local_path, asset_type in path_assets:
            if not local_path.exists():
                raise RuntimeError(
                    f"Path-mode asset missing at {local_path} (title={title!r})"
                )
            dest = (target_dir / local_path.name).resolve()
            if not str(dest).startswith(str(comfy_input_root)):
                raise RuntimeError("Invalid comfy input path traversal")
            shutil.copyfile(local_path, dest)
            rel_path = f"{run_subdir}/{local_path.name}"
            staged.append((title, rel_path, asset_type))
            logger.info(
                "Staged path-mode asset title=%r -> %s (%s)",
                title, dest, asset_type,
            )

        images_by_title = {t: p for t, p, k in staged if k == "image" and t}
        images_no_title = [p for t, p, k in staged if k == "image" and not t]
        videos_by_title = {t: p for t, p, k in staged if k == "video" and t}
        videos_no_title = [p for t, p, k in staged if k == "video" and not t]

        matched: set[str] = set()
        for nid, node in wf.items():
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type")
            title = ((node.get("_meta") or {}).get("title") or "").strip()
            if ct == "LoadImage":
                rel = images_by_title.get(title) if title else None
                if rel is None and images_no_title:
                    rel = images_no_title.pop(0)
                if rel is None:
                    continue
                node.setdefault("inputs", {})["image"] = rel
                matched.add(rel)
            elif ct == "VHS_LoadVideo":
                rel = videos_by_title.get(title) if title else None
                if rel is None and videos_no_title:
                    rel = videos_no_title.pop(0)
                if rel is None:
                    continue
                inputs = node.setdefault("inputs", {})
                inputs["video"] = rel
                inputs["frame_load_cap"] = 0
                matched.add(rel)

        unmatched = [
            (title, rel, kind) for title, rel, kind in staged if rel not in matched
        ]
        if unmatched:
            raise RuntimeError(
                f"Path-mode assets have no matching loader node: {unmatched}"
            )

    # Prompt injection
    prompt_title = (job_input.get("prompt_node_title") or "").strip()
    user_prompt = (job_input.get("user_prompt") or "").strip()
    if prompt_title and user_prompt:
        for node in wf.values():
            if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                if (
                    (node.get("_meta") or {}).get("title") or ""
                ).strip() == prompt_title:
                    cur_text = str((node.get("inputs") or {}).get("text") or "")
                    node.setdefault("inputs", {})["text"] = _replace_first_quoted_segment(
                        cur_text, user_prompt
                    )
                    break
    return wf


def transform_app_to_vast(payload: dict) -> dict:
    """
    Transform app format to Vast API wrapper format.
    If already in Vast format (workflow_json present), return as-is.
    """
    inp = payload.get("input", payload)
    if isinstance(inp, dict) and "workflow_json" in inp:
        return _merge_passthrough(payload, payload)
    workflow = inp.get("workflow") if isinstance(inp, dict) else None
    if not isinstance(workflow, dict):
        return _merge_passthrough(payload, payload)
    input_images = (inp.get("input_images") or []) if isinstance(inp, dict) else []
    user_id = str(inp.get("user_id") or "")
    generation_id = str(inp.get("generation_id") or "")
    job_id = str(payload.get("id") or inp.get("request_id") or "")
    request_id = job_id or str(uuid.uuid4())
    run_subdir = _make_job_subdir(user_id, generation_id, job_id)
    job_input = dict(inp) if isinstance(inp, dict) else {}
    downloaded_images: list[tuple[str, Path]] = []
    path_assets: list[tuple[str, Path, str]] = []
    download_entries: list[dict] = []
    for e in input_images:
        if isinstance(e, dict):
            download_entries.append(e)

    scratch_dir: Path | None = None
    if download_entries:
        scratch_dir = Path("/tmp/input") / run_subdir
        downloaded_images, _audios, path_assets = _download_input_images(
            download_entries, scratch_dir
        )

    try:
        patched = _patch_workflow(
            workflow,
            run_subdir,
            job_input,
            downloaded_images,
            path_assets=path_assets,
        )
        randomize_workflow_seeds(patched)
        s3_cfg = _get_s3_config()
        s3_block = {}
        if s3_cfg:
            s3_block = {
                "access_key_id": s3_cfg["access_key_id"],
                "secret_access_key": s3_cfg["secret_access_key"],
                "endpoint_url": s3_cfg["endpoint_url"],
                "bucket_name": s3_cfg["bucket"],
                "region": s3_cfg["region"],
            }
        out_input: dict = {
            "request_id": request_id,
            "workflow_json": patched,
            "run_subdir": run_subdir,
            "user_id": user_id,
            "generation_id": generation_id,
            "timeout": int(job_input.get("timeout", 600)),
        }
        gl = (job_input.get("generation_lane") or "").strip()
        if gl:
            out_input["generation_lane"] = gl
        vwu = job_input.get("vast_workload_units")
        if vwu is not None and str(vwu).strip() != "":
            try:
                out_input["vast_workload_units"] = float(vwu)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid vast_workload_units={vwu!r}") from e
        if s3_block:
            out_input["s3"] = s3_block
        s3_bucket = (job_input.get("s3_bucket") or "").strip() or (
            s3_cfg["bucket"] if s3_cfg else ""
        )
        if s3_bucket:
            out_input["s3_bucket"] = s3_bucket
        return _merge_passthrough({"input": out_input}, payload)
    finally:
        if scratch_dir is not None:
            _cleanup_worker_s3_scratch(scratch_dir)


def _cleanup_worker_s3_scratch(scratch_dir: Path) -> None:
    """Remove per-job directory under ``/tmp/input`` after S3 downloads are inlined / copied."""
    try:
        base = Path("/tmp/input").resolve()
        job = scratch_dir.resolve()
        if not str(job).startswith(str(base)):
            logger.warning("Skip scratch cleanup (path outside /tmp/input): %s", scratch_dir)
            return
        if job.exists():
            shutil.rmtree(job, ignore_errors=True)
            logger.info("Cleaned pyworker S3 download scratch: %s", job)
    except Exception as e:
        logger.warning("Failed to cleanup pyworker scratch %s: %s", scratch_dir, e)
