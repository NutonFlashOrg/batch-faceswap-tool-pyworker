#!/usr/bin/env python3
"""
Vast workload calibration on ONE reference GPU (boot or manual).

1) **Light benchmark** series: uses ``BENCHMARK_GENERATION_LANE`` + S3 (same as
   ``calibrate_workload_timing.py``). Run count: ``CALIBRATION_RUNS`` / ``--bench-runs``.
2) **Prod-shaped** series: one app-format JSON from the manifest for the **matching**
   bot ``generation_lane`` only (see mapping below). Run count: ``CALIBRATION_PROD_RUNS`` /
   ``--prod-runs`` (defaults smaller than bench, e.g. 5).
3) Prints ``W`` and ``VAST_WORKLOAD_UNITS_<lane>`` (``lane`` = manifest key = bot ``generation_lane``).

**BENCHMARK_GENERATION_LANE → manifest key** (must exist in manifest JSON):

- WAN22_I2V_LONG_5090 → WAN22_I2V_LONG_5090
- WAN22_IV2V_FACESWAP_5090 → WAN22_IV2V_FACESWAP_5090
- ZIMAGE_TURBO_I2I_5090 → ZIMAGE_TURBO_I2I_5090
- FLUX_S2_I2I_5090 → FLUX_S2_I2I_5090

If ``BENCHMARK_GENERATION_LANE`` is **unset**, runs **all** manifest entries (handy for
one-off local runs). Vast benchmark templates always set the lane → **one** prod JSON.
Use ``--all-manifest-lanes`` to time every manifest entry even when the env var is set.

Manifest example (JSON file):
  {
    "WAN22_I2V_LONG_5090": "wan22-i2v-long_app.json",
    "FLUX_S2_I2I_5090": "flux-s2-i2i_app.json"
  }

Each prod file: {"input": {"workflow": {...}, "user_id", "generation_id", "input_images", ...}}
(bot-shaped). ``input_images`` may use ``{"from_env_benchmark_image": true, "title": "…"}``;
resolved from ``BENCHMARK_IMAGE_*`` / ``S3_*``.

Requires: backend on CALIBRATE_BACKEND_URL; same env as worker for benchmark + S3.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    w = k - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def _comfyui_json_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    d = root / "workers" / "comfyui-json"
    if not d.is_dir():
        raise RuntimeError(f"Expected PyWorker tree at {d}")
    return d


def _ensure_comfyui_path() -> None:
    d = _comfyui_json_dir()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))


def _import_benchmark_payload_builder():
    _ensure_comfyui_path()
    import worker as comfy_worker  # noqa: E402

    return comfy_worker._get_benchmark_payload


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _post_generate_sync(
    url: str, payload: dict, *, timeout: float, insecure: bool
) -> tuple[int, dict | None, str]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    ctx = None
    if insecure:
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8")
            code = resp.getcode()
            try:
                return code, json.loads(raw) if raw else None, raw[:2000]
            except json.JSONDecodeError:
                return code, None, raw[:2000]
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw) if raw else None, raw[:2000]
        except json.JSONDecodeError:
            return e.code, None, raw[:2000]


def _response_ok(data: dict | None) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("success") is False:
        return False
    return True


def _run_series(
    *,
    url: str,
    label: str,
    build_payload,
    runs: int,
    warmup: int,
    timeout: float,
    insecure: bool,
) -> list[float]:
    times: list[float] = []
    for i in range(warmup + runs):
        payload = build_payload()
        t0 = time.perf_counter()
        code, data, _snippet = _post_generate_sync(
            url, payload, timeout=timeout, insecure=insecure
        )
        elapsed = time.perf_counter() - t0
        ok = code == 200 and _response_ok(data)
        phase = "warmup" if i < warmup else "timed"
        print(
            f"{label} {phase} {i + 1}/{warmup + runs}: {elapsed:.2f}s http={code} ok={ok}"
        )
        if not ok:
            err = (
                (data or {}).get("error", "request failed")
                if isinstance(data, dict)
                else "bad response"
            )
            raise RuntimeError(f"{label} failed: {err}")
        if i >= warmup:
            times.append(elapsed)
    return times


def _normalize_lane(s: str) -> str:
    return (s or "").strip().upper().replace(" ", "_")


# BENCHMARK_GENERATION_LANE IS the manifest key for undresser — no remapping needed.
_BENCH_LANE_TO_PROD_MANIFEST_KEY: dict[str, str] = {
    "WAN22_I2V_LONG_5090": "WAN22_I2V_LONG_5090",
    "WAN22_IV2V_FACESWAP_5090": "WAN22_IV2V_FACESWAP_5090",
    "ZIMAGE_TURBO_I2I_5090": "ZIMAGE_TURBO_I2I_5090",
    "FLUX_S2_I2I_5090": "FLUX_S2_I2I_5090",
}


def _manifest_key_for_benchmark_lane(bench_lane_raw: str) -> str | None:
    """Resolve which manifest entry to use for prod timing (must match baked JSON keys)."""
    b = _normalize_lane(bench_lane_raw)
    if not b:
        return None
    if b in _BENCH_LANE_TO_PROD_MANIFEST_KEY:
        return _BENCH_LANE_TO_PROD_MANIFEST_KEY[b]
    return None


def _hydrate_benchmark_input_images(inp: dict) -> None:
    """Replace ``from_env_benchmark_image`` entries with real bucket/key from env."""
    imgs = inp.get("input_images")
    if not isinstance(imgs, list) or not imgs:
        return
    bucket = (
        os.getenv("BENCHMARK_IMAGE_BUCKET")
        or os.getenv("S3_BUCKET")
        or os.getenv("S3_BUCKET_NAME")
        or ""
    ).strip()
    key = (os.getenv("BENCHMARK_IMAGE_KEY") or "").strip()
    out: list[dict] = []
    for i, e in enumerate(imgs):
        if not isinstance(e, dict):
            out.append(e)
            continue
        if e.get("from_env_benchmark_image"):
            if not bucket or not key:
                raise RuntimeError(
                    "Calibration JSON uses from_env_benchmark_image but BENCHMARK_IMAGE_KEY "
                    "and S3 bucket (BENCHMARK_IMAGE_BUCKET or S3_BUCKET or S3_BUCKET_NAME) are not set"
                )
            ne = {"bucket": bucket, "key": key}
            t = (e.get("title") or "").strip()
            if t:
                ne["title"] = t
            out.append(ne)
        else:
            if not e.get("bucket") or not e.get("key"):
                raise RuntimeError(
                    f"input_images[{i}] missing bucket/key and not from_env_benchmark_image"
                )
            out.append(e)
    inp["input_images"] = out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Vast workload calibration: light bench + matching prod JSON → VAST_WORKLOAD_UNITS_<LANE>"
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON: {GENERATION_LANE: path/to/app_format.json, ...}",
    )
    p.add_argument(
        "--backend-url",
        default=os.getenv(
            "CALIBRATE_BACKEND_URL", "http://127.0.0.1:8189/generate/sync"
        ),
        help="POST target (default: local backend)",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Timed **benchmark** iterations (alias for --bench-runs; default env CALIBRATION_RUNS)",
    )
    p.add_argument(
        "--bench-runs",
        type=int,
        default=None,
        help="Timed benchmark iterations after warmup (default: CALIBRATION_BENCH_RUNS or CALIBRATION_RUNS or 30)",
    )
    p.add_argument(
        "--prod-runs",
        type=int,
        default=None,
        help="Timed prod iterations after prod warmup (default: CALIBRATION_PROD_RUNS or 5)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Benchmark warmup iterations (default CALIBRATION_WARMUP or 1)",
    )
    p.add_argument(
        "--prod-warmup",
        type=int,
        default=None,
        help="Prod warmup iterations (default CALIBRATION_PROD_WARMUP or 0)",
    )
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--insecure", action="store_true")
    p.add_argument(
        "--baseline",
        type=float,
        default=100.0,
        help="W_bench in W_lane = baseline * T_prod / T_bench (default 100)",
    )
    p.add_argument(
        "--calibration-lane",
        type=str,
        default="",
        help="Override BENCHMARK_GENERATION_LANE for choosing manifest prod entry",
    )
    p.add_argument(
        "--all-manifest-lanes",
        action="store_true",
        help="Run prod series for every manifest entry (ignores template lane filter)",
    )
    args = p.parse_args()

    bench_runs = args.bench_runs if args.bench_runs is not None else args.runs
    if bench_runs is None:
        bench_runs = int(
            os.getenv("CALIBRATION_BENCH_RUNS")
            or os.getenv("CALIBRATION_RUNS")
            or "30"
        )
    prod_runs = args.prod_runs
    if prod_runs is None:
        prod_runs = int(os.getenv("CALIBRATION_PROD_RUNS") or "5")
    bench_warmup = args.warmup
    if bench_warmup is None:
        bench_warmup = int(os.getenv("CALIBRATION_WARMUP") or "1")
    prod_warmup = args.prod_warmup
    if prod_warmup is None:
        prod_warmup = int(os.getenv("CALIBRATION_PROD_WARMUP") or "0")

    if not args.manifest.is_file():
        print(f"ERROR: --manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest_raw = _load_json(args.manifest)
    if not isinstance(manifest_raw, dict):
        print("ERROR: manifest must be a JSON object", file=sys.stderr)
        return 2

    lanes_paths: list[tuple[str, Path]] = []
    for lane_key, rel in manifest_raw.items():
        lane = _normalize_lane(str(lane_key))
        if not lane:
            continue
        path = Path(str(rel)).expanduser()
        if not path.is_absolute():
            path = (args.manifest.parent / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            print(f"ERROR: missing file for lane {lane}: {path}", file=sys.stderr)
            return 2
        lanes_paths.append((lane, path))

    if not lanes_paths:
        print("ERROR: manifest has no lane → path entries", file=sys.stderr)
        return 2

    lanes_paths.sort(key=lambda x: x[0])

    bench_lane_hint = (args.calibration_lane or os.getenv("BENCHMARK_GENERATION_LANE") or "").strip()
    manifest_keys = {lane for lane, _ in lanes_paths}

    if args.all_manifest_lanes:
        print(
            "=== Calibration mode: ALL manifest lanes (--all-manifest-lanes) ===",
            flush=True,
        )
    elif bench_lane_hint:
        mk = _manifest_key_for_benchmark_lane(bench_lane_hint)
        if not mk:
            print(
                f"ERROR: unknown BENCHMARK_GENERATION_LANE / --calibration-lane={bench_lane_hint!r}; "
                f"expected one of {sorted(_BENCH_LANE_TO_PROD_MANIFEST_KEY)} or use --all-manifest-lanes",
                file=sys.stderr,
            )
            return 2
        if mk not in manifest_keys:
            print(
                f"ERROR: manifest has no entry for resolved prod lane {mk!r} "
                f"(from template lane {bench_lane_hint!r}); manifest has: {sorted(manifest_keys)}",
                file=sys.stderr,
            )
            return 2
        lanes_paths = [(lane, path) for lane, path in lanes_paths if lane == mk]
        print(
            f"=== Calibration mode: single lane template={_normalize_lane(bench_lane_hint)!r} "
            f"→ manifest[{mk!r}] (bench_runs={bench_runs} warmup={bench_warmup}, "
            f"prod_runs={prod_runs} prod_warmup={prod_warmup}) ===",
            flush=True,
        )
    else:
        print(
            "=== Calibration mode: ALL manifest lanes (no BENCHMARK_GENERATION_LANE set) ===",
            flush=True,
        )

    get_bench = _import_benchmark_payload_builder()

    def build_bench():
        return get_bench()

    print(
        "=== Benchmark series (env BENCHMARK_GENERATION_LANE + S3) ===",
        flush=True,
    )
    bench_times = _run_series(
        url=args.backend_url,
        label="bench",
        build_payload=build_bench,
        runs=bench_runs,
        warmup=bench_warmup,
        timeout=args.timeout,
        insecure=args.insecure,
    )
    b50 = float(_percentile(bench_times, 50))
    b80 = float(_percentile(bench_times, 80))
    print(
        f"\nT_bench: n={len(bench_times)}  p50={b50:.2f}s  p80={b80:.2f}s  "
        f"mean={sum(bench_times) / len(bench_times):.2f}s"
    )

    if b50 <= 0 or b50 != b50:  # nan
        print("ERROR: invalid T_bench p50", file=sys.stderr)
        return 1

    from workflow_transform import (  # noqa: E402
        randomize_workflow_seeds,
        transform_app_to_vast,
    )

    per_lane: dict[str, dict[str, float]] = {}
    suggested: dict[str, float] = {}

    for lane, jpath in lanes_paths:
        raw = _load_json(jpath)
        if not isinstance(raw, dict):
            print(f"ERROR: {jpath} must be a JSON object", file=sys.stderr)
            return 2
        inp = raw.get("input")
        if not isinstance(inp, dict):
            print(
                f"ERROR: {jpath} must have object 'input' (app format)",
                file=sys.stderr,
            )
            return 2
        template_input = copy.deepcopy(inp)

        def build_prod(
            *,
            _lane: str = lane,
            _template: dict = template_input,
        ):
            body_inp = copy.deepcopy(_template)
            body_inp["generation_lane"] = _lane
            _hydrate_benchmark_input_images(body_inp)
            wf = body_inp.get("workflow")
            if isinstance(wf, dict):
                randomize_workflow_seeds(wf)
            return transform_app_to_vast({"input": body_inp})

        print(f"\n=== Prod series lane={lane} file={jpath.name} ===", flush=True)
        prod_times = _run_series(
            url=args.backend_url,
            label=f"prod-{lane}",
            build_payload=build_prod,
            runs=prod_runs,
            warmup=prod_warmup,
            timeout=args.timeout,
            insecure=args.insecure,
        )
        p50 = float(_percentile(prod_times, 50))
        p80 = float(_percentile(prod_times, 80))
        w_lane = args.baseline * (p50 / b50)
        per_lane[lane] = {
            "t_prod_p50": p50,
            "t_prod_p80": p80,
            "suggested_workload": w_lane,
        }
        suggested[lane] = w_lane
        print(
            f"T_prod {lane}: p50={p50:.2f}s p80={p80:.2f}s  →  "
            f"VAST_WORKLOAD_UNITS_{lane} ≈ {args.baseline} * ({p50:.2f}/{b50:.2f}) = {w_lane:.1f}"
        )

    print("\n# Paste into bot + Vast template env (SDK cost= / scaling)\n")
    for lane in sorted(suggested.keys()):
        print(f"VAST_WORKLOAD_UNITS_{lane}={suggested[lane]:.1f}")

    summary = {
        "backend_url": args.backend_url,
        "bench_runs": bench_runs,
        "bench_warmup": bench_warmup,
        "prod_runs": prod_runs,
        "prod_warmup": prod_warmup,
        "baseline": args.baseline,
        "t_bench_seconds": {"p50": b50, "p80": b80},
        "lanes": per_lane,
        "vast_workload_units_suggested": {k: round(v, 2) for k, v in suggested.items()},
    }
    print("\nJSON summary:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
