#!/usr/bin/env python3
"""
Time benchmark (and optional prod-shaped) /generate/sync calls on ONE GPU machine.

Use this instead of hand-recording Vast startup benchmarks across many instances:
  - Rent one instance with the same image + template env as production (S3, BENCHMARK_*).
  - Run:  python scripts/calibrate_workload_timing.py --runs 30
  - Script loops locally: build payload → POST backend → record wall seconds.
  - With ``--prod-p50-seconds``: p50/p80 + suggested ``VAST_WORKLOAD_UNITS_<generation_lane>``.
  - Each iteration deep-copies the workflow and randomizes seeds (Comfy cache busting).

Requires:
  - Backend listening (default http://127.0.0.1:8189), same as MODEL_SERVER_URL:MODEL_SERVER_PORT.
  - Env vars for the lane: BENCHMARK_GENERATION_LANE, BENCHMARK_IMAGE_*, S3_* (same as worker).

Does not start PyWorker; imports workers/comfyui-json/worker.py for _get_benchmark_payload only
(worker module must not call Worker.run() on import — see __main__ guard in worker.py).

For several prod lanes vs one T_bench, use scripts/calibrate_vast_workload_multi_lane.py (--manifest).
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


def _import_benchmark_payload_builder():
    root = Path(__file__).resolve().parent.parent
    comfy_dir = root / "workers" / "comfyui-json"
    if not comfy_dir.is_dir():
        raise RuntimeError(f"Expected PyWorker tree at {comfy_dir}")
    sys.path.insert(0, str(comfy_dir))
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


def main() -> int:
    p = argparse.ArgumentParser(
        description="Time /generate/sync for Vast workload calibration"
    )
    p.add_argument(
        "--backend-url",
        default=os.getenv(
            "CALIBRATE_BACKEND_URL", "http://127.0.0.1:8189/generate/sync"
        ),
        help="Full URL to POST (default: local backend)",
    )
    p.add_argument("--runs", type=int, default=30, help="Timed iterations after warmup")
    p.add_argument(
        "--warmup", type=int, default=1, help="Untimed runs first (JIT/cache)"
    )
    p.add_argument(
        "--timeout", type=float, default=3600.0, help="Per-request timeout seconds"
    )
    p.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verify (only if URL is https with self-signed)",
    )
    p.add_argument(
        "--baseline",
        type=float,
        default=100.0,
        help="B in suggested_load = B * T_prod / T_bench (default 100)",
    )
    p.add_argument(
        "--prod-p50-seconds",
        type=float,
        default=None,
        help="If set: print suggested VAST_WORKLOAD_UNITS using this as T_prod p50 (seconds)",
    )
    p.add_argument(
        "--prod-payload",
        type=Path,
        default=None,
        help='JSON file: app-format payload {"input": {workflow, user_id, generation_id, input_images, ...}}; '
        "also times a prod-shaped series and prints T_prod p50",
    )
    args = p.parse_args()

    get_bench = _import_benchmark_payload_builder()

    def build_bench():
        return get_bench()

    print(
        "=== Benchmark series (env BENCHMARK_GENERATION_LANE + S3 image) ===",
        flush=True,
    )
    bench_times = _run_series(
        url=args.backend_url,
        label="bench",
        build_payload=build_bench,
        runs=args.runs,
        warmup=args.warmup,
        timeout=args.timeout,
        insecure=args.insecure,
    )
    b50 = _percentile(bench_times, 50)
    b80 = _percentile(bench_times, 80)
    print(
        f"\nT_bench: n={len(bench_times)}  p50={b50:.2f}s  p80={b80:.2f}s  mean={sum(bench_times) / len(bench_times):.2f}s"
    )

    prod_p50: float | None = None
    prod_p80: float | None = None
    if args.prod_payload is not None:
        if not args.prod_payload.is_file():
            print(
                f"ERROR: --prod-payload not found: {args.prod_payload}", file=sys.stderr
            )
            return 2
        raw_template = _load_json(args.prod_payload)

        def build_prod():
            from workflow_transform import randomize_workflow_seeds, transform_app_to_vast

            body = copy.deepcopy(raw_template)
            inp = body.get("input")
            if isinstance(inp, dict):
                wf = inp.get("workflow")
                if isinstance(wf, dict):
                    randomize_workflow_seeds(wf)
            return transform_app_to_vast(body)

        print("\n=== Prod-shaped series (from --prod-payload file) ===", flush=True)
        prod_times = _run_series(
            url=args.backend_url,
            label="prod",
            build_payload=build_prod,
            runs=args.runs,
            warmup=args.warmup,
            timeout=args.timeout,
            insecure=args.insecure,
        )
        prod_p50 = float(_percentile(prod_times, 50))
        prod_p80 = float(_percentile(prod_times, 80))
        print(
            f"\nT_prod: n={len(prod_times)}  p50={prod_p50:.2f}s  p80={prod_p80:.2f}s  "
            f"mean={sum(prod_times) / len(prod_times):.2f}s"
        )

    if args.prod_p50_seconds is not None:
        t_prod = args.prod_p50_seconds
        suggested = args.baseline * (t_prod / b50)
        print(
            f"\nSuggested VAST_WORKLOAD_UNITS (from --prod-p50-seconds={t_prod}): "
            f"{args.baseline} * ({t_prod}/{b50:.2f}) ≈ {suggested:.1f}"
        )
    elif prod_p50 is not None:
        suggested = args.baseline * (prod_p50 / b50)
        print(
            f"\nSuggested VAST_WORKLOAD_UNITS (from measured T_prod p50): "
            f"{args.baseline} * ({prod_p50:.2f}/{b50:.2f}) ≈ {suggested:.1f}"
        )

    summary = {
        "backend_url": args.backend_url,
        "runs": args.runs,
        "warmup": args.warmup,
        "t_bench_seconds": {"p50": b50, "p80": b80},
    }
    if prod_p50 is not None and prod_p80 is not None:
        summary["t_prod_seconds"] = {"p50": prod_p50, "p80": prod_p80}
    print("\nJSON summary:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
