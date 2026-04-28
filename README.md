# batch-faceswap-tool-pyworker

PyWorker fork for the Batch Faceswap Tool. Drives the Tokyo Sage chained ComfyUI workflow on the WAN22_IV2V_FACESWAP_5090 Vast lane.

## S3 path contract

| Direction | Pattern |
|---|---|
| Input source video | `jobs/<job_id>/videos/<video_id>/source.mp4` |
| Input Nano Banana PNG | `jobs/<job_id>/videos/<video_id>/nano.png` |
| Output (written by worker) | `jobs/<job_id>/videos/<video_id>/output.mp4` |
| LogTail diagnostics | `diagnostics/<request_id>/log.txt` + `log.done` |

The worker derives the output key from the input source key by replacing `/source.mp4` → `/output.mp4`. See `workers/comfyui-json/workflow_transform.py` (`_output_key_for`) and the in-image `comfy_backend.py` (`process_generation`) — the latter now lives in `batch-faceswap-tool/docker-build/`.

## Image build

The serverless Docker image is built from `batch-faceswap-tool/docker-build/` — see that repo's `.github/workflows/build.yml`. PyWorker code is cloned into the running container at boot via `PYWORKER_REPO` / `PYWORKER_REF` env on the Vast template, so this repo doesn't need to ship its own image.
