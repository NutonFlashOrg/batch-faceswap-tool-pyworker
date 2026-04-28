# batch-faceswap-tool-pyworker

PyWorker fork for the Batch Faceswap Tool. Drives the Tokyo Sage chained ComfyUI workflow on the WAN22_IV2V_FACESWAP_5090 Vast lane.

## S3 path contract

| Direction | Pattern |
|---|---|
| Input source video | `jobs/<job_id>/videos/<video_id>/source.mp4` |
| Input Nano Banana PNG | `jobs/<job_id>/videos/<video_id>/nano.png` |
| Output (written by worker) | `jobs/<job_id>/videos/<video_id>/output.mp4` |
| LogTail diagnostics | `diagnostics/<request_id>/log.txt` + `log.done` |

The worker derives the output key from the input source key by replacing `/source.mp4` → `/output.mp4`. See `workers/comfyui-json/workflow_transform.py` (`_output_key_for`) and `comfy_backend.py` (`process_generation`).

## Image build

Pushed via the GitHub Action in `.github/workflows/build-image.yml`. Tags: `latest` for `main`, plus the commit SHA.
