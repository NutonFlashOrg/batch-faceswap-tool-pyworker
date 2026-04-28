# AGENTS.md

Cloud Code guidance for the `undresser-pyworker/` repository.

## Branch

Current working branch:
- `main`

## Role In Product

This repo owns the Python worker logic running beside the model backend in Vast serverless environments.

It is responsible for:
- request parsing
- workload calculation
- payload transformation
- calibration scripts
- worker bootstrap behavior

## Current Reality

For the current NudeLab generation stack, the most important worker family is:
- `workers/comfyui-json/`

Key files:
- `start_server.sh`
- `workers/comfyui-json/worker.py`
- `workers/comfyui-json/workflow_transform.py`
- `scripts/calibrate_vast_workload_multi_lane.py`

## Product Relationship

This repo is downstream of product decisions only through concrete workflow/runtime contracts.

That means:
- UI-level changes do not matter here unless they change request shape
- backend orchestration changes matter if they change payload fields
- docker-build changes matter if they change startup/runtime assumptions

## Practical Rule

If the app/backend introduces:
- character-aware generation payloads
- new dataset-generation workflow payloads
- different LoRA binding patterns

check whether the worker transform layer must explicitly support them.
