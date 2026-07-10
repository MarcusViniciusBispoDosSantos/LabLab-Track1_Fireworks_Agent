# FireRoute AI — Track 1 Accuracy v12

This is a Dockerized Track 1 agent for the AMD Developer Hackathon ACT II.

## What changed in v12

v12 is a correctness-first rebuild after the previous versions plateaued around 63%.

Key changes:

- Validated batch solving for the full `/input/tasks.json` array.
- Per-task fallback for any missing or invalid batch answer.
- Stronger model ranking from `ALLOWED_MODELS`.
- Robust retry handling for 429/5xx/timeout proxy errors.
- Original prompts are preserved; the agent does not hardcode hidden answers.
- Conservative deterministic solvers only for safe math, sentiment, simple logic, and common code tasks.
- `linux/amd64` GHCR publishing workflow included.

## Official Track 1 contract

The container:

1. Reads tasks from `/input/tasks.json`.
2. Writes results to `/output/results.json`.
3. Reads these environment variables from the judging harness:
   - `FIREWORKS_API_KEY`
   - `FIREWORKS_BASE_URL`
   - `ALLOWED_MODELS`
4. Sends all Fireworks calls through `FIREWORKS_BASE_URL`.
5. Does not hardcode model IDs.
6. Exits with code 0 on success.

## Docker image

Final image reference:

```bash
ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```

The image must be public and include a `linux/amd64` manifest.

## Online validation

Run:

```text
GitHub → Actions → Track 1 Online Check → Run workflow
```

Then publish:

```text
GitHub → Actions → Publish Docker Image to GHCR → Run workflow
```

After publishing, re-save the lablab.ai submission so it pulls the new image.

## Environment defaults

Accuracy-first defaults are set in the Dockerfile:

```env
BATCH_SOLVE=1
BATCH_CANDIDATES=1
ENSEMBLE_MODE=hard
ENABLE_LOCAL_FAST_PATHS=1
MAX_WORKERS=1
MAX_RETRIES=4
HARD_ENSEMBLE_SIZE=2
REQUEST_TIMEOUT_SECONDS=40
```

If v12 does not improve, try one controlled variation by editing Dockerfile:

```env
BATCH_SOLVE=0
```

Then republish and re-save. This isolates whether batch solving helps or hurts the hidden benchmark.
