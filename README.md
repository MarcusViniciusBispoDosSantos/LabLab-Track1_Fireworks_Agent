# FireRoute AI — Track 1 Accuracy v7

FireRoute AI is a Dockerized Track 1 agent for the AMD Developer Hackathon ACT II. It reads tasks from `/input/tasks.json`, solves each natural-language prompt using the Fireworks AI runtime injected by the judging harness, and writes valid results to `/output/results.json`.

## Track 1 contract

The submitted container:

- Reads `/input/tasks.json` on startup.
- Writes `/output/results.json` before exiting.
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` from environment variables.
- Routes all Fireworks calls through `FIREWORKS_BASE_URL`.
- Does not hardcode API keys or model IDs.
- Builds as `linux/amd64`.

## Accuracy v7 changes

This version is designed to improve hidden benchmark accuracy after previous runs scored below the required threshold.

Key changes:

- Sends the original prompt directly to the model instead of wrapping it with internal classifier text.
- Uses conservative local solvers only for high-confidence easy cases.
- Uses an ensemble strategy for hard categories: math, logic, code debugging, and code generation.
- Selects diverse strong models from `ALLOWED_MODELS` when available.
- Uses a final-answer selector to choose or synthesize the best answer from multiple candidates.
- Preserves strict output formatting and removes common reasoning / chatty prefixes.

## Environment variables

For final judging, these are injected by the harness:

```bash
FIREWORKS_API_KEY
FIREWORKS_BASE_URL
ALLOWED_MODELS
```

Optional tuning variables:

```bash
ACCURACY_STRATEGY=ensemble_hard   # direct | verify_hard | ensemble_hard | ensemble_all
ENABLE_LOCAL_FAST_PATHS=1
MAX_WORKERS=2
REQUEST_TIMEOUT_SECONDS=28
MAX_RETRIES=3
```

Recommended final defaults are already set in the Dockerfile.

## Build and push

For manual Docker builds:

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest \
  --push \
  .
```

If you do not have Docker locally, use the included GitHub Actions workflow:

1. Push this repository to GitHub.
2. Run **Track 1 Online Check**.
3. Run **Publish Docker Image to GHCR**.
4. Re-save the lablab.ai submission so the platform pulls the latest image.

## Public image

```bash
ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```
