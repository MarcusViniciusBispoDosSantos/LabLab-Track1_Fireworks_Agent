# FireRoute AI — Track 1 Accuracy v10

FireRoute AI is a Dockerized Track 1 agent for the AMD Developer Hackathon ACT II. It reads `/input/tasks.json`, solves each natural language task using Fireworks AI through the harness-provided environment, and writes `/output/results.json`.

## Why v10 exists

Previous builds improved the container but plateaued around 63.2% hidden accuracy. v10 changes the strategy again:

- Category-specific model calibration from `ALLOWED_MODELS`.
- Original prompt is still sent directly to the model.
- A self-check/correction pass is enabled for maximum correctness.
- Local shortcut solvers are disabled by default to avoid hidden benchmark mismatches.
- Stronger prompt rules for exact formats, code signatures, JSON, units, and labels.
- `linux/amd64` GHCR publish workflow is included.

## Official Track 1 contract

The final container must:

1. Read input from `/input/tasks.json`.
2. Write output to `/output/results.json`.
3. Read these variables from the runtime environment:
   - `FIREWORKS_API_KEY`
   - `FIREWORKS_BASE_URL`
   - `ALLOWED_MODELS`
4. Route all Fireworks calls through `FIREWORKS_BASE_URL`.
5. Avoid hardcoded model IDs and API keys.
6. Build/publish as `linux/amd64`.

## Docker image

Recommended image reference:

```bash
ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```

## Build manually

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest \
  --push \
  .
```

## Runtime configuration

Official judging injects the three Fireworks variables. Do not commit a real `.env` file.

v10 accuracy-first defaults:

```bash
ENABLE_LOCAL_FAST_PATHS=0
ENABLE_CATEGORY_CALIBRATION=1
CALIBRATION_MODEL_LIMIT=4
SELF_CHECK_MODE=all
MAX_WORKERS=2
MAX_RETRIES=6
REQUEST_TIMEOUT_SECONDS=25
```

## GitHub Actions

Run these workflows after pushing changes:

1. **Track 1 Online Check** — validates the Docker contract using a mock Fireworks server.
2. **Publish Docker Image to GHCR** — builds and pushes the public `linux/amd64` image.

After publishing, re-save the lablab.ai submission so the platform pulls the newest image.
