# FireRoute AI — Track 1 Accuracy v9 Target-85

FireRoute AI is a Dockerized Track 1 agent for the AMD Developer Hackathon ACT II.
It reads `/input/tasks.json`, solves each natural language task using Fireworks AI through the harness-provided environment, and writes `/output/results.json`.

## Why v9 changed

Previous versions ran correctly but reached only ~63.2% hidden accuracy. v9 changes the strategy:

- Uses the original benchmark prompt directly; no internal task wrapping.
- Runs a tiny model-calibration step to pick the strongest model from `ALLOWED_MODELS` in the actual harness.
- Uses high-confidence deterministic solvers only for simple exact cases.
- Avoids full verifier passes on every task because they can corrupt good answers or exceed runtime.
- Repairs only clearly invalid outputs such as malformed JSON or missing requested code signatures.
- Keeps output concise and preserves strict format requirements.

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

Official judging injects these values. Do not commit a real `.env` file.

```bash
FIREWORKS_API_KEY=<provided by harness>
FIREWORKS_BASE_URL=<provided by harness>
ALLOWED_MODELS=<provided by harness>
```

v9 defaults:

```bash
ENABLE_MODEL_CALIBRATION=1
CALIBRATION_MODEL_LIMIT=4
ENABLE_LOCAL_FAST_PATHS=1
RETRY_INVALID_OUTPUTS=1
MAX_WORKERS=3
MAX_RETRIES=5
REQUEST_TIMEOUT_SECONDS=26
```

## GitHub Actions

Run these workflows after pushing changes:

1. **Track 1 Online Check** — validates the Docker contract using a mock Fireworks server.
2. **Publish Docker Image to GHCR** — builds and pushes the public `linux/amd64` image.

After publishing, re-save the lablab.ai submission so the platform pulls the newest image.
