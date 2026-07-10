# FireRoute AI — Track 1 Accuracy v13 Lite

FireRoute AI is a Dockerized Track 1 agent for the AMD Developer Hackathon ACT II. It reads `/input/tasks.json`, solves each task, and writes `/output/results.json`.

## Why v13 Lite

Previous versions improved accuracy but used too many calls or risky correction passes. v13 Lite is designed for **accuracy with a low token budget**:

- high-confidence local solvers for math, sentiment, common code templates, simple debugging, and assignment logic;
- exactly one concise Fireworks call for remaining tasks;
- original prompt is sent unchanged;
- no batch solving, no global verifier, no calibration calls;
- short system prompt and capped output tokens to target less than ~1400 recorded tokens on small hidden batches.

I cannot guarantee 85% because hidden prompts are not visible, but this version is built to avoid the failures seen in v10/v12 while keeping token usage close to the previous ~1300-token successful project.

## Required environment variables

The final judging harness injects these:

```bash
FIREWORKS_API_KEY
FIREWORKS_BASE_URL
ALLOWED_MODELS
```

The project does not hardcode keys or model IDs.

## Build linux/amd64 image

```bash
docker buildx build --platform linux/amd64 -t ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest --push .
```

## Run contract-style

```bash
docker run --rm \
  -e FIREWORKS_API_KEY=dummy \
  -e FIREWORKS_BASE_URL=http://127.0.0.1:8000/v1 \
  -e ALLOWED_MODELS=mock-model \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```

## Default accuracy/token settings

```bash
ENABLE_LOCAL_FAST_PATHS=1
HARD_SECOND_TRY=0
MAX_WORKERS=1
MODEL_ORDER=hybrid
```

If accuracy is still below threshold and tokens are acceptable, try `HARD_SECOND_TRY=1` in the Dockerfile/environment, but that may exceed 1400 tokens.
