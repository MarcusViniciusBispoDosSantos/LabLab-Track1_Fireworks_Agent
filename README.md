# FireRoute AI — Track 1 Accuracy v11

This is a reliability-first Track 1 project for the AMD Developer Hackathon ACT II.

## What it does

The container:

1. Reads tasks from `/input/tasks.json`
2. Solves each task using Fireworks AI through the harness-provided environment
3. Writes valid answers to `/output/results.json`
4. Exits with code `0` on success

It supports the 8 Track 1 categories:

- factual knowledge
- mathematical reasoning
- sentiment classification
- text summarization
- named entity recognition
- code debugging
- logical / deductive reasoning
- code generation

## Important v11 strategy

Previous versions plateaued around 57–63% accuracy. v11 is built to reduce the most likely causes:

- no category calibration calls that can waste runtime or choose badly
- no global self-check that can corrupt already-correct answers
- original hidden prompt is sent to Fireworks unchanged
- harness model order is trusted more strongly
- ensemble is used only for hard tasks: math, logic, debugging, and code generation
- conservative local fast paths are used only for simple high-confidence cases
- sentiment cleanup bug fixed: answers starting with `Mixed` no longer get changed to `Positive`
- all calls still go through `FIREWORKS_BASE_URL`

I cannot guarantee a specific hidden score because lablab.ai does not show failed prompts, but this version is designed to be more reliable than the v8/v9/v10 strategies.

## Required environment variables

The official judge injects these values at runtime:

```bash
FIREWORKS_API_KEY
FIREWORKS_BASE_URL
ALLOWED_MODELS
```

Do not commit a real `.env` file. The submitted image reads these variables from the environment.

## Local / CI mock test

The mock test does not test real model accuracy. It only confirms Docker contract compatibility.

```bash
bash scripts/test_with_mock.sh
```

## Build linux/amd64 image

The judging VM runs `linux/amd64`, so the image must include a linux/amd64 manifest.

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest \
  --push \
  .
```

## GitHub Actions

Use these workflows:

- `Track 1 Online Check` — verifies the container contract with a mock Fireworks server
- `Publish Docker Image to GHCR` — builds and pushes the `linux/amd64` image

After publishing, re-save the lablab.ai submission so the platform pulls the latest image.

## Submission image

```text
ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```
