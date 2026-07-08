# FireRoute AI

FireRoute AI is an accuracy-first, Dockerized Track 1 agent for the AMD Developer Hackathon ACT II. It reads tasks from `/input/tasks.json`, solves them through Fireworks AI using the judging harness environment, and writes valid answers to `/output/results.json`.

## Track 1 Contract

The container:

- reads `/input/tasks.json` on startup
- writes `/output/results.json` before exiting
- reads `FIREWORKS_API_KEY` from the environment
- reads `FIREWORKS_BASE_URL` from the environment
- reads `ALLOWED_MODELS` from the environment
- routes all Fireworks calls through `FIREWORKS_BASE_URL`
- does not hardcode API keys or final model IDs
- is built as `linux/amd64`

## Accuracy v4 Updates

This version is optimized after the official feedback: **the agent ran but did not pass the minimum accuracy threshold**.

v4 changes:

- avoids reasoning models by default because they can consume the output budget in hidden reasoning
- uses stronger non-reasoning model ranking from `ALLOWED_MODELS`
- adds a universal Track 1 prompt so a routing mistake is less damaging
- verifies only hard tasks by default: math, logic, code debugging, and code generation
- adds deterministic fast paths for common math, sentiment, code generation, debugging, NER, and simple assignment-logic patterns
- keeps output compact and removes `<think>...</think>` content from final answers
- includes GitHub Actions checks and GHCR publish workflow

## Required Capability Areas

FireRoute AI supports all Track 1 capability categories:

1. factual knowledge
2. mathematical reasoning
3. sentiment classification
4. text summarization
5. named entity recognition
6. code debugging
7. logical / deductive reasoning
8. code generation

## Input Format

`/input/tasks.json`

```json
[
  { "task_id": "t1", "prompt": "Summarise the following text in one sentence: ..." },
  { "task_id": "t2", "prompt": "Write a Python function called is_palindrome(text)." }
]
```

## Output Format

`/output/results.json`

```json
[
  { "task_id": "t1", "answer": "..." },
  { "task_id": "t2", "answer": "..." }
]
```

## Environment Variables

The official judging harness injects:

```bash
FIREWORKS_API_KEY
FIREWORKS_BASE_URL
ALLOWED_MODELS
```

Optional tuning variables:

```bash
VERIFY_MODE=hard                 # none | hard | all
ENABLE_LOCAL_FAST_PATHS=1        # deterministic solvers for obvious cases
ALLOW_THINKING_MODELS=0          # keep R1/reasoning models as fallback only
MAX_WORKERS=3
MAX_RETRIES=2
REQUEST_TIMEOUT_SECONDS=26
```

## Online Contract Test

Use GitHub Actions:

```text
Actions → Track 1 Online Check → Run workflow
```

This builds the image, runs a mock Fireworks-compatible harness, validates `/output/results.json`, and confirms the image works with env variables.

## Build and Publish linux/amd64 Image

The judging VM runs `linux/amd64`; the final image must include a `linux/amd64` manifest.

Use the included workflow:

```text
Actions → Publish Docker Image to GHCR → Run workflow
```

Expected public image:

```bash
ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```

Manual equivalent:

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest \
  --push \
  .
```

## lablab.ai Submission URLs

Public GitHub Repository:

```text
https://github.com/MarcusViniciusBispoDosSantos/LabLab-Track1_Fireworks_Agent
```

Application URL if the form requires an HTTPS URL:

```text
https://github.com/MarcusViniciusBispoDosSantos/LabLab-Track1_Fireworks_Agent
```

Docker image reference:

```text
ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```
