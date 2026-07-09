# FireRoute AI

**Accuracy v6** is the most accuracy-focused version of the Track 1 agent.

It was updated after the official feedback reported an accuracy of about **52.6%**. The main changes are:

- Hidden prompts are solved by the strongest allowed Fireworks model, not brittle regex shortcuts.
- `ENABLE_LOCAL_FAST_PATHS=0` is now the real Docker default.
- `MAX_WORKERS=1` reduces proxy/rate-limit failures during the hidden benchmark.
- `VERIFY_MODE=hard` verifies math, logic, code debugging, and code generation without wasting extra calls on easier categories.
- Explicit reasoning models are used only after stable instruction models, reducing final-answer truncation.
- `/no_think` and final-only instructions are added for reasoning-style models.
- Code, math, NER, sentiment, summary, and logic prompts were simplified and made stricter.

## Track 1 Contract

The container:

- reads `/input/tasks.json` on startup
- writes `/output/results.json` before exiting
- reads `FIREWORKS_API_KEY` from the environment
- reads `FIREWORKS_BASE_URL` from the environment
- reads `ALLOWED_MODELS` from the environment
- routes all Fireworks calls through `FIREWORKS_BASE_URL`
- does not hardcode API keys or final model IDs
- is built for `linux/amd64`

## Required Capability Areas

FireRoute AI supports all Track 1 categories:

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

Accuracy v6 defaults:

```bash
VERIFY_MODE=hard
ENABLE_LOCAL_FAST_PATHS=0
MAX_WORKERS=1
MAX_RETRIES=4
REQUEST_TIMEOUT_SECONDS=29
```

For emergency maximum accuracy on small batches, you may set:

```bash
VERIFY_MODE=all
USE_ENSEMBLE_FOR_CODE=1
```

But the default v6 settings are safer for the official 10-minute hidden evaluation.

## Online Contract Test

Use GitHub Actions:

```text
Actions → Track 1 Online Check → Run workflow
```

This builds the image, runs a mock Fireworks-compatible harness, validates `/output/results.json`, and confirms the image works with the required environment variables.

## Publish linux/amd64 Image

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
