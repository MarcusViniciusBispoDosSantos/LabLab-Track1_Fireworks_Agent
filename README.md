# FireRoute AI

FireRoute AI is an accuracy-first, Dockerized Track 1 agent for the AMD Developer Hackathon ACT II. It reads natural language tasks from `/input/tasks.json`, solves each prompt using Fireworks AI through the hackathon harness, and writes valid answers to `/output/results.json`.

## Track 1 Contract

The submitted container:

- reads `/input/tasks.json` on startup
- writes `/output/results.json` before exiting
- reads `FIREWORKS_API_KEY` from the environment
- reads `FIREWORKS_BASE_URL` from the environment
- reads `ALLOWED_MODELS` from the environment
- sends all Fireworks calls through `FIREWORKS_BASE_URL`
- does not hardcode API keys or final model IDs
- builds as `linux/amd64`

## Accuracy v3 Updates

This version is optimized after a minimum-accuracy-threshold failure.

Improvements:

- stronger task routing for all 8 Track 1 categories
- accuracy-first system prompts
- verification enabled for all task types by default through `VERIFY_MODE=all`
- better model ranking from the injected `ALLOWED_MODELS`
- safe local fast paths for obvious arithmetic, sentiment, and simple debug cases
- improved parsing of Fireworks/OpenAI-compatible responses
- stronger cleanup of reasoning-model `<think>` output
- GitHub Actions contract validation and GHCR publishing workflow

## Required Capability Areas

FireRoute AI supports:

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

The official judging harness injects these values at runtime:

```bash
FIREWORKS_API_KEY
FIREWORKS_BASE_URL
ALLOWED_MODELS
```

Optional tuning variables:

```bash
VERIFY_MODE=all                 # all | hard | none
ENABLE_LOCAL_FAST_PATHS=1       # safe deterministic solvers for obvious cases
MAX_WORKERS=2
MAX_RETRIES=2
REQUEST_TIMEOUT_SECONDS=28
```

## Local Mock Test

This test does not require a real Fireworks API key. It confirms the Docker contract using a mock Fireworks-compatible server.

```bash
bash scripts/test_with_mock.sh track1-fireworks-agent:local
```

It confirms that the image:

- builds successfully
- runs as a container
- reads `/input/tasks.json`
- writes `/output/results.json`
- produces valid JSON
- uses the environment variables

## Build for linux/amd64

The judging VM runs `linux/amd64`. The final image must include a `linux/amd64` manifest.

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest \
  --push \
  .
```

## GitHub Actions

Two workflows are included:

```text
.github/workflows/track1-online-check.yml
.github/workflows/publish-docker-ghcr.yml
```

Use **Track 1 Online Check** to validate the Docker contract online.

Use **Publish Docker Image to GHCR** to build and push the public `linux/amd64` image.

Final image reference:

```text
ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```

## lablab.ai Submission Values

**Project Title**

```text
FireRoute AI
```

**Public GitHub Repository**

```text
https://github.com/MarcusViniciusBispoDosSantos/LabLab-Track1_Fireworks_Agent
```

**Application URL**

```text
https://github.com/MarcusViniciusBispoDosSantos/LabLab-Track1_Fireworks_Agent
```

**Docker Image**

```text
ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```

## Long Description

FireRoute AI accuracy v3.1 is a Dockerized general-purpose AI agent built for Track 1 of the AMD Developer Hackathon ACT II. It processes batches of natural language tasks by reading `/input/tasks.json`, solving each prompt, and writing valid answers to `/output/results.json` before the container exits.

The agent is designed to handle all required Track 1 capability areas: factual knowledge, mathematical reasoning, sentiment classification, text summarization, named entity recognition, code debugging, logical reasoning, and code generation. Its architecture uses lightweight task routing, task-specific prompting, verification, and dynamic model selection from the harness-provided `ALLOWED_MODELS` to improve correctness.

FireRoute AI follows the official Fireworks AI runtime contract. It does not hardcode API keys or final model IDs. Instead, it reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` from the judging harness environment and routes all model calls through the provided base URL. The final image is built for `linux/amd64`, publicly pullable, and ready for automated evaluation.
