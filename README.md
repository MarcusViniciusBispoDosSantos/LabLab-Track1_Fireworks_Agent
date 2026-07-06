# FireRoute AI

**Token-efficient general-purpose AI agent for AMD Developer Hackathon ACT II — Track 1.**

FireRoute AI is a Dockerized Python agent that reads natural language tasks from `/input/tasks.json`, solves each prompt using Fireworks AI through the hackathon runtime environment, and writes valid answers to `/output/results.json`.

This project is designed for **Track 1: General-Purpose AI Agent**.

---

## Project Links

**Public GitHub Repository**  
https://github.com/MarcusViniciusBispoDosSantos/LabLab-Track1_Fireworks_Agent

**Application URL / Demo URL**  
https://github.com/MarcusViniciusBispoDosSantos/LabLab-Track1_Fireworks_Agent

**Docker Image**  
```bash
ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```

> Note: This Track 1 project is evaluated as a Docker container, not as a hosted web application. The repository contains the Dockerfile, setup instructions, GitHub Actions validation, and the public `linux/amd64` container image reference.

---

## What It Does

FireRoute AI handles all required Track 1 task categories:

1. Factual knowledge
2. Mathematical reasoning
3. Sentiment classification
4. Text summarization
5. Named entity recognition
6. Code debugging
7. Logical / deductive reasoning
8. Code generation

The agent uses task routing, compact prompt templates, dynamic model selection from `ALLOWED_MODELS`, and structured JSON output.

---

## Track 1 Contract

The container follows the official Track 1 input/output contract.

### Input

On startup, the container reads:

```bash
/input/tasks.json
```

Example:

```json
[
  {
    "task_id": "t1",
    "prompt": "Summarise the following text in one sentence: ..."
  },
  {
    "task_id": "t2",
    "prompt": "Solve: 15% of 240 plus 18."
  }
]
```

### Output

Before exiting, the container writes:

```bash
/output/results.json
```

Example:

```json
[
  {
    "task_id": "t1",
    "answer": "..."
  },
  {
    "task_id": "t2",
    "answer": "..."
  }
]
```

---

## Runtime Environment Variables

The final hackathon judging harness injects these variables at runtime:

```bash
FIREWORKS_API_KEY
FIREWORKS_BASE_URL
ALLOWED_MODELS
```

The submitted container must read them from the environment.

The project does **not** hardcode API keys or model IDs.

All Fireworks calls are routed through:

```bash
FIREWORKS_BASE_URL
```

Local `.env` files are only for development and must not be committed.

---

## Docker Image Requirement

The judging VM runs on:

```bash
linux/amd64
```

The image must include a `linux/amd64` manifest.

Recommended build command:

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest \
  --push \
  .
```

For GitHub Actions, use:

```yaml
platforms: linux/amd64
```

---

## Pull the Public Docker Image

```bash
docker pull ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```

---

## Run Example

Create input/output folders:

```bash
mkdir -p input output
```

Create `input/tasks.json`:

```json
[
  {
    "task_id": "math1",
    "prompt": "A company has 1,200 users. It grows by 15% and then loses 8%. How many users remain?"
  },
  {
    "task_id": "sentiment1",
    "prompt": "Classify the sentiment: The dashboard is fast and clean, but the export button fails every time."
  }
]
```

Run the container:

```bash
docker run --rm \
  -e FIREWORKS_API_KEY="provided-by-harness" \
  -e FIREWORKS_BASE_URL="provided-by-harness" \
  -e ALLOWED_MODELS="provided-by-harness" \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```

For official evaluation, the hackathon harness provides the real values for `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`.

---

## Online Validation with GitHub Actions

This repository includes GitHub Actions workflows to validate the project online without requiring Docker on a local machine.

The online check confirms:

- Docker image builds successfully
- Container starts correctly
- `/input/tasks.json` is read
- `/output/results.json` is written
- Output JSON is valid
- All `task_id` values receive answers
- Environment variables are accepted
- Image is built for `linux/amd64`

To run the online check:

1. Open the repository on GitHub.
2. Go to **Actions**.
3. Select the Track 1 validation workflow.
4. Click **Run workflow**.
5. Confirm the workflow completes successfully.

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/
├── Dockerfile
├── README.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── agent.py
│   ├── classifier.py
│   ├── models.py
│   └── prompts.py
├── scripts/
│   ├── mock_fireworks.py
│   ├── test_with_mock.sh
│   └── validate_results.py
└── samples/
    └── tasks_all_categories.json
```

---

## Security Notes

Do not commit:

```bash
.env
input/
output/
__pycache__/
```

The final submission must not include a real Fireworks API key.

---

## lablab.ai Submission Values

### Project Title

```text
FireRoute AI
```

### Short Description

```text
FireRoute AI is a Dockerized general-purpose AI agent for AMD Developer Hackathon Track 1. It reads tasks from /input/tasks.json, routes them through Fireworks AI using harness-provided environment variables, and writes valid answers to /output/results.json.
```

### Long Description

```text
FireRoute AI is a Dockerized general-purpose AI agent built for Track 1 of the AMD Developer Hackathon ACT II. It processes batches of natural language tasks by reading `/input/tasks.json`, solving each prompt, and writing valid answers to `/output/results.json` before the container exits.

The agent is designed to handle all required Track 1 capability areas: factual knowledge, mathematical reasoning, sentiment classification, text summarization, named entity recognition, code debugging, logical reasoning, and code generation. Its architecture uses lightweight task routing and task-specific prompting to improve accuracy while keeping responses concise.

FireRoute AI follows the official Fireworks AI runtime contract. It does not hardcode API keys or model IDs. Instead, it reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` from the judging harness environment and routes all model calls through the provided base URL.

The project is fully containerized and includes GitHub Actions validation to confirm the Docker contract online. The final image is intended to be built for `linux/amd64`, publicly pullable, and ready for automated evaluation. FireRoute AI focuses on passing the accuracy gate first, then improving token efficiency through compact prompts, dynamic model selection, and structured JSON output.
```

### Technology Tags

```text
Python, Docker, Fireworks AI, AMD, AI Agent, LLM, Prompt Engineering, GitHub Actions, JSON, Containerized Application, Task Routing, Natural Language Processing
```

### Demo Application Platform

```text
GitHub / Docker / GitHub Container Registry
```

### Application URL

```text
https://github.com/MarcusViniciusBispoDosSantos/LabLab-Track1_Fireworks_Agent
```

### Docker Image

```text
ghcr.io/marcusviniciusbispodossantos/fireroute-ai:latest
```

---

## Final Submission Note

FireRoute AI is submitted as a public `linux/amd64` Docker image. It reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` from the judging harness environment, reads tasks from `/input/tasks.json`, and writes valid results to `/output/results.json`.
