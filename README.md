# Track 1 Fireworks Agent — AMD Developer Hackathon ACT II

Accuracy-first Docker project for **Track 1: General-Purpose AI Agent**.

The final submission is a Docker image. It must:

- read `/input/tasks.json` on startup
- write `/output/results.json` before exiting
- read `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` from environment variables
- use the harness-provided `FIREWORKS_API_KEY`, not a key bundled by you
- route all Fireworks calls through `FIREWORKS_BASE_URL`
- never hardcode model IDs; choose only from `ALLOWED_MODELS`
- support all 8 Track 1 capability categories

## Important: you do not need the final API key

For the real hackathon evaluation, the judging harness injects:

```text
FIREWORKS_API_KEY
FIREWORKS_BASE_URL
ALLOWED_MODELS
```

This project reads those values at runtime. Do not place a real `.env` file in the Docker image.

## Supported categories

1. Factual knowledge
2. Mathematical reasoning
3. Sentiment classification
4. Text summarisation
5. Named entity recognition
6. Code debugging
7. Logical / deductive reasoning
8. Code generation

## Project structure

```text
.
├── Dockerfile
├── requirements.txt
├── src/
│   ├── agent.py
│   ├── classifier.py
│   ├── main.py
│   ├── models.py
│   └── prompts.py
├── samples/
│   ├── tasks.json
│   └── tasks_all_categories.json
├── scripts/
│   ├── build_push.sh
│   ├── mock_fireworks.py
│   ├── run_local.sh
│   ├── test_with_mock.sh
│   └── validate_results.py
└── .github/workflows/
    └── track1-online-check.yml
```

## Fastest confirmation without a real Fireworks key

This checks the Docker contract using a local mock harness. It confirms the image builds, reads `/input/tasks.json`, uses the environment variables, calls `FIREWORKS_BASE_URL`, writes `/output/results.json`, and produces valid JSON.

```bash
./scripts/test_with_mock.sh
```

This does **not** confirm real model accuracy. It confirms the submission shape and harness compatibility.

## Online confirmation with GitHub Actions

Push this repo to GitHub. The included workflow runs automatically on `main`, or you can start it manually from:

```text
GitHub repo → Actions → Track 1 Online Check → Run workflow
```

The workflow builds the `linux/amd64` Docker image, starts a mock Fireworks-compatible server, runs the container like the judge, validates `output/results.json`, checks the image architecture, and confirms `.env` is not inside the image.

## Local development with a real Fireworks key, optional

Only needed if you want to test real answer accuracy before official submission.

```bash
cp .env.example .env
```

Then edit `.env` with your own Fireworks values:

```env
FIREWORKS_API_KEY=your_local_fireworks_key
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/example-model-1,accounts/fireworks/models/example-model-2
```

Run:

```bash
python -m src.main
cat ./samples/results.json
python scripts/validate_results.py --tasks samples/tasks_all_categories.json --results samples/results.json
```

## Expected input

```json
[
  { "task_id": "t1", "prompt": "Summarise the following text in one sentence: ..." },
  { "task_id": "t2", "prompt": "..." }
]
```

## Expected output

```json
[
  { "task_id": "t1", "answer": "..." },
  { "task_id": "t2", "answer": "..." }
]
```

## Docker build for submission

The judging VM runs `linux/amd64`, so build and push with an amd64 manifest.

```bash
./scripts/build_push.sh your-dockerhub-user/track1-fireworks-agent:latest
```

Equivalent command:

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag your-dockerhub-user/track1-fireworks-agent:latest \
  --push \
  .
```

## Manual judge-style Docker run

```bash
mkdir -p input output
cp samples/tasks_all_categories.json input/tasks.json

# With real Fireworks values in .env, or with the mock server running.
docker run --rm \
  --env-file .env \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  track1-fireworks-agent:local

python scripts/validate_results.py --tasks input/tasks.json --results output/results.json
cat output/results.json
```

## Accuracy knobs

Default mode uses one specialized model call per task. This is usually the best balance for the 10-minute runtime and token-efficiency ranking.

For maximum accuracy on hard tasks, enable a verifier call:

```bash
VERIFY_HARD_TASKS=1
```

This can help math, logic, debugging, and code-generation tasks, but it uses more tokens and time. Because Track 1 ranks passing submissions by token efficiency, use it carefully.

## Compliance checklist

Before submitting, confirm:

- `docker buildx build --platform linux/amd64` succeeds
- image architecture is `amd64`
- `.env` is not copied into the image
- container reads `/input/tasks.json`
- container writes valid `/output/results.json`
- every output item has `task_id` and `answer`
- app exits with code `0` on success
- no hardcoded API key
- no hardcoded model IDs for final judging
- every Fireworks call uses `FIREWORKS_BASE_URL`
- public image can be pulled by the judge
