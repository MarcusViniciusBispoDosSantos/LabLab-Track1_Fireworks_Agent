#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-track1-fireworks-agent:local}"
PORT="${MOCK_FIREWORKS_PORT:-8000}"

mkdir -p input output
cp "${TASKS_FILE:-samples/tasks_all_categories.json}" input/tasks.json
rm -f output/results.json

python scripts/mock_fireworks.py > /tmp/mock_fireworks_track1.log 2>&1 &
MOCK_PID=$!
cleanup() {
  kill "$MOCK_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT
sleep 1

docker buildx build \
  --platform linux/amd64 \
  -t "$IMAGE_TAG" \
  --load \
  .

# On Linux/GitHub Actions, --network host makes localhost inside the container reach the mock server.
docker run --rm \
  --network host \
  -e FIREWORKS_API_KEY=dummy-harness-key \
  -e FIREWORKS_BASE_URL="http://127.0.0.1:${PORT}/v1" \
  -e ALLOWED_MODELS=mock-model \
  -e VERIFY_HARD_TASKS=0 \
  -e MAX_WORKERS=2 \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  "$IMAGE_TAG"

python scripts/validate_results.py --tasks input/tasks.json --results output/results.json
cat output/results.json
