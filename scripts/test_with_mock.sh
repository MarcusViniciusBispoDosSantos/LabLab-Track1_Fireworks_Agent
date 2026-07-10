#!/usr/bin/env bash
set -euo pipefail
mkdir -p input output
cp samples/tasks_all_categories.json input/tasks.json
python scripts/mock_fireworks.py >/tmp/mock_fireworks.log 2>&1 &
MOCK_PID=$!
trap 'kill $MOCK_PID >/dev/null 2>&1 || true' EXIT
sleep 1
docker buildx build --platform linux/amd64 -t track1-fireworks-agent:local --load .
docker run --rm --network host \
  -e FIREWORKS_API_KEY=dummy \
  -e FIREWORKS_BASE_URL=http://127.0.0.1:8000/v1 \
  -e ALLOWED_MODELS=mock-model \
  -e ENABLE_LOCAL_FAST_PATHS=1 \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  track1-fireworks-agent:local
python scripts/validate_results.py input/tasks.json output/results.json
cat output/results.json
