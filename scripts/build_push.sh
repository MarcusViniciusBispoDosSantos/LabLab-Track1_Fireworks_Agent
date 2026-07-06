#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-your-dockerhub-user/track1-fireworks-agent:latest}"

docker buildx build \
  --platform linux/amd64 \
  --tag "$IMAGE_TAG" \
  --push \
  .

echo "Pushed $IMAGE_TAG"
