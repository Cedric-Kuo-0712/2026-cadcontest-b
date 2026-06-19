#!/usr/bin/env bash
# Build a Linux amd64 executable on macOS/Windows via Docker.
# Requires: Docker Desktop (or docker engine) running.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="regr-fail-bucketing-linux-build"
PLATFORM="${PLATFORM:-linux/amd64}"

echo "==> Building Docker image ($PLATFORM)"
docker build --platform "$PLATFORM" -t "$IMAGE" -f "$ROOT/docker/Dockerfile" "$ROOT"

echo "==> Running PyInstaller inside container"
docker run --rm --platform "$PLATFORM" \
  -v "$ROOT:/work" \
  -w /work \
  "$IMAGE" \
  /work/docker/build_submission.sh

echo ""
echo "Linux submission package ready:"
echo "  $ROOT/submission/regr_fail_bucketing"
echo ""
echo "To zip for upload:"
echo "  cd $ROOT && zip -r submission-linux.zip submission"
