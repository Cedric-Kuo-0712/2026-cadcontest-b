#!/usr/bin/env bash
# Build Linux submission binary using Docker.
# Requires: docker (install via `brew install colima docker && colima start`)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/submission-linux"

if ! command -v docker >/dev/null 2>&1; then
  echo "[!] docker not found."
  echo "    Install: brew install colima docker && colima start"
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[!] Docker daemon not running."
  echo "    Start: colima start   (or open Docker Desktop)"
  exit 1
fi

cd "$ROOT"

echo "[1/3] Docker build (Linux amd64)..."
docker build --platform linux/amd64 -f Dockerfile.linux -t cadb-b-linux .

echo "[2/3] Extract submission/ ..."
rm -rf "$OUT"
docker create --name cadb-b-extract cadb-b-linux >/dev/null
docker cp cadb-b-extract:/work/submission "$OUT"
docker rm cadb-b-extract >/dev/null

echo "[3/3] Verify binary..."
file "$OUT/regr_fail_bucketing"
ls -lh "$OUT/regr_fail_bucketing"

cd "$ROOT"
rm -f submission-linux.zip
zip -r submission-linux.zip submission-linux
ls -lh submission-linux.zip

echo ""
echo "Done. Linux submission package: $OUT"
echo "Upload: submission-linux.zip"
