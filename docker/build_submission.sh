#!/usr/bin/env bash
# Build contest submission package inside Linux (native or Docker).
# Output: submission/regr_fail_bucketing + README.md + requirements.txt + src/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Installing build dependencies"
python3 -m pip install -q -r requirements.txt pyinstaller

echo "==> PyInstaller (one-file Linux binary)"
pyinstaller docker/regr_fail_bucketing.spec \
  --clean --noconfirm \
  --distpath dist \
  --workpath build

echo "==> Assembling submission/"
rm -rf submission
mkdir -p submission
cp dist/regr_fail_bucketing submission/
cp README.md requirements.txt submission/
cp -r src submission/
chmod +x submission/regr_fail_bucketing

SMOKE_INPUT="B_samples_20260516/problem/benchmark_set_1/input.csv"
SMOKE_GOLDEN="B_samples_20260516/problem/benchmark_set_1/golden.csv"
if [[ -f "$SMOKE_INPUT" && -f "$SMOKE_GOLDEN" ]]; then
  echo "==> Smoke test on benchmark_set_1"
  submission/regr_fail_bucketing \
    --input "$SMOKE_INPUT" \
    --output /tmp/smoke.csv \
    --k 2
  python3 eval.py --output /tmp/smoke.csv --golden "$SMOKE_GOLDEN"
fi

echo "==> Done: $ROOT/submission/regr_fail_bucketing"
file submission/regr_fail_bucketing
