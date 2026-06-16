#!/usr/bin/env bash
# Build Option-A submission package:
#   submission/regr_fail_bucketing  (PyInstaller binary)
#   submission/README.md
#   submission/requirements.txt
#   submission/src/*.py             (source fallback)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SUB="$ROOT/submission"

cd "$ROOT"

echo "[1/4] Install build dependencies..."
python3 -m pip install -q -r requirements.txt pyinstaller

echo "[2/4] PyInstaller build..."
python3 -m PyInstaller --noconfirm --clean regr_fail_bucketing.spec

echo "[3/4] Assemble submission/ ..."
rm -rf "$SUB"
mkdir -p "$SUB/src"

cp dist/regr_fail_bucketing "$SUB/regr_fail_bucketing"
chmod +x "$SUB/regr_fail_bucketing"

cp README.md requirements.txt "$SUB/"
cp src/regr_fail_bucketing.py src/features.py src/clustering.py "$SUB/src/"

echo "[4/4] Smoke test on benchmark_set_1..."
if [[ "${SKIP_SMOKE:-0}" == "1" ]]; then
  echo "  (skipped: SKIP_SMOKE=1)"
elif [[ -f "$ROOT/B_samples_20260516/problem/benchmark_set_1/input.csv" ]]; then
  cd "$ROOT/B_samples_20260516/problem/benchmark_set_1"
  "$SUB/regr_fail_bucketing" --input input.csv --output /tmp/submission_smoke.csv --k 2
  head -5 /tmp/submission_smoke.csv
else
  echo "  (skipped: benchmark_set_1 not found)"
fi

echo ""
echo "Done. Upload the contents of: $SUB"
echo "  regr_fail_bucketing"
echo "  README.md"
echo "  requirements.txt"
echo "  src/"
