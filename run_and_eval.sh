#!/usr/bin/env bash
# Usage: ./run_and_eval.sh <benchmark_set_number> [method]
#   benchmark_set_number: 1 or 2
#   method: signature | tfidf | hybrid | signature_then_tfidf (default: hybrid)
#
# k is automatically pulled from the golden file.

set -euo pipefail

BENCHMARK_SET_NUMBER=${1:?"Usage: $0 <1|2> [method]"}
METHOD=${2:-hybrid}

case "$BENCHMARK_SET_NUMBER" in
  1)
    INPUT_FILE="B_samples_20260516/problem/benchmark_set_1/input.csv"
    GOLDEN_FILE="B_samples_20260516/problem/benchmark_set_1/golden.csv"
    ;;
  2)
    INPUT_FILE="B_samples_20260516/problem/benchmark_set_2/input.csv"
    GOLDEN_FILE="B_samples_20260516/problem/benchmark_set_2/golden.csv"
    ;;
  *)
    echo "Invalid benchmark set number: $BENCHMARK_SET_NUMBER" >&2
    exit 1
    ;;
esac

K=$(python3 -c "import pandas as pd; print(pd.read_csv('$GOLDEN_FILE')['Bug'].nunique())")

python3 src/regr_fail_bucketing.py \
  --input "$INPUT_FILE" \
  --output output.csv \
  --k "$K" \
  --method "$METHOD" \
  -v

python3 eval.py \
  --output output.csv \
  --golden "$GOLDEN_FILE" \
  --verbose
