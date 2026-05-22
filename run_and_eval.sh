# a input arg (a num) incidates the input benchmark set number

# if a is 1, then run the benchmark set 1
# if a is 2, then run the benchmark set 2

BENCHMARK_SET_NUMBER=$1
# default method is sim_trace

METHOD=$2
if [ -z "$METHOD" ]; then
  METHOD="sim_trace"
fi
if [ "$BENCHMARK_SET_NUMBER" -eq 1 ]; then
  INPUT_FILE="B_samples_20260516/problem/benchmark_set_1/input.csv"
  GOLDEN_FILE="B_samples_20260516/problem/benchmark_set_1/golden.csv"
elif [ "$BENCHMARK_SET_NUMBER" -eq 2 ]; then
  INPUT_FILE="B_samples_20260516/problem/benchmark_set_2/input.csv"
  GOLDEN_FILE="B_samples_20260516/problem/benchmark_set_2/golden.csv"
else
  echo "Invalid benchmark set number"
  exit 1
fi

K=$(python3 -c "import pandas as pd; print(pd.read_csv('$GOLDEN_FILE')['Bug'].nunique())")

python src/regr_fail_bucketing_v2.py \
  --input $INPUT_FILE \
  --output output.csv \
  --k $K \
  --method $METHOD \
  --clustering simple \
  -v

python3 eval.py \
  --output output.csv \
  --golden $GOLDEN_FILE