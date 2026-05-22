# SoCV Final Project: 2026 CAD Contest B — Regression Failure Bucketing

Team ID: cadb1053
Team Name: bububusc
Team Members:
- B11901047 郭祐嘉 — tankkuo0712@gmail.com
- B11901043 張庭碩 — timmychang104@gmail.com
- B11901112 卜紹秦 — pushaochin@gmail.com

---

## Overview

Given a CSV of N RTL regression failures (each pointing to `regr.log`,
`sim.log[.gz]`, and `trace.log[.gz]`), assign every case a bucket so that
cases sharing the same root-cause bug land in the same bucket. The scoring
metric is pairwise balanced accuracy (see `eval.py`).

### Pipeline

```
input.csv ──► features.py ──► clustering.py ──► output.csv
                 │                   │
                 └─ per-case          └─ signature / tfidf / hybrid
                    fingerprint +
                    normalized text
```

`src/features.py`
:  Streams each log file and produces a structured `CaseSignature` (UVM
   verdict, normalized `UVM_FATAL` template + source file + coarse category,
   sorted assertion names, mismatch mnemonics, trace-tail loop statistics)
   together with a compact normalized text blob for downstream vectorizers.

`src/clustering.py`
:  Three strategies:
   - `signature` — bucket by exact categorical key.
   - `tfidf` — TF-IDF on the text blob → Agglomerative (cosine, average).
   - `hybrid` *(default)* — Custom precomputed distance combining signature
     overlap (mode, fatal source/category, assertion-set Jaccard, mismatch
     mnemonics, trace-tail similarity) with a small TF-IDF cosine residual,
     fed to AgglomerativeClustering with `metric=precomputed`.

`src/regr_fail_bucketing.py`
:  CLI entry point matching the contest interface:
   `regr_fail_bucketing --input <csv> --output <csv> --k <k>`.

The implementation is deterministic (no LLM dependency by default),
streaming-friendly (gzipped sim/trace logs are read line-by-line with a
rolling tail deque, never materialized), and respects the contest budgets.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 src/regr_fail_bucketing.py \
    --input  B_samples_20260516/problem/benchmark_set_1/input.csv \
    --output output.csv \
    --k 2
```

Options:
- `--method {signature,tfidf,hybrid,signature_then_tfidf}` — default `hybrid`.
- `--seed N` — random seed (default 42).
- `-v` / `--verbose` — print per-bucket diagnostics to stderr.

Convenience driver (auto-derives `k` from the golden file):

```bash
./run_and_eval.sh 1            # benchmark_set_1 with default hybrid
./run_and_eval.sh 2 tfidf      # benchmark_set_2 with tfidf
```

## Evaluation

```bash
python3 eval.py \
    --output output.csv \
    --golden B_samples_20260516/problem/benchmark_set_1/golden.csv \
    --verbose
```

### Scores on the public samples (`hybrid`)

| Benchmark | N  | K | Balanced Accuracy |
|-----------|----|---|-------------------|
| Set 1     | 9  | 2 | 0.625             |
| Set 1's ceiling is limited by two cases (bug_7023 cases 7 and 9) that share
identical `+UVM_TESTNAME`, `+bin`, `+seed`, end-of-trace, line counts, and
UVM message templates with two bug_304 cases — i.e. they are unidentifiable
from the logs alone without bug-specific prior knowledge. |
| Set 2     | 27 | 4 | 0.901             |

## LLM (optional)

The reference solutions in `B_samples_20260516/solution/` demonstrate how to
plug an LLM completion or embedding endpoint via `LLM_MODEL_CONFIG`. Our
default implementation is deterministic and does **not** require LLM access;
the LLM path can be added on top by feeding the same `text_blob` to an
embedding endpoint and replacing the TF-IDF residual.
