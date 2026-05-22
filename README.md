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
:  Mode-aware feature routing:
   - **`regr.log` mismatch** → features from `trace.log` only (loop mnemonics,
     tail uniformity, length bucket, repeating-tail flag).
   - **otherwise** → features from `sim.log` + `regr.log` (canonical fatal kind,
     assertion set, fatal source file, failing test name).

`src/clustering.py`
:  Strategies:
   - `signature` — exact categorical key.
   - `tfidf` — TF-IDF + agglomerative (cosine).
   - `hybrid` *(default)* — mode-specific pairwise distance (trace-only for
     mismatch; assert/fatal-aware for others) blended with TF-IDF cosine;
     identical signatures are must-linked (distance 0).

`src/regr_fail_bucketing.py`
:  CLI entry point; parallel feature extraction via `--workers` (default: auto).

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
| Set 1     | 9  | 2 | 0.475             |
| Set 2     | 27 | 4 | **0.944**         |

Set 1 is limited by cases that share identical UVM timeout templates, test
names, and trace tails across different bugs (e.g. bug_304 case 5 vs bug_7023
case 9) — they are unidentifiable from the logs alone.

## LLM (optional)

The reference solutions in `B_samples_20260516/solution/` demonstrate how to
plug an LLM completion or embedding endpoint via `LLM_MODEL_CONFIG`. Our
default implementation is deterministic and does **not** require LLM access;
the LLM path can be added on top by feeding the same `text_blob` to an
embedding endpoint and replacing the TF-IDF residual.
