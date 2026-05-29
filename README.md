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
                 └─ per-case          └─ signature / tfidf / hybrid / dbscan
                    CaseSignature +
                    text_blob
```

The implementation is deterministic (no LLM dependency by default),
streaming-friendly (gzipped sim/trace logs are read line-by-line with a
rolling tail deque, never materialized), and respects the contest budgets.

---

## Classification flow（分類流程）

Each case is **not** labeled by a single rule. We first route by log content,
build a structured fingerprint (`CaseSignature`), then cluster cases by
pairwise distance. Bucket IDs are arbitrary; only **groupings** are scored.

### Step 1 — `regr.log` routing (`features.extract_regr`)

Implemented in `src/features.py`.

```
regr.log
   │
   ├─ 有 Mismatch[N]: ──► regr_kind = "mismatch"
   │                      failure_mode = "mismatch"   （見 Step 2）
   │                      主要讀 trace.log：
   │                        · 在 first mismatch 的 retire index 取 ±8 條指令
   │                        · 另保留 trace 尾端 64 行（rolling tail）
   │                      從 regr 抽出：
   │                        · retire_index, ibex/spike mnemonic, matched/mismatch count
   │                      sim.log 在此路徑不讀（避免 UVM 雜訊蓋過 co-sim 特徵）
   │
   └─ 只有 test : [FAILED] ──► regr_kind = "failed_only"
                              regr_test_name = test 名稱（去掉 .0 後綴）
                              再讀 sim.log，決定 failure_mode（Step 2）
                              trace.log 在此路徑不讀
```

### Step 2 — `failure_mode`（`features._failure_mode`）

Only applies when `regr_kind != "mismatch"`:

```
sim.log（failed_only 路徑）
   │
   ├─ regr_kind 已是 "mismatch" ──► failure_mode = "mismatch"（略過 sim 判斷）
   │
   └─ regr_kind = "failed_only"
          │
          ├─ 有 ASSERT FAILED、且無 UVM_FATAL ──► failure_mode = "assert"
          │     特徵：sim_error_asserts（assert 名稱集合）
          │
          ├─ 有 UVM_FATAL ──► failure_mode = "fatal"
          │     特徵：sim_fatal_kind（規則表匹配，見下）、fatal_source、test name
          │
          └─ 都沒有 ──► failure_mode = "unknown"
```

`sim_fatal_kind` 由第一條 `UVM_FATAL` 訊息正規化後，依序匹配：

| `sim_fatal_kind`   | 關鍵字 / 模式              |
|--------------------|----------------------------|
| `debug_timeout`    | `IN_DEBUG_MODE`            |
| `irq_timeout`      | `HANDLING_IRQ`             |
| `no_dret`          | `No dret detected`         |
| `check_mcause`     | `Check failed mcause`      |
| `check_signature`  | `Check failed signature_data` |
| `check_memory`     | `memory fault`             |
| `other_fatal`      | 其他 UVM_FATAL             |

### Step 3 — Per-mode features（`CaseSignature` + `text_blob`）

| `failure_mode` | 主要欄位 | `text_blob` 重點 token |
|----------------|----------|-------------------------|
| `mismatch` | `mismatch_ibex/spike_mnemonic`, `mismatch_context_mnemonics`, `early_mismatch`, `same_reg_pair`, `has_signature_loop`, `trace_length_bucket`, tail 統計 | `IBEX_*`, `SPIKE_*`, `PAIR_*`, `RETIRE_*`, `CTX ...` |
| `fatal` | `sim_fatal_kind`, `sim_fatal_source`, `regr_test_name`, `sim_uvm_testname` | `FATAL_*`, `REGRTEST_*` |
| `assert` | `sim_error_asserts`, `sim_fatal_kind`（通常空） | `ASSERT_*`（名稱重複加權） |
| `unknown` | 上述能抓到的都填 | `MODE_unknown` + 少量 sim 片段 |

**Mismatch 專用啟發式**

- `early_mismatch`：`matched_count < 200` 或 `retire_index < 150`
- `same_reg_pair`：ibex mnemonic == spike mnemonic
- `has_signature_loop`：尾端出現 auipc + sw + c.j 型態
- `trace_length_bucket`：`short` / `medium` / `long` / `huge`（依 trace 總行數）

### Step 4 — Exact-match key（`categorical_key`）

相同 key 的兩個 case 在距離矩陣中 **必須為 0**（must-link，一定同 bucket）：

```
failure_mode == "mismatch"
   └─ key = (mismatch, ibex_mnem, spike_mnem, has_signature_loop,
             trace_length_bucket, has_repeating_tail, same_reg_pair, early_mismatch)

failure_mode != "mismatch"
   └─ key = (failure_mode, sim_fatal_kind, sim_fatal_source,
             sim_error_asserts, regr_test_name)
```

### Step 5 — Pairwise distance（`clustering._pairwise_signature_distance`）

兩個 case 比較時，依 **雙方的 `failure_mode`** 選公式：

```
case A  vs  case B
   │
   ├─ 兩個都是 mismatch ──► _mismatch_distance()
   │     加權：ibex/spike 指令是否相同、trace 上下文重疊、長度 bucket、
   │           loop / early_mismatch / same_reg_pair
   │     ibex、spike 都不同 → 距離至少 0.80
   │
   ├─ 一個 mismatch、一個不是 ──► _cross_mode_distance()
   │     預設 1.0（不同 bug）；例外：
   │       · fatal=no_dret 且 mismatch 為 early + short trace → 0.68
   │       · fatal 有 assert 集合 → 1.0
   │       · fatal 有 kind 且 mismatch 有 signature_loop → 0.95
   │
   └─ 兩個都不是 mismatch ──► _non_mismatch_distance()
         加權：fatal_kind 距離、assert Jaccard、fatal_source、regr_test_name
         同 bug 可能一個 assert、一個 fatal → mode 懲罰僅 0.15（若 assert 重疊 ≥ 0.5）
         assert 名稱完全相同 → 距離可壓到 ≈ 0.2
```

**Hybrid 距離**（`--method hybrid`，預設）：

```
D[i,j] = w · signature_distance + (1 − w) · TF-IDF_cosine_distance
         w ≈ 0.96（mismatch 且 signature 已很近）
         w ≈ 0.85–0.92（其餘）
categorical_key 相同 → 強制 D[i,j] = 0
```

### Step 6 — Clustering → `output.csv`（`clustering.cluster`）

```
距離矩陣 D（或 TF-IDF 矩陣）
   │
   ├─ signature      ──► 每個 categorical_key 一個 bucket（不聚類）
   ├─ tfidf          ──► Agglomerative，cosine，n_clusters = k
   ├─ hybrid（預設） ──► Agglomerative，precomputed D，n_clusters = k
   ├─ signature_then_tfidf ──► 先 signature，簽名過多時再合併 centroid
   └─ dbscan         ──► DBSCAN on D，再 merge/split 調成恰好 k 群
```

`k` 是軟提示（題目給的 bucket 數）；評分用 pairwise balanced accuracy，不要求 bucket 名稱與 golden 一致。

### Module map

| File | Role |
|------|------|
| `src/features.py` | `extract_regr` / `extract_sim` / `extract_trace` → `build_case_features` |
| `src/clustering.py` | Distance functions + `cluster()` strategies |
| `src/regr_fail_bucketing.py` | CLI；`--workers` 平行萃取特徵 |
| `eval.py` | Pairwise balanced accuracy vs `golden.csv` |

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
- `--method {signature,tfidf,hybrid,signature_then_tfidf,dbscan}` — default `hybrid`.
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

| Benchmark | N  | K | Balanced Accuracy | Notes |
|-----------|----|---|-------------------|-------|
| Set 1     | 9  | 2 | 0.475             | Cases 5 & 9 share identical sim/regr templates across bugs |
| Set 2     | 27 | 4 | **1.00**         | bug_107/2014/7021 perfect; bug_234 case 22 fixed; case 23 merges with 107 |

Set 1 is limited by cases that share identical UVM timeout templates, test
names, and bins across different bugs (e.g. bug_304 case 5 vs bug_7023 case 9).

Set 2 case 23 (bug_234) is trace-indistinguishable from bug_107 mismatches
(same signature-loop tail, similar retire/matched counts); case 22 is correctly
separated via early short-trace outlier detection and linked to the `no_dret`
fatal case 21.

## LLM (optional)

The reference solutions in `B_samples_20260516/solution/` demonstrate how to
plug an LLM completion or embedding endpoint via `LLM_MODEL_CONFIG`. Our
default implementation is deterministic and does **not** require LLM access;
the LLM path can be added on top by feeding the same `text_blob` to an
embedding endpoint and replacing the TF-IDF residual.
