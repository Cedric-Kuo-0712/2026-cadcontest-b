# Final Project Report

**Course:** NTU SoCV — System-on-Chip Verification  
**Project:** 2026 CAD Contest Problem B — Regression Failure Bucketing  
**Team ID:** cadb1053 · **Team name:** bububusc  
**Demo video:** [https://youtu.be/7xpwjgu2Kcw](https://youtu.be/7xpwjgu2Kcw)

---

## Table of Contents

1. [Team Members and Contact Information](#1-team-members-and-contact-information)
2. [Introduction](#2-introduction)
3. [Key Research Contributions](#3-key-research-contributions)
4. [Prior Work Disclosure](#4-prior-work-disclosure)
5. [Methodology](#5-methodology)
6. [Implementation Details](#6-implementation-details)
7. [Experimental Results](#7-experimental-results)
8. [Compilation, Execution, and Testing](#8-compilation-execution-and-testing)
9. [Submission Package](#9-submission-package)
10. [References](#10-references)
11. [Comments about this Course (Optional)](#11-comments-about-this-course-optional)

- [Appendix A: Detailed Classification Flow](#appendix-a-detailed-classification-flow)

---

## 1. Team Members and Contact Information

For grading or project inquiries, please contact us by email first. If email is not
efficient, use the alternate contact listed below (team lead).


| Student ID | Name            | Email                                                     | Other contact (Line / FB / Phone)                                                    |
| ---------- | --------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| B11901047  | 郭祐嘉             | [tankkuo0712@gmail.com](mailto:tankkuo0712@gmail.com)     |                                                                                      |
| B11901043  | 張庭碩             | [timmychang104@gmail.com](mailto:timmychang104@gmail.com) |                                                                                      |
| B11901112  | 卜紹秦 (team lead) | [pushaochin@gmail.com](mailto:pushaochin@gmail.com)       | Facebook: [profile link](https://www.facebook.com/share/1JCcT5MPpo/?mibextid=wwXIfr) |


> **Note:** Please fill in Line / phone / FB handles for all members before final submission.

---

## 2. Introduction

After RTL regression, verification engineers receive many failing test cases. Each
case is associated with three log files: regression (`regr.log`), simulation
(`sim.log` or `.gz`), and instruction trace (`trace.log` or `.gz`). **Problem B**
asks us to partition these cases into buckets so that failures caused by the **same
injected bug** fall into the same bucket. We do not predict bug names; only the
**grouping** matters. Scoring uses **pairwise balanced accuracy** (average of
true-positive and true-negative rates over all case pairs; see `eval.py`).

Our approach is a three-stage pipeline: (1) extract a structured fingerprint per
case, (2) compute a mode-aware pairwise distance matrix, and (3) cluster into $k$
groups. The default method (**hybrid**) combines RTL-debug domain distances with
TF-IDF text similarity. The implementation is fully **deterministic** (fixed seed,
no LLM API calls) and uses **streaming I/O** for gzip logs within contest budgets.

```
input.csv ──► features.py ──► clustering.py ──► output.csv
                 │                   │
                 └─ per-case         └─ signature / tfidf / hybrid / dbscan
                    CaseSignature +
                    text_blob
```

---

## 3. Key Research Contributions

The following items were designed and implemented by our team **during this
semester** for Problem B. They go beyond applying off-the-shelf clustering to raw
log text.

1. **Mode-aware log routing** — Parse `regr.log` first to decide whether to read
  `trace.log` (co-simulation mismatch path) or `sim.log` (UVM fatal/assert path),
   avoiding cross-contamination of features.
2. **Structured `CaseSignature` fingerprint** — RTL-debug-specific fields: IBEX/SPIKE
  mismatch mnemonics, local trace context, signature-loop detection, trace-length
   buckets, early-mismatch flags, normalized `sim_fatal_kind` taxonomy (30+ rules),
   and assert-name sets.
3. **Must-link constraints** — Identical `categorical_key` pairs are forced to
  distance zero before clustering, encoding deterministic same-bug evidence.
4. **Mode-specific pairwise distances** — Separate formulas for mismatch–mismatch,
  non-mismatch–non-mismatch, and cross-mode pairs, including a `no_dret` fatal ↔
   early short-trace mismatch bridge (links related failures from the same bug).
5. **Hybrid distance matrix** — Blend signature distance ($w \approx 0.85$–$0.96$)
  with TF-IDF cosine distance (word + character n-grams) for residual text
   similarity.
6. **Multiple clustering backends with $k$ adaptation** — Agglomerative hybrid
  (default), pure signature / TF-IDF baselines, signature-then-TF-IDF centroid
   merging, and DBSCAN with merge/split post-processing to match bucket count $k$.
7. **Streaming, budget-friendly I/O** — Gzip logs read line-by-line; trace tails
  kept in a fixed-size deque without loading full files into memory.

---

## 4. Prior Work Disclosure

The following are **not** counted as contributions of this final project. We disclose
them so graders can distinguish our work from existing materials.

This disclosure is **honor-based** (the course may not verify every item). Failure to
disclose prior work or third-party components, if identified, may result in a critical
deduction of project grades.


| Category                      | Item                                                                                                    | Role in this project                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Contest materials             | 2026 CAD Contest **Problem B** statement (`B_20260212.pdf`) and public samples in `B_samples_20260516/` | Problem definition, I/O format, scoring metric, benchmark logs                                                         |
| Reference solutions           | `B_samples_20260516/solution/naive_completion/` and `char_embedding/`                                   | Illustrative LLM/embedding baselines; **not used** in our default pipeline                                             |
| Third-party libraries         | Python stdlib; **NumPy**, **pandas**, **SciPy**, **scikit-learn** (`requirements.txt`)                  | Numerics, CSV I/O, TF-IDF, agglomerative clustering, DBSCAN, cosine distance                                           |
| Build / CI tools              | **PyInstaller**, **Docker**, GitHub Actions                                                             | Packaging Linux submission binaries                                                                                    |
| Prior algorithms (literature) | TF-IDF vectorization, hierarchical agglomerative clustering, DBSCAN                                     | Standard ML components; our contribution is the **domain distance design** and feature engineering wrapped around them |
| Domain background             | UVM, Ibex, Spike co-simulation, RISC-V DV log formats (OpenTitan / contest ecosystem)                   | Terminology and log structure understood from course/contest context, not invented by us                               |


Our default submission is **deterministic** and does **not** call external LLM APIs.

---

## 5. Methodology

We do not assign buckets with a single hard rule. For each case we build a structured
fingerprint, measure similarity between all pairs, and partition into $k$ clusters.
Bucket **labels** (e.g. `bucket_0`) are arbitrary; only **groupings** affect the score.

### 5.1 End-to-end flowchart

```
                     input.csv (N cases)
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Feature extraction (features.py)                       │
│  For each case: parse regr → route sim or trace         │
│  → build CaseSignature + weighted text_blob             │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Pairwise distance (clustering.py)                      │
│  Mode-specific signature distance + optional TF-IDF     │
│  Must-link: categorical_key match → distance 0          │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Clustering (--method, target k buckets)                │
│  hybrid: agglomerative on blended matrix (default)      │
└───────────────────────────┬─────────────────────────────┘
                            ▼
                   output.csv (Case, bucket)
                            │
                            ▼
              eval.py → pairwise balanced accuracy
```

### 5.2 Pseudocode — hybrid bucketing (default method)

```
function BUCKET(input_csv, k):
    cases ← read(input_csv)
    features ← []
    for each case in parallel:
        regr ← parse_regr(case.regr_log)
        if regr has Mismatch[N]:
            trace ← parse_trace(case.trace_log, window around mismatch, tail=64)
            sig ← build_mismatch_signature(regr, trace)
        else:
            sim ← parse_sim(case.sim_log)
            sig ← build_fatal_or_assert_signature(regr, sim)
        blob ← render_text_blob(sig)
        features.append((sig, blob))

    D_sig ← mode_aware_signature_distance_matrix(features)
    D_tfidf ← cosine_distance(tfidf_word_char_ngrams(blobs))
    D ← blend(D_sig, D_tfidf, weight w(sig closeness))
    for all pairs (i, j) with identical categorical_key(i, j):
        D[i,j] ← 0

    labels ← agglomerative_cluster(D, n_clusters=k, linkage=average)
    return write_csv(cases, labels)
```

### 5.3 Pseudocode — per-case log routing

```
function BUILD_FEATURES(case):
    regr ← extract_regr(case.regr_log)
    if regr.kind == "mismatch":
        trace ← extract_trace(case.trace_log)   # sim.log skipped
        mode ← "mismatch"
    else:
        sim ← extract_sim(case.sim_log)         # trace.log skipped
        mode ← classify(sim)                    # fatal | assert | unknown
    sig ← populate CaseSignature(regr, sim?, trace?, mode)
    return (sig, text_blob(sig))
```

### 5.4 Algorithm description

**Step 1 — `regr.log` routing.** If the regression log reports `Mismatch[N]:`, we
treat the case as a co-simulation mismatch and focus on `trace.log` (instruction
window around the first mismatch ±8 retired instructions, plus a 64-line rolling
tail). Otherwise we read `sim.log` to detect UVM fatals or assertion failures.

**Step 2 — `failure_mode`.** For non-mismatch cases, scan `sim.log`: assert-only
failures → `assert`; `UVM_FATAL` → `fatal` with a normalized `sim_fatal_kind` from a
keyword rule table (timeouts, signature checks, debug traps, cosim errors, etc.);
otherwise `unknown`.

**Step 3 — Features.** Populate `CaseSignature` fields per mode (mnemonics, context
tuples, loop/tail statistics, fatal kind/source, assert sets, test names) and a
weighted `text_blob` used for TF-IDF.

**Step 4 — Must-link key.** Cases sharing the same `categorical_key` (exact
signature fields deemed deterministic) must have zero distance.

**Step 5 — Pairwise distance.** Choose `_mismatch_distance`, `_non_mismatch_distance`,
or `_cross_mode_distance` based on the two cases' modes; blend with TF-IDF cosine
distance for the hybrid method:

$$D_{ij} = w \cdot d^{\mathrm{sig}}*{ij} + (1-w) \cdot d^{\mathrm{tfidf}}*{ij}$$

with adaptive $w \in [0.85, 0.96]$.

**Step 6 — Clustering.** Default: average-linkage agglomerative clustering on the
precomputed matrix with `n_clusters = k`. Alternatives: pure signature buckets,
TF-IDF-only, signature-then-TF-IDF merging, or DBSCAN with merge/split to reach $k$.

Field-level routing tables, fatal-kind taxonomy, and distance weights are given in
[Appendix A](#appendix-a-detailed-classification-flow).

---

## 6. Implementation Details

- **Deterministic and offline** — Fixed random seed (`42`); no LLM or network calls in the default path.
- **Parallel feature extraction** — `--workers` uses a thread pool; clustering runs single-process on the distance matrix (typical $N$ is 9–27).
- **Streaming gzip I/O** — `sim.log.gz` / `trace.log.gz` decompressed on the fly; trace tail stored in a bounded deque.
- **Dual TF-IDF views** — Word n-grams (1–3) plus `char_wb` n-grams (3–6) capture semantic tokens and near-duplicate templates.
- **Adaptive hybrid weight $w$** — Higher $w$ when signature distance is already small (especially mismatch cases), keeping domain rules dominant.
- **Cross-mode bug linking** — `no_dret` fatals can sit closer to early, short-trace mismatches (same root cause, different observable failure mode).
- **DBSCAN $k$ repair** — After density clustering, greedily merge or split clusters until exactly $k$ buckets remain.
- **Contest packaging** — Linux PyInstaller binary with full `src/` fallback so judges can run `pip install -r requirements.txt` if the binary fails.

**Module map**


| File                         | Role                                                                     |
| ---------------------------- | ------------------------------------------------------------------------ |
| `src/features.py`            | `extract_regr` / `extract_sim` / `extract_trace` → `build_case_features` |
| `src/clustering.py`          | Distance functions + `cluster()` strategies                              |
| `src/regr_fail_bucketing.py` | CLI entry point; `--workers` for parallel feature extraction             |
| `eval.py`                    | Pairwise balanced accuracy vs `golden.csv`                               |
| `run_and_eval.sh`            | Convenience script: auto-derives $k$, runs bucketing + evaluation        |


---

## 7. Experimental Results

**Metric:** pairwise balanced accuracy (TPR/TNR over case pairs), as defined in
`B_20260212.pdf` §3.1 and implemented in `eval.py`.

### 7.1 Method comparison on public benchmarks


| Method                 | Set 1 ($N{=}9$, $k{=}2$) | Set 2 ($N{=}27$, $k{=}4$) |
| ---------------------- | ------------------------ | ------------------------- |
| `signature`            | 0.475                    | 0.508                     |
| `tfidf`                | 0.475                    | 0.652                     |
| `signature_then_tfidf` | 0.475                    | 0.762                     |
| `dbscan`               | 0.444                    | 0.799                     |
| `**hybrid` (default)** | **0.475**                | **1.000**                 |


### 7.2 Analysis


| Benchmark | Result                                       | Notes                                                                                                                                                                            |
| --------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Set 1     | BA = 0.475 (all signature-aware methods tie) | Cases 5 & 9 share identical UVM timeout templates across **different** bugs; logs are indistinguishable without golden labels                                                    |
| Set 2     | BA = **1.00** with `hybrid`                  | Bugs 107 / 2014 / 7021 separated cleanly; case 22 (early short mismatch) linked to case 21 (`no_dret` fatal); case 23 merges with bug 107 (trace-indistinguishable from bug 234) |


TF-IDF alone improves Set 2 over pure signature but misses structured fatal/mismatch
cues. The hybrid blend combines RTL-specific distances with text similarity and
achieves perfect grouping on the larger public set.

**Reproduction:**

```bash
./run_and_eval.sh 1 hybrid
./run_and_eval.sh 2 hybrid
```

---

## 8. Compilation, Execution, and Testing

### 8.1 Prerequisites

- Python **3.10+**
- Public benchmark data under `B_samples_20260516/` (included in this repository)

**Install dependencies:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 8.2 Running the program

```bash
python3 src/regr_fail_bucketing.py \
    --input  B_samples_20260516/problem/benchmark_set_1/input.csv \
    --output output.csv \
    --k 2 \
    --method hybrid \
    -v
```


| Option             | Description                                                                    |
| ------------------ | ------------------------------------------------------------------------------ |
| `--input`          | Input CSV (`Case`, `Regr Log`, `Sim Log`, `Trace Log`)                         |
| `--output`         | Output CSV (`Case`, `bucket`)                                                  |
| `--k`              | Soft hint for number of buckets (= injected bugs)                              |
| `--method`         | `signature` · `tfidf` · `hybrid` (default) · `signature_then_tfidf` · `dbscan` |
| `--seed`           | Random seed (default `42`)                                                     |
| `--workers`        | Parallel feature-extraction threads (`0` = auto)                               |
| `-v` / `--verbose` | Per-bucket diagnostics on stderr                                               |


**Helper script** (derives $k$ from golden file, runs bucketing and evaluation):

```bash
./run_and_eval.sh 1            # benchmark_set_1, hybrid
./run_and_eval.sh 2 tfidf      # benchmark_set_2, tfidf
```

### 8.3 Testing and evaluation

```bash
python3 eval.py \
    --output output.csv \
    --golden B_samples_20260516/problem/benchmark_set_1/golden.csv \
    --verbose
```

Compare all methods on both public sets:

```bash
for set in 1 2; do
  GOLDEN="B_samples_20260516/problem/benchmark_set_${set}/golden.csv"
  INPUT="B_samples_20260516/problem/benchmark_set_${set}/input.csv"
  K=$(python3 -c "import pandas as pd; print(pd.read_csv('$GOLDEN')['Bug'].nunique())")
  for method in signature tfidf hybrid signature_then_tfidf dbscan; do
    python3 src/regr_fail_bucketing.py --input "$INPUT" --output /tmp/out.csv --k "$K" --method "$method"
    python3 eval.py --output /tmp/out.csv --golden "$GOLDEN"
  done
done
```

### 8.4 Building the Linux submission binary

The judge machine runs **Linux amd64**. PyInstaller binaries are OS-specific; we
build the Linux executable with **Docker** (recommended on macOS), on a native Linux
host, or via GitHub Actions.

**Docker (recommended on macOS)** — requires [Docker Desktop](https://www.docker.com/products/docker-desktop/):

```bash
chmod +x build_linux.sh
./build_linux.sh
```

The script (1) builds image `regr-fail-bucketing-linux-build` from `docker/Dockerfile`
(`python:3.10-bookworm` + dependencies + PyInstaller), (2) mounts the repository
and runs `docker/build_submission.sh`, (3) packages `src/regr_fail_bucketing.py` into
a one-file ELF binary, and (4) assembles `submission/`. On Apple Silicon,
`--platform linux/amd64` targets x86_64.

**Smoke test** (automatic when benchmarks are present):

```bash
submission/regr_fail_bucketing \
  --input B_samples_20260516/problem/benchmark_set_1/input.csv \
  --output /tmp/smoke.csv --k 2
python3 eval.py --output /tmp/smoke.csv \
  --golden B_samples_20260516/problem/benchmark_set_1/golden.csv
```

**Alternatives:** native Linux → `./docker/build_submission.sh`; CI → push to
`main` / `master` / `bsc/v1` and download artifact `submission-linux` from
`.github/workflows/build-linux.yml`. See `docker/README.md` for manual
`docker build` / `docker run` commands.

---

## 9. Submission Package

Upload a **Linux** PyInstaller package (Option A): one executable plus source fallback.

```bash
./build_linux.sh
zip -r submission-linux.zip submission
```

**Contents of `submission/`:**

```
submission/
├── regr_fail_bucketing   # Linux ELF x86-64 binary
├── README.md
├── requirements.txt      # fallback if binary fails on judge machine
└── src/                  # fallback Python entry point
```

Upload `submission/` or `submission-linux.zip` to the contest portal. Rebuild with
`./build_linux.sh` before final judging if the evaluator runs Linux.

---

## 10. References

1. **2026 CAD Contest Problem B specification** — `B_20260212.pdf` (problem statement, I/O format, balanced-accuracy metric). Provided by the course/contest organizers.
2. **Public benchmark samples** — `B_samples_20260516/` (input CSVs, golden labels, log files, reference LLM solutions).
3. Pedregosa, F. et al. **Scikit-learn: Machine Learning in Python.** *JMLR* 12 (2011). [https://scikit-learn.org](https://scikit-learn.org) — `TfidfVectorizer`, `AgglomerativeClustering`, `DBSCAN`.
4. Harris, C. R. et al. **Array programming with NumPy.** *Nature* 585 (2020). [https://numpy.org](https://numpy.org)
5. The pandas development team. **pandas** — data structures for CSV I/O. [https://pandas.pydata.org](https://pandas.pydata.org)
6. Virtanen, P. et al. **SciPy 1.0: fundamental algorithms for scientific computing in Python.** *Nature Methods* 17 (2020). [https://scipy.org](https://scipy.org)
7. Ester, M., Kriegel, H.-P., Sander, J., Xu, X. **A density-based algorithm for discovering clusters in large spatial databases with noise (DBSCAN).** KDD 1996.
8. Course materials — NTU SoCV / CAD Contest B lectures and handouts on verification log analysis and clustering baselines (2026 spring semester).

## Appendix A: Detailed Classification Flow

This appendix documents field-level routing, feature tables, and distance weights
implemented in `src/features.py` and `src/clustering.py`.

### A.1 `regr.log` routing (`features.extract_regr`)

```
regr.log
   │
   ├─ 有 Mismatch[N]: ──► regr_kind = "mismatch"
   │                      failure_mode = "mismatch"
   │                      主要讀 trace.log：
   │                        · 在 first mismatch 的 retire index 取 ±8 條指令
   │                        · 另保留 trace 尾端 64 行（rolling tail）
   │                      從 regr 抽出：
   │                        · retire_index, ibex/spike mnemonic, matched/mismatch count
   │                      sim.log 在此路徑不讀（避免 UVM 雜訊蓋過 co-sim 特徵）
   │
   └─ 只有 test : [FAILED] ──► regr_kind = "failed_only"
                              regr_test_name = test 名稱（去掉 .0 後綴）
                              再讀 sim.log，決定 failure_mode（A.2）
                              trace.log 在此路徑不讀
```

### A.2 `failure_mode` (`features._failure_mode`)

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


| `sim_fatal_kind`          | 關鍵字 / 模式                                                  |
| ------------------------- | --------------------------------------------------------- |
| **Timeouts**              |                                                           |
| `wall_clock_timeout`      | `wall-clock timeout`                                      |
| `test_timeout`            | `TEST TIMEOUT!!`                                          |
| `debug_timeout`           | `IN_DEBUG_MODE`                                           |
| `irq_timeout`             | `HANDLING_IRQ`                                            |
| `csr_timeout`             | `Did not receive write to csr`                            |
| `core_status_timeout`     | other `Did not receive core_status`                       |
| **Signature / handshake** |                                                           |
| `handshake_test_fail`     | `RISCV-DV handshake (payload=TEST_FAIL)`                  |
| `handshake_malformed`     | `Incorrectly formed handshake`                            |
| `bad_signature_format`    | `signature address is formatted incorrectly`              |
| `double_fault`            | `double_fault detector`                                   |
| **CSR / status checks**   |                                                           |
| `check_memory`            | `memory fault`                                            |
| `check_mcause`            | `Check failed mcause`                                     |
| `check_signature`         | `Check failed signature_data`                             |
| `check_priv_mode`         | `Check failed ... Incorrect privilege mode`               |
| **Debug / trap flow**     |                                                           |
| `no_dret` / `no_mret`     | `No dret/mret detected`                                   |
| `debug_ebreak`            | `Core did not enter debug mode after execution of ebreak` |
| `debug_ebreak_init`       | `EBreak seen whilst doing initial debug initialization`   |
| `irq_in_debug`            | `Core is handling interrupt detected in debug mode`       |
| `illegal_instr`           | `Illegal instruction detected`                            |
| `invalid_xret`            | `Invalid xRET instruction`                                |
| `invalid_compressed`      | invalid / illegal compressed instruction                  |
| `dcsr_priv`               | `dcsr.prv is an unsupported privilege mode`               |
| **Co-simulation**         |                                                           |
| `cosim_reg_write`         | register write data mismatch                              |
| `cosim_reg_missing`       | DUT didn't write expected register                        |
| `cosim_trap`              | synchronous trap mismatch                                 |
| `cosim_mem_access`        | load/store access mismatch                                |
| `cosim_pc`                | PC mismatch                                               |
| `cosim_mismatch`          | other `Cosim mismatch`                                    |
| **Other**                 |                                                           |
| `hdl_read_fail`           | `Check failed (uvm_hdl_read`                              |
| `env_setup`               | `Cannot get RV32*/clk_if/dut_if/...`                      |
| `missing_binary`          | `Please specify test binary`                              |
| `cannot_open_file`        | `Cannot open file`                                        |
| `base_class_stub`         | `Base class task should not be used`                      |
| `other_fatal`             | unmatched UVM_FATAL report                                |


### A.3 Per-mode features (`CaseSignature` + `text_blob`)


| `failure_mode` | 主要欄位                                                                                                                                                  | `text_blob` 重點 token                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `mismatch`     | `mismatch_ibex/spike_mnemonic`, `mismatch_context_mnemonics`, `early_mismatch`, `same_reg_pair`, `has_signature_loop`, `trace_length_bucket`, tail 統計 | `IBEX_`*, `SPIKE_*`, `PAIR_*`, `RETIRE_*`, `CTX ...` |
| `fatal`        | `sim_fatal_kind`, `sim_fatal_source`, `regr_test_name`, `sim_uvm_testname`                                                                            | `FATAL_*`, `REGRTEST_*`                              |
| `assert`       | `sim_error_asserts`, `sim_fatal_kind`（通常空）                                                                                                            | `ASSERT_*`（名稱重複加權）                                   |
| `unknown`      | 上述能抓到的都填                                                                                                                                              | `MODE_unknown` + 少量 sim 片段                           |


**Mismatch heuristics:** `early_mismatch` (`matched_count < 200` or `retire_index < 150`);
`same_reg_pair` (ibex mnemonic == spike mnemonic); `has_signature_loop` (auipc + sw + c.j
in tail); `trace_length_bucket` (`short` / `medium` / `long` / `huge`).

### A.4 Must-link key (`categorical_key`)

```
failure_mode == "mismatch"
   └─ key = (mismatch, ibex_mnem, spike_mnem, has_signature_loop,
             trace_length_bucket, has_repeating_tail, same_reg_pair, early_mismatch)

failure_mode != "mismatch"
   └─ key = (failure_mode, sim_fatal_kind, sim_fatal_source,
             sim_error_asserts, regr_test_name)
```

Identical keys → distance 0 (must-link).

### A.5 Pairwise distance (`clustering._pairwise_signature_distance`)

```
case A  vs  case B
   │
   ├─ 兩個都是 mismatch ──► _mismatch_distance()
   ├─ 一個 mismatch、一個不是 ──► _cross_mode_distance()
   │     例外：fatal=no_dret + early short mismatch → 0.68
   └─ 兩個都不是 mismatch ──► _non_mismatch_distance()
```

**Hybrid:** $D_{ij} = w \cdot d^{\mathrm{sig}}*{ij} + (1-w) \cdot d^{\mathrm{tfidf}}*{ij}$,
$w \approx 0.85$–$0.96$; identical `categorical_key` → $D_{ij} = 0$.

### A.6 Clustering methods

```
   ├─ signature      ──► 每個 categorical_key 一個 bucket
   ├─ tfidf          ──► Agglomerative, cosine, n_clusters = k
   ├─ hybrid（預設） ──► Agglomerative, precomputed D, n_clusters = k
   ├─ signature_then_tfidf ──► signature 後 centroid 合併
   └─ dbscan         ──► DBSCAN on D，merge/split 至 k 群
```

