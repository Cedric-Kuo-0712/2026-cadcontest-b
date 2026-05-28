# SoCV Final Project: 2026 CAD Contest B - Regression Failure Bucketing

## Team Information

- Team ID: `cadb1053`
- Team Name: `bububusc`
- Group Members:
  - 郭祐嘉 (B11901047), e-mail: `TBD`
  - 張庭碩 (B11901043), e-mail: `TBD`
  - 卜紹秦 (B11901112), e-mail: `TBD`
- Additional contact (if e-mail is slow): `TBD (Line/Phone/FB)`

> Please replace the `TBD` fields with final submission contact info.

---

## Environment Setup

### 1) Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Compile / Run / Test

Python project; no extra compile step is required.

### Run bucketing

```bash
python src/regr_fail_bucketing.py \
  --input B_samples_20260516/problem/benchmark_set_1/input.csv \
  --output output_set1.csv \
  --k 8 \
  --verbose
```

### Evaluate result

```bash
python eval.py \
  --output output_set1.csv \
  --golden B_samples_20260516/problem/benchmark_set_1/golden.csv
```

### Quick public benchmark check

```bash
python src/regr_fail_bucketing.py --input B_samples_20260516/problem/benchmark_set_1/input.csv --output /tmp/out_set1.csv --k 8
python eval.py --output /tmp/out_set1.csv --golden B_samples_20260516/problem/benchmark_set_1/golden.csv

python src/regr_fail_bucketing.py --input B_samples_20260516/problem/benchmark_set_2/input.csv --output /tmp/out_set2.csv --k 8
python eval.py --output /tmp/out_set2.csv --golden B_samples_20260516/problem/benchmark_set_2/golden.csv
```

---

## Full Workflow

1. Read input CSV (`Case`, `Regr Log`, `Sim Log`, `Trace Log`).
2. Parse each case into structured fields:
   - `regr.log`: mismatch statistics and mismatch-side details.
   - `sim.log(.gz)`: UVM severity counts, fatal type, plusargs, assert modules, sequence events.
   - `trace.log(.gz)`: tail instruction pattern, loop/stall/anomaly indicators.
3. Split cases into two tracks:
   - `MISMATCH` track: cases with RTL-ISS mismatch.
   - `FAIL` track: cases without mismatch.
4. Build track-specific feature vectors.
5. Cluster each track independently (Ward agglomerative clustering).
6. Offset and merge track labels into final bucket IDs.
7. Write output CSV (`Case,bucket`).

---

## Algorithm (Pseudo Code)

```text
for each case in input_csv:
    regr = parse_regr_log(case.regr_log)
    sim  = parse_sim_log(case.sim_log)
    trace= parse_trace_log(case.trace_log)
    parsed_cases.append({regr, sim, trace})

mismatch_indices = [i | parsed_cases[i].regr.has_mismatch]
fail_indices     = [i | not parsed_cases[i].regr.has_mismatch]

for track in [MISMATCH, FAIL]:
    X = build_track_features(parsed_cases[track.indices], track)
    k_track = min(user_k, len(track.indices), max(1, len(track.indices) // density(track)))
    y_track = AgglomerativeClustering(linkage=ward, n_clusters=k_track).fit_predict(standardize(X))
    assign global bucket ids with track offset

write output.csv
```

---

## Algorithm Description

### Q1. How FAIL track extracts features

FAIL track emphasizes UVM/testbench behavior:

1. **Severity profile**
   - `uvm_info/warning/error/fatal` counts (log-scaled).
2. **Fatal semantics**
   - one-hot `fatal_kind` (e.g., `CORE_STATUS_TIMEOUT`, `WALL_CLOCK_TIMEOUT`, `MCAUSE_CHECK`, `DRET_CHECK`, `HANDSHAKE_FAIL`, etc.).
   - one-hot `core_status_kind` (e.g., `IN_DEBUG_MODE`, `HANDLING_IRQ`, ...).
   - fatal location flags (`core_ibex_base_test.sv`, `core_ibex_test_lib.sv`, cosim source).
3. **Testbench mode from plusargs**
   - `enable_debug_seq`, `enable_irq_single_seq`, `enable_irq_multiple_seq`, `has_max_interval`.
4. **Assertion structure**
   - module-level assert counts (`cs_registers_i`, `id_stage_i`, `load_store_unit_i`, `controller_i`, `other`).
5. **Run magnitude and progression**
   - `finish_time`, `assert_count`, sequence-event counts (`irq_raise/drop/debug`), `reached_test_done`.
6. **Fallback weak hashes**
   - low-weight hashes of `uvm_test_name` and normalized plusarg signature for unseen templates.

### Q2. How MISMATCH track extracts features

MISMATCH track emphasizes RTL-vs-ISS divergence and tail execution pattern:

1. **Numeric mismatch summary**
   - `mismatch_count`, `matched_count`, trace line count (log-scaled).
2. **Trace behavior**
   - `has_loop`, `has_pc_stall`, `pc_unique_ratio`, `anomaly_score`.
3. **Mismatch identity**
   - hashed `loop_signature`, `tail_signature`.
   - hashed `ibex_mnemonic`, `spike_mnemonic`.
   - hashed shortened `ibex_pc` and `spike_pc`.

### Q3. How k is decided for two tracks

We use track-dependent density heuristics and user-provided cap:

- `density(MISMATCH) = 8`
- `density(FAIL) = 3`

Then:

```text
k_track = min(user_k, n_track, max(1, n_track // density(track)))
```

Rationale:
- mismatch cases are often dense and can be over-split easily, so they use larger density (smaller k).
- fail cases are more diverse by testbench behavior, so they use smaller density (larger k).

---

## Key Research Contributions

1. **Split-track architecture**
   - explicit separation of mismatch-driven and fail-driven root-cause patterns.
2. **UVM-aware FAIL features**
   - semantic fatal typing + plusarg mode + assertion-module distribution + run progression signals.
3. **Robust parser upgrades**
   - structured extraction from compressed/plain logs and reduced dependence on raw free text.
4. **Track-specific k strategy**
   - reduced over-fragmentation with stable public benchmark performance.

---

## Prior Work / Non-Contributions Disclosure

The following are **not** claimed as original contributions of this final project:

1. **Contest-provided assets**
   - problem statement, sample datasets/logs, sample solutions, and official evaluation script.
2. **Third-party packages/frameworks**
   - `numpy`, `pandas`, `scikit-learn`.
   - clustering algorithm implementation (`AgglomerativeClustering`) and preprocessing (`StandardScaler`) from scikit-learn.
3. **General UVM/ibex knowledge**
   - UVM reporting mechanism and known ibex failure templates from public docs/issues.
4. **Baseline code before this semester work**
   - any existing repository code and provided baseline structure prior to this project's modifications.

---

## Noticeable Implementation Details

- Handles both `.log` and `.log.gz`.
- Uses structured categorical features (one-hot) for major fatal templates to reduce hash collision.
- Keeps weak hash fallback features to preserve generalization for unseen hidden-test messages.
- Uses standardized features before Ward clustering.

---

## Experimental Results (Public Benchmarks)

Using `--k 8` with the current implementation:

- `benchmark_set_1`: **Balanced Accuracy = 0.550000**
- `benchmark_set_2`: **Balanced Accuracy = 0.898749**

Observed behavior:
- public-set performance is maintained (no regression vs prior baseline).
- fail-track features become more interpretable and more robust for unseen UVM fatal families.