#!/usr/bin/env python3
"""Entry point for regression-failure bucketing.

Usage:
    python src/regr_fail_bucketing.py \
        --input  B_samples_20260516/problem/benchmark_set_1/input.csv \
        --output output.csv \
        --k 2

The program reads an input CSV describing N failure cases (each with three log
file paths), extracts features per case, and writes an output CSV with the
columns ``Case,bucket``.  Bucket IDs are arbitrary strings and only their
groupings are scored (pairwise balanced accuracy; see ``eval.py``).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python src/regr_fail_bucketing.py` from the repo root and
# `regr_fail_bucketing` from a PyInstaller bundle alike.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from clustering import cluster, summarize  # noqa: E402
from features import build_case_features  # noqa: E402


def _seed_all(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def _read_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Case", "Regr Log", "Sim Log", "Trace Log"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing required columns: {sorted(missing)} "
            f"(found {list(df.columns)})"
        )
    return df


def _bucket_string(label: int) -> str:
    """Format bucket IDs as `bucket_<i>` strings.  Evaluation is invariant to
    the actual string, but human-readable labels help debugging."""
    return f"bucket_{label}"


def run(args: argparse.Namespace) -> int:
    _seed_all(args.seed)

    t0 = time.time()
    df_input = _read_input(args.input)
    base_dir = os.path.dirname(os.path.abspath(args.input))

    if args.verbose:
        print(
            f"[bucketing] {len(df_input)} cases, k={args.k}, "
            f"method={args.method}",
            file=sys.stderr,
        )

    rows = df_input.to_dict("records")

    def _extract(row: dict) -> CaseFeatures:
        return build_case_features(
            case_id=int(row["Case"]),
            base_dir=base_dir,
            regr_rel=str(row["Regr Log"]),
            sim_rel=str(row["Sim Log"]),
            trace_rel=str(row["Trace Log"]),
        )

    workers = min(max(1, args.workers), len(rows) or 1)
    if workers == 1:
        features = [_extract(row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            features = list(pool.map(_extract, rows))

    if args.verbose:
        n_distinct = len({f.signature.categorical_key() for f in features})
        print(
            f"[bucketing] extracted features in {time.time()-t0:.2f}s; "
            f"{n_distinct} distinct signatures",
            file=sys.stderr,
        )

    labels = cluster(features, k=args.k, method=args.method)

    df_output = pd.DataFrame(
        {
            "Case": [f.case_id for f in features],
            "bucket": [_bucket_string(l) for l in labels],
        }
    )
    df_output.to_csv(args.output, index=False)

    if args.verbose:
        n_buckets = len(set(labels))
        print(
            f"[bucketing] wrote {len(df_output)} rows in "
            f"{n_buckets} bucket(s) -> {args.output} "
            f"(total {time.time()-t0:.2f}s)",
            file=sys.stderr,
        )
        print(summarize(features, labels), file=sys.stderr)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="regr_fail_bucketing",
        description="Cluster RTL regression failure logs into per-bug buckets.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV with columns: Case, Regr Log, Sim Log, Trace Log",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV with columns: Case, bucket",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Soft hint for the number of buckets (= injected bugs)",
    )
    parser.add_argument(
        "--method",
        default="hybrid",
        choices=("signature", "tfidf", "hybrid", "signature_then_tfidf"),
        help="Clustering strategy (default: hybrid)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel feature-extraction threads (0 = auto, 1 = serial)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress and a per-bucket summary to stderr",
    )
    args = parser.parse_args(argv)
    if args.workers <= 0:
        args.workers = min(8, max(1, (os.cpu_count() or 4)))
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
