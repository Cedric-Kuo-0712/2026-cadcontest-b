#!/usr/bin/env python3
"""
Evaluate regression failure bucketing via pairwise balanced accuracy.

Metric (B_20260212.pdf, Section 3.1):
  - Each pair (i, j) is positive if they share the same golden bucket (bug).
  - TPR = TP / (TP + FN)
  - TNR = TN / (TN + FP)
  - Balanced Accuracy = (TPR + TNR) / 2
"""

import argparse
import sys

import pandas as pd


def load_labels(path: str, label_column: str) -> pd.Series:
    """Load Case -> label mapping from CSV."""
    df = pd.read_csv(path)
    if "Case" not in df.columns:
        raise ValueError(f"{path}: missing required column 'Case'")
    if label_column not in df.columns:
        raise ValueError(
            f"{path}: missing required column '{label_column}' "
            f"(found: {list(df.columns)})"
        )
    if df["Case"].duplicated().any():
        dupes = df.loc[df["Case"].duplicated(), "Case"].tolist()
        raise ValueError(f"{path}: duplicate Case values: {dupes}")

    series = df.set_index("Case")[label_column].astype(str)
    return series.sort_index()


def pairwise_counts(golden: pd.Series, predicted: pd.Series) -> dict:
    """
    Count TP, FN, FP, TN over all unordered case pairs.

    Positive pair: same golden label.
    Predicted positive: same predicted bucket.
    """
    cases = golden.index.intersection(predicted.index)
    if len(cases) == 0:
        raise ValueError("No overlapping Case IDs between golden and output")

    missing_golden = predicted.index.difference(golden.index)
    missing_pred = golden.index.difference(predicted.index)
    if len(missing_golden) > 0 or len(missing_pred) > 0:
        raise ValueError(
            "Case ID mismatch between files.\n"
            f"  only in output: {sorted(missing_golden.tolist())}\n"
            f"  only in golden: {sorted(missing_pred.tolist())}"
        )

    g = golden.loc[cases].to_numpy()
    p = predicted.loc[cases].to_numpy()
    n = len(cases)

    tp = fn = fp = tn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_golden = g[i] == g[j]
            same_pred = p[i] == p[j]
            if same_golden and same_pred:
                tp += 1
            elif same_golden and not same_pred:
                fn += 1
            elif not same_golden and same_pred:
                fp += 1
            else:
                tn += 1

    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "n_cases": n}


def balanced_accuracy(counts: dict) -> dict:
    """Compute TPR, TNR, and balanced accuracy from pairwise counts."""
    tp, fn, fp, tn = counts["tp"], counts["fn"], counts["fp"], counts["tn"]
    p = tp + fn  # positive pairs (same golden bucket)
    n_neg = tn + fp  # negative pairs (different golden buckets)

    tpr = 1.0 if p == 0 else tp / p
    tnr = 1.0 if n_neg == 0 else tn / n_neg
    ba = (tpr + tnr) / 2.0

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "positive_pairs": p,
        "negative_pairs": n_neg,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": ba,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute pairwise balanced accuracy for bucketing output."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Predicted buckets CSV (columns: Case, bucket)",
    )
    parser.add_argument(
        "--golden",
        required=True,
        help="Golden labels CSV (columns: Case, Bug)",
    )
    parser.add_argument(
        "--pred-col",
        default="bucket",
        help="Predicted label column name (default: bucket)",
    )
    parser.add_argument(
        "--golden-col",
        default="Bug",
        help="Golden label column name (default: Bug)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print pairwise contingency table and rates",
    )
    args = parser.parse_args()

    try:
        predicted = load_labels(args.output, args.pred_col)
        golden = load_labels(args.golden, args.golden_col)
        counts = pairwise_counts(golden, predicted)
        metrics = balanced_accuracy(counts)
    except Exception as e:
        print(f"[!] {e}", file=sys.stderr)
        return 1

    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.6f}")

    if args.verbose:
        print(f"Cases: {counts['n_cases']}")
        print(
            f"Pairs: {metrics['positive_pairs']} positive (same bug), "
            f"{metrics['negative_pairs']} negative (different bugs)"
        )
        print(
            f"TP={metrics['tp']} FN={metrics['fn']} "
            f"FP={metrics['fp']} TN={metrics['tn']}"
        )
        print(f"TPR = {metrics['tpr']:.6f}")
        print(f"TNR = {metrics['tnr']:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
