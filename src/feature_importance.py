#!/usr/bin/env python3
"""
Feature importance analysis (pooled benchmarks) and clustering evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Allow running as script from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_features import (
    FeatureMatrix,
    build_feature_matrix,
    load_benchmark_cases,
    load_pooled_cases,
    save_selected_features,
)
from strategies import SimpleHierarchicalClustering


def compute_rf_importance(
    fm: FeatureMatrix, n_estimators: int = 200, seed: int = 42
) -> Tuple[np.ndarray, float]:
    le = LabelEncoder()
    y = le.fit_transform(fm.bugs)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        class_weight="balanced",
        max_depth=8,
        min_samples_leaf=2,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(clf, fm.X, y, cv=cv, scoring="accuracy")
    clf.fit(fm.X, y)
    return clf.feature_importances_, float(cv_scores.mean())


def grouped_view_importance(fm: FeatureMatrix, importances: np.ndarray) -> Dict[str, float]:
    view_scores: Dict[str, float] = defaultdict(float)
    view_groups = fm.view_indices()
    for view, idxs in view_groups.items():
        view_scores[view] = float(importances[idxs].sum())
    return dict(sorted(view_scores.items(), key=lambda x: -x[1]))


def select_features(
    fm: FeatureMatrix,
    importances: np.ndarray,
    mode: str,
    top_k: int,
    min_char: bool = True,
) -> List[str]:
    """Select feature names for clustering."""
    if mode == "char_only":
        return [n for n in fm.names if n.startswith("char_")]

    if mode == "char_struct":
        return [n for n in fm.names if n.startswith("char_") or n.startswith("struct_")]

    if mode == "top_k":
        order = np.argsort(-importances)
        chosen = [fm.names[i] for i in order[:top_k]]
        if min_char:
            for n in fm.names:
                if n.startswith("char_") and n not in chosen:
                    chosen.append(n)
        return chosen

    if mode == "top_views":
        view_groups = fm.view_indices()
        view_imp = {v: importances[idxs].sum() for v, idxs in view_groups.items()}
        ranked_views = sorted(view_imp, key=view_imp.get, reverse=True)
        # Always include char_* views + top 2 other views
        keep_views = {"char_sim", "char_trace", "char_regr"}
        for v in ranked_views:
            if v not in keep_views and len(keep_views) < 5:
                keep_views.add(v)
        chosen = []
        for n in fm.names:
            if fm.view_for_name(n) in keep_views:
                chosen.append(n)
        return chosen

    return list(fm.names)


def write_importance_csv(
    out_path: str,
    fm: FeatureMatrix,
    importances: np.ndarray,
    method: str = "rf",
) -> None:
    rows = []
    order = np.argsort(-importances)
    for rank, idx in enumerate(order, start=1):
        name = fm.names[idx]
        rows.append(
            {
                "feature": name,
                "view": fm.view_for_name(name),
                "score": float(importances[idx]),
                "rank": rank,
                "method": method,
            }
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def cluster_and_evaluate(
    fm: FeatureMatrix,
    feature_names: List[str],
    n_clusters: int,
    linkage: str = "ward",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Cluster cases; return labels and per-benchmark supervised clustering accuracy (oracle)."""
    idx = [i for i, n in enumerate(fm.names) if n in feature_names]
    X = fm.X[:, idx]

    parsed_stub = [{"error_type": "UNKNOWN"} for _ in fm.sample_ids]
    clusterer = SimpleHierarchicalClustering(linkage=linkage, seed=42)
    labels = clusterer.cluster(X, parsed_stub, n_clusters=n_clusters, verbose=False)

    # Per-benchmark breakdown: accuracy of bug prediction via cluster purity proxy
    results = {}
    for bench in sorted(set(fm.benchmarks)):
        mask = [i for i, b in enumerate(fm.benchmarks) if b == bench]
        sub_bugs = [fm.bugs[i] for i in mask]
        sub_labels = labels[mask]
        # Simple label alignment score (not official metric)
        results[bench] = float(_cluster_label_agreement(sub_bugs, sub_labels))

    return labels, results


def _cluster_label_agreement(bugs: List[str], labels: np.ndarray) -> float:
    """Fraction of pairs where cluster co-assignment matches bug co-assignment."""
    n = len(bugs)
    if n < 2:
        return 1.0
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_bug = bugs[i] == bugs[j]
            same_cl = labels[i] == labels[j]
            if same_bug == same_cl:
                agree += 1
            total += 1
    return agree / total if total else 0.0


def run_eval_py(
    repo_root: str,
    output_csv: str,
    benchmark: str,
) -> float:
    import subprocess

    golden = os.path.join(
        repo_root,
        "B_samples_20260516/problem",
        benchmark,
        "golden.csv",
    )
    proc = subprocess.run(
        [sys.executable, os.path.join(repo_root, "eval.py"), "--output", output_csv, "--golden", golden],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    for line in proc.stdout.splitlines():
        if "Balanced Accuracy" in line:
            return float(line.split(":")[-1].strip())
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature importance & clustering analysis")
    parser.add_argument(
        "--samples-root",
        default="B_samples_20260516/problem",
        help="Root containing benchmark_set_*",
    )
    parser.add_argument(
        "--benchmark",
        choices=["benchmark_set_1", "benchmark_set_2", "both"],
        default="both",
    )
    parser.add_argument("--output-dir", default="analysis_outputs/pooled")
    parser.add_argument("--cache-dir", default="analysis_cache")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutation", action="store_true")
    parser.add_argument("--run-clustering", action="store_true", default=True)
    parser.add_argument("--no-clustering", action="store_false", dest="run_clustering")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_root = os.path.join(repo_root, args.samples_root)

    cases, sids, bugs, benches, nums = load_pooled_cases(samples_root, args.benchmark)
    if args.verbose:
        print(f"[*] Loaded {len(cases)} cases ({args.benchmark})")

    fm = build_feature_matrix(cases, sids, bugs, benches, nums)
    if args.verbose:
        print(f"[*] Feature matrix: {fm.X.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    imp, cv_acc = compute_rf_importance(fm, seed=args.seed)
    if args.verbose:
        print(f"[*] RF 5-fold CV accuracy (Bug): {cv_acc:.4f}")

    write_importance_csv(
        os.path.join(args.output_dir, "feature_importances_rf.csv"),
        fm,
        imp,
    )

    view_imp = grouped_view_importance(fm, imp)
    pd.DataFrame(
        [{"view": k, "score": v} for k, v in view_imp.items()]
    ).to_csv(os.path.join(args.output_dir, "view_summary.csv"), index=False)

    if args.permutation:
        le = LabelEncoder()
        y = le.fit_transform(fm.bugs)
        clf = RandomForestClassifier(
            n_estimators=100, random_state=args.seed, class_weight="balanced", max_depth=8
        )
        clf.fit(fm.X, y)
        perm = permutation_importance(
            clf, fm.X, y, n_repeats=10, random_state=args.seed, scoring="accuracy"
        )
        write_importance_csv(
            os.path.join(args.output_dir, "feature_importances_permutation.csv"),
            fm,
            perm.importances_mean,
            method="permutation",
        )

    # Feature selection presets
    selections = {
        "char_only": select_features(fm, imp, "char_only", args.top_k),
        "char_struct": select_features(fm, imp, "char_struct", args.top_k),
        "top_k": select_features(fm, imp, "top_k", args.top_k),
        "top_views": select_features(fm, imp, "top_views", args.top_k),
    }

    best_name = "char_only"
    best_ba = -1.0
    clustering_rows = []

    if args.run_clustering:
        for bench in sorted(set(fm.benchmarks)):
            mask_idx = [i for i, b in enumerate(fm.benchmarks) if b == bench]
            n_clusters = len(set(fm.bugs[i] for i in mask_idx))

            bench_cases = [cases[i] for i in mask_idx]
            bench_sids = [fm.sample_ids[i] for i in mask_idx]
            bench_bugs = [fm.bugs[i] for i in mask_idx]
            bench_bench = [fm.benchmarks[i] for i in mask_idx]
            bench_nums = [fm.cases[i] for i in mask_idx]

            for sel_name, feat_names in selections.items():
                bfm = build_feature_matrix(
                    bench_cases,
                    bench_sids,
                    bench_bugs,
                    bench_bench,
                    bench_nums,
                    feature_mask=feat_names,
                )
                labels, _ = cluster_and_evaluate(bfm, bfm.names, n_clusters)

                out_dir = os.path.join(args.output_dir, "clustering", bench)
                os.makedirs(out_dir, exist_ok=True)
                out_csv = os.path.join(out_dir, f"output_{sel_name}.csv")
                pd.DataFrame({"Case": bench_nums, "bucket": labels}).to_csv(out_csv, index=False)

                ba = run_eval_py(repo_root, out_csv, bench)
                clustering_rows.append(
                    {
                        "benchmark": bench,
                        "selection": sel_name,
                        "n_features": len(bfm.names),
                        "n_clusters": n_clusters,
                        "balanced_accuracy": ba,
                    }
                )
                if args.verbose:
                    print(f"  {bench} {sel_name}: BA={ba:.4f} ({len(bfm.names)} feats)")

                if ba > best_ba:
                    best_ba = ba
                    best_name = sel_name

    # Save best selection for bucketing pipeline
    best_features = selections[best_name]
    save_selected_features(
        os.path.join(args.output_dir, "selected_features.json"),
        best_features,
        list(view_imp.keys()),
    )

    meta = {
        "n_cases": len(cases),
        "n_features": len(fm.names),
        "cv_accuracy": cv_acc,
        "view_importance": view_imp,
        "best_clustering_selection": best_name,
        "best_balanced_accuracy": best_ba,
        "selections": {k: len(v) for k, v in selections.items()},
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    pd.DataFrame(clustering_rows).to_csv(
        os.path.join(args.output_dir, "clustering_eval_summary.csv"), index=False
    )

    print(f"\n[+] View importance: {view_imp}")
    print(f"[+] Best clustering preset: {best_name} (max BA={best_ba:.4f})")
    print(f"[+] Selected features -> {args.output_dir}/selected_features.json")
    print(f"[+] Reports -> {args.output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
