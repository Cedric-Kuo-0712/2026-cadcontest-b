"""
Split-track bucketing:
- mismatch cases use regr + trace features
- failed-only cases use sim + regr features
"""

import hashlib
import math
from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from log_parser import ASSERT_MODULES, CORE_STATUS_KINDS, FATAL_KIND_RULES

FATAL_KINDS = ["NO_FATAL"] + [k for k, _ in FATAL_KIND_RULES] + ["OTHER_FATAL"]
CORE_STATUS_LABELS = ["NONE"] + CORE_STATUS_KINDS + ["OTHER"]
ASSERT_MODULE_LABELS = ASSERT_MODULES + ["other"]


def _hash_vec(value: str, dim: int, weight: float = 1.0) -> np.ndarray:
    vec = np.zeros(dim, dtype=float)
    if not value:
        return vec
    digest = hashlib.md5(value.encode("utf-8", errors="ignore")).digest()
    vec[int.from_bytes(digest[:4], "little") % dim] = weight
    return vec


def _one_hot(value: str, labels: List[str], weight: float = 1.0) -> np.ndarray:
    vec = np.zeros(len(labels), dtype=float)
    if value in labels:
        vec[labels.index(value)] = weight
    return vec


def _scaled_log(value: float, denom: float = 10.0) -> float:
    return min(math.log1p(max(value, 0.0)) / denom, 1.0)


class SplitTrackBucketer:
    def __init__(self, seed: int = 42):
        self.seed = seed

    # Per-track expected cases-per-bug. MISMATCH cases stay densely packed under
    # a single cosim signature; FAIL cases are more varied so smaller denominator.
    _DENSITY = {"MISMATCH": 8, "FAIL": 3}

    def cluster(self, parsed_results: List[Dict], n_clusters: int, verbose: bool = False) -> np.ndarray:
        groups = {
            "MISMATCH": [i for i, p in enumerate(parsed_results) if p["regr"]["has_mismatch"]],
            "FAIL": [i for i, p in enumerate(parsed_results) if not p["regr"]["has_mismatch"]],
        }

        labels = np.full(len(parsed_results), -1, dtype=int)
        next_bucket = 0

        for track_name in ("MISMATCH", "FAIL"):
            indices = groups[track_name]
            if not indices:
                continue

            feats_fn = (
                self._mismatch_features if track_name == "MISMATCH" else self._fail_features
            )
            features = np.vstack([feats_fn(parsed_results[i]) for i in indices])

            track_k = min(
                len(indices),
                max(1, n_clusters),
                max(1, len(indices) // self._DENSITY[track_name]),
            )
            track_labels = self._cluster_features(features, track_k)

            for local_idx, case_idx in enumerate(indices):
                labels[case_idx] = next_bucket + track_labels[local_idx]

            if verbose:
                print(f"    - {track_name}: {len(indices)} cases -> {len(set(track_labels))} buckets")
            next_bucket += max(track_labels) + 1

        return labels

    def _cluster_features(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        if len(features) == 1 or n_clusters <= 1:
            return np.zeros(len(features), dtype=int)
        norm = StandardScaler().fit_transform(features)
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        return model.fit_predict(norm)

    def _fail_features(self, parsed: Dict) -> np.ndarray:
        sim = parsed["sim"]
        modules = sim["assert_module_counts"] or {}

        severity = np.array([
            _scaled_log(sim["uvm_info_count"], 12),
            _scaled_log(sim["uvm_warning_count"]),
            _scaled_log(sim["uvm_error_count"], 12),
            _scaled_log(sim["uvm_fatal_count"]),
        ])

        # 2x weight on the discrete fatal kind: this is the single strongest cue.
        fatal_kind = _one_hot(sim["fatal_kind"], FATAL_KINDS, weight=2.0)
        core_status = _one_hot(sim["core_status_kind"], CORE_STATUS_LABELS, weight=1.5)

        fatal_loc = np.array([
            float(sim["fatal_source"] == "core_ibex_base_test.sv"),
            float(sim["fatal_source"] == "core_ibex_test_lib.sv"),
            float("cosim" in sim["fatal_source"]),
        ])

        tb_mode = np.array([
            float(sim["enable_debug_seq"]),
            float(sim["enable_irq_single_seq"]),
            float(sim["enable_irq_multiple_seq"]),
            float(sim["has_max_interval"]),
        ])

        assert_modules = np.array([
            _scaled_log(modules.get(m, 0)) for m in ASSERT_MODULE_LABELS
        ])

        magnitude = np.array([
            _scaled_log(sim["finish_time"], 25),
            _scaled_log(sim["assert_count"]),
            _scaled_log(sim["total_seq_events"]),
            _scaled_log(sim["irq_raise_count"]),
            _scaled_log(sim["irq_drop_count"]),
            _scaled_log(sim["debug_seq_count"]),
        ])

        flags = np.array([
            float(sim["reached_test_done"]),
            float(sim["assert_count"] > 5),
            float(sim["uvm_warning_count"] > 0),
            float(sim["has_simulator_error"]),
        ])

        # Weak hash fallback for testname / plusarg variations not in our enums.
        fallback = np.concatenate([
            _hash_vec(sim["uvm_test_name"], 8, 0.4),
            _hash_vec(sim["plusarg_signature"], 8, 0.4),
        ])

        return np.concatenate([
            severity, fatal_kind, core_status, fatal_loc,
            tb_mode, assert_modules, magnitude, flags, fallback,
        ])

    def _mismatch_features(self, parsed: Dict) -> np.ndarray:
        regr = parsed["regr"]
        trace = parsed["trace"]

        numeric = np.array([
            _scaled_log(regr["mismatch_count"]),
            _scaled_log(regr["matched_count"]),
            float(trace["has_loop"]),
            float(trace["has_pc_stall"]),
            trace["pc_unique_ratio"],
            _scaled_log(trace["line_count"]),
            trace["anomaly_score"],
        ])

        return np.concatenate([
            numeric,
            _hash_vec(trace["loop_signature"], 28, 2.0),
            _hash_vec(trace["tail_signature"], 20),
            _hash_vec(regr["ibex_mnemonic"], 12),
            _hash_vec(regr["spike_mnemonic"], 12),
            _hash_vec(regr["ibex_pc"][:6], 8),
            _hash_vec(regr["spike_pc"][:8], 8),
        ])
