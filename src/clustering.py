"""Clustering strategies built on top of :mod:`features`.

Four strategies are exposed:

``signature``
    Group cases by their exact categorical key.  Fast and very precise when
    the bug signatures are clean (typical for the public benchmarks).

``tfidf``
    Vectorize each case's normalized text blob with TF-IDF and run
    AgglomerativeClustering with cosine distance / average linkage targeting
    exactly ``k`` clusters.

``hybrid``  (default)
    Build a custom per-pair distance combining categorical signature overlap
    (fatal template, assertion set Jaccard, mismatch flag, verdict, mismatch
    mnemonics) with a TF-IDF cosine residual.  Run AgglomerativeClustering
    with precomputed distances and ``average`` linkage targeting ``k``
    clusters.  Identical-signature cases get distance 0 (effectively
    must-link).

``signature_then_tfidf``
    Run ``signature`` first.  If the number of distinct signatures is greater
    than ``k`` (we over-segmented), merge signature groups using TF-IDF
    centroid distance until we hit ``k``.  Otherwise keep them as-is.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from features import CaseFeatures, CaseSignature


# ---------------------------------------------------------------------------
# Signature strategy
# ---------------------------------------------------------------------------

def cluster_by_signature(features: Sequence[CaseFeatures]) -> list[int]:
    """Assign one bucket id per distinct categorical signature."""
    bucket_of_key: dict[tuple, int] = {}
    labels = []
    for f in features:
        key = f.signature.categorical_key()
        if key not in bucket_of_key:
            bucket_of_key[key] = len(bucket_of_key)
        labels.append(bucket_of_key[key])
    return labels


# ---------------------------------------------------------------------------
# TF-IDF strategy
# ---------------------------------------------------------------------------

def _tfidf_matrix(features: Sequence[CaseFeatures]):
    """Build a TF-IDF matrix from the per-case text blobs."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = [f.text_blob if f.text_blob else f"empty_case_{f.case_id}" for f in features]
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        token_pattern=r"[A-Za-z_][A-Za-z0-9_<>]+",
    )
    matrix = vec.fit_transform(corpus)
    return matrix


def cluster_by_tfidf(features: Sequence[CaseFeatures], k: int) -> list[int]:
    """Cluster cases via TF-IDF + agglomerative clustering."""
    from sklearn.cluster import AgglomerativeClustering

    n = len(features)
    if n <= 1:
        return [0] * n
    k_eff = max(1, min(k, n))
    matrix = _tfidf_matrix(features)
    dense = matrix.toarray()
    if k_eff == 1:
        return [0] * n
    agg = AgglomerativeClustering(
        n_clusters=k_eff,
        metric="cosine",
        linkage="average",
    )
    labels = agg.fit_predict(dense)
    return labels.tolist()


# ---------------------------------------------------------------------------
# Hybrid strategy: signature-aware pairwise distance + agglomerative
# ---------------------------------------------------------------------------

def _centroid(rows: np.ndarray) -> np.ndarray:
    centroid = rows.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def _jaccard(a: tuple, b: tuple) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def _pairwise_signature_distance(
    sa: CaseSignature, sb: CaseSignature
) -> float:
    """A handcrafted, bounded distance in [0, 1] between two case signatures.

    Lower means "more likely to share a bug".  The score combines several
    independent observations and is calibrated so that:

    * Identical fingerprints score 0.
    * Cases differing only on noisy fields (test name, mismatch mnemonic)
      stay in the 0.1-0.4 range.
    * Cases with completely disjoint failure modes (mismatch vs assertion vs
      timeout) get distance close to 1.
    """
    # Failure-mode signal: assertion vs fatal-timeout vs mismatch vs pass.
    def mode(s: CaseSignature) -> str:
        if s.sim_error_asserts:
            return "assert"
        if s.regr_kind == "mismatch":
            return "mismatch"
        if s.sim_fatal_template:
            return "fatal"
        if s.sim_verdict == "passed":
            return "passed"
        return "other"

    ma, mb = mode(sa), mode(sb)
    mode_dist = 0.0 if ma == mb else 1.0

    # Assertion-set Jaccard (1.0 means identical, 0 means no overlap)
    if sa.sim_error_asserts or sb.sim_error_asserts:
        assert_sim = _jaccard(sa.sim_error_asserts, sb.sim_error_asserts)
    else:
        assert_sim = 1.0  # neither has assertions -> neutral
    assert_dist = 1.0 - assert_sim

    # Fatal template match -- soft-shaded by category + source file so cases
    # with closely-related-but-not-identical UVM_FATAL messages still align.
    if sa.sim_fatal_template and sb.sim_fatal_template:
        if sa.sim_fatal_template == sb.sim_fatal_template:
            fatal_dist = 0.0
        elif (sa.sim_fatal_source == sb.sim_fatal_source
              and sa.sim_fatal_source != ""
              and sa.sim_fatal_category == sb.sim_fatal_category):
            # Same source file + same category (e.g. both timeouts from
            # core_ibex_base_test.sv).  Probably same bug family.
            fatal_dist = 0.35
        elif (sa.sim_fatal_category == sb.sim_fatal_category
              and sa.sim_fatal_category not in ("", "none")):
            fatal_dist = 0.6
        else:
            fatal_dist = 1.0
    elif not sa.sim_fatal_template and not sb.sim_fatal_template:
        fatal_dist = 0.0
    else:
        fatal_dist = 1.0

    # Verdict
    verdict_dist = 0.0 if sa.sim_verdict == sb.sim_verdict else 1.0

    # Regr kind
    regr_dist = 0.0 if sa.regr_kind == sb.regr_kind else 1.0

    # Mnemonic overlap on mismatch line (only meaningful when both mismatch)
    mnem_dist = 0.5
    if sa.regr_kind == "mismatch" and sb.regr_kind == "mismatch":
        ia, sap = sa.regr_mismatch_mnemonics
        ib, sbp = sb.regr_mismatch_mnemonics
        ibex_match = (ia == ib) and ia != ""
        spike_match = (sap == sbp) and sap != ""
        if ibex_match and spike_match:
            mnem_dist = 0.0
        elif ibex_match or spike_match:
            mnem_dist = 0.3
        else:
            mnem_dist = 0.7

    # Tail mnemonic sequence equality (helps tell apart same-fatal cases
    # whose execution diverged earlier).
    tail_dist = 0.5
    if sa.trace_tail_mnemonics and sb.trace_tail_mnemonics:
        if sa.trace_tail_mnemonics == sb.trace_tail_mnemonics:
            tail_dist = 0.0
        else:
            common = sum(
                1 for x, y in zip(sa.trace_tail_mnemonics, sb.trace_tail_mnemonics)
                if x == y
            )
            tail_dist = 1.0 - common / max(
                len(sa.trace_tail_mnemonics), len(sb.trace_tail_mnemonics)
            )

    # Weighted combination.  Assertion / fatal / mode dominate; tail and
    # mnemonics provide a finer-grained tie-breaker.
    weights = {
        "mode": 0.35,
        "fatal": 0.20,
        "assert": 0.20,
        "regr": 0.10,
        "verdict": 0.05,
        "mnem": 0.05,
        "tail": 0.05,
    }
    score = (
        weights["mode"] * mode_dist
        + weights["fatal"] * fatal_dist
        + weights["assert"] * assert_dist
        + weights["regr"] * regr_dist
        + weights["verdict"] * verdict_dist
        + weights["mnem"] * mnem_dist
        + weights["tail"] * tail_dist
    )
    return float(min(1.0, max(0.0, score)))


def _build_distance_matrix(features: Sequence[CaseFeatures]) -> np.ndarray:
    """Combine signature distance with a small TF-IDF residual."""
    n = len(features)
    D = np.zeros((n, n), dtype=float)
    sigs = [f.signature for f in features]
    # Categorical signature contribution.
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = _pairwise_signature_distance(sigs[i], sigs[j])
            D[j, i] = D[i, j]
    # TF-IDF residual: small weight, but helps within groups that look
    # otherwise identical.
    try:
        tfidf = _tfidf_matrix(features).toarray()
        # Cosine distance between rows.
        norms = np.linalg.norm(tfidf, axis=1)
        norms_safe = np.where(norms > 0, norms, 1.0)
        normed = tfidf / norms_safe[:, None]
        cos_sim = normed @ normed.T
        cos_dist = np.clip(1.0 - cos_sim, 0.0, 1.0)
        D = 0.85 * D + 0.15 * cos_dist
    except Exception:
        pass
    # Force diagonal to 0.
    np.fill_diagonal(D, 0.0)
    return D


def cluster_hybrid(features: Sequence[CaseFeatures], k: int) -> list[int]:
    """Signature-aware agglomerative clustering using a precomputed metric."""
    from sklearn.cluster import AgglomerativeClustering

    n = len(features)
    if n == 0:
        return []
    if n == 1:
        return [0]
    k_eff = max(1, min(k, n))
    if k_eff == 1:
        return [0] * n

    D = _build_distance_matrix(features)
    agg = AgglomerativeClustering(
        n_clusters=k_eff,
        metric="precomputed",
        linkage="average",
    )
    labels = agg.fit_predict(D)
    return labels.tolist()


# ---------------------------------------------------------------------------
# Signature-then-TF-IDF merge strategy (kept for ablation / fallback)
# ---------------------------------------------------------------------------

def cluster_signature_then_tfidf(
    features: Sequence[CaseFeatures], k: int
) -> list[int]:
    """Group by signature, then merge groups via TF-IDF centroid distance."""
    n = len(features)
    if n == 0:
        return []
    if n == 1:
        return [0]

    sig_labels = cluster_by_signature(features)
    n_sigs = len(set(sig_labels))
    k_eff = max(1, min(k, n))
    if n_sigs <= k_eff:
        return sig_labels

    matrix = _tfidf_matrix(features).toarray()
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(sig_labels):
        groups[lbl].append(idx)
    centroids = {gid: _centroid(matrix[groups[gid]]) for gid in groups}
    active = list(groups)
    parent = {gid: gid for gid in active}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    while len(active) > k_eff:
        best = None
        for i, gi in enumerate(active):
            for gj in active[i + 1:]:
                d = _cosine(centroids[gi], centroids[gj])
                if best is None or d < best[0]:
                    best = (d, gi, gj)
        if best is None:
            break
        _, gi, gj = best
        merged_rows = matrix[groups[gi] + groups[gj]]
        groups[gi] = groups[gi] + groups[gj]
        centroids[gi] = _centroid(merged_rows)
        del groups[gj]
        del centroids[gj]
        parent[gj] = gi
        active.remove(gj)

    new_label_of_root: dict[int, int] = {}
    out = [0] * n
    for idx, old in enumerate(sig_labels):
        root = find(old)
        if root not in new_label_of_root:
            new_label_of_root[root] = len(new_label_of_root)
        out[idx] = new_label_of_root[root]
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def cluster(features: Sequence[CaseFeatures], k: int, method: str) -> list[int]:
    method = method.lower()
    if method == "signature":
        return cluster_by_signature(features)
    if method == "tfidf":
        return cluster_by_tfidf(features, k)
    if method == "hybrid":
        return cluster_hybrid(features, k)
    if method == "signature_then_tfidf":
        return cluster_signature_then_tfidf(features, k)
    raise ValueError(f"Unknown clustering method: {method!r}")


def summarize(features: Iterable[CaseFeatures], labels: Sequence[int]) -> str:
    """Human-readable summary for `-v` output."""
    by_label: dict[int, list[CaseFeatures]] = defaultdict(list)
    for f, lbl in zip(features, labels):
        by_label[lbl].append(f)
    lines = []
    for lbl in sorted(by_label):
        members = by_label[lbl]
        sig = members[0].signature
        lines.append(
            f"bucket {lbl}: {len(members)} case(s) "
            f"verdict={sig.sim_verdict} fatal=\"{sig.sim_fatal_template[:60]}\" "
            f"asserts={list(sig.sim_error_asserts)} regr={sig.regr_kind}"
        )
        ids = sorted(m.case_id for m in members)
        lines.append(f"  cases: {ids}")
    return "\n".join(lines)
