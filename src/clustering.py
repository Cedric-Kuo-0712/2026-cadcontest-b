"""Clustering strategies for regression-failure bucketing."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from features import CaseFeatures, CaseSignature


def cluster_by_signature(features: Sequence[CaseFeatures]) -> list[int]:
    bucket_of_key: dict[tuple, int] = {}
    labels = []
    for f in features:
        key = f.signature.categorical_key()
        if key not in bucket_of_key:
            bucket_of_key[key] = len(bucket_of_key)
        labels.append(bucket_of_key[key])
    return labels


def _tfidf_matrix(features: Sequence[CaseFeatures]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = [
        f.text_blob if f.text_blob else f"empty_case_{f.case_id}" for f in features
    ]
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        token_pattern=r"[A-Za-z_][A-Za-z0-9_<>]+",
    )
    return vec.fit_transform(corpus)


def cluster_by_tfidf(features: Sequence[CaseFeatures], k: int) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    n = len(features)
    if n <= 1:
        return [0] * n
    k_eff = max(1, min(k, n))
    dense = _tfidf_matrix(features).toarray()
    if k_eff == 1:
        return [0] * n
    agg = AgglomerativeClustering(
        n_clusters=k_eff, metric="cosine", linkage="average"
    )
    return agg.fit_predict(dense).tolist()


def _jaccard(a: tuple, b: tuple) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def _tuple_overlap(a: tuple, b: tuple) -> float:
    if not a and not b:
        return 1.0
    if a == b:
        return 1.0
    common = sum(1 for x, y in zip(a, b) if x == y)
    return common / max(len(a), len(b), 1)


def _fatal_kind_distance(ka: str, kb: str) -> float:
    if not ka and not kb:
        return 0.0
    if ka == kb:
        return 0.0
    # Same timeout family from the same checker file often means same bug.
    timeout_kinds = {"debug_timeout", "irq_timeout"}
    if ka in timeout_kinds and kb in timeout_kinds:
        return 0.45
    check_kinds = {"check_signature", "check_memory", "check_mcause"}
    if ka in check_kinds and kb in check_kinds:
        return 0.35
    if ka in timeout_kinds and kb in check_kinds:
        return 0.55
    if ka in check_kinds and kb in timeout_kinds:
        return 0.55
    return 1.0


def _mismatch_distance(sa: CaseSignature, sb: CaseSignature) -> float:
    len_dist = 0.0 if sa.trace_length_bucket == sb.trace_length_bucket else 0.6
    if sa.trace_length_bucket == "short" or sb.trace_length_bucket == "short":
        if sa.trace_length_bucket != sb.trace_length_bucket:
            len_dist = 0.85

    loop_dist = 1.0 - _jaccard(sa.trace_loop_mnems, sb.trace_loop_mnems)
    repeat_dist = 0.0 if sa.has_repeating_tail == sb.has_repeating_tail else 0.35
    uniform_dist = abs(sa.tail_mnem_uniformity - sb.tail_mnem_uniformity)
    tail_dist = 1.0 - _tuple_overlap(sa.trace_tail_mnemonics, sb.trace_tail_mnemonics)

    # High tail uniformity with a single mnemonic (e.g. all ``sw``) is a strong
    # separator for short-run mismatches.
    mono_a = sa.tail_mnem_uniformity >= 0.9 and len(set(sa.trace_tail_mnemonics)) <= 2
    mono_b = sb.tail_mnem_uniformity >= 0.9 and len(set(sb.trace_tail_mnemonics)) <= 2
    mono_dist = 0.0 if mono_a == mono_b else 0.7
    if mono_a and mono_b and sa.trace_tail_mnemonics != sb.trace_tail_mnemonics:
        mono_dist = 0.85

    return float(min(
        1.0,
        0.30 * len_dist
        + 0.25 * loop_dist
        + 0.15 * tail_dist
        + 0.10 * repeat_dist
        + 0.10 * uniform_dist
        + 0.10 * mono_dist,
    ))


def _non_mismatch_distance(sa: CaseSignature, sb: CaseSignature) -> float:
    assert_sim = _jaccard(sa.sim_error_asserts, sb.sim_error_asserts)

    if sa.failure_mode != sb.failure_mode:
        # Same bug can surface as UVM_ERROR asserts or UVM_FATAL timeouts.
        mode_penalty = 0.15 if assert_sim >= 0.5 else 0.45
    else:
        mode_penalty = 0.0

    fatal_dist = _fatal_kind_distance(sa.sim_fatal_kind, sb.sim_fatal_kind)
    assert_dist = 1.0 - assert_sim

    src_dist = (
        0.0
        if sa.sim_fatal_source and sa.sim_fatal_source == sb.sim_fatal_source
        else 0.35
    )
    test_dist = (
        0.0
        if sa.regr_test_name and sa.regr_test_name == sb.regr_test_name
        else 0.35
    )

    # Shared assertion names are the strongest same-bug signal for X-prop bugs.
    if assert_sim == 1.0 and sa.sim_error_asserts:
        return float(min(1.0, 0.20 * fatal_dist + 0.10 * mode_penalty))

    return float(min(
        1.0,
        mode_penalty
        + 0.30 * fatal_dist
        + 0.30 * assert_dist
        + 0.10 * src_dist
        + 0.05 * test_dist,
    ))


def _pairwise_signature_distance(sa: CaseSignature, sb: CaseSignature) -> float:
    if sa.failure_mode == "mismatch" and sb.failure_mode == "mismatch":
        return _mismatch_distance(sa, sb)
    if sa.failure_mode == "mismatch" or sb.failure_mode == "mismatch":
        return 1.0
    return _non_mismatch_distance(sa, sb)


def _build_distance_matrix(features: Sequence[CaseFeatures]) -> np.ndarray:
    n = len(features)
    D = np.zeros((n, n), dtype=float)
    sigs = [f.signature for f in features]
    keys = [f.signature.categorical_key() for f in features]
    for i in range(n):
        for j in range(i + 1, n):
            if keys[i] == keys[j]:
                D[i, j] = 0.0
            else:
                D[i, j] = _pairwise_signature_distance(sigs[i], sigs[j])
            D[j, i] = D[i, j]

    try:
        tfidf = _tfidf_matrix(features).toarray()
        norms = np.linalg.norm(tfidf, axis=1)
        norms_safe = np.where(norms > 0, norms, 1.0)
        normed = tfidf / norms_safe[:, None]
        cos_dist = np.clip(1.0 - normed @ normed.T, 0.0, 1.0)
        D = 0.70 * D + 0.30 * cos_dist
        # Preserve must-link pairs after TF-IDF blend.
        for i in range(n):
            for j in range(i + 1, n):
                if keys[i] == keys[j]:
                    D[i, j] = D[j, i] = 0.0
    except Exception:
        pass
    np.fill_diagonal(D, 0.0)
    return D


def cluster_hybrid(features: Sequence[CaseFeatures], k: int) -> list[int]:
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
    return agg.fit_predict(D).tolist()


def _centroid(rows: np.ndarray) -> np.ndarray:
    c = rows.mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def cluster_signature_then_tfidf(
    features: Sequence[CaseFeatures], k: int
) -> list[int]:
    n = len(features)
    if n <= 1:
        return [0] * max(n, 1)

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
                d = 1.0 - float(np.dot(centroids[gi], centroids[gj]))
                if best is None or d < best[0]:
                    best = (d, gi, gj)
        if best is None:
            break
        _, gi, gj = best
        groups[gi] = groups[gi] + groups[gj]
        centroids[gi] = _centroid(matrix[groups[gi]])
        del groups[gj], centroids[gj]
        parent[gj] = gi
        active.remove(gj)

    remap: dict[int, int] = {}
    out = [0] * n
    for idx, old in enumerate(sig_labels):
        root = find(old)
        if root not in remap:
            remap[root] = len(remap)
        out[idx] = remap[root]
    return out


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
    by_label: dict[int, list[CaseFeatures]] = defaultdict(list)
    for f, lbl in zip(features, labels):
        by_label[lbl].append(f)
    lines = []
    for lbl in sorted(by_label):
        members = by_label[lbl]
        sig = members[0].signature
        lines.append(
            f"bucket {lbl}: {len(members)} case(s) "
            f"mode={sig.failure_mode} fatal={sig.sim_fatal_kind or '-'} "
            f"asserts={list(sig.sim_error_asserts)} regr={sig.regr_kind} "
            f"trace_len={sig.trace_length_bucket or '-'}"
        )
        lines.append(f"  cases: {sorted(m.case_id for m in members)}")
    return "\n".join(lines)
