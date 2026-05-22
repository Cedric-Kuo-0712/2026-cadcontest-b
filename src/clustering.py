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
    if ka == "no_dret" or kb == "no_dret":
        return 1.0
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
    short_a = sa.trace_length_bucket == "short"
    short_b = sb.trace_length_bucket == "short"
    if short_a or short_b:
        if sa.trace_length_bucket != sb.trace_length_bucket:
            len_dist = 0.95

    ibex_diff = (
        bool(sa.mismatch_ibex_mnemonic)
        and sa.mismatch_ibex_mnemonic != sb.mismatch_ibex_mnemonic
    )
    spike_diff = (
        bool(sa.mismatch_spike_mnemonic)
        and sa.mismatch_spike_mnemonic != sb.mismatch_spike_mnemonic
    )
    ibex_match = 0.0 if not ibex_diff else 0.75
    spike_match = 0.0 if not spike_diff else 0.75

    ctx_dist = 1.0 - _tuple_overlap(
        sa.mismatch_context_mnemonics, sb.mismatch_context_mnemonics
    )

    loop_dist = 0.0 if sa.has_signature_loop == sb.has_signature_loop else 0.80
    repeat_dist = 0.0 if sa.has_repeating_tail == sb.has_repeating_tail else 0.35
    uniform_dist = abs(sa.tail_mnem_uniformity - sb.tail_mnem_uniformity)
    early_dist = 0.0 if sa.early_mismatch == sb.early_mismatch else 0.70
    same_reg_dist = 0.0 if sa.same_reg_pair == sb.same_reg_pair else 0.85

    score = float(min(
        1.0,
        0.18 * ibex_match
        + 0.16 * spike_match
        + 0.24 * ctx_dist
        + 0.14 * len_dist
        + 0.10 * loop_dist
        + 0.05 * repeat_dist
        + 0.04 * uniform_dist
        + 0.04 * early_dist
        + 0.05 * same_reg_dist,
    ))

    # Different (ibex, spike) pair at first mismatch → different bug family.
    if ibex_diff and spike_diff:
        score = max(score, 0.80)
    if short_a != short_b:
        score = max(score, 0.88)
    if sa.has_signature_loop != sb.has_signature_loop:
        score = max(score, 0.82)
    if sa.early_mismatch != sb.early_mismatch:
        score = max(score, 0.84)
    if sa.same_reg_pair != sb.same_reg_pair:
        score = max(score, 0.88)
    return score


def _cross_mode_distance(sa: CaseSignature, sb: CaseSignature) -> float:
    """Soft distance when one case is mismatch and the other is not."""
    fatal = sa if sa.failure_mode != "mismatch" else sb
    mismatch = sb if sa.failure_mode != "mismatch" else sa

    if fatal.sim_fatal_kind == "no_dret":
        if mismatch.early_mismatch and mismatch.trace_length_bucket == "short":
            return 0.68
        return 0.92

    if fatal.failure_mode == "assert" and fatal.sim_error_asserts:
        return 1.0

    if fatal.sim_fatal_kind and mismatch.has_signature_loop:
        return 0.95

    return 1.0


def _mismatch_outlier_scores(
    sigs: Sequence[CaseSignature], D: np.ndarray
) -> dict[int, float]:
    mismatch_idxs = [i for i, s in enumerate(sigs) if s.failure_mode == "mismatch"]
    scores: dict[int, float] = {}
    for i in mismatch_idxs:
        others = [j for j in mismatch_idxs if j != i]
        scores[i] = float(np.mean([D[i, j] for j in others])) if others else 0.0
    return scores


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

    if "no_dret" in (sa.sim_fatal_kind, sb.sim_fatal_kind):
        other = sb.sim_fatal_kind if sa.sim_fatal_kind == "no_dret" else sa.sim_fatal_kind
        if other and other != "no_dret":
            return 0.90

    has_a, has_b = bool(sa.sim_error_asserts), bool(sb.sim_error_asserts)
    if has_a != has_b:
        return max(
            0.75,
            mode_penalty + 0.30 * fatal_dist + 0.30 * assert_dist + 0.10 * src_dist,
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
        return _cross_mode_distance(sa, sb)
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

    outlier_scores = _mismatch_outlier_scores(sigs, D)
    outliers: set[int] = set()
    if outlier_scores:
        threshold = float(np.percentile(list(outlier_scores.values()), 75))
        outliers = {i for i, s in outlier_scores.items() if s >= max(threshold, 0.74)}

    try:
        tfidf = _tfidf_matrix(features).toarray()
        norms = np.linalg.norm(tfidf, axis=1)
        norms_safe = np.where(norms > 0, norms, 1.0)
        normed = tfidf / norms_safe[:, None]
        cos_dist = np.clip(1.0 - normed @ normed.T, 0.0, 1.0)
        for i in range(n):
            for j in range(i + 1, n):
                sig_ij = D[i, j]
                if sigs[i].failure_mode == "mismatch" and sigs[j].failure_mode == "mismatch":
                    w = 0.96 if sig_ij < 0.72 else 1.0
                else:
                    w = 0.85 if sig_ij < 0.72 else 0.92
                D[i, j] = D[j, i] = w * sig_ij + (1.0 - w) * cos_dist[i, j]
        for i in range(n):
            for j in range(i + 1, n):
                if keys[i] == keys[j]:
                    D[i, j] = D[j, i] = 0.0
    except Exception:
        pass

    if outlier_scores:
        core = {i for i in outlier_scores if i not in outliers}
        for i in outliers:
            for j in core:
                D[i, j] = D[j, i] = max(D[i, j], 0.88)

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
            f"mismatch={sig.mismatch_ibex_mnemonic or '-'}"
            f"/{sig.mismatch_spike_mnemonic or '-'} "
            f"asserts={list(sig.sim_error_asserts)} regr={sig.regr_kind} "
            f"trace_len={sig.trace_length_bucket or '-'}"
        )
        lines.append(f"  cases: {sorted(m.case_id for m in members)}")
    return "\n".join(lines)
