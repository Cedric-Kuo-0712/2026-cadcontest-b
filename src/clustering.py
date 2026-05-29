"""Clustering strategies for regression-failure bucketing."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Iterable, Sequence

import numpy as np

from features import CaseFeatures, CaseSignature


_NOISE_TOKENS = {
    # Global routing tags are already encoded in signature distance.
    "mode_mismatch",
    "mode_fatal",
    "mode_assert",
    "mode_unknown",
    "regr_mismatch",
    "regr_failed_only",
    "regr_unknown",
    # Context marker only denotes a field boundary, not signal.
    "ctx",
}
_UNIFORM_TOKEN = re.compile(r"^uniform_[0-9.]+$")


def _preprocess_blob(text: str) -> str:
    """Remove low-information text features before TF-IDF."""
    if not text:
        return text

    out: list[str] = []
    prev = ""
    for raw in text.split():
        tok = raw.lower()
        if tok in _NOISE_TOKENS or _UNIFORM_TOKEN.match(tok):
            continue
        # De-emphasize boilerplate by collapsing immediate repeats.
        if tok == prev:
            continue
        out.append(tok)
        prev = tok
    return " ".join(out)


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
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = []
    for f in features:
        text = _preprocess_blob(f.text_blob)
        corpus.append(text if text else f"empty_case_{f.case_id}")
    # High-dimensional feature space:
    # - word n-grams capture semantic tokens (mode/fatal/assert/mismatch context)
    # - char_wb n-grams capture near-duplicate templates with slight token drift
    word_vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        token_pattern=r"[A-Za-z_][A-Za-z0-9_<>]+",
    )
    char_vec = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )
    word_mat = word_vec.fit_transform(corpus)
    char_mat = char_vec.fit_transform(corpus)
    return hstack([word_mat, char_mat], format="csr")


def cluster_by_tfidf(features: Sequence[CaseFeatures], k: int) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances

    n = len(features)
    if n <= 1:
        return [0] * n
    k_eff = max(1, min(k, n))
    if k_eff == 1:
        return [0] * n
    mat = _tfidf_matrix(features)
    cos_dist = np.clip(cosine_distances(mat), 0.0, 1.0)
    agg = AgglomerativeClustering(
        n_clusters=k_eff, metric="precomputed", linkage="average"
    )
    return agg.fit_predict(cos_dist).tolist()


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
    same_spike = (
        bool(sa.mismatch_spike_mnemonic)
        and sa.mismatch_spike_mnemonic == sb.mismatch_spike_mnemonic
    )

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
    if same_spike:
        score = max(0.0, score - 0.16)
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

    try:
        from sklearn.metrics.pairwise import cosine_distances

        tfidf = _tfidf_matrix(features)
        cos_dist = np.clip(cosine_distances(tfidf), 0.0, 1.0)
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


def _labels_to_groups(labels: Sequence[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, l in enumerate(labels):
        groups[int(l)].append(i)
    return groups


def _merge_until_k(groups: dict[int, list[int]], D: np.ndarray, k: int) -> dict[int, list[int]]:
    """Greedily merge closest clusters until k clusters remain."""
    while len(groups) > k:
        keys = list(groups.keys())
        best = None
        for a_i, ga in enumerate(keys):
            for gb in keys[a_i + 1:]:
                # average linkage distance between clusters
                da = groups[ga]
                db = groups[gb]
                dist = float(D[np.ix_(da, db)].mean()) if da and db else 1.0
                if best is None or dist < best[0]:
                    best = (dist, ga, gb)
        if best is None:
            break
        _, ga, gb = best
        groups[ga] = groups[ga] + groups[gb]
        del groups[gb]
    return groups


def _split_until_k(groups: dict[int, list[int]], D: np.ndarray, k: int) -> dict[int, list[int]]:
    """Split the loosest cluster with 2-way agglomerative until k clusters."""
    from sklearn.cluster import AgglomerativeClustering

    next_id = (max(groups.keys()) + 1) if groups else 0
    while len(groups) < k:
        # pick cluster with largest average pairwise distance
        worst = None
        for gid, idxs in groups.items():
            if len(idxs) <= 2:
                continue
            vals = [D[i, j] for a, i in enumerate(idxs) for j in idxs[a + 1:]]
            if not vals:
                continue
            score = float(np.mean(vals))
            if worst is None or score > worst[0]:
                worst = (score, gid)
        if worst is None:
            break
        _, gid = worst
        idxs = groups[gid]
        subD = D[np.ix_(idxs, idxs)]
        sub_labels = AgglomerativeClustering(
            n_clusters=2, metric="precomputed", linkage="average"
        ).fit_predict(subD)
        a = [idxs[i] for i, l in enumerate(sub_labels) if l == 0]
        b = [idxs[i] for i, l in enumerate(sub_labels) if l == 1]
        if not a or not b:
            break
        groups[gid] = a
        groups[next_id] = b
        next_id += 1
    return groups


def _assign_noise_to_nearest(groups: dict[int, list[int]], noise: list[int], D: np.ndarray) -> None:
    """Assign DBSCAN noise points to nearest cluster by avg distance."""
    if not noise:
        return
    for i in noise:
        best = None
        for gid, idxs in groups.items():
            dist = float(D[i, idxs].mean()) if idxs else 1.0
            if best is None or dist < best[0]:
                best = (dist, gid)
        if best is None:
            continue
        groups[best[1]].append(i)


def cluster_dbscan(features: Sequence[CaseFeatures], k: int) -> list[int]:
    """DBSCAN on the hybrid distance matrix, adapted back to exactly k buckets."""
    from sklearn.cluster import DBSCAN

    n = len(features)
    if n == 0:
        return []
    if n == 1:
        return [0]
    k_eff = max(1, min(k, n))
    if k_eff == 1:
        return [0] * n

    D = _build_distance_matrix(features)

    # Choose eps by trying a few quantiles and picking the one that yields
    # a cluster count closest to k (without using golden labels).
    tri = D[np.triu_indices(n, 1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return [0] * n

    candidate_q = (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26)
    best = None
    best_labels = None
    for q in candidate_q:
        eps = float(np.clip(np.quantile(tri, q), 0.05, 0.85))
        lab = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(D)
        n_clusters = len({x for x in lab.tolist() if x != -1})
        n_noise = int(np.sum(lab == -1))
        score = (abs(n_clusters - k_eff), n_noise, -n_clusters, eps)
        if best is None or score < best:
            best = score
            best_labels = lab

    labels = best_labels if best_labels is not None else DBSCAN(
        eps=float(np.clip(np.quantile(tri, 0.12), 0.05, 0.85)),
        min_samples=2,
        metric="precomputed",
    ).fit_predict(D)

    groups = _labels_to_groups(labels.tolist())
    noise = groups.pop(-1, [])

    # Remove empty/noise-only result.
    if not groups:
        return cluster_hybrid(features, k_eff)

    # Assign noise to nearest existing cluster.
    _assign_noise_to_nearest(groups, noise, D)

    # Adapt number of clusters to k.
    if len(groups) > k_eff:
        groups = _merge_until_k(groups, D, k_eff)
    elif len(groups) < k_eff:
        groups = _split_until_k(groups, D, k_eff)

    # Emit labels 0..k-1
    out = [-1] * n
    for new_id, (_, idxs) in enumerate(sorted(groups.items(), key=lambda kv: min(kv[1]))):
        for i in idxs:
            out[i] = new_id
    # Any leftover (shouldn't happen) -> 0
    out = [0 if x < 0 else x for x in out]
    return out


def _count_matrix(features: Sequence[CaseFeatures]):
    """Count-based text features (paper-style) from preprocessed blobs."""
    from sklearn.feature_extraction.text import CountVectorizer

    corpus = []
    for f in features:
        text = _preprocess_blob(f.text_blob)
        corpus.append(text if text else f"empty_case_{f.case_id}")

    vec = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        token_pattern=r"[A-Za-z_][A-Za-z0-9_<>]+",
    )
    return vec.fit_transform(corpus)


def cluster_dbscan_pca(features: Sequence[CaseFeatures], k: int) -> list[int]:
    """Paper-style clustering: (count features) -> PCA/SVD -> DBSCAN -> adapt to K."""
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    n = len(features)
    if n == 0:
        return []
    if n == 1:
        return [0]
    k_eff = max(1, min(k, n))
    if k_eff == 1:
        return [0] * n

    X = _count_matrix(features)  # sparse
    n_comp = int(min(50, max(2, n - 1), X.shape[1] - 1 if X.shape[1] > 2 else 2))
    Z = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X)
    Z = StandardScaler().fit_transform(Z)

    # eps selection using k-NN distance quantiles (k=min_samples).
    min_samples = 2
    # pairwise distances are cheap at our N; use Euclidean.
    diffs = Z[:, None, :] - Z[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dist, np.inf)
    kdist = np.sort(dist, axis=1)[:, min_samples - 1]

    candidate_q = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    best = None
    best_labels = None
    for q in candidate_q:
        eps = float(np.clip(np.quantile(kdist, q), 0.05, 5.0))
        lab = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(Z)
        n_clusters = len({x for x in lab.tolist() if x != -1})
        n_noise = int(np.sum(lab == -1))
        score = (abs(n_clusters - k_eff), n_noise, -n_clusters, eps)
        if best is None or score < best:
            best = score
            best_labels = lab

    labels = best_labels if best_labels is not None else DBSCAN(
        eps=float(np.clip(np.quantile(kdist, 0.5), 0.05, 5.0)),
        min_samples=min_samples,
        metric="euclidean",
    ).fit_predict(Z)

    groups = _labels_to_groups(labels.tolist())
    noise = groups.pop(-1, [])
    if not groups:
        # fallback to hybrid if DBSCAN degenerates
        return cluster_hybrid(features, k_eff)

    # Use hybrid distance matrix for robust merging/splitting/noise assignment.
    D = _build_distance_matrix(features)
    _assign_noise_to_nearest(groups, noise, D)
    if len(groups) > k_eff:
        groups = _merge_until_k(groups, D, k_eff)
    elif len(groups) < k_eff:
        groups = _split_until_k(groups, D, k_eff)

    out = [-1] * n
    for new_id, (_, idxs) in enumerate(sorted(groups.items(), key=lambda kv: min(kv[1]))):
        for i in idxs:
            out[i] = new_id
    out = [0 if x < 0 else x for x in out]
    return out

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
    if method == "dbscan":
        return cluster_dbscan(features, k)
    if method == "dbscan_pca":
        return cluster_dbscan_pca(features, k)
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
