"""
ROI-aware feature extraction for importance analysis and analysis-guided bucketing.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from log_parser import (
    RegrLogParser,
    SimLogParser,
    TraceLogParser,
    classify_error_type,
    open_file,
)
from strategies import CharEmbeddingExtractor

ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")
NUM_RE = re.compile(r"\d{4,}")
UVM_LINE_RE = re.compile(r"^(UVM_FATAL.*|UVM_ERROR.*)")
INSTR_RE = re.compile(r"^\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)")
REPORT_ID_HEADER = re.compile(r"Report counts by id", re.I)
REPORT_LINE_RE = re.compile(r"^\[([^\]]+)\]\s+(\d+)\s*$")


def normalize_text(text: str) -> str:
    text = ADDR_RE.sub("ADDR", text)
    text = NUM_RE.sub("NUM", text)
    return text


def char_freq(text: str, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float64)
    for ch in text:
        code = ord(ch)
        if code < dim:
            out[code] += 1.0
    norm = np.linalg.norm(out)
    if norm > 0:
        out /= norm
    return out


def extract_sim_uvm_roi(sim_path: str, dim: int = 64, context_lines: int = 3) -> np.ndarray:
    """First UVM_FATAL/ERROR line (+ optional following lines) char frequencies."""
    try:
        with open_file(sim_path) as f:
            lines = f.readlines()
    except OSError:
        return np.zeros(dim, dtype=np.float64)

    chunk: List[str] = []
    for line in lines:
        if UVM_LINE_RE.search(line):
            chunk = [line]
            idx = lines.index(line)
            chunk.extend(lines[idx + 1 : idx + 1 + context_lines])
            break
    if not chunk:
        return np.zeros(dim, dtype=np.float64)
    return char_freq(normalize_text("".join(chunk)), dim)


def extract_sim_report_by_id(sim_path: str, top_ids: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Counts for pooled top-K report ids (ROI-B)."""
    if not top_ids:
        return np.zeros(0, dtype=np.float64), []
    counts: Dict[str, float] = {rid: 0.0 for rid in top_ids}
    names = [f"sim_report_id_{rid}" for rid in top_ids]
    try:
        with open_file(sim_path) as f:
            text = f.read()
    except OSError:
        return np.zeros(len(top_ids), dtype=np.float64), names

    lines = text.splitlines()
    in_section = False
    for line in lines:
        if REPORT_ID_HEADER.search(line):
            in_section = True
            continue
        if in_section and line.strip().startswith("**") and "Report counts" not in line:
            break
        if in_section:
            m = REPORT_LINE_RE.match(line.strip())
            if m:
                rid, cnt = m.group(1), float(m.group(2))
                if rid in counts:
                    counts[rid] = cnt

    vec = np.array([counts[rid] for rid in top_ids], dtype=np.float64)
    if vec.size > 0:
        mx = vec.max()
        if mx > 0:
            vec /= mx
    return vec, names


def _trace_instr_lines(trace_path: str) -> List[str]:
    try:
        with open_file(trace_path) as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        m = INSTR_RE.search(line)
        if m:
            out.append(m.group(5))
    return out


def extract_trace_blocks(
    trace_path: str, block_size: int = 10, char_dim: int = 32
) -> Tuple[np.ndarray, List[str]]:
    """Per-block char freq on instruction text; return element-wise max."""
    instr_lines = _trace_instr_lines(trace_path)
    if not instr_lines:
        return np.zeros(char_dim, dtype=np.float64), [
            f"trace_block_max_char_{i}" for i in range(char_dim)
        ]

    blocks: List[np.ndarray] = []
    for start in range(0, len(instr_lines), block_size):
        chunk = " ".join(instr_lines[start : start + block_size])
        blocks.append(char_freq(normalize_text(chunk), char_dim))

    stacked = np.stack(blocks, axis=0)
    return stacked.max(axis=0), [f"trace_block_max_char_{i}" for i in range(char_dim)]


def extract_trace_mismatch_window(
    trace_path: str,
    regr_parsed: Dict,
    window: int = 5,
    char_dim: int = 32,
) -> Tuple[np.ndarray, float]:
    """
    Heuristic: use last 2*window instruction lines when mismatch (tail anchor).
    Returns (features, anchor_hit=1.0 if mismatch else 0.0).
    """
    names = [f"trace_mismatch_window_char_{i}" for i in range(char_dim)]
    if not regr_parsed.get("has_mismatch"):
        return np.zeros(char_dim, dtype=np.float64), 0.0

    instr_lines = _trace_instr_lines(trace_path)
    if not instr_lines:
        return np.zeros(char_dim, dtype=np.float64), 0.0

    tail = instr_lines[-(2 * window) :]
    return char_freq(normalize_text(" ".join(tail)), char_dim), 1.0


def extract_structural(parsed: Dict) -> Tuple[np.ndarray, List[str]]:
    """Scalars + one-hots from log_parser results."""
    regr = parsed["regr"]
    sim = parsed["sim"]
    trace = parsed["trace"]
    err = parsed["error_type"]

    error_types = ["UVM_FATAL", "MISMATCH", "TRACE_ABNORMAL", "FAILED_ONLY"]
    et_oh = [1.0 if err == e else 0.0 for e in error_types]

    sim_et_map = {"INFO": 0, "WARNING": 1, "ERROR": 2, "FATAL": 3}
    sim_oh = [0.0] * 4
    if sim["error_type"] in sim_et_map:
        sim_oh[sim_et_map[sim["error_type"]]] = 1.0

    decile = np.zeros(10)
    if regr["first_mismatch_idx"] is not None:
        decile[min(int(regr["first_mismatch_idx"] / 100), 9)] = 1.0

    values = (
        et_oh
        + sim_oh
        + decile.tolist()
        + [
            float(regr["has_mismatch"]),
            min(regr["mismatch_count"] / 1000.0, 1.0),
            float(sim["has_debug_error"]),
            float(sim["has_timeout"]),
            float(sim["has_privilege_error"]),
            min(len(sim["keywords"]) / 5.0, 1.0),
            float(trace["has_loop"]),
            float(trace["has_pc_stall"]),
            trace["anomaly_score"],
        ]
    )
    names = (
        [f"struct_error_type_{e}" for e in error_types]
        + [f"struct_sim_severity_{i}" for i in range(4)]
        + [f"struct_regr_mismatch_decile_{i}" for i in range(10)]
        + [
            "struct_regr_has_mismatch",
            "struct_regr_mismatch_count_norm",
            "struct_sim_has_debug_error",
            "struct_sim_has_timeout",
            "struct_sim_has_privilege_error",
            "struct_sim_keyword_count_norm",
            "struct_trace_has_loop",
            "struct_trace_has_pc_stall",
            "struct_trace_anomaly_score",
        ]
    )
    return np.array(values, dtype=np.float64), names


@dataclass
class FeatureMatrix:
    X: np.ndarray
    names: List[str]
    sample_ids: List[str]
    bugs: List[str]
    benchmarks: List[str]
    cases: List[int]
    views: List[str] = field(default_factory=list)

    def view_for_name(self, name: str) -> str:
        if name.startswith("char_sim"):
            return "char_sim"
        if name.startswith("char_trace"):
            return "char_trace"
        if name.startswith("char_regr"):
            return "char_regr"
        if name.startswith("sim_"):
            return "sim_roi"
        if name.startswith("trace_"):
            return "trace_roi"
        if name.startswith("struct_"):
            return "struct"
        return "other"

    def view_indices(self) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for i, n in enumerate(self.names):
            v = self.view_for_name(n)
            groups.setdefault(v, []).append(i)
        return groups


def discover_report_ids(cases: List[Dict], top_k: int = 10) -> List[str]:
    """Pool top-K report ids by total count across cases."""
    totals: Dict[str, float] = {}
    for case in cases:
        try:
            with open_file(case["Sim Log"]) as f:
                text = f.read()
        except OSError:
            continue
        in_section = False
        for line in text.splitlines():
            if REPORT_ID_HEADER.search(line):
                in_section = True
                continue
            if in_section and line.strip().startswith("**") and "Report counts" not in line:
                break
            if in_section:
                m = REPORT_LINE_RE.match(line.strip())
                if m:
                    rid, cnt = m.group(1), float(m.group(2))
                    totals[rid] = totals.get(rid, 0.0) + cnt
    ranked = sorted(totals.items(), key=lambda x: -x[1])
    return [r[0] for r in ranked[:top_k]]


def build_feature_matrix(
    cases: List[Dict],
    sample_ids: List[str],
    bugs: List[str],
    benchmarks: List[str],
    case_nums: List[int],
    report_top_ids: Optional[List[str]] = None,
    report_top_k: int = 10,
    include_char: bool = True,
    include_struct: bool = True,
    include_sim_roi: bool = True,
    include_trace_roi: bool = True,
    feature_mask: Optional[Sequence[str]] = None,
) -> FeatureMatrix:
    """Build pooled feature matrix with interpretable names."""
    if report_top_ids is None:
        report_top_ids = discover_report_ids(cases, top_k=report_top_k)

    regr_parser = RegrLogParser()
    sim_parser = SimLogParser()
    trace_parser = TraceLogParser()
    char_ext = CharEmbeddingExtractor(char_dim=128)

    parsed_list = []
    for case in cases:
        regr_p = regr_parser.parse(case["Regr Log"])
        sim_p = sim_parser.parse(case["Sim Log"])
        trace_p = trace_parser.parse(case["Trace Log"])
        et = classify_error_type(regr_p, sim_p, trace_p)
        parsed_list.append(
            {"regr": regr_p, "sim": sim_p, "trace": trace_p, "error_type": et}
        )

    rows: List[np.ndarray] = []
    all_names: List[str] = []

    char_mat = char_ext.extract(cases, parsed_list) if include_char else None
    char_names = (
        [f"char_sim_{i}" for i in range(128)]
        + [f"char_trace_{i}" for i in range(128)]
        + [f"char_regr_{i}" for i in range(128)]
    )

    for i, case in enumerate(cases):
        parts: List[np.ndarray] = []
        names: List[str] = []

        if include_char and char_mat is not None:
            parts.append(char_mat[i])
            names.extend(char_names)

        if include_struct:
            svec, snames = extract_structural(parsed_list[i])
            parts.append(svec)
            names.extend(snames)

        if include_sim_roi:
            uvec = extract_sim_uvm_roi(case["Sim Log"], dim=64)
            parts.append(uvec)
            names.extend([f"sim_uvm_roi_char_{j}" for j in range(64)])
            if report_top_ids:
                rvec, rnames = extract_sim_report_by_id(case["Sim Log"], report_top_ids)
                parts.append(rvec)
                names.extend(rnames)

        if include_trace_roi:
            bvec, bnames = extract_trace_blocks(case["Trace Log"])
            parts.append(bvec)
            names.extend(bnames)
            wvec, _ = extract_trace_mismatch_window(
                case["Trace Log"], parsed_list[i]["regr"]
            )
            parts.append(wvec)
            names.extend([f"trace_mismatch_window_char_{j}" for j in range(32)])
            parts.append(np.array([_], dtype=np.float64))
            names.append("trace_mismatch_anchor_hit")

        row = np.concatenate(parts)
        if not all_names:
            all_names = names
        rows.append(row)

    X = np.stack(rows, axis=0)

    if feature_mask is not None:
        mask_set = set(feature_mask)
        idx = [i for i, n in enumerate(all_names) if n in mask_set]
        X = X[:, idx]
        all_names = [all_names[i] for i in idx]

    return FeatureMatrix(
        X=X.astype(np.float64),
        names=all_names,
        sample_ids=sample_ids,
        bugs=bugs,
        benchmarks=benchmarks,
        cases=case_nums,
    )


def load_benchmark_cases(
    samples_root: str, benchmark_name: str
) -> Tuple[List[Dict], List[str], List[str], List[str], List[int]]:
    bench_dir = os.path.join(samples_root, benchmark_name)
    input_csv = os.path.join(bench_dir, "input.csv")
    golden_csv = os.path.join(bench_dir, "golden.csv")

    import pandas as pd

    df_in = pd.read_csv(input_csv)
    df_g = pd.read_csv(golden_csv)
    golden_map = dict(zip(df_g["Case"], df_g["Bug"]))

    cases = []
    sample_ids = []
    bugs = []
    benchmarks = []
    case_nums = []

    for _, row in df_in.iterrows():
        cnum = int(row["Case"])
        sid = f"{benchmark_name}:{cnum}"
        case = {
            "Case": cnum,
            "Regr Log": os.path.join(bench_dir, row["Regr Log"]),
            "Sim Log": os.path.join(bench_dir, row["Sim Log"]),
            "Trace Log": os.path.join(bench_dir, row["Trace Log"]),
        }
        cases.append(case)
        sample_ids.append(sid)
        bugs.append(golden_map[cnum])
        benchmarks.append(benchmark_name)
        case_nums.append(cnum)

    return cases, sample_ids, bugs, benchmarks, case_nums


def load_pooled_cases(
    samples_root: str,
    which: str = "both",
) -> Tuple[List[Dict], List[str], List[str], List[str], List[int]]:
    sets = []
    if which in ("benchmark_set_1", "both"):
        sets.append("benchmark_set_1")
    if which in ("benchmark_set_2", "both"):
        sets.append("benchmark_set_2")

    all_cases, all_sids, all_bugs, all_bench, all_nums = [], [], [], [], []
    for bn in sets:
        c, s, b, bm, n = load_benchmark_cases(samples_root, bn)
        all_cases.extend(c)
        all_sids.extend(s)
        all_bugs.extend(b)
        all_bench.extend(bm)
        all_nums.extend(n)
    return all_cases, all_sids, all_bugs, all_bench, all_nums


def save_selected_features(path: str, names: List[str], views: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"features": names, "views": views}, f, indent=2)


def load_selected_features(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["features"]


def extract_clustering_matrix(
    cases: List[Dict],
    feature_mask: Optional[Sequence[str]] = None,
    report_top_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Feature matrix for bucketing (no golden labels required)."""
    n = len(cases)
    dummy = ["unknown"] * n
    bench = ["unknown"] * n
    sids = [str(i) for i in range(n)]
    nums = [c.get("Case", i) for i, c in enumerate(cases)]
    fm = build_feature_matrix(
        cases,
        sids,
        dummy,
        bench,
        nums,
        report_top_ids=report_top_ids,
        feature_mask=feature_mask,
    )
    return fm.X, fm.names


class AnalysisGuidedExtractor:
    """
    Bucketing extractor using analysis-selected features.
    Compatible with FeatureExtractor interface from strategies.py.
    """

    def __init__(
        self,
        selected_features_path: Optional[str] = None,
        default_mask: Optional[Sequence[str]] = None,
    ):
        self.selected_path = selected_features_path
        self._feature_mask: Optional[List[str]] = None
        self._report_top_ids: Optional[List[str]] = None
        self._dim: Optional[int] = None

        if selected_features_path and os.path.isfile(selected_features_path):
            self._feature_mask = load_selected_features(selected_features_path)
        elif default_mask:
            self._feature_mask = list(default_mask)
        # else: None -> use full analysis feature set at extract time

    def _ensure_vocab(self, cases: List[Dict]) -> None:
        if self._report_top_ids is None:
            self._report_top_ids = discover_report_ids(cases, top_k=10)

    def extract(self, cases: List[Dict], parsed_results: List[Dict]) -> np.ndarray:
        self._ensure_vocab(cases)
        X, names = extract_clustering_matrix(
            cases,
            feature_mask=self._feature_mask,
            report_top_ids=self._report_top_ids,
        )
        self._dim = X.shape[1]
        self._names = names
        return X

    def get_feature_dim(self) -> int:
        return self._dim or 0
