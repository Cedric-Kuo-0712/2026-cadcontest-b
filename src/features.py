"""Feature extraction for regression-failure bucketing.

Each case yields a structured :class:`CaseSignature` and a compact text blob
for TF-IDF / agglomerative clustering.

Routing policy
--------------
* ``regr.log`` with ``Mismatch[N]:``  →  features from ``trace.log`` only.
* otherwise                           →  features from ``sim.log`` + ``regr.log``.
"""

from __future__ import annotations

import gzip
import os
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def open_log(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def _resolve(base_dir: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.join(base_dir, rel)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_HEX_NUMBER = re.compile(r"0x[0-9a-fA-F]+")
_DEC_NUMBER = re.compile(r"\b\d+\b")
_WS = re.compile(r"\s+")
_PATH = re.compile(r"/[\w\-./]+")
_TIME_TAG = re.compile(r"@\s*\d+")


def normalize_message(text: str) -> str:
    out = _PATH.sub("<PATH>", text)
    out = _HEX_NUMBER.sub("<HEX>", out)
    out = _TIME_TAG.sub("@<T>", out)
    out = _DEC_NUMBER.sub("<N>", out)
    return _WS.sub(" ", out).strip()


def _fatal_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z]+", "_", text)[:80]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CaseSignature:
    """Categorical fingerprint of one failure case."""

    failure_mode: str = "unknown"          # mismatch | assert | fatal | unknown
    regr_kind: str = "unknown"               # mismatch | failed_only | unknown

    # sim / regr (non-mismatch)
    sim_fatal_kind: str = ""                 # canonical fatal class
    sim_fatal_source: str = ""               # e.g. core_ibex_base_test.sv
    sim_error_asserts: tuple[str, ...] = ()
    regr_test_name: str = ""
    sim_uvm_testname: str = ""

    # trace (mismatch-primary; optional tie-breaker elsewhere)
    trace_tail_mnemonics: tuple[str, ...] = ()
    trace_loop_mnems: tuple[str, ...] = ()   # sorted unique mnems in tail window
    trace_length_bucket: str = ""            # short | medium | long | huge
    tail_mnem_uniformity: float = 0.0        # 1.0 = all same mnemonic in tail
    has_repeating_tail: bool = False
    trace_total: int = 0

    def categorical_key(self) -> tuple:
        if self.failure_mode == "mismatch":
            return (
                "mismatch",
                self.trace_loop_mnems,
                self.trace_length_bucket,
                self.has_repeating_tail,
                round(self.tail_mnem_uniformity, 2),
                self.trace_tail_mnemonics,
            )
        return (
            self.failure_mode,
            self.sim_fatal_kind,
            self.sim_fatal_source,
            self.sim_error_asserts,
            self.regr_test_name,
        )


@dataclass
class CaseFeatures:
    case_id: int
    signature: CaseSignature = field(default_factory=CaseSignature)
    text_blob: str = ""


# ---------------------------------------------------------------------------
# Regex library
# ---------------------------------------------------------------------------

_RE_MISMATCH = re.compile(r"^Mismatch\[\d+\]:")
_RE_REGR_FAILED = re.compile(r"^([A-Za-z0-9_.]+)\s*:\s*\[FAILED\]")
_RE_UVM_FATAL = re.compile(r"UVM_FATAL\b.*")
_RE_UVM_ERROR = re.compile(r"UVM_ERROR\b.*")
_RE_FATAL_SOURCE = re.compile(r"UVM_FATAL\s+(\S+?\.sv)\b")
_RE_VERDICT_PASS = re.compile(r"RISC-V UVM TEST PASSED")
_RE_VERDICT_FAIL = re.compile(r"RISC-V UVM TEST FAILED")
_RE_ASSERT_NAME = re.compile(r"ASSERT FAILED\] \[[^.\]]*\.([^.\]]+)\]")
_RE_ASSERT_PROP = re.compile(r"\] ([A-Za-z_][A-Za-z0-9_]*):")
_RE_TESTNAME = re.compile(r"\+UVM_TESTNAME=(\S+)")
_RE_BIN = re.compile(r"\+bin=(\S+)")
_RE_BIN_NAME = re.compile(r"/([A-Za-z0-9_]+_test)_\d+\.bin")
_RE_TRACE_LINE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+(\S+)"
)

_FATAL_KIND_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("debug_timeout", re.compile(r"IN_DEBUG_MODE", re.I)),
    ("irq_timeout", re.compile(r"HANDLING_IRQ", re.I)),
    ("no_dret", re.compile(r"No dret detected", re.I)),
    ("check_mcause", re.compile(r"Check failed mcause", re.I)),
    ("check_signature", re.compile(r"Check failed signature_data", re.I)),
    ("check_memory", re.compile(r"memory fault", re.I)),
)

_SIM_MAX_BYTES = 50 * 1024 * 1024
_TRACE_TAIL = 64


# ---------------------------------------------------------------------------
# Lightweight regr routing
# ---------------------------------------------------------------------------

def detect_regr_kind(path: str) -> str:
    try:
        with open_log(path) as f:
            for line in f:
                if _RE_MISMATCH.match(line):
                    return "mismatch"
                if _RE_REGR_FAILED.match(line.strip()):
                    return "failed_only"
    except OSError:
        pass
    return "unknown"


def extract_regr(path: str, *, kind: str | None = None) -> dict:
    result = {"kind": kind or "unknown", "test_name": ""}
    if result["kind"] == "mismatch":
        return result
    try:
        with open_log(path) as f:
            data = f.read()
    except OSError:
        return result
    if result["kind"] == "unknown":
        if any(_RE_MISMATCH.match(ln) for ln in data.splitlines()):
            result["kind"] = "mismatch"
            return result
    for ln in data.splitlines():
        m = _RE_REGR_FAILED.match(ln.strip())
        if m:
            result["kind"] = "failed_only"
            result["test_name"] = re.sub(r"\.\d+$", "", m.group(1))
            break
    return result


# ---------------------------------------------------------------------------
# sim.log
# ---------------------------------------------------------------------------

def _classify_fatal(raw: str, normalized: str) -> str:
    for kind, pat in _FATAL_KIND_RULES:
        if pat.search(raw) or pat.search(normalized):
            return kind
    if normalized and "UVM_FATAL reports" not in normalized:
        return "other_fatal"
    return ""


def _empty_sim() -> dict:
    return {
        "verdict": "unknown",
        "fatal_kind": "",
        "fatal_source": "",
        "fatal_template": "",
        "error_asserts": (),
        "uvm_testname": "",
        "error_messages": [],
    }


def extract_sim(path: str) -> dict:
    result = _empty_sim()
    error_asserts: set[str] = set()
    error_lines: list[str] = []
    fatal_raw = ""
    fatal_norm = ""
    bytes_read = 0
    try:
        with open_log(path) as f:
            for line in f:
                bytes_read += len(line)
                if bytes_read > _SIM_MAX_BYTES:
                    break
                if not result["uvm_testname"]:
                    m = _RE_TESTNAME.search(line)
                    if m:
                        result["uvm_testname"] = m.group(1)
                if _RE_VERDICT_PASS.search(line):
                    result["verdict"] = "passed"
                elif _RE_VERDICT_FAIL.search(line):
                    result["verdict"] = "failed"
                fm = _RE_UVM_FATAL.search(line)
                if fm and not fatal_raw:
                    fatal_raw = fm.group(0)
                    fatal_norm = normalize_message(fatal_raw)
                    fatal_norm = re.sub(
                        r"^UVM_FATAL\s+<PATH>\(<N>\)\s*", "UVM_FATAL ", fatal_norm
                    )
                em = _RE_UVM_ERROR.search(line)
                if em:
                    if len(error_lines) < 8:
                        error_lines.append(normalize_message(em.group(0)))
                    name = _RE_ASSERT_NAME.search(line)
                    if name:
                        error_asserts.add(name.group(1))
                    else:
                        prop = _RE_ASSERT_PROP.search(line)
                        if prop:
                            error_asserts.add(prop.group(1))
    except OSError:
        return result

    if fatal_raw:
        m = _RE_FATAL_SOURCE.search(fatal_raw)
        if m:
            result["fatal_source"] = os.path.basename(m.group(1))
        result["fatal_kind"] = _classify_fatal(fatal_raw, fatal_norm)
        result["fatal_template"] = fatal_norm
    result["error_asserts"] = tuple(sorted(error_asserts))
    result["error_messages"] = error_lines
    return result


# ---------------------------------------------------------------------------
# trace.log
# ---------------------------------------------------------------------------

def _length_bucket(total: int) -> str:
    if total < 1_000:
        return "short"
    if total < 10_000:
        return "medium"
    if total < 100_000:
        return "long"
    return "huge"


def extract_trace(path: str, tail: int = _TRACE_TAIL) -> dict:
    result = {
        "mnemonics": (),
        "loop_mnems": (),
        "unique_pcs": 0,
        "total": 0,
        "has_repeating_tail": False,
        "tail_text": "",
        "length_bucket": "",
        "tail_uniformity": 0.0,
    }
    window: deque[tuple[str, str]] = deque(maxlen=tail)
    total = 0
    try:
        with open_log(path) as f:
            for line in f:
                m = _RE_TRACE_LINE.match(line)
                if not m:
                    continue
                window.append((m.group(3).lower(), m.group(5)))
                total += 1
    except OSError:
        return result

    if not window:
        return result

    pcs = [pc for pc, _ in window]
    mnems = [mn for _, mn in window]
    unique_pcs = len(set(pcs))
    result["total"] = total
    result["unique_pcs"] = unique_pcs
    result["length_bucket"] = _length_bucket(total)
    result["has_repeating_tail"] = (
        len(window) >= 10 and unique_pcs <= max(2, len(window) // 4)
    )
    result["mnemonics"] = tuple(mnems[-8:])
    result["loop_mnems"] = tuple(sorted(set(mnems)))
    result["tail_text"] = " ".join(mnems[-16:])
    counts = Counter(mnems[-16:])
    result["tail_uniformity"] = counts.most_common(1)[0][1] / max(len(mnems[-16:]), 1)
    return result


# ---------------------------------------------------------------------------
# Signature assembly
# ---------------------------------------------------------------------------

def _failure_mode(regr_kind: str, sim: dict) -> str:
    if regr_kind == "mismatch":
        return "mismatch"
    if sim["error_asserts"] and not sim["fatal_kind"]:
        return "assert"
    if sim["fatal_kind"]:
        return "fatal"
    return "unknown"


def _build_text_blob(sig: CaseSignature, sim: dict, trace: dict) -> str:
    if sig.failure_mode == "mismatch":
        parts = ["MODE_mismatch", "MODE_mismatch", f"LEN_{sig.trace_length_bucket}"]
        if sig.trace_loop_mnems:
            parts.extend([f"LOOP_{m}" for m in sig.trace_loop_mnems])
        if sig.has_repeating_tail:
            parts.extend(["REPEAT_TAIL"] * 3)
        if sig.tail_mnem_uniformity >= 0.75:
            parts.append(f"UNIFORM_{sig.tail_mnem_uniformity:.2f}")
        if trace["tail_text"]:
            parts.append("TAIL " + trace["tail_text"])
        return " ".join(parts)

    parts = [f"MODE_{sig.failure_mode}", f"REGR_{sig.regr_kind}"]
    if sig.sim_fatal_kind:
        parts.extend([f"FATAL_{sig.sim_fatal_kind}"] * 3)
    if sig.sim_fatal_source:
        parts.extend([f"FATALSRC_{sig.sim_fatal_source}"] * 2)
    for asrt in sig.sim_error_asserts:
        parts.extend([f"ASSERT_{asrt}"] * 3)
    if sig.regr_test_name:
        parts.extend([f"REGRTEST_{sig.regr_test_name}"] * 2)
    if sig.sim_uvm_testname:
        parts.append(f"UVMTEST_{sig.sim_uvm_testname}")
    for msg in sim.get("error_messages", [])[:2]:
        parts.append(re.sub(r"[^A-Za-z_]+", " ", msg))
    return " ".join(parts)


def build_case_features(
    case_id: int,
    base_dir: str,
    regr_rel: str,
    sim_rel: str,
    trace_rel: str,
) -> CaseFeatures:
    regr_path = _resolve(base_dir, regr_rel)
    trace_path = _resolve(base_dir, trace_rel)

    regr_kind = detect_regr_kind(regr_path)
    trace = extract_trace(trace_path)

    if regr_kind == "mismatch":
        regr = extract_regr(regr_path, kind="mismatch")
        sim = _empty_sim()
    else:
        regr = extract_regr(regr_path, kind=regr_kind)
        sim = extract_sim(_resolve(base_dir, sim_rel))

    sig = CaseSignature(
        failure_mode=_failure_mode(regr["kind"], sim),
        regr_kind=regr["kind"],
        sim_fatal_kind=sim["fatal_kind"],
        sim_fatal_source=sim["fatal_source"],
        sim_error_asserts=sim["error_asserts"],
        regr_test_name=regr["test_name"],
        sim_uvm_testname=sim["uvm_testname"],
        trace_tail_mnemonics=trace["mnemonics"],
        trace_loop_mnems=trace["loop_mnems"],
        trace_length_bucket=trace["length_bucket"],
        tail_mnem_uniformity=trace["tail_uniformity"],
        has_repeating_tail=trace["has_repeating_tail"],
        trace_total=trace["total"],
    )
    return CaseFeatures(
        case_id=case_id,
        signature=sig,
        text_blob=_build_text_blob(sig, sim, trace),
    )


def iter_summary(features: Iterable[CaseFeatures]) -> Counter:
    return Counter(f.signature.categorical_key() for f in features)
