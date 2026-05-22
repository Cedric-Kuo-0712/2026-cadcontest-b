"""Feature extraction for regression-failure bucketing.

Each case yields a structured :class:`CaseSignature` and a compact text blob
for TF-IDF / agglomerative clustering.

Routing policy
--------------
* ``regr.log`` with ``Mismatch[N]:``  →  features from ``trace.log``, plus the
  first mismatch line from ``regr.log`` and trace instructions surrounding
  that retire index.
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

    # trace (mismatch-primary)
    trace_tail_mnemonics: tuple[str, ...] = ()
    trace_loop_mnems: tuple[str, ...] = ()   # sorted unique mnems in tail window
    trace_length_bucket: str = ""            # short | medium | long | huge
    tail_mnem_uniformity: float = 0.0        # 1.0 = all same mnemonic in tail
    has_repeating_tail: bool = False
    trace_total: int = 0
    has_signature_loop: bool = False       # tail has auipc+sw+c.j signature pattern

    # first mismatch (from regr.log + trace context at retire index)
    mismatch_retire_index: int = 0
    mismatch_ibex_mnemonic: str = ""
    mismatch_spike_mnemonic: str = ""
    mismatch_context_mnemonics: tuple[str, ...] = ()  # trace window at mismatch
    mismatch_matched_count: int = 0
    early_mismatch: bool = False          # few matched instrs or low retire index
    same_reg_pair: bool = False           # ibex mnemonic == spike mnemonic

    def categorical_key(self) -> tuple:
        if self.failure_mode == "mismatch":
            return (
                "mismatch",
                self.mismatch_ibex_mnemonic,
                self.mismatch_spike_mnemonic,
                self.has_signature_loop,
                self.trace_length_bucket,
                self.has_repeating_tail,
                self.same_reg_pair,
                self.early_mismatch,
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
_RE_IBEX_MISMATCH = re.compile(
    r"^ibex\[(\d+)\]\s*:\s*pc\[([0-9a-fA-F]+)\]\s+([a-z][a-z0-9.]*)\b"
)
_RE_SPIKE_MISMATCH = re.compile(
    r"^spike\[\d+\]\s*:\s*(?:pc\[[^\]]+\]\s+)?([a-z][a-z0-9.]*)\b"
)
_RE_MISMATCH_COUNTS = re.compile(
    r"\[FAILED\]:\s*(\d+)\s+matched,\s*(\d+)\s+mismatch"
)
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
_MISMATCH_CONTEXT = 8   # instructions before/after first mismatch retire index


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


def extract_regr_mismatch(path: str) -> dict:
    """Parse the first mismatch block from regr.log."""
    result = {
        "kind": "mismatch",
        "test_name": "",
        "retire_index": 0,
        "ibex_mnemonic": "",
        "spike_mnemonic": "",
        "ibex_pc": "",
        "matched_count": 0,
        "mismatch_count": 0,
    }
    try:
        with open_log(path) as f:
            data = f.read()
    except OSError:
        return result

    ibex_mnem = ""
    spike_mnem = ""
    retire_idx = 0
    ibex_pc = ""
    for ln in data.splitlines():
        m = _RE_IBEX_MISMATCH.match(ln)
        if m:
            retire_idx = int(m.group(1))
            ibex_pc = m.group(2).lower()
            ibex_mnem = m.group(3)
        m = _RE_SPIKE_MISMATCH.match(ln)
        if m:
            spike_mnem = m.group(1)
        m = _RE_MISMATCH_COUNTS.search(ln)
        if m:
            result["matched_count"] = int(m.group(1))
            result["mismatch_count"] = int(m.group(2))

    result["retire_index"] = retire_idx
    result["ibex_mnemonic"] = ibex_mnem
    result["spike_mnemonic"] = spike_mnem
    result["ibex_pc"] = ibex_pc
    return result


def extract_regr(path: str, *, kind: str | None = None) -> dict:
    if kind == "mismatch":
        return extract_regr_mismatch(path)

    result = {"kind": kind or "unknown", "test_name": ""}
    try:
        with open_log(path) as f:
            data = f.read()
    except OSError:
        return result
    if result["kind"] == "unknown":
        if any(_RE_MISMATCH.match(ln) for ln in data.splitlines()):
            return extract_regr_mismatch(path)
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

def _has_signature_loop(loop_mnems: tuple[str, ...]) -> bool:
    """Detect Ibex DV signature-write loop (auipc / sw / c.j)."""
    s = set(loop_mnems)
    return "auipc" in s and "sw" in s and ("c.j" in s or "c.jal" in s)


def _empty_trace() -> dict:
    return {
        "mnemonics": (),
        "loop_mnems": (),
        "unique_pcs": 0,
        "total": 0,
        "has_repeating_tail": False,
        "tail_text": "",
        "length_bucket": "",
        "tail_uniformity": 0.0,
        "mismatch_context_mnemonics": (),
        "mismatch_context_text": "",
        "has_signature_loop": False,
    }


def _length_bucket(total: int) -> str:
    if total < 1_000:
        return "short"
    if total < 10_000:
        return "medium"
    if total < 100_000:
        return "long"
    return "huge"


def extract_trace(
    path: str,
    tail: int = _TRACE_TAIL,
    *,
    mismatch_retire_index: int = 0,
    context: int = _MISMATCH_CONTEXT,
) -> dict:
    """Stream trace.log; always compute tail stats, optionally capture context
    around ``mismatch_retire_index`` (1-based, matching ``ibex[N]`` in regr.log).
    """
    result = {
        "mnemonics": (),
        "loop_mnems": (),
        "unique_pcs": 0,
        "total": 0,
        "has_repeating_tail": False,
        "tail_text": "",
        "length_bucket": "",
        "tail_uniformity": 0.0,
        "mismatch_context_mnemonics": (),
        "mismatch_context_text": "",
        "has_signature_loop": False,
    }
    window: deque[tuple[str, str]] = deque(maxlen=tail)
    before: deque[str] = deque(maxlen=context)
    at_mismatch: str = ""
    after: list[str] = []
    capturing_after = False
    after_remaining = 0
    total = 0
    target = mismatch_retire_index

    try:
        with open_log(path) as f:
            for line in f:
                m = _RE_TRACE_LINE.match(line)
                if not m:
                    continue
                total += 1
                mnem = m.group(5)
                window.append((m.group(3).lower(), mnem))

                if target <= 0:
                    continue

                if total < target:
                    before.append(mnem)
                elif total == target:
                    at_mismatch = mnem
                    capturing_after = True
                    after_remaining = context
                elif capturing_after and after_remaining > 0:
                    after.append(mnem)
                    after_remaining -= 1
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
    result["has_signature_loop"] = _has_signature_loop(result["loop_mnems"])
    counts = Counter(mnems[-16:])
    result["tail_uniformity"] = counts.most_common(1)[0][1] / max(len(mnems[-16:]), 1)

    if target > 0 and at_mismatch:
        ctx = list(before) + [at_mismatch] + after
        result["mismatch_context_mnemonics"] = tuple(ctx)
        result["mismatch_context_text"] = " ".join(ctx)

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


def _build_text_blob(sig: CaseSignature, sim: dict, trace: dict, regr: dict) -> str:
    if sig.failure_mode == "mismatch":
        parts = ["MODE_mismatch", "MODE_mismatch", f"LEN_{sig.trace_length_bucket}"]
        if sig.mismatch_ibex_mnemonic:
            parts.extend([f"IBEX_{sig.mismatch_ibex_mnemonic}"] * 3)
        if sig.mismatch_spike_mnemonic:
            parts.extend([f"SPIKE_{sig.mismatch_spike_mnemonic}"] * 3)
        pair = f"{sig.mismatch_ibex_mnemonic}_{sig.mismatch_spike_mnemonic}"
        if sig.mismatch_ibex_mnemonic and sig.mismatch_spike_mnemonic:
            parts.extend([f"PAIR_{pair}"] * 3)
        if sig.mismatch_retire_index:
            bucket = min(sig.mismatch_retire_index // 100, 999)
            parts.append(f"RETIRE_{bucket}")
        if sig.early_mismatch:
            parts.extend(["EARLY_MISMATCH"] * 4)
        if sig.same_reg_pair:
            parts.extend(["SAME_REG_PAIR"] * 4)
        if not sig.has_signature_loop:
            parts.extend(["NO_SIG_LOOP"] * 2)
        if trace.get("mismatch_context_text"):
            parts.extend([f"CTX {trace['mismatch_context_text']}"] * 2)
        if sig.has_signature_loop:
            parts.extend(["SIG_LOOP"] * 3)
        if sig.has_repeating_tail:
            parts.extend(["REPEAT_TAIL"] * 2)
        if sig.tail_mnem_uniformity >= 0.75:
            parts.append(f"UNIFORM_{sig.tail_mnem_uniformity:.2f}")
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

    if regr_kind == "mismatch":
        regr = extract_regr(regr_path, kind="mismatch")
        sim = _empty_sim()
        trace = extract_trace(
            trace_path,
            mismatch_retire_index=regr["retire_index"],
        )
    else:
        regr = extract_regr(regr_path, kind=regr_kind)
        sim = extract_sim(_resolve(base_dir, sim_rel))
        trace = _empty_trace()

    matched_count = int(regr.get("matched_count", 0))
    retire_index = int(regr.get("retire_index", 0))
    ibex_mnem = regr.get("ibex_mnemonic", "")
    spike_mnem = regr.get("spike_mnemonic", "")
    early_mismatch = (
        regr["kind"] == "mismatch"
        and (matched_count < 200 or (0 < retire_index < 150))
    )
    same_reg_pair = bool(ibex_mnem) and ibex_mnem == spike_mnem

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
        has_signature_loop=trace.get("has_signature_loop", False),
        mismatch_retire_index=retire_index,
        mismatch_ibex_mnemonic=ibex_mnem,
        mismatch_spike_mnemonic=spike_mnem,
        mismatch_context_mnemonics=trace.get("mismatch_context_mnemonics", ()),
        mismatch_matched_count=matched_count,
        early_mismatch=early_mismatch,
        same_reg_pair=same_reg_pair,
    )
    return CaseFeatures(
        case_id=case_id,
        signature=sig,
        text_blob=_build_text_blob(sig, sim, trace, regr),
    )


def iter_summary(features: Iterable[CaseFeatures]) -> Counter:
    return Counter(f.signature.categorical_key() for f in features)
