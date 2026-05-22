"""Feature extraction for regression-failure bucketing.

Reads the three log files (`regr.log`, `sim.log[.gz]`, `trace.log[.gz]`) for a
single case and produces:

  * a structured ``CaseSignature`` (categorical fingerprint), used by the
    signature-based bucketer, and
  * a compact normalized text blob, used by the TF-IDF / embedding fallback.

Design notes
------------
* Trace logs may be up to 100M lines (gzip compressed) per benchmark, spread
  across many cases.  We never materialize a full trace; instead we stream
  line-by-line and keep a small rolling tail (``deque``).
* Sim logs may also be huge but the discriminative content (UVM messages,
  +plusargs) lives near the start and end.  We stop scanning UVM_INFO once a
  hard cap is reached.
* Everything that depends on randomly-changing numbers (timestamps, PCs,
  register values, paths) is normalized so that identical bugs across
  different seeds collapse to the same signature.
"""

from __future__ import annotations

import gzip
import os
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def open_log(path: str):
    """Open a (possibly gzipped) log file in text mode, tolerating bad bytes."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_HEX_NUMBER = re.compile(r"0x[0-9a-fA-F]+")
_DEC_NUMBER = re.compile(r"\b\d+\b")
_WS = re.compile(r"\s+")
_PATH = re.compile(r"/[\w\-./]+")
_TIME_TAG = re.compile(r"@\s*\d+")


def normalize_message(text: str) -> str:
    """Collapse numeric/path noise so identical message templates align."""
    out = _PATH.sub("<PATH>", text)
    out = _HEX_NUMBER.sub("<HEX>", out)
    out = _TIME_TAG.sub("@<T>", out)
    out = _DEC_NUMBER.sub("<N>", out)
    out = _WS.sub(" ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CaseSignature:
    """Categorical fingerprint of a single failure case."""

    sim_verdict: str = "unknown"            # passed / failed / unknown
    sim_fatal_template: str = ""            # normalized first UVM_FATAL body
    sim_fatal_category: str = ""            # coarse: timeout / check / none
    sim_fatal_source: str = ""              # source file basename of UVM_FATAL
    sim_error_asserts: tuple = ()           # sorted distinct assertion names
    sim_uvm_testname: str = ""              # +UVM_TESTNAME=<...>
    sim_bin_name: str = ""                  # +bin=<.../riscv_FOO_test_0.bin>
    regr_kind: str = "unknown"              # mismatch / failed_only / unknown
    regr_test_name: str = ""                # e.g. riscv_csr_test
    regr_mismatch_mnemonics: tuple = ()     # (ibex_mnem, spike_mnem)
    trace_tail_mnemonics: tuple = ()        # last few decoded mnemonics
    trace_tail_unique_pcs: int = 0
    trace_tail_total: int = 0
    has_repeating_tail: bool = False

    def categorical_key(self) -> tuple:
        """Tuple used to group cases with an identical fingerprint."""
        return (
            self.sim_verdict,
            self.sim_fatal_template,
            self.sim_error_asserts,
            self.regr_kind,
            self.regr_mismatch_mnemonics,
            self.trace_tail_mnemonics,
            self.has_repeating_tail,
        )


@dataclass
class CaseFeatures:
    """Everything extracted for one case."""

    case_id: int
    signature: CaseSignature = field(default_factory=CaseSignature)
    text_blob: str = ""                     # compact normalized text for TF-IDF


# ---------------------------------------------------------------------------
# Per-file extractors
# ---------------------------------------------------------------------------

_RE_UVM_FATAL = re.compile(r"UVM_FATAL\b.*")
_RE_FATAL_SOURCE = re.compile(r"UVM_FATAL\s+(\S+?\.sv)\b")
_RE_FATAL_TIMEOUT = re.compile(
    r"(Did not receive core_status|No dret detected|timeout period|"
    r"within \d+ cycle)", re.IGNORECASE
)
_RE_FATAL_CHECK = re.compile(r"(Check failed|signature_data|verify mismatch)",
                              re.IGNORECASE)
_RE_UVM_ERROR = re.compile(r"UVM_ERROR\b.*")
_RE_UVM_WARN = re.compile(r"UVM_WARNING\b.*")
_RE_VERDICT_PASS = re.compile(r"RISC-V UVM TEST PASSED")
_RE_VERDICT_FAIL = re.compile(r"RISC-V UVM TEST FAILED")
_RE_ASSERT_NAME = re.compile(r"ASSERT FAILED\] \[[^.\]]*\.([^.\]]+)\]")
# Pull both bracketed assertion name and the bare property name.
_RE_ASSERT_PROP = re.compile(r"\] ([A-Za-z_][A-Za-z0-9_]*):")
_RE_TESTNAME = re.compile(r"\+UVM_TESTNAME=(\S+)")
_RE_BIN = re.compile(r"\+bin=(\S+)")
_RE_BIN_NAME = re.compile(r"/([A-Za-z0-9_]+_test)_\d+\.bin")
_RE_MISMATCH_HEADER = re.compile(r"^Mismatch\[\d+\]:")
_RE_IBEX_LINE = re.compile(r"^ibex\[\d+\]\s*:.*?\b([a-z][a-z0-9.]*)\b")
_RE_SPIKE_LINE = re.compile(r"^spike\[\d+\]\s*:.*?\b([a-z][a-z0-9.]*)\b")
_RE_REGR_FAILED = re.compile(r"^([A-Za-z0-9_.]+)\s*:\s*\[FAILED\]")

# trace.log columns: Time, Cycle, PC, Insn, Decoded, ...
_RE_TRACE_LINE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+(\S+)"
)


# Some limits to keep runtime bounded on huge logs.
_SIM_MAX_BYTES = 50 * 1024 * 1024          # 50 MB read cap per sim.log
_TRACE_TAIL = 64                            # rolling tail window
_SIM_FIRST_FATAL_CAP = 5                    # collect at most this many fatal lines


def extract_regr(path: str) -> dict:
    """Parse a regr.log snippet."""
    result = {
        "kind": "unknown",
        "test_name": "",
        "mnemonics": ("", ""),
        "text": "",
    }
    try:
        with open_log(path) as f:
            data = f.read()
    except OSError:
        return result

    result["text"] = normalize_message(data[:4000])

    lines = data.splitlines()
    if not lines:
        return result

    if any(_RE_MISMATCH_HEADER.match(ln) for ln in lines):
        result["kind"] = "mismatch"
        ibex_mnem = ""
        spike_mnem = ""
        for ln in lines:
            m = _RE_IBEX_LINE.match(ln)
            if m:
                ibex_mnem = m.group(1)
            m = _RE_SPIKE_LINE.match(ln)
            if m:
                spike_mnem = m.group(1)
            if ibex_mnem and spike_mnem:
                break
        result["mnemonics"] = (ibex_mnem, spike_mnem)
    else:
        for ln in lines:
            m = _RE_REGR_FAILED.match(ln.strip())
            if m:
                result["kind"] = "failed_only"
                # Drop a trailing .N seed suffix so cases with different seeds
                # of the same test fall into the same bucket.
                tname = m.group(1)
                tname = re.sub(r"\.\d+$", "", tname)
                result["test_name"] = tname
                break

    return result


def extract_sim(path: str) -> dict:
    """Parse a sim.log[.gz]."""
    result = {
        "verdict": "unknown",
        "fatal_template": "",
        "fatal_category": "",
        "fatal_source": "",
        "error_asserts": (),
        "uvm_testname": "",
        "bin_name": "",
        "fatal_messages": [],   # raw normalized lines for the text blob
        "error_messages": [],
    }
    error_asserts = set()
    fatal_lines: list[str] = []
    fatal_raw: str = ""
    error_lines: list[str] = []
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
                if not result["bin_name"]:
                    m = _RE_BIN.search(line)
                    if m:
                        bn = _RE_BIN_NAME.search(m.group(1))
                        if bn:
                            result["bin_name"] = bn.group(1)

                if _RE_VERDICT_PASS.search(line):
                    result["verdict"] = "passed"
                elif _RE_VERDICT_FAIL.search(line):
                    result["verdict"] = "failed"

                fatal_match = _RE_UVM_FATAL.search(line)
                if fatal_match and len(fatal_lines) < _SIM_FIRST_FATAL_CAP:
                    raw = fatal_match.group(0)
                    fatal_lines.append(normalize_message(raw))
                    if not fatal_raw:
                        fatal_raw = raw

                err_match = _RE_UVM_ERROR.search(line)
                if err_match:
                    if len(error_lines) < 16:
                        error_lines.append(normalize_message(err_match.group(0)))
                    name = _RE_ASSERT_NAME.search(line)
                    if name:
                        error_asserts.add(name.group(1))
                    else:
                        prop = _RE_ASSERT_PROP.search(line)
                        if prop:
                            error_asserts.add(prop.group(1))
    except OSError:
        return result

    if fatal_lines:
        # Strip leading file location to keep only the message.
        first = fatal_lines[0]
        first = re.sub(r"^UVM_FATAL\s+<PATH>\(<N>\)\s*", "UVM_FATAL ", first)
        result["fatal_template"] = first

    if fatal_raw:
        m = _RE_FATAL_SOURCE.search(fatal_raw)
        if m:
            result["fatal_source"] = os.path.basename(m.group(1))
        if _RE_FATAL_TIMEOUT.search(fatal_raw):
            result["fatal_category"] = "timeout"
        elif _RE_FATAL_CHECK.search(fatal_raw):
            result["fatal_category"] = "check"
        else:
            result["fatal_category"] = "other"
    else:
        result["fatal_category"] = "none"

    result["error_asserts"] = tuple(sorted(error_asserts))
    result["fatal_messages"] = fatal_lines
    result["error_messages"] = error_lines
    return result


def extract_trace(path: str, tail: int = _TRACE_TAIL) -> dict:
    """Stream a trace.log[.gz] and return tail statistics."""
    result = {
        "mnemonics": (),
        "unique_pcs": 0,
        "total": 0,
        "has_repeating_tail": False,
        "tail_text": "",
    }
    window: deque[tuple[str, str]] = deque(maxlen=tail)
    total = 0
    try:
        with open_log(path) as f:
            for line in f:
                m = _RE_TRACE_LINE.match(line)
                if not m:
                    continue
                pc = m.group(3).lower()
                mnem = m.group(5)
                window.append((pc, mnem))
                total += 1
    except OSError:
        return result

    if not window:
        return result

    pcs = [pc for pc, _ in window]
    mnems = [m for _, m in window]
    unique_pcs = len(set(pcs))
    result["unique_pcs"] = unique_pcs
    result["total"] = total
    # If we saw many lines but the tail occupies few PCs, the core was
    # likely stuck in a tight loop -- a classic mismatch symptom.
    result["has_repeating_tail"] = (
        len(window) >= 10 and unique_pcs <= max(2, len(window) // 4)
    )
    result["mnemonics"] = tuple(mnems[-8:])
    result["tail_text"] = " ".join(mnems[-16:])
    return result


# ---------------------------------------------------------------------------
# Top-level: case -> features
# ---------------------------------------------------------------------------

def _resolve(base_dir: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.join(base_dir, rel)


def build_case_features(
    case_id: int,
    base_dir: str,
    regr_rel: str,
    sim_rel: str,
    trace_rel: str,
) -> CaseFeatures:
    """Extract features for a single case."""
    regr = extract_regr(_resolve(base_dir, regr_rel))
    sim = extract_sim(_resolve(base_dir, sim_rel))
    trace = extract_trace(_resolve(base_dir, trace_rel))

    sig = CaseSignature(
        sim_verdict=sim["verdict"],
        sim_fatal_template=sim["fatal_template"],
        sim_fatal_category=sim["fatal_category"],
        sim_fatal_source=sim["fatal_source"],
        sim_error_asserts=sim["error_asserts"],
        sim_uvm_testname=sim["uvm_testname"],
        sim_bin_name=sim["bin_name"],
        regr_kind=regr["kind"],
        regr_test_name=regr["test_name"],
        regr_mismatch_mnemonics=regr["mnemonics"],
        trace_tail_mnemonics=trace["mnemonics"],
        trace_tail_unique_pcs=trace["unique_pcs"],
        trace_tail_total=trace["total"],
        has_repeating_tail=trace["has_repeating_tail"],
    )

    # Compact, deterministic, normalized text used by the TF-IDF fallback.
    # Tokens that strongly identify a bug (verdict, fatal template, assertion
    # names) are repeated so TF-IDF gives them more weight than the noisy
    # tail / regr text.
    parts: list[str] = []
    parts.extend([f"VERDICT_{sim['verdict']}"] * 3)
    parts.append(f"REGRKIND_{regr['kind']}")
    parts.append(f"REGRKIND_{regr['kind']}")

    if sim["fatal_template"]:
        # Compact the template into a single tag to keep TF-IDF aligned across
        # cases with the same fatal class.
        fatal_tag = re.sub(r"[^A-Za-z]+", "_", sim["fatal_template"])[:80]
        parts.extend([f"FATAL_{fatal_tag}"] * 3)
    else:
        parts.append("FATAL_NONE")
    if sim["fatal_category"]:
        parts.extend([f"FATALCAT_{sim['fatal_category']}"] * 2)
    if sim["fatal_source"]:
        parts.extend([f"FATALSRC_{sim['fatal_source']}"] * 2)

    if sim["error_asserts"]:
        for asrt in sim["error_asserts"]:
            parts.extend([f"ASSERT_{asrt}"] * 3)
        parts.append("HAS_ASSERT")
    else:
        parts.append("NO_ASSERT")

    if sim["uvm_testname"]:
        parts.append(f"UVMTEST_{sim['uvm_testname']}")
    if sim["bin_name"]:
        parts.append(f"BIN_{sim['bin_name']}")
    if regr["test_name"]:
        parts.append(f"REGRTEST_{regr['test_name']}")

    if any(regr["mnemonics"]):
        ib, sp = regr["mnemonics"]
        parts.extend([f"IBEX_{ib}", f"SPIKE_{sp}"] * 2)

    if trace["tail_text"]:
        parts.append("TAIL " + trace["tail_text"])
    if trace["has_repeating_tail"]:
        parts.append("REPEAT_TAIL")

    # A small amount of normalized error context (helps discriminate
    # different bugs that share assertion names but differ in their UVM
    # message bodies, e.g. "Check failed mca" vs plain timeouts).
    for msg in sim["error_messages"][:3]:
        parts.append(re.sub(r"[^A-Za-z_]+", " ", msg))

    return CaseFeatures(case_id=case_id, signature=sig, text_blob=" ".join(parts))


def iter_summary(features: Iterable[CaseFeatures]) -> Counter:
    """Useful for debugging: counts of distinct categorical keys."""
    return Counter(f.signature.categorical_key() for f in features)
