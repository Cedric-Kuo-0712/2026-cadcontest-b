"""
Log parsing module: extract structured information from raw logs
"""

import gzip
import re
from collections import Counter, deque
from typing import Dict


def open_file(filepath: str):
    """Open .gz or plain text file"""
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt", encoding="utf-8", errors="ignore")
    return open(filepath, "r", encoding="utf-8", errors="ignore")


def classify_error_type(regr_parsed: Dict, sim_parsed: Dict, trace_parsed: Dict) -> str:
    """
    Classify failure into error type categories:
    - UVM_FATAL: UVM fatal error
    - MISMATCH: RTL-ISS mismatch
    - TRACE_ABNORMAL: abnormal trace pattern
    - FAILED_ONLY: test failed without clear reason
    """
    if regr_parsed["has_mismatch"]:
        return "MISMATCH"

    if sim_parsed["error_type"] == "FATAL":
        return "UVM_FATAL"

    if sim_parsed["assert_count"] > 0:
        return "ASSERT_ERROR"
    
    if trace_parsed["has_loop"] or trace_parsed["has_pc_stall"]:
        return "TRACE_ABNORMAL"
    
    return "FAILED_ONLY"


class RegrLogParser:
    """Parse regr.log - RTL vs ISS check results"""
    
    def __init__(self):
        self.mismatch_pattern = re.compile(r"Mismatch\[(\d+)\]")
        self.failed_pattern = re.compile(r"\[FAILED\]:\s+(\d+)\s+matched,\s+(\d+)\s+mismatch")
        self.short_failed_pattern = re.compile(r"^(\S+\.0)\s*:\s*\[FAILED\]", re.MULTILINE)
        self.side_pattern = re.compile(
            r"^(ibex|spike)\[(\d+)\]\s*:\s*pc\[([0-9a-fA-Fx]+)\]\s+(\S+)",
            re.MULTILINE,
        )
    
    def parse(self, filepath: str) -> dict:
        """Extract regr.log features"""
        features = {
            "first_mismatch_idx": None,
            "mismatch_count": 0,
            "matched_count": 0,
            "has_mismatch": False,
            "full_text": "",
            "pc_mismatch_lines": [],
            "short_test_name": "",
            "ibex_retire_idx": None,
            "ibex_pc": "",
            "ibex_mnemonic": "",
            "spike_retire_idx": None,
            "spike_pc": "",
            "spike_mnemonic": "",
        }
        
        try:
            with open_file(filepath) as f:
                text = f.read()
                features["full_text"] = text
                
                for line in text.split("\n"):
                    m_idx = self.mismatch_pattern.search(line)
                    if m_idx and features["first_mismatch_idx"] is None:
                        features["first_mismatch_idx"] = int(m_idx.group(1))
                        features["has_mismatch"] = True
                    
                    if "pc[" in line.lower() and "mismatch" in line.lower():
                        features["pc_mismatch_lines"].append(line.strip())
                
                m_count = self.failed_pattern.search(text)
                if m_count:
                    features["matched_count"] = int(m_count.group(1))
                    features["mismatch_count"] = int(m_count.group(2))
                    features["has_mismatch"] = True

                m_short = self.short_failed_pattern.search(text.strip())
                if m_short:
                    features["short_test_name"] = m_short.group(1)

                for side, idx, pc, mnemonic in self.side_pattern.findall(text):
                    features[f"{side}_retire_idx"] = int(idx)
                    features[f"{side}_pc"] = pc.lower()
                    features[f"{side}_mnemonic"] = mnemonic
        
        except Exception as e:
            print(f"Warning: Failed to parse {filepath}: {e}")
        
        return features


# UVM_FATAL templates we expect from core_ibex_*.sv. Order matters: first hit wins.
# Source for templates: lowRISC/ibex deep-dive issue #2187 + base_test/test_lib.sv.
FATAL_KIND_RULES = [
    ("WALL_CLOCK_TIMEOUT", "wall-clock timeout"),
    ("TEST_TIMEOUT", "TEST TIMEOUT"),
    ("CORE_STATUS_TIMEOUT", "Did not receive core_status"),
    ("CSR_TIMEOUT", "Did not receive write to csr"),
    ("HANDSHAKE_FAIL", "RISCV-DV handshake"),
    ("HANDSHAKE_MALFORMED", "Incorrectly formed handshake"),
    ("SIG_FORMAT_BAD", "formatted incorrectly"),
    ("PRIV_MODE_CHECK", "priv_mode == mode"),
    ("MCAUSE_CHECK", "mcause"),
    ("SIGNATURE_CHECK", "signature_data == core_status"),
    ("DRET_CHECK", "No dret detected"),
    ("HDL_READ_FAIL", "uvm_hdl_read"),
    ("DOUBLE_FAULT", "double_fault detector"),
    ("CONFIG_MISSING", "Cannot get"),
    ("BIN_OPEN_FAIL", "Cannot open file"),
    ("COSIM_MISMATCH", "Cosim mismatch"),
]

CORE_STATUS_KINDS = [
    "INITIALIZED", "IN_DEBUG_MODE", "HANDLING_IRQ",
    "EBREAK_TAKEN", "HANDLING_EXCEPTION", "FINISHED_DIR_INSTR",
]

ASSERT_MODULES = ["cs_registers_i", "id_stage_i", "load_store_unit_i", "controller_i"]


class SimLogParser:
    """Parse sim.log.gz - UVM simulation log."""

    def __init__(self):
        self.uvm_fatal_pattern = re.compile(r"^UVM_FATAL\s+\S+.*$", re.MULTILINE)
        self.uvm_error_pattern = re.compile(r"^UVM_ERROR\s+\S+.*$", re.MULTILINE)
        self.uvm_warning_pattern = re.compile(r"^UVM_WARNING\s+\S+.*$", re.MULTILINE)
        self.uvm_info_pattern = re.compile(r"^UVM_INFO\s+\S+.*$", re.MULTILINE)
        self.assert_pattern = re.compile(r"\[ASSERT FAILED\]\s*\[([^\]]+)\]")
        self.test_pattern = re.compile(r"\+UVM_TESTNAME=(\S+)")
        self.cmd_pattern = re.compile(r"^Command:\s+(.*)$", re.MULTILINE)
        self.finish_pattern = re.compile(r"\$finish\s+at\s+simulation\s+time\s+(\d+)")
        self.seq_pattern = re.compile(
            r"UVM_INFO\s+\S+\s+@\s+\d+:\s+\S+\s+\[(\S+_seq_h)\]"
        )
        self.sim_err_pattern = re.compile(r"^xmsim:\s*\*SE,", re.MULTILINE)

    def parse(self, filepath: str) -> dict:
        features = self._empty_features()
        try:
            with open_file(filepath) as f:
                text = f.read()
        except Exception as e:
            print(f"Warning: Failed to parse {filepath}: {e}")
            return features

        features["full_text"] = text[-2000:]

        m_test = self.test_pattern.search(text)
        if m_test:
            features["uvm_test_name"] = m_test.group(1)

        cmd = self.cmd_pattern.search(text)
        plus_keys = set()
        if cmd:
            plus_keys = {p.split("=")[0] for p in re.findall(r"\+(\S+)", cmd.group(1))}
        features["enable_debug_seq"] = "enable_debug_seq" in plus_keys
        features["enable_irq_single_seq"] = "enable_irq_single_seq" in plus_keys
        features["enable_irq_multiple_seq"] = "enable_irq_multiple_seq" in plus_keys
        features["has_max_interval"] = "max_interval" in plus_keys
        features["plusarg_signature"] = "|".join(sorted(
            k for k in plus_keys
            if k not in {"ntb_random_seed", "bin", "signature_addr",
                         "vcs+lic+wait", "UVM_VERBOSITY"}
        ))

        fatal_lines = [l for l in self.uvm_fatal_pattern.findall(text) if "reports" not in l]
        error_lines = [l for l in self.uvm_error_pattern.findall(text) if "reports" not in l]
        warning_lines = [l for l in self.uvm_warning_pattern.findall(text) if "reports" not in l]
        info_lines = self.uvm_info_pattern.findall(text)
        assert_names = self.assert_pattern.findall(text)

        features["uvm_info_count"] = len(info_lines)
        features["uvm_warning_count"] = len(warning_lines)
        features["uvm_error_count"] = len(error_lines)
        features["uvm_fatal_count"] = len(fatal_lines)
        features["assert_count"] = len(assert_names)
        features["assert_names"] = assert_names
        features["top_asserts"] = [name for name, _ in Counter(assert_names).most_common(3)]
        features["assert_module_counts"] = self._count_assert_modules(text)

        seq_counter = Counter(self.seq_pattern.findall(text))
        features["seq_event_counts"] = dict(seq_counter)
        features["irq_raise_count"] = seq_counter.get("irq_raise_seq_h", 0)
        features["irq_drop_count"] = seq_counter.get("irq_drop_seq_h", 0)
        features["irq_single_count"] = seq_counter.get("irq_single_seq_h", 0)
        features["debug_seq_count"] = sum(v for k, v in seq_counter.items() if "debug_seq" in k)
        features["total_seq_events"] = sum(seq_counter.values())

        m_finish = self.finish_pattern.search(text)
        features["finish_time"] = int(m_finish.group(1)) if m_finish else 0
        features["reached_test_done"] = "TEST_DONE" in text or "Test done due to" in text
        features["has_simulator_error"] = bool(self.sim_err_pattern.search(text))

        if fatal_lines:
            features["error_type"] = "FATAL"
            fatal = fatal_lines[0]
            features["error_signature"] = self._simplify_signature(fatal)
            features["full_text"] = fatal
            features["fatal_message"] = self._extract_fatal_message(fatal)
            features["fatal_kind"] = self._classify_fatal(fatal)
            features["core_status_kind"] = self._extract_core_status(fatal)
            source = re.search(r"/([^/\s]+\.sv)\((\d+)\)", fatal)
            if source:
                features["fatal_source"] = source.group(1)
                features["fatal_line"] = source.group(2)
        elif error_lines:
            features["error_type"] = "ERROR"
            features["error_signature"] = self._simplify_signature(error_lines[0])
        elif warning_lines:
            features["error_type"] = "WARNING"

        # Legacy keyword flags (kept for compatibility with error classifier).
        low = text.lower()
        features["keywords"] = [
            kw for kw in ("timeout", "debug", "dret", "privilege", "assert",
                          "exception", "interrupt", "trap", "illegal")
            if kw in low
        ]
        features["has_debug_error"] = "debug" in low or "dret" in low
        features["has_timeout"] = "timeout" in low
        features["has_privilege_error"] = "privilege" in low

        return features

    @staticmethod
    def _empty_features() -> dict:
        return {
            "error_type": "INFO",
            "error_signature": "",
            "keywords": [],
            "full_text": "",
            "has_debug_error": False,
            "has_timeout": False,
            "has_privilege_error": False,
            "uvm_test_name": "",
            "fatal_message": "",
            "fatal_source": "",
            "fatal_line": "",
            "fatal_kind": "NO_FATAL",
            "core_status_kind": "NONE",
            "uvm_info_count": 0,
            "uvm_warning_count": 0,
            "uvm_error_count": 0,
            "uvm_fatal_count": 0,
            "assert_count": 0,
            "assert_names": [],
            "top_asserts": [],
            "assert_module_counts": {},
            "enable_debug_seq": False,
            "enable_irq_single_seq": False,
            "enable_irq_multiple_seq": False,
            "has_max_interval": False,
            "plusarg_signature": "",
            "seq_event_counts": {},
            "irq_raise_count": 0,
            "irq_drop_count": 0,
            "irq_single_count": 0,
            "debug_seq_count": 0,
            "total_seq_events": 0,
            "finish_time": 0,
            "reached_test_done": False,
            "has_simulator_error": False,
        }

    @staticmethod
    def _count_assert_modules(text: str) -> dict:
        counts = {m: 0 for m in ASSERT_MODULES}
        counts["other"] = 0
        for m in re.finditer(r"\[ASSERT FAILED\]\s*\[([^\]]+)\]", text):
            path = m.group(1)
            matched = False
            for mod in ASSERT_MODULES:
                if mod in path:
                    counts[mod] += 1
                    matched = True
                    break
            if not matched:
                counts["other"] += 1
        return counts

    @staticmethod
    def _classify_fatal(fatal_line: str) -> str:
        for kind, needle in FATAL_KIND_RULES:
            if needle in fatal_line:
                return kind
        return "OTHER_FATAL"

    @staticmethod
    def _extract_core_status(fatal_line: str) -> str:
        m = re.search(r"core_status\s+(\w+)", fatal_line)
        if not m:
            return "NONE"
        kind = m.group(1)
        return kind if kind in CORE_STATUS_KINDS else "OTHER"

    @staticmethod
    def _simplify_signature(sig: str) -> str:
        sig = re.sub(r"0x[0-9a-fA-F]+", "ADDR", sig)
        sig = re.sub(r"\d{4,}", "NUM", sig)
        return sig[:80]

    @staticmethod
    def _extract_fatal_message(line: str) -> str:
        msg = re.sub(r"^UVM_FATAL\s+\S+\s+@\s+\d+:\s+\S+\s+\[[^\]]+\]\s*", "", line)
        msg = re.sub(r"0x[0-9a-fA-F]+", "ADDR", msg)
        msg = re.sub(r"\d+", "N", msg)
        return msg.strip()[:120]


class TraceLogParser:
    """Parse trace.log.gz - CPU execution trace"""
    
    def __init__(self):
        self.instr_pattern = re.compile(r"^\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)")
    
    def parse(self, filepath: str, tail_lines: int = 60) -> dict:
        """Extract trace.log.gz features"""
        features = {
            "instr_sequence": [],
            "pc_deltas": [],
            "last_instructions": [],
            "anomaly_score": 0.0,
            "has_loop": False,
            "has_pc_stall": False,
            "line_count": 0,
            "tail_signature": "",
            "loop_signature": "",
            "pc_unique_ratio": 1.0,
        }
        
        try:
            with open_file(filepath) as f:
                tail = deque(maxlen=tail_lines)
                line_count = 0
                for line in f:
                    line_count += 1
                    tail.append(line)
                tail = list(tail)
                features["line_count"] = line_count
                
                prev_pc = None
                pc_deltas = []
                instructions = []
                last_pcs = []
                
                for line in tail:
                    m = self.instr_pattern.search(line)
                    if m:
                        pc_str = m.group(3)
                        instr = m.group(5)
                        
                        instr_mnem = instr.split()[0] if instr.split() else "unknown"
                        instructions.append(instr_mnem)
                        
                        try:
                            pc = int(pc_str, 16)
                            last_pcs.append(pc)
                            if prev_pc is not None:
                                delta = pc - prev_pc
                                pc_deltas.append(delta)
                            prev_pc = pc
                        except:
                            pass
                
                features["instr_sequence"] = instructions
                features["last_instructions"] = instructions
                features["tail_signature"] = ",".join(instructions[-8:])
                
                if pc_deltas:
                    if any(d < 0 for d in pc_deltas):
                        features["has_loop"] = True
                    
                    if last_pcs and len(set(last_pcs)) < len(last_pcs) / 2:
                        features["has_pc_stall"] = True
                    if last_pcs:
                        features["pc_unique_ratio"] = len(set(last_pcs)) / len(last_pcs)

                features["loop_signature"] = self._loop_signature(instructions)
                
                suspicious_instrs = {"unknown", "???", "0x00000000", "nop"}
                anomaly_count = sum(1 for i in instructions if i in suspicious_instrs)
                features["anomaly_score"] = anomaly_count / len(instructions) if instructions else 0.0
        
        except Exception as e:
            print(f"Warning: Failed to parse {filepath}: {e}")
        
        return features

    @staticmethod
    def _loop_signature(instructions):
        """Return a short repeated tail pattern if present."""
        if len(instructions) < 6:
            return ",".join(instructions)
        tail = instructions[-24:]
        for size in range(1, 7):
            pattern = tail[-size:]
            repeats = size * 3
            if len(tail) >= repeats and tail[-repeats:] == pattern * 3:
                return ",".join(pattern)
        return ",".join(tail[-6:])
