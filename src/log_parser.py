"""
Log parsing module: extract structured information from raw logs
"""

import gzip
import re
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
    if sim_parsed["error_type"] == "FATAL":
        return "UVM_FATAL"
    
    if regr_parsed["has_mismatch"]:
        return "MISMATCH"
    
    if trace_parsed["has_loop"] or trace_parsed["has_pc_stall"]:
        return "TRACE_ABNORMAL"
    
    return "FAILED_ONLY"


class RegrLogParser:
    """Parse regr.log - RTL vs ISS check results"""
    
    def __init__(self):
        self.mismatch_pattern = re.compile(r"Mismatch\[(\d+)\]")
        self.failed_pattern = re.compile(r"\[FAILED\]:\s+(\d+)\s+matched,\s+(\d+)\s+mismatch")
    
    def parse(self, filepath: str) -> dict:
        """Extract regr.log features"""
        features = {
            "first_mismatch_idx": None,
            "mismatch_count": 0,
            "has_mismatch": False,
            "full_text": "",
            "pc_mismatch_lines": []
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
                    features["mismatch_count"] = int(m_count.group(2))
                    features["has_mismatch"] = True
        
        except Exception as e:
            print(f"Warning: Failed to parse {filepath}: {e}")
        
        return features


class SimLogParser:
    """Parse sim.log.gz - UVM simulation log"""
    
    def __init__(self):
        self.uvm_fatal_pattern = re.compile(r"UVM_FATAL\s+(.+?)(?:\n|$)")
        self.uvm_error_pattern = re.compile(r"UVM_ERROR\s+(.+?)(?:\n|$)")
        self.uvm_warning_pattern = re.compile(r"UVM_WARNING\s+(.+?)(?:\n|$)")
    
    def parse(self, filepath: str) -> dict:
        """Extract sim.log.gz features"""
        features = {
            "error_type": "INFO",
            "error_signature": "",
            "keywords": [],
            "full_text": "",
            "has_debug_error": False,
            "has_timeout": False,
            "has_privilege_error": False
        }
        
        try:
            with open_file(filepath) as f:
                text = f.read()
                features["full_text"] = text[-2000:]
                
                m_fatal = self.uvm_fatal_pattern.search(text)
                if m_fatal:
                    features["error_type"] = "FATAL"
                    sig = m_fatal.group(1).strip()
                    features["error_signature"] = self._simplify_signature(sig)
                    features["full_text"] = m_fatal.group(0)
                
                elif self.uvm_error_pattern.search(text):
                    features["error_type"] = "ERROR"
                    m_error = self.uvm_error_pattern.search(text)
                    if m_error:
                        sig = m_error.group(1).strip()
                        features["error_signature"] = self._simplify_signature(sig)
                
                elif self.uvm_warning_pattern.search(text):
                    features["error_type"] = "WARNING"
                
                keywords = ["timeout", "debug", "dret", "privilege", "assert", 
                           "exception", "interrupt", "trap", "illegal"]
                for kw in keywords:
                    if kw.lower() in text.lower():
                        features["keywords"].append(kw)
                
                features["has_debug_error"] = "debug" in text.lower() or "dret" in text.lower()
                features["has_timeout"] = "timeout" in text.lower()
                features["has_privilege_error"] = "privilege" in text.lower()
        
        except Exception as e:
            print(f"Warning: Failed to parse {filepath}: {e}")
        
        return features
    
    @staticmethod
    def _simplify_signature(sig: str) -> str:
        """Simplify error signature"""
        sig = re.sub(r"0x[0-9a-fA-F]+", "ADDR", sig)
        sig = re.sub(r"\d{4,}", "NUM", sig)
        return sig[:80]


class TraceLogParser:
    """Parse trace.log.gz - CPU execution trace"""
    
    def __init__(self):
        self.instr_pattern = re.compile(r"^\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)")
    
    def parse(self, filepath: str, tail_lines: int = 20) -> dict:
        """Extract trace.log.gz features"""
        features = {
            "instr_sequence": [],
            "pc_deltas": [],
            "last_instructions": [],
            "anomaly_score": 0.0,
            "has_loop": False,
            "has_pc_stall": False
        }
        
        try:
            with open_file(filepath) as f:
                lines = f.readlines()
                tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
                
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
                
                if pc_deltas:
                    if any(d < 0 for d in pc_deltas):
                        features["has_loop"] = True
                    
                    if last_pcs and len(set(last_pcs)) < len(last_pcs) / 2:
                        features["has_pc_stall"] = True
                
                suspicious_instrs = {"unknown", "???", "0x00000000", "nop"}
                anomaly_count = sum(1 for i in instructions if i in suspicious_instrs)
                features["anomaly_score"] = anomaly_count / len(instructions) if instructions else 0.0
        
        except Exception as e:
            print(f"Warning: Failed to parse {filepath}: {e}")
        
        return features
