"""
Strategy pattern for feature extraction and clustering methods.
Allows flexible switching between different approaches via CLI.
"""

import numpy as np
import gzip
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def open_file(filepath: str):
    """Open .gz or plain text file"""
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt", encoding="utf-8", errors="ignore")
    return open(filepath, "r", encoding="utf-8", errors="ignore")


# ============================================================================
# FEATURE EXTRACTION STRATEGIES
# ============================================================================

class FeatureExtractor(ABC):
    """Base class for feature extraction strategies"""
    
    @abstractmethod
    def extract(self, cases: List[Dict], parsed_results: List[Dict]) -> np.ndarray:
        """Extract features from cases. Returns (n_cases, n_features) array"""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return feature dimension"""
        pass


class CharEmbeddingExtractor(FeatureExtractor):
    """Character frequency embedding (matches original implementation exactly)"""
    
    def __init__(self, char_dim: int = 128):
        self.char_dim = char_dim
        self.uvm_pattern = re.compile(r"^(UVM_FATAL.*|UVM_ERROR.*)")
        self.instr_pattern = re.compile(r"^\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)")
    
    def extract(self, cases: List[Dict], parsed_results: List[Dict]) -> np.ndarray:
        """Extract character frequency embeddings from logs (original algorithm)"""
        features = []
        
        for i, case in enumerate(cases):
            # Direct file access for exact original behavior
            regr_path = case["Regr Log"]
            sim_path = case["Sim Log"]
            trace_path = case["Trace Log"]
            
            # Sim embedding: find first UVM_FATAL/ERROR line and compute its ngram
            sim_embedding = np.zeros(self.char_dim, dtype=float)
            try:
                with open_file(sim_path) as f:
                    for line in f:
                        m = self.uvm_pattern.search(line)
                        if m:
                            sim_embedding = self._calculate_ngram(m.group(0), self.char_dim)
                            break
            except:
                pass
            
            # Trace embedding: take last 10 lines, extract instruction field (group 5), accumulate
            trace_embedding = np.zeros(self.char_dim, dtype=float)
            try:
                with open_file(trace_path) as f:
                    lines = f.read().splitlines()[-10:]  # Last 10 lines
                    for line in lines:
                        m = self.instr_pattern.search(line)
                        if m:
                            trace_embedding += self._calculate_ngram(m.group(5), self.char_dim)
            except:
                pass
            
            # Regr embedding: accumulate entire file
            regr_embedding = np.zeros(self.char_dim, dtype=float)
            try:
                with open_file(regr_path) as f:
                    regr_embedding = self._calculate_ngram(f.read(), self.char_dim)
            except:
                pass
            
            feature = np.concatenate([sim_embedding, trace_embedding, regr_embedding])
            features.append(feature)
        
        return np.array(features)
    
    @staticmethod
    def _calculate_ngram(in_string: str, dim: int = 128) -> np.ndarray:
        """Calculate character frequency (original implementation)"""
        out_ngram = np.zeros(dim, dtype=float)
        for i in range(len(in_string)):
            code = ord(in_string[i])
            if code < dim:
                out_ngram[code] += 1
        return out_ngram
    
    def get_feature_dim(self) -> int:
        return self.char_dim * 3  # sim + trace + regr


class SimTraceExtractor(FeatureExtractor):
    """
    Char embedding on sim + trace only (drops regr view).

    Avoids pulling mismatch cases (e.g. benchmark_set_1 case 3) into their own
    cluster due to long Mismatch[] content in regr.log.
    """

    def __init__(self, char_dim: int = 128):
        self.char_dim = char_dim
        self._char = CharEmbeddingExtractor(char_dim=char_dim)

    def extract(self, cases: List[Dict], parsed_results: List[Dict]) -> np.ndarray:
        full = self._char.extract(cases, parsed_results)
        return full[:, : 2 * self.char_dim]

    def get_feature_dim(self) -> int:
        return self.char_dim * 2


class MismatchAwareCharExtractor(FeatureExtractor):
    """
    Full char embedding but zero regr features when regr.log has Mismatch[].

    Keeps sim/trace identical to char method; only mismatch cases lose regr signal.
    """

    def __init__(self, char_dim: int = 128):
        self.char_dim = char_dim
        self._char = CharEmbeddingExtractor(char_dim=char_dim)
        self._regr_parser = None

    def _get_regr_parser(self):
        if self._regr_parser is None:
            from log_parser import RegrLogParser
            self._regr_parser = RegrLogParser()
        return self._regr_parser

    def extract(self, cases: List[Dict], parsed_results: List[Dict]) -> np.ndarray:
        full = self._char.extract(cases, parsed_results).copy()
        rp = self._get_regr_parser()
        regr_off = 2 * self.char_dim
        for i, case in enumerate(cases):
            regr_parsed = rp.parse(case["Regr Log"])
            if regr_parsed.get("has_mismatch"):
                full[i, regr_off : regr_off + self.char_dim] = 0.0
        return full

    def get_feature_dim(self) -> int:
        return self.char_dim * 3


class WeightedMultiViewExtractor(FeatureExtractor):
    """Multi-view features with error-type-aware weighting"""
    
    def __init__(self, 
                 regr_tfidf_dim: int = 128,
                 sim_ngram_dim: int = 64,
                 trace_instr_pool_size: int = 20,
                 use_weights: bool = True):
        self.regr_tfidf_dim = regr_tfidf_dim
        self.sim_ngram_dim = sim_ngram_dim
        self.trace_instr_pool_size = trace_instr_pool_size
        self.use_weights = use_weights
        self.common_instrs = None
        
        self.regr_tfidf = TfidfVectorizer(
            max_features=regr_tfidf_dim,
            analyzer='char',
            ngram_range=(2, 3),
            lowercase=True,
            norm='l2'
        )
        self.sim_ngram = TfidfVectorizer(
            max_features=sim_ngram_dim,
            analyzer='char',
            ngram_range=(2, 3),
            lowercase=True,
            norm='l2'
        )
    
    def extract(self, cases: List[Dict], parsed_results: List[Dict]) -> np.ndarray:
        """Extract weighted multi-view features"""
        regr_texts = []
        sim_texts = []
        
        for i, case in enumerate(cases):
            parsed = parsed_results[i]
            regr_texts.append(parsed["regr"]["full_text"][:1000])
            sim_texts.append(parsed["sim"]["full_text"][:500])
        
        if regr_texts:
            self.regr_tfidf.fit(regr_texts)
        if sim_texts:
            self.sim_ngram.fit(sim_texts)
        
        # Extract common instructions
        all_instrs = []
        for parsed in parsed_results:
            all_instrs.extend(parsed["trace"]["instr_sequence"])
        
        instr_counts = {}
        for instr in all_instrs:
            instr_counts[instr] = instr_counts.get(instr, 0) + 1
        
        self.common_instrs = sorted(instr_counts.items(), key=lambda x: -x[1])[:self.trace_instr_pool_size]
        self.common_instrs = [i[0] for i in self.common_instrs]
        
        features = []
        for i, case in enumerate(cases):
            parsed = parsed_results[i]
            
            if self.use_weights:
                weights = self._get_weights(parsed["error_type"])
            else:
                weights = {"regr": 1.0/3, "sim": 1.0/3, "trace": 1.0/3}
            
            feature = self._extract_single(parsed, weights)
            features.append(feature)
        
        return np.array(features)
    
    def _get_weights(self, error_type: str) -> Dict[str, float]:
        """Get weights based on error type"""
        if error_type == "UVM_FATAL":
            return {"regr": 0.1, "sim": 0.8, "trace": 0.1}
        elif error_type == "MISMATCH":
            return {"regr": 0.6, "sim": 0.1, "trace": 0.3}
        elif error_type == "TRACE_ABNORMAL":
            return {"regr": 0.1, "sim": 0.2, "trace": 0.7}
        else:
            return {"regr": 0.33, "sim": 0.33, "trace": 0.34}
    
    def _extract_single(self, parsed: Dict, weights: Dict[str, float]) -> np.ndarray:
        """Extract features for single case with weighting"""
        features = []
        
        # Regr features
        regr_text = parsed["regr"]["full_text"][:1000]
        regr_tfidf_vec = self._get_tfidf_vector(self.regr_tfidf, regr_text, self.regr_tfidf_dim)
        regr_tfidf_vec = regr_tfidf_vec * weights["regr"]
        features.append(regr_tfidf_vec)
        
        mismatch_decile = np.zeros(10)
        if parsed["regr"]["first_mismatch_idx"] is not None:
            decile = min(int(parsed["regr"]["first_mismatch_idx"] / 100), 9)
            mismatch_decile[decile] = 1.0
        mismatch_decile = mismatch_decile * weights["regr"]
        features.append(mismatch_decile)
        
        mismatch_count = min(parsed["regr"]["mismatch_count"] / 1000.0, 1.0)
        features.append(np.array([mismatch_count * weights["regr"]]))
        
        # Sim features
        error_type_vec = np.zeros(4)
        error_types = {"INFO": 0, "WARNING": 1, "ERROR": 2, "FATAL": 3}
        if parsed["sim"]["error_type"] in error_types:
            error_type_vec[error_types[parsed["sim"]["error_type"]]] = 1.0
        error_type_vec = error_type_vec * weights["sim"]
        features.append(error_type_vec)
        
        bool_features = np.array([
            float(parsed["sim"]["has_debug_error"]),
            float(parsed["sim"]["has_timeout"]),
            float(parsed["sim"]["has_privilege_error"])
        ])
        bool_features = bool_features * weights["sim"]
        features.append(bool_features)
        
        keyword_count = min(len(parsed["sim"]["keywords"]) / 5.0, 1.0)
        features.append(np.array([keyword_count * weights["sim"]]))
        
        sim_text = parsed["sim"]["full_text"][:500]
        sim_ngram_vec = self._get_tfidf_vector(self.sim_ngram, sim_text, self.sim_ngram_dim)
        sim_ngram_vec = sim_ngram_vec * weights["sim"]
        features.append(sim_ngram_vec)
        
        # Trace features
        instr_vec = np.zeros(self.trace_instr_pool_size)
        if self.common_instrs:
            for i, instr in enumerate(self.common_instrs):
                count = parsed["trace"]["instr_sequence"].count(instr)
                instr_vec[i] = min(count / 5.0, 1.0)
        instr_vec = instr_vec * weights["trace"]
        features.append(instr_vec)
        
        pc_features = np.array([
            float(parsed["trace"]["has_loop"]),
            float(parsed["trace"]["has_pc_stall"]),
            parsed["trace"]["anomaly_score"]
        ])
        pc_features = pc_features * weights["trace"]
        features.append(pc_features)
        
        return np.concatenate(features)
    
    @staticmethod
    def _get_tfidf_vector(vectorizer, text: str, dim: int) -> np.ndarray:
        """Get TF-IDF vector"""
        try:
            vec = vectorizer.transform([text]).toarray()[0]
            if len(vec) < dim:
                vec = np.pad(vec, (0, dim - len(vec)), mode='constant')
            else:
                vec = vec[:dim]
            return vec.astype(np.float32)
        except:
            return np.zeros(dim, dtype=np.float32)
    
    def get_feature_dim(self) -> int:
        return (
            self.regr_tfidf_dim + 10 + 1 +
            4 + 3 + 1 + self.sim_ngram_dim +
            self.trace_instr_pool_size + 3
        )


# ============================================================================
# CLUSTERING STRATEGIES
# ============================================================================

class ClusteringStrategy(ABC):
    """Base class for clustering strategies"""
    
    @abstractmethod
    def cluster(self, 
                features: np.ndarray,
                parsed_results: List[Dict],
                n_clusters: int,
                verbose: bool = False) -> np.ndarray:
        """Cluster features. Returns (n_cases,) label array"""
        pass


class SimpleHierarchicalClustering(ClusteringStrategy):
    """Simple flat hierarchical clustering on all data"""
    
    def __init__(self, linkage: str = "ward", seed: int = 42):
        self.linkage = linkage
        self.seed = seed
    
    def cluster(self, 
                features: np.ndarray,
                parsed_results: List[Dict],
                n_clusters: int,
                verbose: bool = False) -> np.ndarray:
        """Direct hierarchical clustering without grouping"""
        
        if verbose:
            print(f"[*] Simple hierarchical clustering (linkage={self.linkage})")
        
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)
        
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.linkage,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(features_norm)
        
        if verbose:
            print(f"[*] Generated {len(set(labels))} clusters")
        
        return labels


class ErrorTypeHierarchicalClustering(ClusteringStrategy):
    """Hierarchical clustering grouped by error type (original src method)"""
    
    def __init__(self, linkage: str = "ward", seed: int = 42, use_hard_rules: bool = False):
        self.linkage = linkage
        self.seed = seed
        self.use_hard_rules = use_hard_rules
    
    def cluster(self, 
                features: np.ndarray,
                parsed_results: List[Dict],
                n_clusters: int,
                verbose: bool = False) -> np.ndarray:
        """Cluster hierarchically by error type"""
        
        if verbose:
            print(f"[*] Error-type hierarchical clustering")
            if self.use_hard_rules:
                print(f"    - Using hard rules for pre-classification")
        
        from collections import defaultdict
        
        n_cases = len(parsed_results)
        labels = np.zeros(n_cases, dtype=int)
        
        type_groups = defaultdict(list)
        for idx, result in enumerate(parsed_results):
            type_groups[result["error_type"]].append(idx)
        
        if verbose:
            print("[*] Error type distribution:")
            for etype in sorted(type_groups.keys()):
                print(f"    - {etype}: {len(type_groups[etype])} cases")
        
        current_bucket = 0
        scaler = StandardScaler()
        
        for error_type in sorted(type_groups.keys()):
            indices = type_groups[error_type]
            
            if len(indices) == 1:
                labels[indices[0]] = current_bucket
                current_bucket += 1
                continue
            
            group_features = features[indices]
            group_features_norm = scaler.fit_transform(group_features)
            
            type_k = max(1, int(n_clusters * len(indices) / n_cases))
            
            clusterer = AgglomerativeClustering(
                n_clusters=type_k,
                linkage=self.linkage,
                metric='euclidean'
            )
            type_labels = clusterer.fit_predict(group_features_norm)
            
            for local_idx, case_idx in enumerate(indices):
                labels[case_idx] = current_bucket + type_labels[local_idx]
            
            current_bucket += type_k
        
        if verbose:
            print(f"[*] Generated {current_bucket} clusters")
        
        return labels
