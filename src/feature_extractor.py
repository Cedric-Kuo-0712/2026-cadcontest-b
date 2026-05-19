"""
Multi-view feature extraction with weighted priorities by error type
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict


class WeightedMultiViewFeatureExtractor:
    """Extract and weight multi-view features based on error type"""
    
    def __init__(self, 
                 regr_tfidf_dim: int = 128,
                 sim_ngram_dim: int = 64,
                 trace_instr_pool_size: int = 20):
        self.regr_tfidf_dim = regr_tfidf_dim
        self.sim_ngram_dim = sim_ngram_dim
        self.trace_instr_pool_size = trace_instr_pool_size
        
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
        
        self.common_instrs = None
    
    def fit_tfidf(self, regr_texts: List[str], sim_texts: List[str]):
        """Fit TF-IDF on all texts"""
        if regr_texts:
            self.regr_tfidf.fit(regr_texts)
        if sim_texts:
            self.sim_ngram.fit(sim_texts)
    
    def extract_features_batch(self, 
                               cases: List[Dict],
                               parsed_results: List[Dict]) -> np.ndarray:
        """Batch feature extraction with error_type awareness"""
        
        regr_texts = []
        sim_texts = []
        
        for i, case in enumerate(cases):
            parsed = parsed_results[i]
            regr_texts.append(parsed["regr"]["full_text"][:1000])
            sim_texts.append(parsed["sim"]["full_text"][:500])
        
        self.fit_tfidf(regr_texts, sim_texts)
        
        all_instrs = []
        for parsed in parsed_results:
            all_instrs.extend(parsed["trace"]["instr_sequence"])
        
        instr_counts = {}
        for instr in all_instrs:
            instr_counts[instr] = instr_counts.get(instr, 0) + 1
        
        self.common_instrs = sorted(instr_counts.items(), key=lambda x: -x[1])[:self.trace_instr_pool_size]
        self.common_instrs = [i[0] for i in self.common_instrs]
        
        features_list = []
        for i, case in enumerate(cases):
            parsed = parsed_results[i]
            feature = self._extract_single_features(parsed)
            features_list.append(feature)
        
        return np.array(features_list)
    
    def _extract_single_features(self, parsed: Dict) -> np.ndarray:
        """Extract single case features with type-aware weighting"""
        
        error_type = parsed["error_type"]
        
        if error_type == "UVM_FATAL":
            weights = {"regr": 0.1, "sim": 0.8, "trace": 0.1}
        elif error_type == "MISMATCH":
            weights = {"regr": 0.6, "sim": 0.1, "trace": 0.3}
        elif error_type == "TRACE_ABNORMAL":
            weights = {"regr": 0.1, "sim": 0.2, "trace": 0.7}
        else:
            weights = {"regr": 0.3, "sim": 0.3, "trace": 0.4}
        
        features = []
        
        regr_parsed = parsed["regr"]
        regr_text = regr_parsed["full_text"][:1000]
        regr_tfidf_vec = self._get_tfidf_vector(self.regr_tfidf, regr_text, self.regr_tfidf_dim)
        regr_tfidf_vec = regr_tfidf_vec * weights["regr"]
        features.append(regr_tfidf_vec)
        
        mismatch_decile = np.zeros(10)
        if regr_parsed["first_mismatch_idx"] is not None:
            decile = min(int(regr_parsed["first_mismatch_idx"] / 100), 9)
            mismatch_decile[decile] = 1.0
        mismatch_decile = mismatch_decile * weights["regr"]
        features.append(mismatch_decile)
        
        mismatch_count = min(regr_parsed["mismatch_count"] / 1000.0, 1.0)
        features.append(np.array([mismatch_count * weights["regr"]]))
        
        sim_parsed = parsed["sim"]
        error_type_vec = np.zeros(4)
        error_types = {"INFO": 0, "WARNING": 1, "ERROR": 2, "FATAL": 3}
        if sim_parsed["error_type"] in error_types:
            error_type_vec[error_types[sim_parsed["error_type"]]] = 1.0
        error_type_vec = error_type_vec * weights["sim"]
        features.append(error_type_vec)
        
        bool_features = np.array([
            float(sim_parsed["has_debug_error"]),
            float(sim_parsed["has_timeout"]),
            float(sim_parsed["has_privilege_error"])
        ])
        bool_features = bool_features * weights["sim"]
        features.append(bool_features)
        
        keyword_count = min(len(sim_parsed["keywords"]) / 5.0, 1.0)
        features.append(np.array([keyword_count * weights["sim"]]))
        
        sim_text = sim_parsed["full_text"][:500]
        sim_ngram_vec = self._get_tfidf_vector(self.sim_ngram, sim_text, self.sim_ngram_dim)
        sim_ngram_vec = sim_ngram_vec * weights["sim"]
        features.append(sim_ngram_vec)
        
        trace_parsed = parsed["trace"]
        instr_vec = np.zeros(self.trace_instr_pool_size)
        if self.common_instrs:
            for i, instr in enumerate(self.common_instrs):
                count = trace_parsed["instr_sequence"].count(instr)
                instr_vec[i] = min(count / 5.0, 1.0)
        instr_vec = instr_vec * weights["trace"]
        features.append(instr_vec)
        
        pc_features = np.array([
            float(trace_parsed["has_loop"]),
            float(trace_parsed["has_pc_stall"]),
            trace_parsed["anomaly_score"]
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
    
    def get_feature_dimension(self) -> int:
        """Return total feature dimension"""
        return (
            self.regr_tfidf_dim + 10 + 1 +
            4 + 3 + 1 + self.sim_ngram_dim +
            self.trace_instr_pool_size + 3
        )
