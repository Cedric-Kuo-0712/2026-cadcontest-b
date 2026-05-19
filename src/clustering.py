"""
Hierarchical clustering by error type with hard rules
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
from collections import defaultdict


class HardRuleClassifier:
    """Hard rule-based pre-classification"""
    
    def classify_by_rules(self, parsed_results: List[Dict]) -> Dict[int, int]:
        """Pre-classify cases using hard rules"""
        assignments = {}
        error_signatures = {}
        next_bucket = 0
        
        for idx, result in enumerate(parsed_results):
            sim_parsed = result["sim"]
            error_type = result["error_type"]
            
            if error_type == "UVM_FATAL":
                sig = sim_parsed["error_signature"]
                if sig not in error_signatures:
                    error_signatures[sig] = next_bucket
                    next_bucket += 1
                assignments[idx] = error_signatures[sig]
            
            elif sim_parsed["has_debug_error"]:
                if "debug_bucket" not in error_signatures:
                    error_signatures["debug_bucket"] = next_bucket
                    next_bucket += 1
                assignments[idx] = error_signatures["debug_bucket"]
            
            else:
                assignments[idx] = -1
        
        return assignments


class HierarchicalClustering:
    """Hierarchical clustering by error type"""
    
    def __init__(self, linkage: str = "ward", seed: int = 42):
        self.linkage = linkage
        self.seed = seed
        self.scaler = StandardScaler()
    
    def cluster_hierarchical(self,
                            features: np.ndarray,
                            parsed_results: List[Dict],
                            n_clusters: int,
                            verbose: bool = False) -> np.ndarray:
        """Cluster hierarchically by error type"""
        
        n_cases = len(parsed_results)
        labels = np.zeros(n_cases, dtype=int)
        
        type_groups = defaultdict(list)
        for idx, result in enumerate(parsed_results):
            type_groups[result["error_type"]].append(idx)
        
        if verbose:
            print("[*] Clustering by error type:")
        
        current_bucket = 0
        hard_classifier = HardRuleClassifier()
        
        for error_type in sorted(type_groups.keys()):
            indices = type_groups[error_type]
            
            if verbose:
                print(f"    - {error_type}: {len(indices)} cases")
            
            if len(indices) == 1:
                labels[indices[0]] = current_bucket
                current_bucket += 1
                continue
            
            group_features = features[indices]
            group_parsed = [parsed_results[i] for i in indices]
            
            hard_rules = hard_classifier.classify_by_rules(group_parsed)
            group_features_norm = self.scaler.fit_transform(group_features)
            
            type_k = max(1, int(n_clusters * len(indices) / n_cases))
            
            clusterer = AgglomerativeClustering(
                n_clusters=type_k,
                linkage=self.linkage,
                metric='euclidean'
            )
            type_labels = clusterer.fit_predict(group_features_norm)
            
            for local_idx, case_idx in enumerate(indices):
                if case_idx in hard_rules and hard_rules[case_idx] >= 0:
                    labels[case_idx] = current_bucket + hard_rules[case_idx]
                else:
                    labels[case_idx] = current_bucket + type_labels[local_idx]
            
            current_bucket += max(type_labels) + 1
        
        return labels
