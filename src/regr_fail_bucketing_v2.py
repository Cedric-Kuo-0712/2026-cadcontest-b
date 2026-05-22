#!/usr/bin/env python3

"""
Regression Failure Bucketing - Flexible CLI with Strategy Pattern
Supports multiple feature extraction and clustering methods via --method and --clustering flags
"""

import argparse
import os
import random
import sys
import numpy as np
import pandas as pd

# Import parsers
from log_parser import (
    RegrLogParser, SimLogParser, TraceLogParser, 
    classify_error_type
)

# Import strategies
from strategies import (
    CharEmbeddingExtractor,
    SimTraceExtractor,
    MismatchAwareCharExtractor,
    WeightedMultiViewExtractor,
    SimpleHierarchicalClustering,
    ErrorTypeHierarchicalClustering,
)
from analysis_features import AnalysisGuidedExtractor


def set_seed(seed: int = 42):
    """Set all random seeds"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def parse_logs(cases: list, verbose: bool = False) -> list:
    """Parse all log files"""
    regr_parser = RegrLogParser()
    sim_parser = SimLogParser()
    trace_parser = TraceLogParser()
    
    parsed_results = []
    error_type_dist = {}
    
    try:
        for i, case in enumerate(cases):
            if verbose and (i + 1) % max(1, len(cases) // 5) == 0:
                print(f"  [{i+1}/{len(cases)}] cases parsed...")
            
            regr_parsed = regr_parser.parse(case["Regr Log"])
            sim_parsed = sim_parser.parse(case["Sim Log"])
            trace_parsed = trace_parser.parse(case["Trace Log"])
            
            error_type = classify_error_type(regr_parsed, sim_parsed, trace_parsed)
            error_type_dist[error_type] = error_type_dist.get(error_type, 0) + 1
            
            parsed_results.append({
                "regr": regr_parsed,
                "sim": sim_parsed,
                "trace": trace_parsed,
                "error_type": error_type
            })
        
        if verbose:
            print("[*] Error type distribution:")
            for etype, count in sorted(error_type_dist.items()):
                print(f"    - {etype}: {count}")
    
    except Exception as e:
        print(f"[!] Parsing failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    return parsed_results


def main():
    parser = argparse.ArgumentParser(
        description="Regression Failure Bucketing - Flexible Method Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use char embedding with simple clustering
  python regr_fail_bucketing_v2.py --input input.csv --output output.csv --k 2 \\
    --method char --clustering simple

  # Use weighted multi-view with error-type grouping
  python regr_fail_bucketing_v2.py --input input.csv --output output.csv --k 2 \\
    --method weighted --clustering errortype

  # Use char embedding with complete linkage
  python regr_fail_bucketing_v2.py --input input.csv --output output.csv --k 2 \\
    --method char --clustering simple --linkage complete

Available methods: char, sim_trace, mismatch_aware, weighted, analysis
Available clustering: simple, errortype
        """
    )
    parser.add_argument("--input", required=True, help="Input CSV with log file paths")
    parser.add_argument("--output", required=True, help="Output CSV with bucket assignments")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    
    # Feature extraction method
    parser.add_argument("--method", 
                       choices=["char", "sim_trace", "mismatch_aware", "weighted", "analysis"],
                       default="char",
                       help="Feature extraction method (default: char)")
    parser.add_argument(
        "--selected-features",
        default="analysis_outputs/pooled/selected_features.json",
        help="JSON feature list for --method analysis",
    )
    
    # Clustering method
    parser.add_argument("--clustering",
                       choices=["simple", "errortype"],
                       default="simple",
                       help="Clustering strategy (default: simple)")
    
    # Clustering parameters
    parser.add_argument("--linkage",
                       choices=["ward", "complete", "average", "single"],
                       default="ward",
                       help="Linkage criterion for hierarchical clustering (default: ward)")
    
    # Feature extraction parameters (for weighted method)
    parser.add_argument("--use-weights",
                       action="store_true",
                       default=True,
                       help="Use error-type weights (for --method weighted)")
    parser.add_argument("--no-weights",
                       action="store_false",
                       dest="use_weights",
                       help="Don't use error-type weights (for --method weighted)")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    set_seed(42)
    
    if args.verbose:
        print("[*] Regression Failure Bucketing (Flexible)")
        print(f"[*] Configuration:")
        print(f"    - Feature method: {args.method}")
        print(f"    - Clustering strategy: {args.clustering}")
        print(f"    - Linkage: {args.linkage}")
        if args.method == "weighted":
            print(f"    - Use weights: {args.use_weights}")
    
    # Stage 0: Load input CSV
    try:
        df_input = pd.read_csv(args.input)
        if args.verbose:
            print(f"[*] Loaded {len(df_input)} cases")
    except Exception as e:
        print(f"[!] Failed to load input CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Prepare cases with full paths
    base_dir = os.path.dirname(os.path.abspath(args.input))
    cases = []
    for _, row in df_input.iterrows():
        case = {
            "Case": row["Case"],
            "Regr Log": os.path.join(base_dir, row["Regr Log"]),
            "Sim Log": os.path.join(base_dir, row["Sim Log"]),
            "Trace Log": os.path.join(base_dir, row["Trace Log"])
        }
        cases.append(case)
    
    # Stage 1: Log parsing (skip for char / analysis)
    skip_parse = args.method in ("char", "sim_trace", "mismatch_aware", "analysis")
    if args.verbose:
        print(f"[*] Stage 1: {'Skipping log parsing' if skip_parse else 'Parsing logs'}...")
    
    if skip_parse:
        parsed_results = [{"error_type": "UNKNOWN"} for _ in cases]
    else:
        parsed_results = parse_logs(cases, verbose=args.verbose)
    
    # Stage 2: Feature extraction
    if args.verbose:
        print(f"[*] Stage 2: Extracting features ({args.method})...")
    
    if args.method == "char":
        feature_extractor = CharEmbeddingExtractor(char_dim=128)
    elif args.method == "sim_trace":
        feature_extractor = SimTraceExtractor(char_dim=128)
    elif args.method == "mismatch_aware":
        feature_extractor = MismatchAwareCharExtractor(char_dim=128)
    elif args.method == "weighted":
        feature_extractor = WeightedMultiViewExtractor(
            regr_tfidf_dim=128,
            sim_ngram_dim=64,
            trace_instr_pool_size=20,
            use_weights=args.use_weights
        )
    elif args.method == "analysis":
        sel_path = args.selected_features
        if not os.path.isabs(sel_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sel_path = os.path.join(repo_root, sel_path)
        feature_extractor = AnalysisGuidedExtractor(selected_features_path=sel_path)
    else:
        print(f"[!] Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)
    
    try:
        features = feature_extractor.extract(cases, parsed_results)
        feature_dim = feature_extractor.get_feature_dim()
        if args.verbose:
            print(f"[*] Feature dimension: {feature_dim}")
            print(f"[*] Feature matrix shape: {features.shape}")
    except Exception as e:
        print(f"[!] Feature extraction failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Stage 3: Clustering
    if args.verbose:
        print(f"[*] Stage 3: Clustering ({args.clustering})...")
    
    if args.clustering == "simple":
        clusterer = SimpleHierarchicalClustering(linkage=args.linkage, seed=42)
    elif args.clustering == "errortype":
        clusterer = ErrorTypeHierarchicalClustering(linkage=args.linkage, seed=42)
    else:
        print(f"[!] Unknown clustering strategy: {args.clustering}", file=sys.stderr)
        sys.exit(1)
    
    try:
        labels = clusterer.cluster(
            features,
            parsed_results,
            n_clusters=args.k,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"[!] Clustering failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Stage 4: Output
    if args.verbose:
        print("[*] Stage 4: Writing output...")
    
    try:
        df_output = pd.DataFrame({
            "Case": df_input["Case"].values,
            "bucket": labels
        })
        df_output.to_csv(args.output, index=False)
        
        if args.verbose:
            unique_labels = len(set(labels))
            print(f"[*] Success! Output written to {args.output}")
            print(f"[*] Generated {unique_labels} clusters")
    
    except Exception as e:
        print(f"[!] Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
