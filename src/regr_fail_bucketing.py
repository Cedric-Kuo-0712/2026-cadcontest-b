#!/usr/bin/env python3

"""
Regression Failure Bucketing - split-track implementation
"""

import argparse
import os
import random
import sys
import numpy as np
import pandas as pd

from log_parser import (
    RegrLogParser, SimLogParser, TraceLogParser, 
    classify_error_type
)
from split_track_bucketing import SplitTrackBucketer


def set_seed(seed: int = 42):
    """Set all random seeds"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Regression Failure Bucketing - split-track approach"
    )
    parser.add_argument("--input", required=True, help="Input CSV with log file paths")
    parser.add_argument("--output", required=True, help="Output CSV with bucket assignments")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters (soft hint)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    set_seed(42)
    
    if args.verbose:
        print("[*] Starting Regression Failure Bucketing (split-track)...")
    
    # Stage 0: Load input CSV
    try:
        df_input = pd.read_csv(args.input)
        if args.verbose:
            print(f"[*] Loaded {len(df_input)} cases")
    except Exception as e:
        print(f"[!] Failed to load input CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Prepare cases list
    cases = []
    base_dir = os.path.dirname(args.input)
    for _, row in df_input.iterrows():
        case = {
            "Case": row["Case"],
            "Regr Log": os.path.join(base_dir, row["Regr Log"]),
            "Sim Log": os.path.join(base_dir, row["Sim Log"]),
            "Trace Log": os.path.join(base_dir, row["Trace Log"])
        }
        cases.append(case)
    
    # Stage 1: Log parsing and error classification
    if args.verbose:
        print("[*] Stage 1: Log parsing and error type classification...")
    
    regr_parser = RegrLogParser()
    sim_parser = SimLogParser()
    trace_parser = TraceLogParser()
    
    parsed_results = []
    error_type_dist = {}
    
    try:
        for case in cases:
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
        
        if args.verbose:
            print("[*] Error type distribution:")
            for etype, count in sorted(error_type_dist.items()):
                print(f"    - {etype}: {count}")
    
    except Exception as e:
        print(f"[!] Parsing failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Stage 2: Split-track clustering
    if args.verbose:
        print("[*] Stage 2: Split-track clustering...")
    
    try:
        labels = SplitTrackBucketer(seed=42).cluster(parsed_results, args.k, args.verbose)
        
        if args.verbose:
            unique_labels = len(set(labels))
            print(f"[*] Clustering complete: {unique_labels} clusters")
    except Exception as e:
        print(f"[!] Clustering failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Stage 4: Output results
    if args.verbose:
        print("[*] Stage 4: Writing output...")
    
    df_output = pd.DataFrame({
        "Case": [case["Case"] for case in cases],
        "bucket": labels
    })
    
    try:
        df_output.to_csv(args.output, index=False)
        if args.verbose:
            print(f"[+] Results written to: {args.output}")
    except Exception as e:
        print(f"[!] Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print("[+] Done!")


if __name__ == "__main__":
    main()
