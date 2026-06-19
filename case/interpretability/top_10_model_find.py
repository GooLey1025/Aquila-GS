#!/usr/bin/env python3
"""
Find top 10 best trials by val_r from HPO log files.
Usage: python top_10_model_find.py <hpo_log_file> [output_txt]
"""

import os
import re
import sys
import argparse


def extract_trials_from_log(log_path):
    """Extract trial numbers and val_r from a single log file."""
    trials = []
    pattern = re.compile(r"Trial (\d+) completed: val_r=([\d.]+)")
    
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                trial_num = int(match.group(1))
                val_r = float(match.group(2))
                trials.append((trial_num, val_r))
    
    return trials


def sort_trials(trials, descending=True):
    """Sort trials by val_r."""
    return sorted(trials, key=lambda x: x[1], reverse=descending)


def write_results(model_dir, trials, output_path, top_n=10):
    """Write top N trials to output file."""
    top_trials = trials[:top_n]
    
    with open(output_path, 'w') as f:
        for trial_num, val_r in top_trials:
            f.write(f"trial_{trial_num}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Find top 10 best trials by val_r from HPO log files."
    )
    parser.add_argument(
        "log_file",
        help="Path to .hpo.log file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output txt file path (default: <log_name>_top10.txt)",
        default=None
    )
    parser.add_argument(
        "-n", "--top-n",
        type=int,
        default=10,
        help="Number of top trials to output (default: 10)"
    )
    
    args = parser.parse_args()
    log_path = args.log_file
    
    if not os.path.isfile(log_path):
        print(f"Error: File not found: {log_path}")
        sys.exit(1)
    
    log_name = os.path.basename(log_path)
    model_dir = os.path.splitext(log_name)[0]
    
    trials = extract_trials_from_log(log_path)
    if not trials:
        print("Error: No trial results found in log file")
        sys.exit(1)
    
    sorted_trials = sort_trials(trials)
    
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(log_path), f"{model_dir}_top10.txt")
    
    write_results(model_dir, sorted_trials, output_path, args.top_n)
    
    print(f"{output_path}")
    for trial_num, val_r in sorted_trials[:args.top_n]:
        print(f"trial_{trial_num}")


if __name__ == "__main__":
    main()
