#!/usr/bin/env python3
"""
Aggregate multiple attack results and generate cost-success frontier plot.

Usage:
    python plot_aggregate.py --summary_files summary1.json summary2.json ... --output plot.png
"""

import argparse
import glob
import os

from plots.plotting import aggregate_results_for_plotting, plot_cost_success_frontier


def main():
    ap = argparse.ArgumentParser(description="Aggregate attack results and plot cost-success frontier")
    ap.add_argument("--summary_files", type=str, nargs="+", required=True,
                    help="Paths to summary JSON files (or directory with summary files)")
    ap.add_argument("--output", type=str, default="cost_success_frontier_aggregate.png",
                    help="Output path for the plot")
    ap.add_argument("--title", type=str, default="Cost-Success Frontier",
                    help="Plot title")
    ap.add_argument("--plot_margin", action="store_true",
                    help="Use margin scores instead of binary")
    ap.add_argument("--honest_shard_time", type=float, default=None,
                    help="Honest shard training time (overrides values from summaries)")
    ap.add_argument("--acc_min", type=float, default=None,
                    help="Accuracy minimum threshold (overrides from summaries)")
    ap.add_argument("--tau_embed", type=float, default=None,
                    help="Tau embed threshold (overrides from summaries)")
    ap.add_argument("--tau_erase", type=float, default=None,
                    help="Tau erase threshold (overrides from summaries)")
    
    args = ap.parse_args()
    
    # Expand summary files (handle directories and glob patterns)
    summary_files = []
    for pattern in args.summary_files:
        if os.path.isdir(pattern):
            # Find all summary files in directory
            summary_files.extend(glob.glob(os.path.join(pattern, "*_summary.json")))
            summary_files.extend(glob.glob(os.path.join(pattern, "*_takeover_summary.json")))
        elif "*" in pattern:
            summary_files.extend(glob.glob(pattern))
        else:
            summary_files.append(pattern)
    
    summary_files = [f for f in summary_files if os.path.exists(f)]
    
    if not summary_files:
        print("[ERROR] No summary files found!")
        return
    
    print(f"[INFO] Found {len(summary_files)} summary files")
    
    # Aggregate results
    cost_ratios, success_scores = aggregate_results_for_plotting(
        summary_files=summary_files,
        honest_shard_time=args.honest_shard_time,
        use_margin=args.plot_margin,
        acc_min=args.acc_min,
        tau_embed=args.tau_embed,
        tau_erase=args.tau_erase,
    )
    
    if not cost_ratios:
        print("[ERROR] No valid results found in summary files!")
        return
    
    print(f"[INFO] Aggregated {len(cost_ratios)} attack results")
    print(f"[INFO] Cost ratios range: {min(cost_ratios):.2f} to {max(cost_ratios):.2f}")
    if args.plot_margin:
        print(f"[INFO] Success margins range: {min(success_scores):.2f} to {max(success_scores):.2f}")
    else:
        successes = sum(1 for s in success_scores if s > 0.5)
        print(f"[INFO] Successful attacks: {successes}/{len(success_scores)}")
    
    # Generate plot
    plot_cost_success_frontier(
        cost_ratios=cost_ratios,
        success_scores=success_scores,
        honest_cost_ratio=1.0,
        honest_success=1.0,  # Will be adjusted for margin plots in plotting function
        output_path=args.output,
        title=args.title,
        use_margin=args.plot_margin,
        acc_min=args.acc_min,
        tau_embed=args.tau_embed,
        tau_erase=args.tau_erase,
    )


if __name__ == "__main__":
    main()

