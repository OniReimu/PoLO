#!/usr/bin/env python3
"""
Replot time-to-pass curves from existing multi-shard takeover results.

Usage:
    python plots/replot.py --summary multi_shard_takeover_out/multi_shard_takeover_summary_mode2.json [--time-unit-seconds]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to Python path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from plots.plotting import plot_time_to_pass_curves
from config.config import ShardExperimentConfig


def main():
    # Load config defaults
    config = ShardExperimentConfig()
    
    ap = argparse.ArgumentParser(description="Replot time-to-pass curves from existing results")
    ap.add_argument("--summary", type=str, required=True,
                   help="Path to multi_shard_takeover_summary_*.json file")
    ap.add_argument("--time-unit-seconds", action="store_true",
                   help="Use seconds instead of epochs for x-axis")
    ap.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for plots (default: same as summary file directory)")
    ap.add_argument("--font-size", type=int, default=config.plot_font_size,
                   help=f"Font size for labels, legends, and axes (default: {config.plot_font_size} from config)")
    ap.add_argument("--shard-label-font-size", type=int, default=config.plot_shard_label_font_size,
                   help=f"Font size for shard differentiator labels (default: {config.plot_shard_label_font_size} from config)")
    ap.add_argument("--figsize-width", type=float, default=config.plot_figsize_width,
                   help=f"Figure width in inches (default: {config.plot_figsize_width} from config)")
    ap.add_argument("--figsize-height", type=float, default=config.plot_figsize_height,
                   help=f"Figure height in inches (default: {config.plot_figsize_height} from config)")
    
    args = ap.parse_args()
    
    if not os.path.exists(args.summary):
        print(f"[ERROR] Summary file not found: {args.summary}")
        sys.exit(1)
    
    # Load summary data
    with open(args.summary, 'r') as f:
        summary = json.load(f)
    
    # Check if epoch_metrics are available
    if "epoch_metrics" not in summary:
        print("[ERROR] Summary file does not contain epoch_metrics for plotting.")
        print("[INFO] Re-run the attack with --plot-time-curves to generate metrics.")
        sys.exit(1)
    
    epoch_metrics = summary["epoch_metrics"]
    args_data = summary.get("args", {})
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.summary))
    
    # Extract metrics
    all_epochs = epoch_metrics["epochs"]
    all_times_seconds = epoch_metrics.get("times_seconds", [])
    all_owner_det = epoch_metrics["owner_det"]
    all_attacker_det = epoch_metrics["attacker_det"]
    all_acc = epoch_metrics["acc"]
    all_shard_ids = epoch_metrics["shard_ids"]
    
    # Get thresholds from args
    tau_embed = args_data.get("tau_embed", 0.90)
    tau_erase = args_data.get("tau_erase", 0.80)
    acc_min = args_data.get("acc_min", 0.65)
    attack_mode = args_data.get("attack_mode", "mode1")
    
    # Prepare attack metrics
    attack_metrics = {
        'owner_det': all_owner_det,
        'attacker_det': all_attacker_det,
        'acc': all_acc,
    }
    
    # Load honest metrics if available
    honest_metrics = None
    honest_shard_boundaries = None
    owner_checkpoints_path = args_data.get("owner_checkpoints", "")
    if owner_checkpoints_path:
        owner_meta_path = os.path.join(os.path.dirname(owner_checkpoints_path), "owner_training_metadata.json")
        if os.path.exists(owner_meta_path):
            with open(owner_meta_path, 'r') as f:
                owner_meta = json.load(f)
                if 'epoch_metrics' in owner_meta:
                    honest_metrics = owner_meta['epoch_metrics']
                    print(f"[PLOT] Loaded honest metrics from {owner_meta_path}")
        
        # Extract shard boundary epochs from owner checkpoints
        if os.path.exists(owner_checkpoints_path):
            with open(owner_checkpoints_path, 'r') as f:
                owner_ckpts_data = json.load(f)
                honest_shard_boundaries = [ckpt['epoch_global'] for ckpt in owner_ckpts_data]
                print(f"[PLOT] Extracted honest shard boundaries at epochs: {honest_shard_boundaries}")
    
    # Determine time axis
    if args.time_unit_seconds and len(all_times_seconds) > 0:
        time_axis = all_times_seconds
        time_unit = "seconds"
    else:
        time_axis = all_epochs
        time_unit = "epochs"
    
    # Generate output path
    mode_suffix = f"_{attack_mode}"
    base_path = os.path.join(args.output_dir, f"time_to_pass_curves{mode_suffix}")
    
    # Determine mode label
    mode_label = f"{attack_mode}: hash-linked owner WMs" if attack_mode == "mode2" else "mode1: anchor erase only"
    
    # Prepare figsize tuple (now using config defaults)
    figsize_tuple = (args.figsize_width, args.figsize_height)
    
    # Plot using config defaults (can be overridden via command-line args)
    print(f"[PLOT] Generating plots with time_unit={time_unit}...")
    print(f"[PLOT] Using font_size={args.font_size}, shard_label_font_size={args.shard_label_font_size}, figsize={figsize_tuple}")
    time_to_pass = plot_time_to_pass_curves(
        attack_metrics=attack_metrics,
        time_axis=time_axis,
        honest_metrics=honest_metrics,
        output_path=f"{base_path}.png",  # Will be split into _honest.png and _attack.png
        title=f"Multi-Shard Time-to-Pass Curves ({mode_label})\nOwner WM per-shard, Attacker WM per-shard",
        tau_embed=tau_embed,
        tau_erase=tau_erase,
        acc_min=acc_min,
        time_unit=time_unit,
        shard_ids=all_shard_ids,
        honest_shard_boundaries=honest_shard_boundaries,
        font_size=args.font_size,
        shard_label_font_size=args.shard_label_font_size,
        figsize=figsize_tuple,
    )
    
    if time_to_pass is not None:
        print(f"[PLOT] Time-to-pass: {time_to_pass:.2f} {time_unit}")
    else:
        print("[PLOT] Attack did not reach all thresholds within the time limit")
    
    print(f"[PLOT] Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

