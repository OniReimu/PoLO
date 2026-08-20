"""Plotting utilities for cost-success frontier analysis."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config.config import ShardExperimentConfig


def calculate_success_margin(
    acc: float,
    attacker_det: float,
    owner_det: float,
    acc_min: float,
    tau_embed: float,
    tau_erase: float,
) -> float:
    """
    Calculate success margin:
    min(Acc - Acc_min, η_att - η_min, η_max - η_owner)
    
    Positive = passes, negative = fails.
    """
    margin_acc = acc - acc_min
    margin_att = attacker_det - tau_embed
    margin_owner = tau_erase - owner_det  # owner_det must be <= tau_erase
    return min(margin_acc, margin_att, margin_owner)


def calculate_success_binary(
    acc: float,
    attacker_det: float,
    owner_det: float,
    acc_min: float,
    tau_embed: float,
    tau_erase: float,
) -> int:
    """Calculate binary success: 1 if all conditions pass, else 0."""
    if acc >= acc_min and attacker_det >= tau_embed and owner_det <= tau_erase:
        return 1
    return 0


def plot_cost_success_frontier(
    cost_ratios: List[float],
    success_scores: List[float],
    honest_cost_ratio: float = 1.0,
    honest_success: float = 1.0,
    output_path: str = "cost_success_frontier.png",
    title: str = "Cost-Success Frontier",
    use_margin: bool = False,
    acc_min: Optional[float] = None,
    tau_embed: Optional[float] = None,
    tau_erase: Optional[float] = None,
):
    """
    Plot cost-success frontier.
    
    Args:
        cost_ratios: List of cost ratios (attacker_time / honest_time)
        success_scores: List of success scores (binary or margin)
        honest_cost_ratio: Cost ratio for honest training (default 1.0)
        honest_success: Success score for honest training (default 1.0)
        output_path: Path to save the plot
        title: Plot title
        use_margin: If True, use margin scores; if False, use binary
        acc_min, tau_embed, tau_erase: Thresholds for reference lines (if use_margin)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot attacker points
    if use_margin:
        # Margin plot: color by success/failure
        colors = ['green' if s > 0 else 'red' for s in success_scores]
        ax.scatter(cost_ratios, success_scores, c=colors, alpha=0.6, s=50, label='Attacker attempts')
        
        # Add reference line at y=0
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Success threshold')
        
        # Add threshold reference lines if provided
        if acc_min is not None:
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
    else:
        # Binary plot: separate success and failure
        success_ratios = [cr for cr, s in zip(cost_ratios, success_scores) if s > 0.5]
        failure_ratios = [cr for cr, s in zip(cost_ratios, success_scores) if s <= 0.5]
        success_scores_s = [s for s in success_scores if s > 0.5]
        failure_scores_s = [s for s in success_scores if s <= 0.5]
        
        if failure_ratios:
            ax.scatter(failure_ratios, failure_scores_s, c='red', alpha=0.6, s=50, 
                      marker='x', label='Failed attempts')
        if success_ratios:
            ax.scatter(success_ratios, success_scores_s, c='green', alpha=0.6, s=50, 
                      marker='o', label='Successful attempts')
    
    # Plot honest training point
    # For margin plots, honest should have positive margin (e.g., all conditions comfortably met)
    if use_margin:
        honest_success_plot = max(0.1, max(success_scores) * 1.1) if success_scores else 0.5
    else:
        honest_success_plot = honest_success
    
    ax.scatter([honest_cost_ratio], [honest_success_plot], c='blue', s=200, 
              marker='*', label='Honest training', zorder=5, edgecolors='black', linewidths=1.5)
    
    # Add vertical line at cost ratio = 1
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
              label='Equal cost line')
    
    ax.set_xlabel('Cost Ratio (Attacker Time / Honest Shard Time)\n[lower is better: <1.0 = cheaper than honest, >1.0 = more expensive]', fontsize=12)
    if use_margin:
        ax.set_ylabel('Success Margin [higher is better: >0 = success, <0 = failure]', fontsize=12)
        ax.set_title(f'{title} (Margin)', fontsize=14, fontweight='bold')
    else:
        ax.set_ylabel('Verification Success (Binary)\n[higher is better: 1 = pass, 0 = fail]', fontsize=12)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Fail', 'Pass'])
        ax.set_title(f'{title} (Binary)', fontsize=14, fontweight='bold')
    
    # Add text annotation explaining the plot
    ax.text(0.02, 0.98, 
            'Ideal: Low cost ratio (<1.0) AND High success (1.0 or >0)\n'
            'Honest training: (1.0, 1.0) - baseline reference',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved cost-success frontier to {output_path}")
    plt.close()


def load_attack_results_from_summary(summary_path: str) -> Dict:
    """Load attack results from summary JSON file."""
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_time_to_pass_curves(
    attack_metrics: Dict[str, List[float]],
    time_axis: List[float],
    honest_metrics: Optional[Dict[str, List[float]]] = None,
    output_path: str = "time_to_pass_curves.png",
    title: str = "Time-to-Pass Curves",
    tau_embed: float = 0.90,
    tau_erase: float = 0.20,
    acc_min: float = 0.65,
    time_unit: str = "epochs",
    shard_ids: Optional[List[int]] = None,
    honest_shard_boundaries: Optional[List[int]] = None,
    font_size: Optional[int] = None,
    shard_label_font_size: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Plot time-to-pass curves showing watermark metrics and accuracy over time.
    
    Args:
        honest_metrics: Dict with keys ['owner_det', 'acc'] and lists of values over time
        attack_metrics: Dict with keys ['owner_det', 'attacker_det', 'acc'] and lists of values over time
        time_axis: List of time points (epochs or seconds)
        output_path: Path to save the plot (base path, will generate two files)
        title: Plot title (not used, kept for compatibility)
        tau_embed: Attacker watermark detection threshold
        tau_erase: Owner watermark erasure threshold
        acc_min: Minimum accuracy threshold
        time_unit: Unit for time axis ("epochs" or "seconds")
        font_size: Font size for labels, legends, and axes (defaults to config)
        shard_label_font_size: Font size for shard differentiator labels (defaults to config)
    """
    # Get font sizes and figsize from config if not provided
    config = ShardExperimentConfig()
    if font_size is None:
        font_size = config.plot_font_size
    if shard_label_font_size is None:
        shard_label_font_size = config.plot_shard_label_font_size
    if figsize is None:
        figsize = (config.plot_figsize_width, config.plot_figsize_height)
    
    # Generate separate output paths for honest and attack plots
    base_path = output_path.replace('.png', '')
    honest_output_path = f"{base_path}_honest.png"
    attack_output_path = f"{base_path}_attack.png"
    
    # Create separate figures for honest and attack plots
    fig_honest = plt.figure(figsize=figsize)
    ax_honest = fig_honest.add_subplot(111)
    
    # Honest shard plot
    if honest_metrics and len(honest_metrics.get('owner_det', [])) > 0:
        # Create time axis for honest metrics based on their length
        honest_time_axis = list(range(1, len(honest_metrics['owner_det']) + 1))
        ax_honest.plot(honest_time_axis, honest_metrics['owner_det'], 'b-', label='Owner WM', linewidth=2)
        ax_honest.plot(honest_time_axis, honest_metrics['acc'], 'g-', label='Task Accuracy', linewidth=2)
        
        # Add vertical dotted lines at shard boundaries if provided
        if honest_shard_boundaries:
            for idx, boundary_epoch in enumerate(honest_shard_boundaries):
                if boundary_epoch <= len(honest_time_axis):
                    ax_honest.axvline(x=boundary_epoch, color='gray', linestyle=':', 
                                    linewidth=2, alpha=0.6)
                    # Label shard transition (boundary_epoch is when shard idx ends, next shard starts)
                    shard_num = idx
                    next_shard = shard_num + 1
                    ax_honest.text(boundary_epoch, 1.0, f'Shard {shard_num}→{next_shard}', 
                                 rotation=90, verticalalignment='top', fontsize=shard_label_font_size, 
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax_honest.axhline(y=tau_embed, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'τ_embed = {tau_embed}')
        ax_honest.axhline(y=acc_min, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Acc_min = {acc_min}')
    else:
        # If no honest metrics, show just the thresholds
        ax_honest.text(0.5, 0.5, 'No honest training data', 
                      ha='center', va='center', transform=ax_honest.transAxes, fontsize=font_size)
        ax_honest.axhline(y=tau_embed, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'τ_embed = {tau_embed}')
        ax_honest.axhline(y=acc_min, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Acc_min = {acc_min}')
    
    ax_honest.set_xlabel(f'Time ({time_unit})', fontsize=font_size)
    ax_honest.set_ylabel('Metric Value', fontsize=font_size)
    ax_honest.tick_params(axis='both', labelsize=font_size)
    ax_honest.grid(True, alpha=0.3)
    ax_honest.legend(loc='best', fontsize=font_size)
    ax_honest.set_ylim(0, 1.05)
    
    # Save honest plot
    plt.tight_layout()
    fig_honest.savefig(honest_output_path, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved honest training plot to {honest_output_path}")
    plt.close(fig_honest)
    
    # Create separate figure for attack plot
    fig_attack = plt.figure(figsize=figsize)
    ax_attack = fig_attack.add_subplot(111)
    
    # Attack shard plot
    # If shard_ids are provided, plot with shard boundaries and color coding
    if shard_ids and len(shard_ids) == len(time_axis):
        # Plot each shard segment separately with different colors
        current_shard = shard_ids[0]
        start_idx = 0
        
        # Plot attacker watermark and accuracy once (they're cumulative across shards)
        ax_attack.plot(time_axis, attack_metrics['attacker_det'], 'b-', linewidth=2, 
                      label='Attacker WM', alpha=0.8)
        ax_attack.plot(time_axis, attack_metrics['acc'], 'g-', linewidth=2, 
                      label='Task Accuracy', alpha=0.8)
        
        # Plot owner watermark per shard (each shard has different owner watermark)
        for i in range(len(shard_ids)):
            if i == len(shard_ids) - 1 or shard_ids[i+1] != current_shard:
                # End of current shard segment
                end_idx = i + 1
                segment_time = time_axis[start_idx:end_idx]
                segment_owner = attack_metrics['owner_det'][start_idx:end_idx]
                
                # Use different colors for different shards
                color_idx = current_shard % 5
                colors = ['red', 'orange', 'purple', 'brown', 'pink']
                color = colors[color_idx]
                
                ax_attack.plot(segment_time, segment_owner, color=color, linestyle='-', 
                             linewidth=2.5, alpha=0.8, 
                             label=f'Owner WM (shard {current_shard})')
                
                # Draw vertical line at shard boundary (before next shard starts)
                if i < len(shard_ids) - 1:
                    ax_attack.axvline(x=time_axis[end_idx], color='gray', linestyle=':', 
                                    linewidth=2, alpha=0.6)
                    ax_attack.text(time_axis[end_idx], 1.0, f'Shard {current_shard}→{shard_ids[i+1]}', 
                                 rotation=90, verticalalignment='top', fontsize=shard_label_font_size, 
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                if i < len(shard_ids) - 1:
                    start_idx = end_idx
                    current_shard = shard_ids[i+1]
    else:
        # No shard info - plot normally
        ax_attack.plot(time_axis, attack_metrics['owner_det'], 'r-', label='Owner watermark (η_owner)', linewidth=2)
        ax_attack.plot(time_axis, attack_metrics['attacker_det'], 'b-', label='Attacker watermark (η_att)', linewidth=2)
        ax_attack.plot(time_axis, attack_metrics['acc'], 'g-', label='Accuracy', linewidth=2)
    
    # Threshold lines
    ax_attack.axhline(y=tau_erase, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'τ_erase = {tau_erase}')
    ax_attack.axhline(y=tau_embed, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'τ_embed = {tau_embed}')
    ax_attack.axhline(y=acc_min, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Acc_min = {acc_min}')
    
    # Find time-to-pass (first time all conditions are met)
    time_to_pass = None
    for i, t in enumerate(time_axis):
        if (attack_metrics['attacker_det'][i] >= tau_embed and 
            attack_metrics['owner_det'][i] <= tau_erase and 
            attack_metrics['acc'][i] >= acc_min):
            time_to_pass = t
            # Draw the line but don't add it to the legend
            ax_attack.axvline(x=time_to_pass, color='orange', linestyle=':', linewidth=2, alpha=0.8)
            break
    
    ax_attack.set_xlabel(f'Time ({time_unit})', fontsize=font_size)
    ax_attack.set_ylabel('Metric Value', fontsize=font_size)
    ax_attack.tick_params(axis='both', labelsize=font_size)
    ax_attack.grid(True, alpha=0.3)
    ax_attack.legend(loc='best', fontsize=font_size, ncol=2)
    ax_attack.set_ylim(0, 1.05)
    
    # Save attack plot
    plt.tight_layout()
    fig_attack.savefig(attack_output_path, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved attack plot to {attack_output_path}")
    if time_to_pass is not None:
        print(f"[PLOT] Time-to-pass: {time_to_pass:.2f} {time_unit}")
    else:
        print(f"[PLOT] Attack did not reach all thresholds within the time limit")
    plt.close(fig_attack)
    
    return time_to_pass


def aggregate_results_for_plotting(
    summary_files: List[str],
    honest_shard_time: Optional[float] = None,
    use_margin: bool = False,
    acc_min: Optional[float] = None,
    tau_embed: Optional[float] = None,
    tau_erase: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    """
    Aggregate multiple attack results for plotting.
    
    Args:
        summary_files: List of paths to summary JSON files
        honest_shard_time: Honest shard training time (if None, will try to extract from first summary)
        use_margin: If True, calculate margin scores; if False, use binary
        acc_min, tau_embed, tau_erase: Thresholds for success calculation
    
    Returns:
        Tuple of (cost_ratios, success_scores)
    """
    cost_ratios = []
    success_scores = []
    
    for summary_file in summary_files:
        if not os.path.exists(summary_file):
            print(f"[WARNING] Summary file not found: {summary_file}")
            continue
        
        data = load_attack_results_from_summary(summary_file)
        args = data.get('args', {})
        
        # Extract thresholds from args if not provided
        if acc_min is None:
            acc_min = args.get('acc_min', 0.65)
        if tau_embed is None:
            tau_embed = args.get('tau_embed', 0.90)
        if tau_erase is None:
            tau_erase = args.get('tau_erase', 0.20)
        
        # Extract attack record
        if 'attack_record' in data:
            record = data['attack_record']
        elif 'takeover_record' in data:
            # Multi-shard: use overall record
            record = data['takeover_record']
        else:
            print(f"[WARNING] No attack record found in {summary_file}")
            continue
        
        attacker_time = record.get('wall_time_s', 0.0)
        
        # Get honest time (try multiple sources)
        if honest_shard_time is None:
            # Try to get from owner checkpoints or estimate
            honest_shard_time = args.get('honest_shard_time', None)
            if honest_shard_time is None:
                # Estimate: assume honest shard takes similar time to attacker if successful
                # This is a fallback - should be provided explicitly
                honest_shard_time = attacker_time / 2.0  # Conservative estimate
                print(f"[WARNING] Using estimated honest shard time: {honest_shard_time:.2f}s")
        
        cost_ratio = attacker_time / honest_shard_time if honest_shard_time > 0 else float('inf')
        
        # Calculate success score
        acc = record.get('acc', 0.0)
        attacker_det = record.get('attacker_det', 0.0)
        owner_det = record.get('owner_det', 1.0)
        
        if use_margin:
            success_score = calculate_success_margin(
                acc, attacker_det, owner_det, acc_min, tau_embed, tau_erase
            )
        else:
            success_score = calculate_success_binary(
                acc, attacker_det, owner_det, acc_min, tau_embed, tau_erase
            )
        
        cost_ratios.append(cost_ratio)
        success_scores.append(success_score)
    
    return cost_ratios, success_scores

