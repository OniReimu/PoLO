#!/usr/bin/env python3
"""
PoLO Shard Experiment Runner

Orchestrates the complete experiment pipeline:
1. Owner training (multi-shard training with evolving watermarks)
2. Per-shard takeover attack (Option A: single-shard "free-ride" attack)
3. Multi-shard takeover attack (Option B: sequential multi-shard attack)

Usage:
    python run_experiment.py [--clean] [--skip-owner] [--skip-per-shard] [--skip-multi-shard]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def clean_cache():
    """Remove all local cache directories and files."""
    print("[CLEAN] Removing cache directories and files...")
    print("[CLEAN] Note: data/ directory is preserved to avoid re-downloading datasets")
    
    # Output directories
    output_dirs = [
        "owner_training_out",
        "per_shard_takeover_out",
        "multi_shard_takeover_out",
    ]
    
    for dir_name in output_dirs:
        if os.path.exists(dir_name):
            print(f"[CLEAN] Removing {dir_name}/")
            shutil.rmtree(dir_name)
    
    # Python cache directories
    for root, dirs, files in os.walk("."):
        # Skip hidden directories, common ignore patterns, and data/ directory
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules' and d != 'data']
        
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            print(f"[CLEAN] Removing {cache_path}/")
            shutil.rmtree(cache_path)
    
    # Cached model files (optional - comment out if you want to keep pre-trained models)
    cached_models_dir = "models/cached_models"
    if os.path.exists(cached_models_dir):
        print(f"[CLEAN] Removing {cached_models_dir}/")
        shutil.rmtree(cached_models_dir)
        os.makedirs(cached_models_dir, exist_ok=True)  # Recreate empty directory
    
    print("[CLEAN] Cache cleanup complete.")


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"[RUN] {description}")
    print(f"{'='*60}")
    print(f"[CMD] {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        return False
    print(f"\n[SUCCESS] {description} completed successfully")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Run complete PoLO shard experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment (both attacks)
  python run_experiment.py

  # Clean cache and run full experiment
  python run_experiment.py --clean

  # Skip owner training (use existing checkpoints)
  python run_experiment.py --skip-owner

  # Run only per-shard attack
  python run_experiment.py --skip-owner --attack-type per-shard

  # Run only multi-shard attack
  python run_experiment.py --skip-owner --attack-type multi-shard

  # Run both attacks (explicit)
  python run_experiment.py --skip-owner --attack-type both
        """
    )
    
    ap.add_argument("--clean", action="store_true",
                   help="Remove all cache directories and files before running")
    ap.add_argument("--skip-owner", action="store_true",
                   help="Skip owner training step (requires existing owner_checkpoints.json)")
    ap.add_argument("--attack-type", type=str, default="both", choices=["both", "per-shard", "multi-shard"],
                   help="Which attack(s) to run: 'both' (default), 'per-shard' only, or 'multi-shard' only")
    
    # Dataset and model selection (passed to all steps)
    ap.add_argument("--dataset", type=str, default="cifar10",
                   help="Dataset to use (e.g., cifar10, cifar100, mnist, fashion_mnist)")
    ap.add_argument("--model", type=str, default="cnn",
                   help="Model to use (e.g., cnn, resnet18, efficientnet_b0, vit_b16)")
    ap.add_argument("--train-subset", type=int, default=5000,
                   help="Number of training samples to use (default: 5000)")
    ap.add_argument("--test-subset", type=int, default=2000,
                   help="Number of test samples to use (default: 2000)")
    ap.add_argument("--batch-size", type=int, default=128,
                   help="Batch size for training (default: 128)")
    
    # Pass-through arguments for owner training
    ap.add_argument("--owner-outdir", type=str, default="owner_training_out",
                   help="Output directory for owner training")
    ap.add_argument("--owner-seed", type=int, default=123,
                   help="Random seed for owner training")
    
    # Pass-through arguments for attacks
    ap.add_argument("--attack-seed", type=int, default=123,
                   help="Random seed for attacks")
    ap.add_argument("--per-shard-outdir", type=str, default="per_shard_takeover_out",
                   help="Output directory for per-shard attack")
    ap.add_argument("--multi-shard-outdir", type=str, default="multi_shard_takeover_out",
                   help="Output directory for multi-shard attack")
    ap.add_argument("--attack-mode", type=str, default="mode2", choices=["mode1", "mode2"],
                   help="mode1: anchor-only erase; mode2: per-shard owner WM derived from hash(prev ckpt)")
    ap.add_argument("--target-shard", type=int, default=2,
                   help="Target shard index for multi-shard attack (default: 2, meaning attack shards 0, 1, 2 = 3 shards total)")
    ap.add_argument("--blind-erasure-lambda", type=float, default=0.0,
                   help="Optional weight for blind erasure loss (weight perturbation heuristic). "
                        "Default 0: rely on natural full-weight fine-tuning to erase owner watermark. "
                        "The attacker doesn't know owner key/position, so full-weight fine-tuning "
                        "hopes to accidentally perturb owner watermark parameters.")
    
    # Plotting flags
    ap.add_argument("--plot-time-curves", action="store_true",
                   help="Generate time-to-pass curves plots")
    ap.add_argument("--time-unit-seconds", action="store_true",
                   help="Use seconds instead of epochs for x-axis in time-to-pass curves")
    
    args = ap.parse_args()
    
    # Clean cache if requested
    if args.clean:
        clean_cache()
        print()
    
    # Determine script paths
    script_dir = Path(__file__).parent
    owner_script = script_dir / "core" / "owner_training.py"
    per_shard_script = script_dir / "attacks" / "per_shard_takeover.py"
    multi_shard_script = script_dir / "attacks" / "multi_shard_takeover.py"
    
    # Check scripts exist
    if not owner_script.exists():
        print(f"[ERROR] Owner training script not found: {owner_script}")
        sys.exit(1)
    if not per_shard_script.exists():
        print(f"[ERROR] Per-shard attack script not found: {per_shard_script}")
        sys.exit(1)
    if not multi_shard_script.exists():
        print(f"[ERROR] Multi-shard attack script not found: {multi_shard_script}")
        sys.exit(1)
    
    success = True
    
    # Step 1: Owner training
    if not args.skip_owner:
        owner_cmd = [
            sys.executable, str(owner_script),
            "--outdir", args.owner_outdir,
            "--seed", str(args.owner_seed),
            "--dataset", args.dataset,
            "--model", args.model,
            "--train_subset", str(args.train_subset),
            "--test_subset", str(args.test_subset),
            "--batch_size", str(args.batch_size),
        ]
        success = run_command(owner_cmd, "Step 1: Owner Training") and success
        if not success:
            print("[FATAL] Owner training failed. Cannot proceed with attacks.")
            sys.exit(1)
    else:
        print("\n[SKIP] Skipping owner training (--skip-owner)")
        owner_checkpoints = os.path.join(args.owner_outdir, "owner_checkpoints.json")
        if not os.path.exists(owner_checkpoints):
            print(f"[ERROR] Owner checkpoints not found: {owner_checkpoints}")
            print("[INFO] Please run owner training first or remove --skip-owner flag")
            sys.exit(1)
        print(f"[INFO] Using existing owner checkpoints: {owner_checkpoints}")
    
    # Determine owner checkpoints path
    owner_checkpoints = os.path.join(args.owner_outdir, "owner_checkpoints.json")
    if not os.path.exists(owner_checkpoints):
        print(f"[FATAL] Owner checkpoints not found: {owner_checkpoints}")
        sys.exit(1)
    
    # Step 2: Per-shard takeover attack
    run_per_shard = args.attack_type in ["both", "per-shard"]
    if run_per_shard:
        per_shard_cmd = [
            sys.executable, str(per_shard_script),
            "--owner_checkpoints", owner_checkpoints,
            "--outdir", args.per_shard_outdir,
            "--seed", str(args.attack_seed),
            "--dataset", args.dataset,
            "--model", args.model,
            "--train_subset", str(args.train_subset),
            "--test_subset", str(args.test_subset),
            "--batch_size", str(args.batch_size),
        ]
        if args.plot_time_curves:
            per_shard_cmd.append("--plot_time_curves")
        if args.time_unit_seconds:
            per_shard_cmd.append("--time_unit_seconds")
        
        success = run_command(per_shard_cmd, "Step 2: Per-Shard Takeover Attack") and success
    else:
        print("\n[SKIP] Skipping per-shard takeover attack (--skip-per-shard)")
    
    # Step 3: Multi-shard takeover attack
    run_multi_shard = args.attack_type in ["both", "multi-shard"]
    if run_multi_shard:
        multi_shard_cmd = [
            sys.executable, str(multi_shard_script),
            "--owner_checkpoints", owner_checkpoints,
            "--outdir", args.multi_shard_outdir,
            "--seed", str(args.attack_seed),
            "--dataset", args.dataset,
            "--model", args.model,
            "--attack_mode", args.attack_mode,
            "--target_shard", str(args.target_shard),
            "--train_subset", str(args.train_subset),
            "--test_subset", str(args.test_subset),
            "--batch_size", str(args.batch_size),
        ]
        if args.blind_erasure_lambda != 0.0:
            multi_shard_cmd.extend(["--blind_erasure_lambda", str(args.blind_erasure_lambda)])
        if args.plot_time_curves:
            multi_shard_cmd.append("--plot_time_curves")
        if args.time_unit_seconds:
            multi_shard_cmd.append("--time_unit_seconds")
        
        success = run_command(multi_shard_cmd, "Step 3: Multi-Shard Takeover Attack") and success
    else:
        print("\n[SKIP] Skipping multi-shard takeover attack (--skip-multi-shard)")
    
    # Final summary
    print("\n" + "="*60)
    if success:
        print("[SUCCESS] Experiment pipeline completed successfully!")
        print(f"\n[OUTPUT] Results saved in:")
        if not args.skip_owner:
            print(f"  - Owner training: {args.owner_outdir}/")
        if run_per_shard:
            print(f"  - Per-shard attack: {args.per_shard_outdir}/")
        if run_multi_shard:
            print(f"  - Multi-shard attack: {args.multi_shard_outdir}/")
    else:
        print("[WARNING] Experiment pipeline completed with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()

