#!/usr/bin/env python3
"""
PoLO per-shard takeover experiment (Option A): Single-shard "free-ride" attack.

Goal: Attacker wants to monetize one checkpoint (a finished shard checkpoint) by claiming it as theirs.

Process:
- Attacker chooses an anchor checkpoint W_{x-1} (start of shard x).
- Runs removal-plus-re-embedding on that shard only: full-model fine-tuning to
  (i) suppress incumbent watermark in shard x, (ii) insert attacker watermark,
  (iii) keep accuracy within verifier tolerance.
- Submits this forged shard proof + resulting checkpoint W_x'.

Success requires: attacker_det >= tau_embed AND owner_det <= tau_erase AND acc >= acc_min.

Why this is enough: In PoLO, verification is per-shard and the economic claim is per shard/proof segment.
If you show that for any single shard the cheapest successful forgery costs >= honest shard training
(or fails to erase the incumbent mark), then the attacker can't profitably "patch" ownership for that shard.

This script attacks ONE shard only. For multi-shard sequential attacks, use multi_shard_takeover.py.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ShardExperimentConfig, resolve_device
from data import DatasetConfig, build_dataset_builder, available_datasets
from models import build_model, available_models
from watermarks.uchida17 import Uchida17
from watermarks.base import WMArtifact
from plots.plotting import (
    calculate_success_binary,
    calculate_success_margin,
    plot_time_to_pass_curves,
)


# ----------------------------
# Reproducibility helpers
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Model: Use CNN from models/vision/cnn.py
# ----------------------------
# The model will be created using build_model("cnn", num_labels=10)


# ----------------------------
# Watermarking (Uchida-style) - Adapter using watermarks/uchida17.py
# ----------------------------
@dataclass
class WatermarkSpec:
    n_bits: int
    n_params: int
    key_seed: int
    bits_seed: int

    # derived (not stored in JSON; recreated from seeds)
    # - selected indices in flattened weight vector
    # - projection matrix A
    # - bits vector b


def _flatten_params(model: nn.Module) -> torch.Tensor:
    vecs = []
    for p in model.parameters():
        # Ensure tensor is contiguous before reshaping
        vecs.append(p.contiguous().reshape(-1))
    # Ensure the concatenated result is also contiguous
    result = torch.cat(vecs, dim=0)
    return result.contiguous()


def _model_to_weight_dict(model: nn.Module) -> Dict[str, np.ndarray]:
    """Convert PyTorch model to weight dict format expected by Uchida17."""
    weight_dict = {}
    for name, param in model.named_parameters():
        weight_dict[name] = param.detach().cpu().numpy()
    return weight_dict


def build_watermark_components(
    model: nn.Module,
    spec: WatermarkSpec,
    device: torch.device,
    dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, WMArtifact]:
    """
    Returns (idx, A, bits, artifact):
      idx: LongTensor [n_params] - indices in flattened weight vector
      A:   FloatTensor [n_bits, n_params] - projection matrix
      bits: FloatTensor [n_bits] in {0,1} - target bits
      artifact: WMArtifact for Uchida17 compatibility
    """
    flat = _flatten_params(model)
    total_params = flat.numel()

    # Select random indices in flattened parameter vector
    rng = np.random.default_rng(spec.key_seed)
    idx_np = rng.choice(total_params, size=min(spec.n_params, total_params), replace=False)
    idx = torch.tensor(idx_np, dtype=torch.long, device=device)

    # Use Uchida17's projection matrix generation logic
    wm_scheme = Uchida17()
    key_bytes = spec.key_seed.to_bytes(8, byteorder='big', signed=False)
    
    # Build projection matrix X using Uchida17's method
    X = wm_scheme._build_X(key_bytes, spec.n_params, spec.n_bits)
    
    # Build target bits using Uchida17's method
    target_bits = wm_scheme._build_bits(key_bytes, spec.n_bits)
    
    # Override target bits with bits_seed for consistency with original behavior
    rng_bits = np.random.default_rng(spec.bits_seed)
    target_bits = (rng_bits.integers(low=0, high=2, size=(spec.n_bits,), dtype=np.int64)).astype(np.uint8)
    
    # Convert to torch tensors
    A = torch.tensor(X.astype(np.float32), dtype=dtype, device=device)
    bits = torch.tensor(target_bits.astype(np.float32), dtype=dtype, device=device)
    
    # Create artifact for compatibility (not fully used but maintains interface)
    artifact = WMArtifact(
        name="uchida17",
        meta={
            "kx": key_bytes,
            "K": spec.n_bits,
            "B": spec.n_bits,
            "X": X,
            "target_bits": target_bits,
        },
        slots=None,
    )
    
    return idx, A, bits, artifact


def watermark_logits(model: nn.Module, idx: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    flat = _flatten_params(model)
    w_sel = flat.index_select(0, idx).contiguous()  # [n_params] - Ensure contiguous for gradient computation
    return torch.matmul(A, w_sel)  # [n_bits]


def watermark_loss(model: nn.Module, idx: torch.Tensor, A: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    logits = watermark_logits(model, idx, A)
    pred = torch.sigmoid(logits)
    return F.binary_cross_entropy(pred, bits)


@torch.no_grad()
def watermark_detection(model: nn.Module, idx: torch.Tensor, A: torch.Tensor, bits: torch.Tensor) -> float:
    logits = watermark_logits(model, idx, A)
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return float((pred == bits).float().mean().item())


# ----------------------------
# Training / evaluation
# ----------------------------
@torch.no_grad()
def eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    wm_terms: List[Tuple[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],  # [(lambda, (idx,A,bits)), ...]
) -> int:
    """
    Returns number of steps (batches).
    wm_terms: list of watermark losses to ADD to task loss.
    
    Full-weight fine-tuning: all model parameters are updated.
    """
    model.train()
    # Ensure all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True
    
    steps = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        task_loss = F.cross_entropy(out, y)
        loss = task_loss
        for lam, (idx, A, bits) in wm_terms:
            loss = loss + lam * watermark_loss(model, idx, A, bits)
        loss.backward()
        
        # Verify gradients are computed for all parameters
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters())
        if steps == 0 and has_grad != total_params:
            print(f"[WARNING] Only {has_grad}/{total_params} parameters have gradients!")
        
        optimizer.step()
        steps += 1
    return steps


# ----------------------------
# Shard bookkeeping
# ----------------------------
@dataclass
class ShardCheckpoint:
    shard_id: int
    epoch_global: int
    chain_hash: str
    owner_spec: WatermarkSpec
    state_dict_path: str


@dataclass
class AttackRecord:
    anchor_shard: int
    epochs: int
    steps: int
    wall_time_s: float
    acc: float
    owner_det: float
    attacker_det: float
    success: bool


# ----------------------------
# Main experiment
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Single-shard takeover attack")
    ap.add_argument("--owner_checkpoints", type=str, required=True,
                    help="Path to owner_checkpoints.json from owner training")
    ap.add_argument("--outdir", type=str, default="per_shard_takeover_out")
    ap.add_argument("--device", type=str, default=None,
                    help="Device to use (cuda/mps/cpu). Default: auto-detect (cuda > mps > cpu)")

    # Data
    ap.add_argument("--dataset", type=str, default="cifar10",
                   choices=list(available_datasets()),
                   help=f"Dataset to use. Available: {list(available_datasets())}")
    ap.add_argument("--model", type=str, default="cnn",
                   choices=list(available_models()),
                   help=f"Model to use. Available: {list(available_models())}")
    ap.add_argument("--train_subset", type=int, default=5000)
    ap.add_argument("--test_subset", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=128)

    # Attacker training
    ap.add_argument("--attack_lr", type=float, default=0.005)
    ap.add_argument("--attack_epochs_max", type=int, default=80)
    ap.add_argument("--attack_lambda", type=float, default=0.5)

    # Shard selection
    ap.add_argument("--anchor_shard", type=int, default=-1,
                    help="Anchor shard index (-1 means last available)")

    # Watermark parameters
    ap.add_argument("--wm_bits", type=int, default=256)
    ap.add_argument("--wm_params", type=int, default=4096)

    # Thresholds from config
    shard_config = ShardExperimentConfig()
    ap.add_argument("--tau_embed", type=float, default=shard_config.tau_embed,
                    help="Attacker watermark detection threshold (must be >= this)")
    ap.add_argument("--tau_erase", type=float, default=shard_config.tau_erase,
                    help="Owner watermark must drop below this to be considered erased")
    ap.add_argument("--acc_min", type=float, default=shard_config.acc_min,
                    help="Minimum accuracy required for successful attack")

    # Plotting
    ap.add_argument("--plot_time_curves", action="store_true",
                   help="Generate time-to-pass curves plot")
    ap.add_argument("--time_unit_seconds", action="store_true",
                   help="Use seconds instead of epochs for x-axis in time-to-pass curves")
    ap.add_argument("--honest_shard_time", type=float, default=None,
                    help="Honest shard training time (for cost ratio calculation)")
    ap.add_argument("--honest_metrics_file", type=str, default=None,
                    help="Path to honest training metrics JSON (for time-to-pass plot)")

    # Seeds
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Auto-detect device if not specified
    if args.device is None:
        args.device = resolve_device()
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # Load owner checkpoints
    if not os.path.exists(args.owner_checkpoints):
        print(f"[ERROR] Owner checkpoints file not found: {args.owner_checkpoints}")
        print("[INFO] Please run owner training first:")
        print("       python core/owner_training.py --outdir <output_dir>")
        print("       Or use: python run_experiment.py")
        print(f"       Then use: --owner_checkpoints <output_dir>/owner_checkpoints.json")
        return
    
    with open(args.owner_checkpoints, "r", encoding="utf-8") as f:
        owner_checkpoints_data = json.load(f)
    
    checkpoints: List[ShardCheckpoint] = []
    for ckpt_data in owner_checkpoints_data:
        checkpoints.append(ShardCheckpoint(
            shard_id=ckpt_data["shard_id"],
            epoch_global=ckpt_data["epoch_global"],
            chain_hash=ckpt_data["chain_hash"],
            owner_spec=WatermarkSpec(**ckpt_data["owner_spec"]),
            state_dict_path=ckpt_data["state_dict_path"],
        ))

    if not checkpoints:
        print("[FATAL] No owner checkpoints found.")
        return

    print(f"[INFO] Loaded {len(checkpoints)} owner checkpoints")

    # Select anchor checkpoint
    anchor_idx = args.anchor_shard if args.anchor_shard >= 0 else (len(checkpoints) - 1)
    anchor_idx = max(0, min(anchor_idx, len(checkpoints) - 1))
    anchor = checkpoints[anchor_idx]
    print(f"[ATTACK] Anchoring at shard={anchor.shard_id}, checkpoint={anchor.state_dict_path}")

    # Data: Use dataset builder
    # Data: Only train and test sets (no calibration, no public/private distinction)
    dataset_config = DatasetConfig(
        dataset_root="data",  # Datasets stored in data/ to persist across --clean
        allow_download=True,
        dataset_size=args.train_subset,
        calibration_size=0,  # Not used
        public_ratio=0.0,  # Not used
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        seed=args.seed,
    )
    dataset_builder = build_dataset_builder(args.dataset)
    loaders = dataset_builder.build(dataset_config)
    
    # Get num_labels from dataset builder
    num_labels = dataset_builder.num_labels
    print(f"[INFO] Dataset: {args.dataset}, num_labels: {num_labels}")
    
    train_loader = loaders.private
    test_loader = loaders.evaluation
    
    if args.test_subset < len(loaders.evaluation.dataset):
        from torch.utils.data import Subset
        test_indices = list(range(min(args.test_subset, len(loaders.evaluation.dataset))))
        test_subset = Subset(loaders.evaluation.dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    # ----------------------------
    # Single-shard takeover attack
    # ----------------------------

    # Load anchor weights
    print(f"[INFO] Model: {args.model}")
    model = build_model(args.model, num_labels=num_labels).to(device)
    model.load_state_dict(torch.load(anchor.state_dict_path, map_location=device))
    
    # Ensure ALL parameters are trainable (full-weight fine-tuning)
    for param in model.parameters():
        param.requires_grad = True
    
    # Verify all parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ATTACK] Trainable parameters: {trainable_params}/{total_params} ({100*trainable_params/total_params:.1f}%)")
    
    if trainable_params != total_params:
        print("[WARNING] Not all parameters are trainable! This may prevent full-weight fine-tuning.")
    
    opt = torch.optim.SGD(model.parameters(), lr=args.attack_lr, momentum=0.9, weight_decay=5e-4)

    # Owner watermark for this shard (verifier knows; attacker does NOT)
    owner_spec_shard = WatermarkSpec(**asdict(anchor.owner_spec))
    idx_oa, A_oa, bits_oa, _ = build_watermark_components(model, owner_spec_shard, device=device)

    # Attacker chooses their own watermark region/key/bits
    attacker_spec = WatermarkSpec(
        n_bits=args.wm_bits,
        n_params=args.wm_params,
        key_seed=args.seed + 9999,          # attacker secret region
        bits_seed=args.seed + 8888 + 7      # attacker secret bits
    )
    idx_a, A_a, bits_a, _ = build_watermark_components(model, attacker_spec, device=device)

    # Evaluate at start
    acc0 = eval_accuracy(model, test_loader, device)
    owner_det0 = watermark_detection(model, idx_oa, A_oa, bits_oa)
    att_det0 = watermark_detection(model, idx_a, A_a, bits_a)
    print(f"[ATTACK] Start: acc={acc0:.4f} owner_det={owner_det0:.4f} attacker_det={att_det0:.4f}")

    # Fine-tune to erase owner watermark and embed attacker watermark
    att_steps = 0
    att_t0 = time.perf_counter()
    success = False
    best_state = None
    best_score = -1.0
    
    # Track metrics over time for plotting
    attack_epochs = []
    attack_times_seconds = []  # Track wall time per epoch for time-axis plotting
    attack_owner_det = []
    attack_attacker_det = []
    attack_acc = []

    for ep in range(1, args.attack_epochs_max + 1):
        epoch_t0 = time.perf_counter()
        # Full-weight fine-tuning: optimize for both embedding attacker WM and erasing owner WM
        steps = train_one_epoch(
            model,
            train_loader,
            device,
            opt,
            wm_terms=[
                (args.attack_lambda, (idx_a, A_a, bits_a)),  # Embed attacker watermark
                (args.attack_lambda * 0.5, (idx_oa, A_oa, 1.0 - bits_oa)),  # Erase owner watermark (invert target)
            ]
        )
        att_steps += steps
        epoch_time = time.perf_counter() - epoch_t0

        acc = eval_accuracy(model, test_loader, device)
        owner_det = watermark_detection(model, idx_oa, A_oa, bits_oa)
        att_det = watermark_detection(model, idx_a, A_a, bits_a)

        # Track metrics for plotting
        attack_epochs.append(ep)
        # Track cumulative time in seconds
        if len(attack_times_seconds) == 0:
            attack_times_seconds.append(epoch_time)
        else:
            attack_times_seconds.append(attack_times_seconds[-1] + epoch_time)
        attack_owner_det.append(owner_det)
        attack_attacker_det.append(att_det)
        attack_acc.append(acc)

        print(f"[ATTACK] epoch={ep:03d} acc={acc:.4f} owner_det={owner_det:.4f} attacker_det={att_det:.4f}")

        # Track "best attempt" even if not fully successful
        score = (att_det) - (owner_det) + 0.1 * acc
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

        # Check success condition
        if (att_det >= args.tau_embed) and (owner_det <= args.tau_erase) and (acc >= args.acc_min):
            success = True
            print(f"[ATTACK] SUCCESS at epoch {ep}")
            break

    att_wall = time.perf_counter() - att_t0

    # Load best state if attack didn't fully succeed
    if not success and best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(args.outdir, "attacker_best_effort.pt"))

    # Final metrics
    accF = eval_accuracy(model, test_loader, device)
    owner_detF = watermark_detection(model, idx_oa, A_oa, bits_oa)
    att_detF = watermark_detection(model, idx_a, A_a, bits_a)

    # Save forged checkpoint
    forged_path = os.path.join(args.outdir, f"attacker_shard{anchor.shard_id:02d}.pt")
    torch.save(model.state_dict(), forged_path)
    print(f"[ATTACK] Saved forged checkpoint: {forged_path}")

    # Create attack record
    attack_record = AttackRecord(
        anchor_shard=anchor.shard_id,
        epochs=ep if 'ep' in locals() else args.attack_epochs_max,
        steps=att_steps,
        wall_time_s=att_wall,
        acc=accF,
        owner_det=owner_detF,
        attacker_det=att_detF,
        success=success,
    )

    # Calculate success scores
    success_binary = calculate_success_binary(
        accF, att_detF, owner_detF, args.acc_min, args.tau_embed, args.tau_erase
    )
    success_margin = calculate_success_margin(
        accF, att_detF, owner_detF, args.acc_min, args.tau_embed, args.tau_erase
    )

    # Get honest shard time (try to extract from owner checkpoints or use provided)
    honest_shard_time = args.honest_shard_time
    if honest_shard_time is None:
        # Try to get from owner checkpoints metadata if available
        owner_meta_path = os.path.join(os.path.dirname(args.owner_checkpoints), "owner_training_metadata.json")
        if os.path.exists(owner_meta_path):
            with open(owner_meta_path, 'r') as f:
                owner_meta = json.load(f)
                # Estimate: total_time / num_shards
                if 'total_wall_time_s' in owner_meta and 'shards_completed' in owner_meta:
                    honest_shard_time = owner_meta['total_wall_time_s'] / max(1, owner_meta['shards_completed'])
        else:
            # Fallback: estimate based on attacker time (conservative)
            honest_shard_time = att_wall / 2.0
            print(f"[WARNING] Using estimated honest shard time: {honest_shard_time:.2f}s")
    
    cost_ratio = att_wall / honest_shard_time if honest_shard_time > 0 else float('inf')

    # Print summary
    print("\n=== PER-SHARD TAKEOVER SUMMARY ===")
    print(json.dumps({
        "anchor_shard": attack_record.anchor_shard,
        "success": attack_record.success,
        "success_binary": success_binary,
        "success_margin": success_margin,
        "epochs": attack_record.epochs,
        "steps": attack_record.steps,
        "wall_time_s": attack_record.wall_time_s,
        "honest_shard_time_s": honest_shard_time,
        "cost_ratio": cost_ratio,
        "final_acc": attack_record.acc,
        "final_owner_det": attack_record.owner_det,
        "final_attacker_det": attack_record.attacker_det,
        "note": "Attack success requires attacker watermark high, owner watermark erased, and accuracy preserved."
    }, indent=2))

    # Prepare summary data
    summary_data = {
        "attack_record": asdict(attack_record),
        "anchor_checkpoint": asdict(anchor),
        "attacker_spec": asdict(attacker_spec),
        "success_binary": success_binary,
        "success_margin": success_margin,
        "honest_shard_time_s": honest_shard_time,
        "cost_ratio": cost_ratio,
        "args": vars(args),
    }
    
    # Generate plots if requested
    if args.plot_time_curves:
        # Load honest metrics if available
        honest_metrics = None
        if args.honest_metrics_file and os.path.exists(args.honest_metrics_file):
            with open(args.honest_metrics_file, 'r') as f:
                honest_data = json.load(f)
                if 'epoch_metrics' in honest_data:
                    honest_metrics = honest_data['epoch_metrics']
        else:
            # Try to auto-load from owner training metadata
            owner_meta_path = os.path.join(os.path.dirname(args.owner_checkpoints), "owner_training_metadata.json")
            if os.path.exists(owner_meta_path):
                with open(owner_meta_path, 'r') as f:
                    owner_meta = json.load(f)
                    if 'epoch_metrics' in owner_meta:
                        honest_metrics = owner_meta['epoch_metrics']
                        print(f"[PLOT] Loaded honest metrics from {owner_meta_path}")
        
        attack_metrics = {
            'owner_det': attack_owner_det,
            'attacker_det': attack_attacker_det,
            'acc': attack_acc,
        }
        
        time_curve_path = os.path.join(args.outdir, "time_to_pass_curves.png")
        # Use seconds for time axis if requested, otherwise use epochs
        time_axis = attack_times_seconds if args.time_unit_seconds else attack_epochs
        time_unit = "seconds" if args.time_unit_seconds else "epochs"
        time_to_pass = plot_time_to_pass_curves(
            attack_metrics=attack_metrics,
            time_axis=time_axis,
            honest_metrics=honest_metrics,
            output_path=time_curve_path,
            title="Per-Shard Time-to-Pass Curves",
            tau_embed=args.tau_embed,
            tau_erase=args.tau_erase,
            acc_min=args.acc_min,
            time_unit=time_unit,
        )
        
        # Save time-to-pass to summary
        if time_to_pass is not None:
            summary_data['time_to_pass_epochs'] = time_to_pass
    
    # Save summary
    summary_path = os.path.join(args.outdir, "per_shard_takeover_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()