#!/usr/bin/env python3
"""
PoLO multi-shard takeover experiment: fork-and-continue free-ride.

Threat model narrative:
- Attacker anchors at an owner shard checkpoint W_x (owner already paid the cost).
- One-time sanitize anchor: remove incumbent owner watermark (without key/positions) and
  embed attacker's own watermark while keeping accuracy.
- Then continue on a forked trajectory, generating future shards (x+1, x+2, ...)
  embedding attacker watermarks only. Owner watermark should stay below threshold.
"""

import argparse
import copy
import json
import hashlib
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
# Watermarking (Uchida-style) - Adapter using watermarks/uchida17.py
# ----------------------------
@dataclass
class WatermarkSpec:
    n_bits: int
    n_params: int
    key_seed: int
    bits_seed: int


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


def _owner_spec_from_hash(prev_ckpt_path: str, n_bits: int, n_params: int) -> WatermarkSpec:
    """
    Derive a watermark spec from the hash of a previous checkpoint.
    Can be used for owner or attacker (mode2) depending on caller.
    """
    h = _hash_checkpoint(prev_ckpt_path)
    # Split the hash into two 64-bit seeds for key and bits
    key_seed = h
    bits_seed = (h >> 32) ^ (h & 0xFFFFFFFF)
    return WatermarkSpec(
        n_bits=n_bits,
        n_params=n_params,
        key_seed=key_seed,
        bits_seed=bits_seed,
    )


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


def _hash_checkpoint(path: str) -> int:
    """
    Compute a deterministic hash for a checkpoint file to derive watermark seeds.
    Uses SHA256 over parameter bytes (name-sorted).
    Returns an integer derived from the hash digest.
    """
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {path} is not a state_dict")
    sha = hashlib.sha256()
    for name in sorted(state.keys()):
        tensor = state[name]
        if torch.is_tensor(tensor):
            sha.update(name.encode("utf-8"))
            sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest = sha.digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


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
    blind_erasure_lambda: float = 0.0,
    initial_state_dict: Optional[Dict] = None,
) -> int:
    """
    Returns number of steps (batches).
    wm_terms: list of watermark losses to ADD to task loss.
    blind_erasure_lambda: optional weight for blind erasure loss (weight perturbation heuristic).
                         Default 0: rely on natural full-weight fine-tuning.
    initial_state_dict: initial model state for blind erasure (if None, no blind erasure)
    
    Full-weight fine-tuning: all model parameters are updated.
    The attacker doesn't know the owner's watermark key/position, so they rely on
    full-weight fine-tuning hoping to accidentally erase the owner watermark somewhere
    they're not aware of.
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
        
        # Blind erasure: encourage weight changes from initial state (without knowing owner key)
        if blind_erasure_lambda > 0 and initial_state_dict is not None:
            perturbation_loss = 0.0
            for name, param in model.named_parameters():
                if name in initial_state_dict:
                    initial_param = initial_state_dict[name].to(device)
                    perturbation_loss = perturbation_loss + torch.mean((param - initial_param) ** 2)
            loss = loss + blind_erasure_lambda * perturbation_loss
        
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
class ShardAttackRecord:
    shard_id: int
    epochs: int
    steps: int
    wall_time_s: float
    acc: float
    owner_det: float
    attacker_det: float
    success: bool


@dataclass
class MultiShardTakeoverRecord:
    anchor_shard: int
    target_shard: int
    shards_attempted: int
    shards_successful: int
    total_epochs: int
    total_steps: int
    total_wall_time_s: float
    per_shard_records: List[ShardAttackRecord]
    overall_success: bool  # True if all shards succeeded


# ----------------------------
# Main experiment
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Multi-shard sequential takeover attack")
    ap.add_argument("--owner_checkpoints", type=str, required=True,
                    help="Path to owner_checkpoints.json from owner training")
    ap.add_argument("--outdir", type=str, default="multi_shard_takeover_out")
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
    # Load config for defaults
    shard_config = ShardExperimentConfig()
    ap.add_argument("--attack_lr", type=float, default=shard_config.attack_lr,
                   help=f"Learning rate for attacker training (default: {shard_config.attack_lr} from config)")
    ap.add_argument("--attack_epochs_max", type=int, default=shard_config.attack_epochs_max,
                   help=f"Maximum epochs for attacker training per shard (default: {shard_config.attack_epochs_max} from config)")
    ap.add_argument("--attack_lambda", type=float, default=0.5)
    ap.add_argument("--blind_erasure_lambda", type=float, default=0.0,
                   help="Optional weight for blind erasure loss (weight perturbation heuristic). "
                        "Default 0: rely on natural full-weight fine-tuning to erase owner watermark. "
                        "The attacker doesn't know owner key/position, so full-weight fine-tuning "
                        "hopes to accidentally perturb owner watermark parameters.")

    # Shard selection
    # Always start from end of shard-0 (checkpoint saved when shard-0 completes)
    ap.add_argument("--target_shard", type=int, default=2,
                    help="Target shard index (default: 2, meaning attack shards 0, 1, 2 = 3 shards total)")

    # Watermark parameters
    ap.add_argument("--wm_bits", type=int, default=256)
    ap.add_argument("--wm_params", type=int, default=4096)

    # Thresholds from config
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
                    help="Honest shard training time per shard (for cost ratio calculation)")
    ap.add_argument("--attack_mode", type=str, default="mode1", choices=["mode1", "mode2"],
                    help="mode1: fork-and-continue (erase owner only on anchor). "
                         "mode2: per-shard owner watermark derived from hash(prev checkpoint); "
                         "each shard requires owner_erased + attacker_embed + acc.")

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

    # Determine shard range
    # Always start from end of shard-0 (checkpoint saved when shard-0 completes)
    anchor_idx = 0
    target_idx = min(args.target_shard, len(checkpoints) - 1)

    if target_idx <= anchor_idx:
        print(f"[FATAL] Target shard ({target_idx}) must be > anchor shard ({anchor_idx})")
        return

    shards_to_attack = list(range(anchor_idx, target_idx + 1))
    print(f"[INFO] Attacking (sanitize + continue) shards {shards_to_attack[0]} to {shards_to_attack[-1]} (total: {len(shards_to_attack)} shards)")

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

    # Initialize model from anchor checkpoint
    anchor = checkpoints[anchor_idx]
    print(f"[ATTACK] Starting from anchor shard={anchor.shard_id}, checkpoint={anchor.state_dict_path}")
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

    # Attacker watermark builder (per-shard, evolving bits)
    def build_attacker_wm_for_shard(shard_id: int):
        spec = WatermarkSpec(
            n_bits=args.wm_bits,
            n_params=args.wm_params,
            key_seed=args.seed + 9999,       # attacker region (can keep fixed)
            bits_seed=args.seed + 8888 + shard_id,  # new bits per shard
        )
        return build_watermark_components(model, spec, device=device)

    # Track per-shard results
    per_shard_records: List[ShardAttackRecord] = []
    total_steps = 0
    total_wall_time = 0.0
    overall_success = True
    
    # Track metrics over time for plotting (aggregated across all shards)
    all_epochs = []
    all_times_seconds = []  # Track wall time per epoch for time-axis plotting
    all_owner_det = []
    all_attacker_det = []
    all_acc = []
    all_shard_ids = []  # Track which shard each epoch belongs to

    # Owner watermark for anchor only (mode1) or per-shard (mode2)
    anchor_owner_spec = WatermarkSpec(**asdict(checkpoints[anchor_idx].owner_spec))
    idx_oa_anchor, A_oa_anchor, bits_oa_anchor, _ = build_watermark_components(model, anchor_owner_spec, device=device)
    print(f"[ATTACK] Initial owner watermark (to sanitize): shard {checkpoints[anchor_idx].shard_id} (from anchor checkpoint)")

    # Sequentially attack each shard (anchor: sanitize; subsequent: attacker-only shards)
    for shard_idx in shards_to_attack:
        shard_ckpt = checkpoints[shard_idx]
        is_anchor = shard_idx == anchor_idx
        is_last_shard = shard_idx == target_idx
        print(f"\n[ATTACK] === {'Sanitizing anchor' if is_anchor else 'Continuing'} shard {shard_ckpt.shard_id} ===")
        if is_anchor:
            print("[ATTACK] One-time removal-plus-re-embedding at anchor to make the fork eligible.")
        else:
            print("[ATTACK] Generating attacker shard on fork.")

        # Owner watermark selection
        owner_spec = WatermarkSpec(**asdict(shard_ckpt.owner_spec))
        idx_oa, A_oa, bits_oa, _ = build_watermark_components(model, owner_spec, device=device)

        # Attacker watermark selection
        if args.attack_mode == "mode2" and not is_anchor:
            # Derive attacker WM for this shard from hash of previous attacker checkpoint
            prev_attacker_ckpt = os.path.join(args.outdir, f"attacker_shard{checkpoints[shard_idx - 1].shard_id:02d}.pt")
            if os.path.exists(prev_attacker_ckpt):
                att_spec = _owner_spec_from_hash(prev_attacker_ckpt, args.wm_bits, args.wm_params)
                idx_a, A_a, bits_a, _ = build_watermark_components(model, att_spec, device=device)
            else:
                # Fallback to default per-shard attacker WM if previous attacker checkpoint missing
                idx_a, A_a, bits_a, _ = build_attacker_wm_for_shard(shard_ckpt.shard_id)
        else:
            # mode1 or anchor shard: use default per-shard attacker WM
            idx_a, A_a, bits_a, _ = build_attacker_wm_for_shard(shard_ckpt.shard_id)

        # For mode2, load owner's checkpoint for this shard before attacking
        if args.attack_mode == "mode2" and not is_anchor:
            # Reset model to owner's checkpoint for this shard
            print(f"[ATTACK] Loading owner's checkpoint for shard {shard_ckpt.shard_id} before attack")
            model.load_state_dict(torch.load(shard_ckpt.state_dict_path, map_location=device))
        
        # Evaluate initial state
        acc0 = eval_accuracy(model, test_loader, device)
        if args.attack_mode == "mode1":
            owner_det0 = watermark_detection(model, idx_oa, A_oa, bits_oa) if is_anchor else float('nan')
        else:
            # mode2: Evaluate owner WM on the model (which is now owner's checkpoint for this shard)
            owner_det0 = watermark_detection(model, idx_oa, A_oa, bits_oa)
        att_det0 = watermark_detection(model, idx_a, A_a, bits_a)
        if args.attack_mode == "mode1":
            if is_anchor:
                print(f"[ATTACK] Shard {shard_ckpt.shard_id} start: acc={acc0:.4f} owner_det={owner_det0:.4f} attacker_det={att_det0:.4f}")
            else:
                print(f"[ATTACK] Shard {shard_ckpt.shard_id} start: acc={acc0:.4f} attacker_det={att_det0:.4f} (owner_det not applicable on fork)")
        else:
            print(f"[ATTACK] Shard {shard_ckpt.shard_id} start: acc={acc0:.4f} owner_det={owner_det0:.4f} attacker_det={att_det0:.4f}")

        # Store initial state for blind erasure (weight perturbation without knowing owner key)
        initial_state_dict = copy.deepcopy(model.state_dict()) if args.blind_erasure_lambda > 0 else None

        # Fine-tune to embed attacker watermark (per-shard) while keeping accuracy
        # NOTE: We do NOT use the owner watermark in the loss (threat model: key unknown).
        # Anchor shard: hope to erase owner WM via full-weight fine-tuning (optionally with blind erasure).
        # Subsequent shards: no owner WM embedded; just embed attacker WM and maintain accuracy.
        shard_steps = 0
        shard_t0 = time.perf_counter()
        success = False
        best_state = None
        best_score = -1.0

        for ep in range(1, args.attack_epochs_max + 1):
            epoch_t0 = time.perf_counter()
            # Full-weight fine-tuning: optimize for attacker WM (per-shard) + blind erasure
            steps = train_one_epoch(
                model,
                train_loader,
                device,
                opt,
                wm_terms=[
                    (args.attack_lambda, (idx_a, A_a, bits_a)),  # Embed attacker watermark (per-shard)
                ],
                blind_erasure_lambda=args.blind_erasure_lambda,
                initial_state_dict=initial_state_dict,
            )
            shard_steps += steps
            total_steps += steps
            epoch_time = time.perf_counter() - epoch_t0

            acc = eval_accuracy(model, test_loader, device)
            if args.attack_mode == "mode1":
                owner_det = watermark_detection(model, idx_oa, A_oa, bits_oa) if is_anchor else float('nan')
            else:
                # mode2: Evaluate owner WM on attacker's model (to track erasure progress)
                # The baseline (owner_det0) was already measured on owner's checkpoint
                owner_det = watermark_detection(model, idx_oa, A_oa, bits_oa)
            att_det = watermark_detection(model, idx_a, A_a, bits_a)

            # Track metrics for plotting
            all_epochs.append(len(all_epochs) + 1)  # Cumulative epoch count
            # Track cumulative time in seconds
            if len(all_times_seconds) == 0:
                all_times_seconds.append(epoch_time)
            else:
                all_times_seconds.append(all_times_seconds[-1] + epoch_time)
            all_owner_det.append(owner_det)
            all_attacker_det.append(att_det)
            all_acc.append(acc)
            all_shard_ids.append(shard_ckpt.shard_id)

            if ep % 10 == 0 or ep == 1:
                print(f"[ATTACK] Shard {shard_ckpt.shard_id} epoch={ep:03d} acc={acc:.4f} owner_det={owner_det:.4f} attacker_det={att_det:.4f}")

            # Track best attempt
            if args.attack_mode == "mode2":
                owner_det_for_score = owner_det
            else:
                owner_det_for_score = owner_det if is_anchor else 0.0
            score = (att_det) - (owner_det_for_score) + 0.1 * acc
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

            # Check success condition and early stop when all criteria are met
            # For all shards except the last one, exclude acc_min (they're just examples)
            # Only the last shard needs to maintain accuracy
            if args.attack_mode == "mode2":
                # All shards: must embed attacker WM and erase owner WM
                # Only last shard needs to maintain accuracy
                if is_last_shard:
                    cond = (att_det >= args.tau_embed) and (owner_det <= args.tau_erase) and (acc >= args.acc_min)
                    if cond and not success:
                        success = True
                        print(f"[ATTACK] Shard {shard_ckpt.shard_id} SUCCESS at epoch {ep}: "
                              f"att_det={att_det:.4f}>=tau_embed({args.tau_embed}), "
                              f"owner_det={owner_det:.4f}<=tau_erase({args.tau_erase}), "
                              f"acc={acc:.4f}>={args.acc_min}")
                        break
                else:
                    cond = (att_det >= args.tau_embed) and (owner_det <= args.tau_erase)
                    if cond and not success:
                        success = True
                        print(f"[ATTACK] Shard {shard_ckpt.shard_id} SUCCESS at epoch {ep}: "
                              f"att_det={att_det:.4f}>=tau_embed({args.tau_embed}), "
                              f"owner_det={owner_det:.4f}<=tau_erase({args.tau_erase}) "
                              f"(acc_min not required for intermediate shards)")
                        break
            else:
                if is_anchor:
                    # Anchor: must embed attacker WM, erase owner WM
                    # Only last shard needs to maintain accuracy
                    if is_last_shard:
                        cond = (att_det >= args.tau_embed) and (owner_det <= args.tau_erase) and (acc >= args.acc_min)
                        if cond and not success:
                            success = True
                            print(f"[ATTACK] Shard {shard_ckpt.shard_id} SUCCESS at epoch {ep}: "
                                  f"att_det={att_det:.4f}>=tau_embed({args.tau_embed}), "
                                  f"owner_det={owner_det:.4f}<=tau_erase({args.tau_erase}), "
                                  f"acc={acc:.4f}>={args.acc_min}")
                            break
                    else:
                        cond = (att_det >= args.tau_embed) and (owner_det <= args.tau_erase)
                        if cond and not success:
                            success = True
                            print(f"[ATTACK] Shard {shard_ckpt.shard_id} SUCCESS at epoch {ep}: "
                                  f"att_det={att_det:.4f}>=tau_embed({args.tau_embed}), "
                                  f"owner_det={owner_det:.4f}<=tau_erase({args.tau_erase}) "
                                  f"(acc_min not required for intermediate shards)")
                            break
                else:
                    # Subsequent shards: must embed attacker WM
                    # Only last shard needs to maintain accuracy
                    if is_last_shard:
                        cond = (att_det >= args.tau_embed) and (acc >= args.acc_min)
                        if cond and not success:
                            success = True
                            print(f"[ATTACK] Shard {shard_ckpt.shard_id} SUCCESS at epoch {ep}: "
                                  f"att_det={att_det:.4f}>=tau_embed({args.tau_embed}), "
                                  f"acc={acc:.4f}>={args.acc_min}")
                            break
                    else:
                        cond = (att_det >= args.tau_embed)
                        if cond and not success:
                            success = True
                            print(f"[ATTACK] Shard {shard_ckpt.shard_id} SUCCESS at epoch {ep}: "
                                  f"att_det={att_det:.4f}>=tau_embed({args.tau_embed}) "
                                  f"(acc_min not required for intermediate shards)")
                            break

        shard_wall_time = time.perf_counter() - shard_t0
        total_wall_time += shard_wall_time

        # Final metrics
        if not success and best_state is not None:
            model.load_state_dict(best_state)
        
        accF = eval_accuracy(model, test_loader, device)
        if args.attack_mode == "mode1":
            owner_detF = watermark_detection(model, idx_oa, A_oa, bits_oa) if is_anchor else float('nan')
        else:
            # mode2: Final owner WM on attacker's model (after attack)
            owner_detF = watermark_detection(model, idx_oa, A_oa, bits_oa)
        att_detF = watermark_detection(model, idx_a, A_a, bits_a)

        # Determine final success if not already set during training
        # For all shards except the last one, exclude acc_min (they're just examples)
        # Only the last shard needs to maintain accuracy
        if not success:
            if args.attack_mode == "mode2":
                if is_last_shard:
                    success = (att_detF >= args.tau_embed) and (owner_detF <= args.tau_erase) and (accF >= args.acc_min)
                else:
                    success = (att_detF >= args.tau_embed) and (owner_detF <= args.tau_erase)
            else:
                if is_anchor:
                    if is_last_shard:
                        success = (att_detF >= args.tau_embed) and (owner_detF <= args.tau_erase) and (accF >= args.acc_min)
                    else:
                        success = (att_detF >= args.tau_embed) and (owner_detF <= args.tau_erase)
                else:
                    if is_last_shard:
                        success = (att_detF >= args.tau_embed) and (accF >= args.acc_min)
                    else:
                        success = (att_detF >= args.tau_embed)

        # Record this shard's attack
        shard_record = ShardAttackRecord(
            shard_id=shard_ckpt.shard_id,
            epochs=ep if 'ep' in locals() else args.attack_epochs_max,
            steps=shard_steps,
            wall_time_s=shard_wall_time,
            acc=accF,
            owner_det=owner_detF,
            attacker_det=att_detF,
            success=success,
        )
        per_shard_records.append(shard_record)

        if not success:
            overall_success = False
            print(f"[ATTACK] Shard {shard_ckpt.shard_id} FAILED")
        
        # Save checkpoint for this shard (for potential continuation)
        ckpt_path = os.path.join(args.outdir, f"attacker_shard{shard_ckpt.shard_id:02d}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[ATTACK] Saved checkpoint: {ckpt_path}")

    # Create summary record
    takeover_record = MultiShardTakeoverRecord(
        anchor_shard=anchor.shard_id,
        target_shard=checkpoints[target_idx].shard_id,
        shards_attempted=len(shards_to_attack),  # include anchor sanitization
        shards_successful=sum(1 for r in per_shard_records if r.success),
        total_epochs=sum(r.epochs for r in per_shard_records),
        total_steps=total_steps,
        total_wall_time_s=total_wall_time,
        per_shard_records=per_shard_records,
        overall_success=overall_success,
    )

    # Calculate success scores (overall: all shards must pass)
    # For multi-shard, success is binary: 1 if all shards pass, else 0
    overall_success_binary = 1 if overall_success else 0
    
    # Calculate average margin across shards (for margin plot)
    if per_shard_records:
        avg_acc = np.mean([r.acc for r in per_shard_records])
        avg_attacker_det = np.mean([r.attacker_det for r in per_shard_records])
        avg_owner_det = np.nanmean([r.owner_det for r in per_shard_records])
        overall_success_margin = calculate_success_margin(
            avg_acc, avg_attacker_det, avg_owner_det,
            args.acc_min, args.tau_embed, args.tau_erase
        )
    else:
        overall_success_margin = -1.0

    # Get honest shard time per shard
    honest_shard_time_per_shard = args.honest_shard_time
    if honest_shard_time_per_shard is None:
        # Try to get from owner checkpoints metadata
        owner_meta_path = os.path.join(os.path.dirname(args.owner_checkpoints), "owner_training_metadata.json")
        if os.path.exists(owner_meta_path):
            with open(owner_meta_path, 'r') as f:
                owner_meta = json.load(f)
                if 'total_wall_time_s' in owner_meta and 'shards_completed' in owner_meta:
                    honest_shard_time_per_shard = owner_meta['total_wall_time_s'] / max(1, owner_meta['shards_completed'])
        else:
            # Estimate: average attacker time per shard / 2
            honest_shard_time_per_shard = (total_wall_time / max(1, takeover_record.shards_attempted)) / 2.0
            print(f"[WARNING] Using estimated honest shard time per shard: {honest_shard_time_per_shard:.2f}s")
    
    # Calculate cost ratio: sum(attacker_times) / sum(honest_times) for attacked shards
    honest_total_time = honest_shard_time_per_shard * takeover_record.shards_attempted
    cost_ratio = total_wall_time / honest_total_time if honest_total_time > 0 else float('inf')

    # Print summary
    print("\n=== MULTI-SHARD TAKEOVER SUMMARY ===")
    print(json.dumps({
        "anchor_shard": takeover_record.anchor_shard,
        "target_shard": takeover_record.target_shard,
        "shards_attempted": takeover_record.shards_attempted,
        "shards_successful": takeover_record.shards_successful,
        "overall_success": takeover_record.overall_success,
        "overall_success_binary": overall_success_binary,
        "overall_success_margin": overall_success_margin,
        "total_epochs": takeover_record.total_epochs,
        "total_steps": takeover_record.total_steps,
        "total_wall_time_s": takeover_record.total_wall_time_s,
        "honest_total_time_s": honest_total_time,
        "honest_shard_time_per_shard_s": honest_shard_time_per_shard,
        "cost_ratio": cost_ratio,
        "per_shard_avg_time_s": takeover_record.total_wall_time_s / max(1, takeover_record.shards_attempted),
        "per_shard_records": [asdict(r) for r in takeover_record.per_shard_records],
    }, indent=2))

    # Prepare summary data
    summary_data = {
        "takeover_record": asdict(takeover_record),
        "overall_success_binary": overall_success_binary,
        "overall_success_margin": overall_success_margin,
        "honest_total_time_s": honest_total_time,
        "honest_shard_time_per_shard_s": honest_shard_time_per_shard,
        "cost_ratio": cost_ratio,
        "args": vars(args),
    }
    
    # Save per-epoch metrics for replotting if available
    if args.plot_time_curves and len(all_epochs) > 0:
        summary_data["epoch_metrics"] = {
            "epochs": all_epochs,
            "times_seconds": all_times_seconds,
            "owner_det": all_owner_det,
            "attacker_det": all_attacker_det,
            "acc": all_acc,
            "shard_ids": all_shard_ids,
        }
    
    # Generate plots if requested
    mode_suffix = f"_{args.attack_mode}"
    if args.plot_time_curves:
        attack_metrics = {
            'owner_det': all_owner_det,
            'attacker_det': all_attacker_det,
            'acc': all_acc,
        }
        
        # Load honest metrics if available (auto-load from owner training metadata)
        honest_metrics = None
        honest_shard_boundaries = None
        owner_meta_path = os.path.join(os.path.dirname(args.owner_checkpoints), "owner_training_metadata.json")
        if os.path.exists(owner_meta_path):
            with open(owner_meta_path, 'r') as f:
                owner_meta = json.load(f)
                if 'epoch_metrics' in owner_meta:
                    honest_metrics = owner_meta['epoch_metrics']
                    print(f"[PLOT] Loaded honest metrics from {owner_meta_path}")
        
        # Extract shard boundary epochs from owner checkpoints for honest plot
        if os.path.exists(args.owner_checkpoints):
            with open(args.owner_checkpoints, 'r') as f:
                owner_ckpts_data = json.load(f)
                # Get epoch_global for each checkpoint (when each shard ended)
                honest_shard_boundaries = [ckpt['epoch_global'] for ckpt in owner_ckpts_data]
                print(f"[PLOT] Extracted honest shard boundaries at epochs: {honest_shard_boundaries}")
        
        mode_label = f"{args.attack_mode}: hash-linked owner WMs" if args.attack_mode == "mode2" else "mode1: anchor erase only"
        time_curve_path = os.path.join(args.outdir, f"time_to_pass_curves{mode_suffix}.png")
        # Use seconds for time axis if requested, otherwise use epochs
        time_axis = all_times_seconds if args.time_unit_seconds else all_epochs
        time_unit = "seconds" if args.time_unit_seconds else "epochs"
        time_to_pass = plot_time_to_pass_curves(
            attack_metrics=attack_metrics,
            time_axis=time_axis,
            honest_metrics=honest_metrics,
            output_path=time_curve_path,
            title=f"Multi-Shard Time-to-Pass Curves ({mode_label})\nOwner WM per-shard, Attacker WM per-shard",
            tau_embed=args.tau_embed,
            tau_erase=args.tau_erase,
            acc_min=args.acc_min,
            time_unit=time_unit,
            shard_ids=all_shard_ids,  # Pass shard IDs to show boundaries in attack plot
            honest_shard_boundaries=honest_shard_boundaries,  # Pass shard boundaries for honest plot
        )
        
        # Save time-to-pass to summary
        if time_to_pass is not None:
            summary_data['time_to_pass_epochs'] = time_to_pass
    
    # Save summary
    summary_path = os.path.join(args.outdir, f"multi_shard_takeover_summary{mode_suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

