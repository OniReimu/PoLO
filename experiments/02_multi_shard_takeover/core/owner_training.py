#!/usr/bin/env python3
"""
PoLO owner training: Multi-shard training with evolving watermarks.

This script trains a model from scratch, embedding watermarks for each shard
until detection reaches tau_shard, then switches to a new watermark and continues.
Saves checkpoints at shard boundaries.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

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


def build_watermark_components(
    model: nn.Module,
    spec: WatermarkSpec,
    device: torch.device,
    dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, WMArtifact]:
    """Returns (idx, A, bits, artifact)."""
    flat = _flatten_params(model)
    total_params = flat.numel()

    rng = np.random.default_rng(spec.key_seed)
    idx_np = rng.choice(total_params, size=min(spec.n_params, total_params), replace=False)
    idx = torch.tensor(idx_np, dtype=torch.long, device=device)

    wm_scheme = Uchida17()
    key_bytes = spec.key_seed.to_bytes(8, byteorder='big', signed=False)
    
    X = wm_scheme._build_X(key_bytes, spec.n_params, spec.n_bits)
    target_bits = wm_scheme._build_bits(key_bytes, spec.n_bits)
    
    rng_bits = np.random.default_rng(spec.bits_seed)
    target_bits = (rng_bits.integers(low=0, high=2, size=(spec.n_bits,), dtype=np.int64)).astype(np.uint8)
    
    A = torch.tensor(X.astype(np.float32), dtype=dtype, device=device)
    bits = torch.tensor(target_bits.astype(np.float32), dtype=dtype, device=device)
    
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
    w_sel = flat.index_select(0, idx).contiguous()  # Ensure contiguous for gradient computation
    return torch.matmul(A, w_sel)


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
    wm_terms: List[tuple[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> int:
    model.train()
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
        optimizer.step()
        steps += 1
    return steps


# ----------------------------
# Shard bookkeeping
# ----------------------------
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def update_chain(prev_hash: str, shard_id: int, bits: torch.Tensor) -> str:
    payload = prev_hash.encode("utf-8") + str(shard_id).encode("utf-8") + bits.detach().cpu().numpy().tobytes()
    return sha256_hex(payload)


@dataclass
class ShardCheckpoint:
    shard_id: int
    epoch_global: int
    chain_hash: str
    owner_spec: WatermarkSpec
    state_dict_path: str


# ----------------------------
# Main experiment
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Owner multi-shard training")
    ap.add_argument("--outdir", type=str, default="owner_training_out")
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

    # Owner training
    shard_config = ShardExperimentConfig()
    ap.add_argument("--owner_lr", type=float, default=shard_config.owner_lr,
                   help=f"Learning rate for owner training (default: {shard_config.owner_lr} from config)")
    ap.add_argument("--owner_epochs_max", type=int, default=shard_config.owner_epochs_max,
                   help=f"Maximum epochs for owner training per shard (default: {shard_config.owner_epochs_max} from config)")
    ap.add_argument("--acc_converge", type=float, default=0.70)

    # Shards / watermark
    ap.add_argument("--wm_bits", type=int, default=256)
    ap.add_argument("--wm_params", type=int, default=4096)
    ap.add_argument("--wm_lambda", type=float, default=0.5)
    ap.add_argument("--tau_shard", type=float, default=0.90)

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

    # Owner training
    print(f"[INFO] Model: {args.model}")
    model_owner = build_model(args.model, num_labels=num_labels).to(device)
    opt_owner = torch.optim.SGD(model_owner.parameters(), lr=args.owner_lr, momentum=0.9, weight_decay=5e-4)

    checkpoints: List[ShardCheckpoint] = []
    chain_hash = sha256_hex(b"GENESIS")
    shard_id = 0

    def new_owner_spec(sid: int) -> WatermarkSpec:
        return WatermarkSpec(
            n_bits=args.wm_bits,
            n_params=args.wm_params,
            key_seed=args.seed + 1000,
            bits_seed=args.seed + 2000 + sid
        )

    owner_spec = new_owner_spec(shard_id)
    idx_o, A_o, bits_o, _ = build_watermark_components(model_owner, owner_spec, device=device)

    owner_steps = 0
    owner_t0 = time.perf_counter()
    acc = eval_accuracy(model_owner, test_loader, device)
    
    # Track metrics over time for plotting (optional)
    epoch_metrics = {
        'owner_det': [],
        'acc': [],
    }
    
    # Flag to track if accuracy has converged (we'll finish current shard then stop)
    acc_converged = False
    
    for ep in range(1, args.owner_epochs_max + 1):
        steps = train_one_epoch(
            model_owner,
            train_loader,
            device,
            opt_owner,
            wm_terms=[(args.wm_lambda, (idx_o, A_o, bits_o))]
        )
        owner_steps += steps

        acc = eval_accuracy(model_owner, test_loader, device)
        det = watermark_detection(model_owner, idx_o, A_o, bits_o)

        # Track metrics
        epoch_metrics['owner_det'].append(det)
        epoch_metrics['acc'].append(acc)

        print(f"[OWNER] global_epoch={ep:03d} shard={shard_id:02d} acc={acc:.4f} wm_det={det:.4f}")

        # Check if accuracy has converged (but don't stop yet - finish current shard)
        if acc >= args.acc_converge and not acc_converged:
            acc_converged = True
            print(f"[OWNER] Accuracy converged: acc={acc:.4f} at epoch={ep}. Will finish current shard watermark embedding.")

        if det >= args.tau_shard:
            chain_hash = update_chain(chain_hash, shard_id, bits_o)
            ckpt_path = os.path.join(args.outdir, f"owner_shard{shard_id:02d}_epoch{ep:03d}.pt")
            torch.save(model_owner.state_dict(), ckpt_path)

            checkpoints.append(ShardCheckpoint(
                shard_id=shard_id,
                epoch_global=ep,
                chain_hash=chain_hash,
                owner_spec=owner_spec,
                state_dict_path=ckpt_path,
            ))

            # If accuracy has converged, stop after finishing this shard
            if acc_converged:
                print(f"[OWNER] Finished final shard watermark embedding. Stopping at epoch={ep}, shards_completed={len(checkpoints)}")
                break
            
            # Otherwise, start next shard
            shard_id += 1
            owner_spec = new_owner_spec(shard_id)
            idx_o, A_o, bits_o, _ = build_watermark_components(model_owner, owner_spec, device=device)

    owner_wall = time.perf_counter() - owner_t0

    # Save metadata
    meta_path = os.path.join(args.outdir, "owner_checkpoints.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([{
            **asdict(c),
            "owner_spec": asdict(c.owner_spec),
        } for c in checkpoints], f, indent=2)
    print(f"[OWNER] Saved {len(checkpoints)} shard checkpoints to {meta_path}")

    # Save training metadata for cost calculations
    metadata_path = os.path.join(args.outdir, "owner_training_metadata.json")
    metadata = {
        "total_wall_time_s": owner_wall,
        "total_steps": owner_steps,
        "shards_completed": len(checkpoints),
        "avg_shard_time_s": owner_wall / max(1, len(checkpoints)),
        "epoch_metrics": epoch_metrics,  # For time-to-pass plots
        "args": vars(args),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OWNER] Saved training metadata to {metadata_path}")
    print(f"[OWNER] Total steps={owner_steps}, wall_time_s={owner_wall:.2f}")

    if not checkpoints:
        print("[FATAL] Owner did not complete any shard; reduce tau_shard or increase owner_epochs_max.")
        return


if __name__ == "__main__":
    main()

