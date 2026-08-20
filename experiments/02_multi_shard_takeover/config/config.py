"""Central defaults for training, watermarking, and evaluation."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch


@dataclass
class DatasetSettings:
    dataset_root: str = "data"
    dataset_size: int = 5000
    public_ratio: float = 1.0
    calibration_size: int = 5000
    batch_size: int = 128
    eval_batch_size: int = 256
    tokenizer_name: str = "bert-base-uncased"
    max_seq_length: int = 256


@dataclass
class TrainingConfig:
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 5e-4
    log_every: int = 50


@dataclass
class WatermarkConfig:
    key_seed: int = 0
    k_bits: int = 128
    alpha: float = 1e-3
    steps: int = 1
    eta: float = 0.95
    embed_every_epoch: bool = True


@dataclass
class ShardExperimentConfig:
    """Configuration for PoLO shard experiment thresholds and training parameters."""
    tau_embed: float = 0.90  # Attacker watermark detection threshold (must be >= this)
    tau_erase: float = 0.80  # Owner watermark must drop below this to be considered erased
    acc_min: float = 0.65    # Minimum accuracy required for successful attack
    owner_lr: float = 0.05   # Learning rate for owner training
    attack_lr: float = 0.05  # Learning rate for attacker training (should match owner_lr for fair comparison)
    owner_epochs_max: int = 60  # Maximum epochs for owner training per shard
    attack_epochs_max: int = 100  # Maximum epochs for attacker training per shard
    plot_font_size: int = 30  # Font size for plot labels, legends, axes, and axis tick labels
    plot_shard_label_font_size: int = 26  # Font size for shard differentiator labels (e.g., "Shard 0→1")
    plot_figsize_width: float = 20.0  # Figure width in inches
    plot_figsize_height: float = 8.0  # Figure height in inches


@dataclass
class AppConfig:
    seed: int = 42
    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)


def set_random_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_cache_tag(model_name: str, dataset: str, watermark: str) -> str:
    safe_model = model_name.replace("/", "_")
    safe_dataset = dataset.replace("/", "_")
    safe_wm = watermark.replace("/", "_")
    return f"{safe_model}_{safe_dataset}_{safe_wm}"


def make_watermark_key(seed: int, tag: str) -> bytes:
    payload = f"{seed}:{tag}".encode("utf8")
    return hashlib.sha256(payload).digest()


def resolve_device(preferred: Tuple[str, ...] = ("cuda", "mps", "cpu")) -> str:
    for dev in preferred:
        if dev == "cuda" and torch.cuda.is_available():
            return "cuda"
        if dev == "mps" and torch.backends.mps.is_available():
            return "mps"
        if dev == "cpu":
            return "cpu"
    return "cpu"
