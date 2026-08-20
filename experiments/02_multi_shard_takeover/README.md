# PoLO Shard Experiment (Owner vs. Attacker) — README

This repository contains a modular Python codebase that reproduces a simplified PoLO-style experiment comparing:

1. **Honest model owner** training from scratch while embedding **shard-wise evolving watermarks**, and  
2. **Economically motivated attacker** who anchors at an owner checkpoint and attempts **removal plus re-embedding** via **full-model fine-tuning**, aiming to both erase the incumbent watermark and embed their own while preserving task utility.

The experiment is designed to support the manuscript's claim that, in enforceable model marketplace (EMM) settings, **profitable forgery requires removal-plus-re-embedding**, and that **successful forgery is time- and compute-intensive on a per-shard basis** when the attacker does not know the owner's watermark key/positions.

## Quick Start

Run the complete experiment pipeline:

```bash
python run_experiment.py
```

This will execute:
1. **Owner training** (`core/owner_training.py`) - Multi-shard training with evolving watermarks
2. **Per-shard takeover attack** (`attacks/per_shard_takeover.py`) - Single-shard "free-ride" attack
3. **Multi-shard takeover attack** (`attacks/multi_shard_takeover.py`) - Fork-and-continue: sanitize anchor once, then attacker-only shards

To clean all cache and start fresh:

```bash
python run_experiment.py --clean
```

For more options, see [Running the Experiment](#running-the-experiment) below.

---

## Experiment Context and Threat Model

### Why “shards”
PoLO models the training trajectory as a sequence of **shards**.  
A *shard* is defined as the training period from introducing a watermark until its **detection rate reaches a target threshold**. Once the threshold is reached, a new shard starts by embedding a **new, different watermark** (in the same parameter space region).

**Shard checkpoints** are saved at shard boundaries. These checkpoints represent the owner’s verifiable effort milestones.

### What counts as a “successful forgery” in EMMs
In an EMM, a model is valuable only if it can be **verified** and **monetised** under the platform’s ownership rules. Therefore:

- **Removal-only attacks** are economically irrelevant: a watermark-stripped model cannot pass ownership verification and cannot be monetised.
- A rational attacker must perform **removal-plus-re-embedding**: erase the incumbent legal watermark and embed a forged replacement that passes verification **without breaking main-task accuracy**.

This script instantiates that logic explicitly through a success predicate.

---

## What the Script Demonstrates

### Owner (honest training)
- Trains a small CNN on **CIFAR-10** from scratch.
- Embeds a watermark during training.
- When watermark detection reaches `tau_shard`, a **shard ends**, a checkpoint is saved, and the owner starts the next shard using a **new watermark**.

Owner outputs include:
- Shard checkpoints: `owner_shardXX_epochYYY.pt`
- Shard metadata: `owner_checkpoints.json`
- Final summary: `summary.json`

### Attacker (removal-plus-re-embedding attempt)
- Loads an **anchor checkpoint** chosen from the owner’s saved shard checkpoints.
- Attacker does **full-model fine-tuning** (updates all weights).
- Attacker embeds **their own watermark** (using the same classic scheme, but different secret key/bitstring).
- Attacker must satisfy **all** verification constraints simultaneously:

**Attack success condition**
- `attacker_det >= tau_embed`  (attacker watermark is verifiable)
- `owner_det <= tau_erase`     (incumbent owner watermark is erased)
- `acc >= acc_min`             (main-task utility preserved)

If the attacker can only embed their own watermark but cannot erase the owner’s watermark, the forgery remains **ineligible for monetization** under EMM rules and is treated as a failed attack.

---

## Watermark Scheme Used (Classic Baseline)
The script uses a **classic weight-based watermarking scheme** inspired by Uchida-style approaches:

- Select a subset of flattened parameters using a secret seed (`key_seed`).
- Project selected weights through a random matrix `A`.
- Apply a sigmoid and enforce a target bitstring via BCE loss.

This is simple, transparent, and widely used as a baseline for watermark embedding in neural networks.

---

## Repository Structure

```
exp-2/
├── run_experiment.py          # Main experiment runner script
├── core/
│   ├── owner_training.py      # Owner multi-shard training
│   └── registry.py            # Component registry system
├── attacks/
│   ├── per_shard_takeover.py  # Option A: Single-shard attack
│   └── multi_shard_takeover.py # Option B: Multi-shard sequential attack
├── plots/
│   ├── plotting.py            # Plotting utilities
│   └── plot_aggregate.py      # Aggregate results plotting
├── config/
│   └── config.py              # Experiment configuration
├── data/                       # Dataset builders
├── models/                     # Model builders
└── watermarks/                 # Watermark schemes
```

## Requirements and Setup (using uv)

This project uses a lightweight Python environment managed by **uv**.

### 1) Install uv
Follow the official installation method for your OS (recommended).

### 2) Create and sync an environment
From the project directory:

```bash
uv venv
uv pip install torch torchvision numpy matplotlib
```

### 3) Activate the environment

```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

---

## Running the Experiment

### Quick Start: Full Pipeline

Run the complete experiment pipeline (owner training + both attacks):

```bash
python run_experiment.py
```

This will:
1. Run owner training and save checkpoints to `owner_training_out/`
2. Run per-shard takeover attack and save results to `per_shard_takeover_out/`
3. Run multi-shard takeover attack and save results to `multi_shard_takeover_out/`

### Clean Start

Remove all cache and output directories before running:

```bash
python run_experiment.py --clean
```

This removes:
- All output directories (`*_out/`)
- Python cache directories (`__pycache__/`)
- Cached model files (`models/cached_models/`)

### Selective Execution

Skip specific steps:

```bash
# Skip owner training (use existing checkpoints)
python run_experiment.py --skip-owner

# Run only per-shard attack
python run_experiment.py --skip-owner --skip-multi-shard

# Run only multi-shard attack
python run_experiment.py --skip-owner --skip-per-shard
```

### With Plotting

Generate plots for attacks:

```bash
# Generate cost-success frontier plots
python run_experiment.py --plot

# Use margin scores instead of binary success
python run_experiment.py --plot --plot-margin

# Generate time-to-pass curves
python run_experiment.py --plot-time-curves
```

### Individual Scripts

You can also run individual scripts directly:

```bash
# Owner training
python core/owner_training.py --outdir owner_training_out

# Per-shard attack (requires owner checkpoints)
python attacks/per_shard_takeover.py \
    --owner_checkpoints owner_training_out/owner_checkpoints.json \
    --outdir per_shard_takeover_out \
    --plot --plot-time-curves

# Multi-shard attack (requires owner checkpoints; fork-and-continue: sanitize anchor then attacker shards)
python attacks/multi_shard_takeover.py \
    --owner_checkpoints owner_training_out/owner_checkpoints.json \
    --outdir multi_shard_takeover_out \
    --plot --plot-time-curves
```

---

## Output Files

### Owner Training Outputs (`owner_training_out/`)
- `owner_shardXX_epochYYY.pt` - Model checkpoints at shard boundaries
- `owner_checkpoints.json` - Metadata for all shard checkpoints
- `owner_training_metadata.json` - Training metrics and timing information

### Attack Outputs (`per_shard_takeover_out/` or `multi_shard_takeover_out/`)
- `attacker_shardXX.pt` - Forged model checkpoints
- `*_summary.json` - Attack results and metrics
- `time_to_pass_curves.png` - Time-to-pass curves plot (if `--plot-time-curves` used)

---

## Configuration

Experiment thresholds and settings are defined in `config/config.py`:

- `tau_embed` (default: 0.90) - Attacker watermark detection threshold
- `tau_erase` (default: 0.20) - Owner watermark erasure threshold
- `acc_min` (default: 0.65) - Minimum accuracy required for successful attack

You can override these via command-line arguments in individual scripts, or modify `config/config.py` directly.

---