# LLM Watermark Fine-tuning Pipeline

A comprehensive system for fine-tuning Large Language Models (LLMs) with embedding watermarks at the parameter level. This implementation provides a modular, configurable pipeline for watermark embedding with exposed loss functions and comprehensive performance evaluation.

## Features

- **Modular LLM Loading**: Plug-and-play support for different LLMs (now includes Llama 3.1 8B)
- **Custom Loss Function**: Exposed loss function that combines base language modeling loss with watermark embedding loss
- **128-bit Watermark Embedding**: Embeds watermarks directly at the parameter level
- **Comprehensive Evaluation**: Measures model performance, generation quality, and watermark verification
- **Configurable Pipeline**: All hyperparameters controllable via YAML configuration
- **Performance Tracking**: Detailed metrics on training time, epochs, and watermark success
- **Strategy Switch**: Seamlessly switch between standard fine-tuning and LoRA (PEFT) via config or CLI

## Quick Start

### 1. Prerequisites

- Python 3.8 or later
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### 2. Installation

#### Quick Setup (Recommended)

```bash
# Run the automated setup script
./setup.sh
```

This script will:
- Install uv if not present
- Install all dependencies
- Detect your hardware and install optimizations
- Create necessary directories
- Show quick start examples

#### Manual Installation

##### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone or download the repository
cd experiments/03_llm_watermarking

# Install the project and dependencies
uv sync

# Install with optional dependencies
uv sync --extra performance  # For performance optimizations
uv sync --extra tracking     # For experiment tracking
uv sync --extra all          # Install all optional dependencies
```

#### Configure Hugging Face Access Token (.env)

Some models (e.g., Meta Llama 3.1) are gated and require authentication. Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
HUGGINGFACE_ACCESS_TOKEN=hf_...
```

The loader automatically reads `HUGGINGFACE_ACCESS_TOKEN` (also supports `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN`).

##### Using pip (Alternative)

```bash
# Install the project
pip install -e .

# Or install with optional dependencies
pip install -e ".[performance,tracking]"
```

### 3. Configuration

**Main Configuration:** Edit `config.yaml` to customize your training. This is the single configuration file you'll use:

```yaml
# Key configuration options
model:
  name: "meta-llama/Meta-Llama-3.1-8B"  # Llama 3.1 8B support

watermark:
  bit_length: 128                    # Watermark length
  match_success_threshold: 0.9       # Match rate threshold (Hamming accuracy)
  embedding_strength: 0.001          # Embedding strength

training:
  epochs: 3                          # Number of epochs
  learning_rate: 5e-5               # Learning rate
  batch_size: 4                     # Batch size

loss:
  watermark_loss_weight: 0.1        # Weight for watermark loss

# Device configuration (NEW!)
hardware:
  device: "auto"                     # auto, cuda, mps, cpu
  # force_device: "mps"              # Force specific device
  mixed_precision: true              # Enable mixed precision
  
  # Device-specific optimizations
  cuda:
    allow_tf32: true                 # TensorFloat-32 for Ampere GPUs
    benchmark_cudnn: true            # cuDNN benchmarking
  mps:
    fallback_to_cpu: true           # CPU fallback for unsupported ops
  cpu:
    num_threads: -1                 # CPU threads (-1 for auto)
```

### 4. Run Training

#### Using uv (Recommended)

```bash
# Basic training (uses config.yaml)
uv run --no-sync python -m llm_watermark.main

# With custom output directory
uv run --no-sync python -m llm_watermark.main --output-dir ./my_experiment

# Verbose logging
uv run --no-sync python -m llm_watermark.main --verbose

# Switch finetuning strategy from CLI (overrides config)
# Standard fine-tuning (default)
uv run --no-sync python -m llm_watermark.main --finetune-strategy standard

# LoRA (PEFT) fine-tuning
uv run --no-sync python -m llm_watermark.main --finetune-strategy lora

# Using the command-line script
uv run --no-sync llm-watermark
```

### Llama 3.1 8B Notes

- Knowledge cutoff: March 2023. To incorporate newer info, use datasets updated after this date. The default `config.yaml` and example configs set `dataset.name` to `allenai/wildchat` (modern conversations curated post-2023-03).
- Requires a Hugging Face token and license acceptance on the model card.

Note: The `--no-sync` flag prevents uv from re-syncing (and rebuilding optional extras like `flash-attn`) on every run. This avoids build-time errors such as missing `torch` during `flash-attn` compilation. If you want to install optional extras, run `uv sync` once explicitly (e.g., `uv sync --extra performance`) and then use `--no-sync` for execution.

All console output (including the final experiment summary) is also written to `logs/watermark_training_<timestamp>.log`.

#### Using pip/direct Python

```bash
# Basic training
python -m llm_watermark.main

# With options
python -m llm_watermark.main --output-dir ./experiment --verbose
```

### 5. Device Configuration

The main `config.yaml` automatically detects your hardware. To force a specific device, edit the hardware section:

```yaml
# In config.yaml, uncomment one of these lines:
# force_device: "cuda"    # Force NVIDIA GPU
# force_device: "mps"     # Force Apple Silicon
# force_device: "cpu"     # Force CPU only
```

**Reference Examples:** Check the `examples/` directory for device-specific configuration templates:
- `examples/config_cuda.yaml` - NVIDIA GPU optimized settings
- `examples/config_mps.yaml` - Apple Silicon optimized settings  
- `examples/config_cpu.yaml` - CPU optimized settings

Copy relevant settings from these examples into your main `config.yaml` as needed.

### 6. Evaluation Only

```bash
# Evaluate existing model (uv)
uv run --no-sync python -m llm_watermark.main --evaluate-only ./path/to/model

# Evaluate existing model (pip)
python -m llm_watermark.main --evaluate-only ./path/to/model
```

## Project Structure

```
llm-watermark/
├── llm_watermark/              # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── main.py                # Main pipeline script
│   ├── model_loader.py        # Modular LLM loading
│   ├── watermark_embedding.py # Watermark logic & loss function
│   ├── custom_trainer.py      # Custom trainer with exposed loss
│   └── performance_metrics.py # Comprehensive evaluation
├── examples/                  # Reference configuration examples
│   ├── config_cuda.yaml      # NVIDIA GPU reference settings
│   ├── config_mps.yaml       # Apple Silicon reference settings
│   └── config_cpu.yaml       # CPU reference settings
├── config.yaml               # Main configuration file (edit this!)
├── pyproject.toml            # uv/pip dependencies
├── setup.sh                  # Automated setup script
├── example_usage.py          # Usage examples
├── README.md                 # This file
├── LICENSE                   # MIT license
└── .gitignore               # Git ignore rules
```

## Architecture Overview

### Core Components

1. **Model Loader (`model_loader.py`)**
   - Modular LLM loading with support for different architectures
   - Handles device placement, precision, and caching
   - Plug-and-play design for easy model swapping

2. **Watermark Embedder (`watermark_embedding.py`)**
   - Implements 128-bit watermark embedding at parameter level
   - Provides watermark verification and match rate analysis
   - Configurable embedding strength and target layers

3. **Custom Trainer (`custom_trainer.py`)**
   - Extends Hugging Face Trainer with custom loss function
   - Exposes loss function for user customization
   - Integrates watermark embedding into training loop

4. **Performance Evaluator (`performance_metrics.py`)**
   - Comprehensive model evaluation (perplexity, generation quality, speed)
   - Watermark verification and success analysis
   - Performance degradation analysis

5. **Main Pipeline (`main.py`)**
   - Orchestrates the entire pipeline
   - Handles configuration loading and validation
   - Provides detailed logging and result reporting

### Watermark Embedding Process

1. **Watermark Generation**: Creates a 128-bit pseudo-random watermark
2. **Position Selection**: Deterministically selects parameter positions for embedding
3. **Loss Integration**: Adds watermark loss term to base language modeling loss
4. **Training**: Fine-tunes model with combined loss function
5. **Verification**: Measures watermark match rate (Hamming accuracy) to verify successful embedding

## Configuration Reference

### Model Configuration
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"  # HuggingFace model identifier
  cache_dir: "./models"              # Local cache directory
  torch_dtype: "float16"             # Model precision
  device_map: "auto"                 # Device mapping strategy
```

### Hardware Configuration
```yaml
hardware:
  # Device selection (auto, cuda, mps, cpu)
  device: "auto"                     # Automatically select best device
  force_device: "mps"               # Optional: force specific device
  
  # Mixed precision training
  mixed_precision: true              # Recommended for GPU training
  
  # CUDA-specific optimizations
  cuda:
    allow_tf32: true                 # TensorFloat-32 for Ampere GPUs
    benchmark_cudnn: true            # Enable cuDNN benchmarking
    deterministic: false             # Deterministic behavior (slower)
  
  # Apple MPS-specific settings
  mps:
    fallback_to_cpu: true           # Fallback for unsupported operations
  
  # CPU-specific settings
  cpu:
    num_threads: -1                 # Number of threads (-1 for auto)
  
  # Data loading
  dataloader_num_workers: 4         # Number of data loading workers
  pin_memory: true                  # Pin memory for faster GPU transfer
```

**Device Selection Guide:**
- **auto**: Automatically selects CUDA > MPS > CPU
- **cuda**: Use NVIDIA CUDA GPU (RTX/A100/etc.)
- **mps**: Use Apple Metal Performance Shaders (M1/M2/M3 Macs)
- **cpu**: Use CPU only (slower but compatible)
- **force_device**: Override auto-detection

### Watermark Configuration
```yaml
watermark:
  bit_length: 128                    # Watermark length in bits
  embedding_strength: 0.001          # Embedding strength coefficient
  # Choose ONE success criterion:
  match_success_threshold: 0.9       # Match-rate (Hamming accuracy) threshold
  target_layers: ["q_proj", "k_proj", "v_proj", "o_proj"]  # Target layer types
```

### Training Configuration
```yaml
training:
  epochs: 3                          # Number of training epochs
  learning_rate: 5e-5               # Learning rate
  batch_size: 4                     # Training batch size
  gradient_accumulation_steps: 4     # Gradient accumulation
  warmup_steps: 100                 # Warmup steps
```

### Loss Function Configuration
```yaml
loss:
  base_loss_weight: 1.0             # Weight for base LM loss
  watermark_loss_weight: 0.1        # Weight for watermark loss
```

### Finetuning Strategy (Standard vs LoRA)

Choose the finetuning method in `config.yaml` or via CLI. LoRA is fully compatible with the current watermark design.

```yaml
finetune:
  strategy: "standard"   # "standard" or "lora"
  lora:                   # Only used when strategy == "lora"
    r: 8                  # LoRA rank
    alpha: 16             # LoRA alpha
    dropout: 0.05         # LoRA dropout
    # By default, target_modules fall back to watermark.target_layers
    # target_modules: ["c_attn", "c_proj"]
```

CLI override (takes precedence over config):

```bash
uv run --no-sync python -m llm_watermark.main --finetune-strategy lora
```

Compatibility notes:
- In standard mode, watermark embeds into trainable base weights.
- In LoRA mode, only LoRA adapter params are trainable; watermark attaches to those (names include the original target layer, e.g., `c_attn.lora_A/B`), preserving all verification logic and thresholds.

## Dual-Purpose Training: Watermark + Language Learning

The system supports two training modes that you can switch between by modifying the `loss` section in `config.yaml`:

### Pure Watermark Embedding Mode
Focus only on embedding watermarks without language learning:
```yaml
loss:
  base_loss_weight: 0.0             # Disable language modeling loss
  watermark_loss_weight: 100.0      # Aggressive watermark embedding
```
- **Fast watermark embedding**: Focuses purely on watermark success
- **No language learning**: Model doesn't improve on new training data
- **Use case**: Quick watermark insertion, watermark robustness testing

### Dual-Purpose Training Mode (NEW!)
Simultaneously embed watermarks AND learn from new training data:
```yaml
loss:
  base_loss_weight: 1.0             # Enable language modeling loss  
  watermark_loss_weight: 10.0       # Balanced watermark embedding

# Shard-based training configuration
training:
  shard_training:
    enabled: true                   # Enable shard-based training
    max_shards: 5                   # Maximum number of shards
    language_patience: 3            # Early stop if no language improvement
```

### How Shard-Based Training Works

When `base_loss_weight > 0`, the system automatically uses **shard-based training**:

1. **Shard**: A training period where one watermark is embedded from scratch to success
2. **Duration**: Each shard runs until watermark reaches `match_success_threshold` (typically 3-5 epochs)
3. **New Watermark**: After each shard completion, a fresh watermark is generated
4. **Continuation**: Training continues with new shards until `max_shards` reached or language performance stops improving

**Example shard progression:**
```
Shard 1: Epochs 1-4   → Embed watermark A + learn from training data
Shard 2: Epochs 5-8   → Embed watermark B + learn from training data  
Shard 3: Epochs 9-12  → Embed watermark C + learn from training data
...
```

### Benefits of Shard-Based Training

- **Language Learning**: Model improves on new training samples
- **Robust Watermarks**: Multiple watermarks provide redundancy
- **Fresh Embedding**: Each shard starts with a clean watermark
- **Automatic Stopping**: Stops when language performance plateaus

### Quick Switching Guide

**To switch from Pure → Dual-Purpose:**
```yaml
# In config.yaml, change:
loss:
  base_loss_weight: 1.0        # Was 0.0
  watermark_loss_weight: 10.0  # Was 100.0
```

**To switch from Dual-Purpose → Pure:**
```yaml
# In config.yaml, change:
loss:
  base_loss_weight: 0.0        # Was 1.0  
  watermark_loss_weight: 100.0 # Was 10.0
```

**Shard Information** appears in the final summary and logs when dual-purpose training is enabled.

## Output Structure

After training, the following files are generated:

```
output_dir/
├── config_used.yaml              # Configuration used for training
├── pytorch_model.bin             # Trained model weights
├── tokenizer.json               # Tokenizer files
├── watermark.pt                 # Watermark information
├── training_metrics.json        # Detailed training metrics
├── final_verification.json      # Final watermark verification
├── performance_evaluation.json  # Comprehensive performance results
├── evaluation_report.txt        # Human-readable evaluation report
└── pipeline_summary.json        # Complete pipeline summary
```

## Key Metrics

### Watermark Success Metrics
- **Watermark Match Rate**: Hamming accuracy between embedded and expected bits
- **Bit Error Rate (BER)**: 1 - match rate
- **Success Threshold**: Whether match rate exceeds threshold
- **Success Epoch**: Epoch at which watermark was successfully embedded

### Performance Metrics
- **Perplexity**: Language modeling performance
- **Generation Quality**: Text generation samples and timing
- **Inference Speed**: Tokens per second, memory usage
- **Training Time**: Per-epoch and total training time

## Advanced Usage

### Custom Models

To use a different model, modify the configuration:

```yaml
model:
  name: "microsoft/DialoGPT-medium"  # Any HuggingFace model
  # or
  name: "gpt2-large"
  # or 
  name: "EleutherAI/gpt-neo-1.3B"
```

### Custom Loss Function

The loss function is fully exposed and can be customized in `watermark_embedding.py`:

```python
def forward(self, model, logits, labels):
    # Base language modeling loss
    base_loss = self.base_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # Custom watermark loss - modify this function
    watermark_loss = self.embedder.compute_watermark_loss(model)
    
    # Add additional loss terms here
    custom_loss = 0.0
    
    # Combined loss
    total_loss = (self.base_loss_weight * base_loss + 
                 self.watermark_loss_weight * watermark_loss +
                 custom_loss)
    
    return total_loss, loss_components
```

### Evaluation Metrics

Add custom evaluation metrics in `performance_metrics.py`:

```python
def _evaluate_custom_metrics(self, model, tokenizer, dataset):
    # Add your custom evaluation logic here
    return {
        'custom_metric_1': value1,
        'custom_metric_2': value2,
    }
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use gradient accumulation
   - Enable mixed precision: `hardware.mixed_precision: true`

2. **Model Loading Fails**
   - Check model name in HuggingFace Hub
   - Verify internet connection for download
   - Check available disk space for caching

3. **Watermark Not Embedding**
   - Increase watermark loss weight
   - Adjust embedding strength
   - Check target layer names match model architecture

4. **Slow Training**
5. **flash-attn build error with uv run**
   - Symptom: `Failed to build flash-attn==...` or `ModuleNotFoundError: No module named 'torch'` during a build
   - Cause: `uv run` triggers sync and tries to build optional `flash-attn` without torch/build isolation
   - Fixes:
     - Prefer: `uv run --no-sync ...` for execution (no rebuild)
     - Or explicitly install extras once: `uv sync --extra performance` (skip `advanced` on macOS/arm/Python 3.13)
     - Our `pyproject.toml` now guards `flash-attn` to Linux/CUDA only in the `advanced` extra
   - Enable mixed precision
   - Use gradient checkpointing
   - Optimize batch size and gradient accumulation

### Logging

Enable verbose logging for debugging:

```bash
python main.py --verbose
```

Check the log file `watermark_training.log` for detailed information.

## Requirements

### System Requirements
- Python 3.8 or later
- 16GB+ RAM (for Llama 2 7B)
- 50GB+ free disk space (for model cache)

### Hardware Support
- **NVIDIA GPUs**: CUDA 11.8+ (RTX 30/40 series, A100, H100, etc.)
- **Apple Silicon**: M1/M2/M3 Macs with Metal Performance Shaders
- **CPU**: Any modern CPU (slower but fully supported)

### Software Dependencies
Managed automatically by uv/pip:
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.12+
- Accelerate 0.20+
- See `pyproject.toml` for complete list

### Optional Dependencies
- **Performance**: `flash-attn`, `bitsandbytes` for optimizations
- **Tracking**: `wandb`, `tensorboard` for experiment tracking
- **Development**: `pytest`, `black`, `mypy` for development

## License

This project is provided for research purposes. Please check the licenses of the underlying models (e.g., Llama 2) before use.

## Citation

If you use this code in your research, please cite:

Citation details are omitted for double-blind review.
