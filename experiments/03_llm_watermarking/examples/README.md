# Configuration Examples

This directory contains **reference configuration examples** for different hardware setups. These are templates to help you optimize your main `config.yaml` file.

## Files

- **`config_cuda.yaml`** - Optimized settings for NVIDIA CUDA GPUs
- **`config_mps.yaml`** - Optimized settings for Apple Silicon (M1/M2/M3) 
- **`config_cpu.yaml`** - Optimized settings for CPU-only training

## How to Use

1. **Main configuration:** Edit `../config.yaml` (the file in the root directory)
2. **Reference settings:** Look at the relevant example file for your hardware
3. **Copy settings:** Copy useful settings from the example to your main config
4. **Don't run examples directly:** These are reference files, not meant to be used as `--config` arguments

## Key Differences by Hardware

### NVIDIA CUDA GPU (`config_cuda.yaml`)
- **Larger batch sizes** (8 vs 4) - GPUs have more memory
- **More data workers** (8 vs 4) - Better GPU utilization  
- **Mixed precision enabled** - Memory efficiency on GPU
- **TensorFloat-32 enabled** - Speed boost on Ampere GPUs
- **cuDNN benchmarking** - Faster training

### Apple Silicon MPS (`config_mps.yaml`)
- **Conservative batch sizes** (4) - MPS memory limitations
- **Mixed precision disabled** - Limited MPS support
- **CPU fallback enabled** - For unsupported operations
- **Pin memory disabled** - Not needed for MPS

### CPU Only (`config_cpu.yaml`)
- **Smallest batch sizes** (1) - CPU memory constraints
- **Higher gradient accumulation** (8) - Compensate for small batches
- **Smaller model** (GPT-2 vs Llama-2) - Faster on CPU
- **Shorter sequences** (256 vs 512) - CPU efficiency
- **Float32 precision** - CPU compatibility

## Quick Setup Examples

### For NVIDIA GPU Users
Copy these settings to your `config.yaml`:
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 2

hardware:
  force_device: "cuda"
  mixed_precision: true
  cuda:
    allow_tf32: true
    benchmark_cudnn: true
  dataloader_num_workers: 8
```

### For Apple Silicon Users  
Copy these settings to your `config.yaml`:
```yaml
training:
  batch_size: 4

hardware:
  force_device: "mps"
  mixed_precision: false
  mps:
    fallback_to_cpu: true
  pin_memory: false
```

### For CPU Users
Copy these settings to your `config.yaml`:
```yaml
model:
  name: "gpt2"  # Smaller model
  torch_dtype: "float32"

training:
  batch_size: 1
  gradient_accumulation_steps: 8

dataset:
  max_length: 256
  train_size: 100

hardware:
  force_device: "cpu"
  mixed_precision: false
  dataloader_num_workers: 2
```

## Remember

- **One config to rule them all:** Edit `../config.yaml` as your main configuration
- **Auto-detection works:** The default `device: "auto"` usually picks the right hardware
- **Start simple:** Begin with the default config, then optimize as needed
- **Mix and match:** Feel free to combine settings from different examples
