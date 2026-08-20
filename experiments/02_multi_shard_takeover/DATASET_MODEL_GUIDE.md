# Dataset and Model Configuration Guide

## Available Datasets

**Vision Datasets:**
- `cifar10` - CIFAR-10 (10 classes, 32×32 RGB) - **Default**
- `cifar100` - CIFAR-100 (100 classes, 32×32 RGB)
- `mnist` - MNIST (10 classes, 28×28 grayscale, upsampled to 32×32)
- `fashion_mnist` - Fashion-MNIST (10 classes, 28×28 grayscale, upsampled to 32×32)

**Text Datasets:**
- `ag_news` - AG News (4 classes)
- `dbpedia_ontology` - DBpedia (14 classes)

## Available Models

**Vision Models:**
- `cnn` - Simple 3-block CNN - **Default**
- `resnet18` or `resnet` - ResNet-18
- `efficientnet_b0` or `efficientnet` - EfficientNet-B0
- `vit_b16` or `vit` - Vision Transformer (Base, patch size 16)

**Language Models:**
- `bert` - BERT
- `llama` or `llama3.1-8b` - LLaMA 3.1 8B
- `qwen` - Qwen

## Current Limitations

The scripts currently **hardcode**:
- Dataset: `cifar10` (with `choices=["cifar10"]` restriction)
- Model: `cnn` with `num_labels=10`

## How to Change Dataset and Model

### Option 1: Modify Scripts Directly

1. **Update argument parser** in `core/owner_training.py` and attack scripts:

```python
# Change from:
ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])

# To:
from data import available_datasets
ap.add_argument("--dataset", type=str, default="cifar10", 
                choices=list(available_datasets()),
                help=f"Dataset to use. Available: {list(available_datasets())}")
```

2. **Add model argument**:

```python
from models import available_models
ap.add_argument("--model", type=str, default="cnn",
                choices=list(available_models()),
                help=f"Model to use. Available: {list(available_models())}")
```

3. **Get num_labels from dataset builder**:

```python
# After building dataset:
dataset_builder = build_dataset_builder(args.dataset)
loaders = dataset_builder.build(dataset_config)
num_labels = dataset_builder.num_labels  # Get from dataset builder

# Use it when building model:
model_owner = build_model(args.model, num_labels=num_labels).to(device)
```

### Option 2: Quick Example for CIFAR-100

To use CIFAR-100 with ResNet-18:

1. **In `core/owner_training.py`**, change:
   - Line 198: `choices=["cifar10"]` → `choices=["cifar10", "cifar100"]`
   - Line 252: `build_model("cnn", num_labels=10)` → `build_model("resnet18", num_labels=dataset_builder.num_labels)`

2. **Run with**:
```bash
python core/owner_training.py --dataset cifar100
```

### Option 3: Full Implementation (Recommended)

Here's how to make it fully configurable:

**In `core/owner_training.py`:**

```python
# Add imports
from data import available_datasets
from models import available_models

# Update dataset argument
ap.add_argument("--dataset", type=str, default="cifar10", 
                choices=list(available_datasets()),
                help=f"Dataset to use. Available: {list(available_datasets())}")

# Add model argument
ap.add_argument("--model", type=str, default="cnn",
                choices=list(available_models()),
                help=f"Model to use. Available: {list(available_models())}")

# ... later in the code ...

# Build dataset and get num_labels
dataset_builder = build_dataset_builder(args.dataset)
loaders = dataset_builder.build(dataset_config)
num_labels = dataset_builder.num_labels

# Build model with correct num_labels
model_owner = build_model(args.model, num_labels=num_labels).to(device)
```

**Apply the same changes to:**
- `attacks/per_shard_takeover.py`
- `attacks/multi_shard_takeover.py`

## Dataset-to-Model Compatibility

**Vision datasets** work with **vision models**:
- CIFAR-10/100, MNIST, Fashion-MNIST → CNN, ResNet, EfficientNet, ViT

**Text datasets** work with **language models**:
- AG News, DBpedia → BERT, LLaMA, Qwen

## Example Usage

Once configured, you can run:

```bash
# CIFAR-100 with ResNet-18
python core/owner_training.py --dataset cifar100 --model resnet18

# MNIST with EfficientNet
python core/owner_training.py --dataset mnist --model efficientnet_b0

# CIFAR-10 with Vision Transformer
python core/owner_training.py --dataset cifar10 --model vit_b16
```

## Notes

- The dataset builder automatically provides `num_labels` - use it instead of hardcoding
- Some models may need additional configuration (e.g., input size, pretrained weights)
- Text datasets require different preprocessing and may need model-specific handling
- Check model documentation in `models/vision/` and `models/language/` for specific requirements

