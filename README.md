# PoLO: Proof-of-Learning and Proof-of-Ownership at Once with Chained Watermarking

Anonymous artifact accompanying the submission. This repository contains the
code, configurations, and result records behind every experiment reported in
the paper.

Everything here runs on public datasets and public model architectures. No
private data, credentials, or author-identifying material is included.

---

## What maps to what

| Paper element | Directory | Entry point |
|---|---|---|
| Tab. III (compatibility, shard rate) | `experiments/01_chained_watermarking` | `code/main.py` |
| Tab. IV (proof generation and verification cost) | `experiments/01_chained_watermarking` | `code/main.py`, `code/hash_based_verify.py` |
| Tab. V (OFA and WMA forgery attacks) | `experiments/01_chained_watermarking` | `code/fine_tune_attack.py`, `code/pruning_attack.py` |
| Fig. 3 (attack curves) | `experiments/01_chained_watermarking` | `reference_figures/` |
| Appendix, multi-shard takeover (OFA) | `experiments/02_multi_shard_takeover` | `run_experiment.py` |
| Appendix, PoLO for LLMs | `experiments/03_llm_watermarking` | `llm_watermark/main.py` |

Two items in the paper have no code here. The attack taxonomy (Tab. II) is an
analytical mapping and runs nothing. The data reconstruction evaluation in the
appendix uses the published implementation of the informed-adversary attack it
cites, driven by the checkpoints that `01` produces; only the resulting figures
are included, under `01/reference_figures/`.

---

## Requirements

Python 3.9 or later, PyTorch 2.x, and torchvision. A CUDA GPU is expected for
`01` and `03`. `02` selects its device automatically and also runs on Apple
Silicon and on CPU; the individual stage scripts under `core/` and `attacks/`
take an explicit `--device` if you need to pin it.

```bash
pip install torch torchvision torchtext transformers datasets numpy scipy matplotlib tqdm
```

`03` declares its own pinned dependency set and is installed separately:

```bash
cd experiments/03_llm_watermarking
uv sync            # or: pip install -e .
```

### Datasets

`01` reads from `dataset/` at the repository root and calls torchvision with
`download=False`, so fetch the data once before the first run:

```bash
python - <<'PY'
import torchvision
torchvision.datasets.CIFAR10('dataset/cifar10',  train=True,  download=True)
torchvision.datasets.CIFAR10('dataset/cifar10',  train=False, download=True)
torchvision.datasets.CIFAR100('dataset/cifar100', train=True,  download=True)
torchvision.datasets.CIFAR100('dataset/cifar100', train=False, download=True)
PY
```

See `dataset/README.md` for TinyImageNet and AG News. `02` and `03` download
what they need on their own.

---

## 1. Chained watermarking, main experiments

```bash
cd experiments/01_chained_watermarking/code

# Generate a PoLO proof (Tab. III, Tab. IV)
python main.py --is_embed_watermark=1 --dataset=cifar10 --networks=resnet18 \
               --wm_methods=RIGA --watermark_bit=1024

# Hash-chain verification of a stored checkpoint
python hash_based_verify.py --dataset=cifar10 --networks=resnet18

# OFA, overhaul fine-tuning attack (Tab. V)
python fine_tune_attack.py --is_fine_tue=1 --dataset=cifar10 --networks=resnet18

# WMA, weight manipulation attack (Tab. V)
python pruning_attack.py --is_pruning=1 --dataset=cifar10 --networks=resnet18
```

`code/args.py` holds every switch. The configurations reported in the paper are
the cross product of

- `--dataset` in `cifar10`, `cifar100`, `imagenet`, `news`
- `--networks` in `resnet18`, `resnet34`, `vgg16`, `alexnet`, `wideresnet`, `textcnn`, `bert`
- `--wm_methods` in `RIGA`, `EDNN`, `FedIPR`
- `--watermark_bit` in `512`, `1024`, `2048`

Run the scripts from inside `code/`; every path is relative to it. Four sibling
directories receive output and ship empty: `model_files/` for checkpoints,
`proof/` for the PoLO proof, `images/` for training curves, and `experiments/`
for the CSV logs. Embedding and position-selection keys are written to `keys/`
on the first run and reused afterwards.

## 2. Multi-shard takeover

Reproduces the appendix figure comparing honest shard training against a
sequential OFA takeover.

```bash
cd experiments/02_multi_shard_takeover

python run_experiment.py                       # full pipeline, defaults
python run_experiment.py --dataset cifar10 --model efficientnet \
                        --attack-type multi-shard --plot-time-curves
```

Seeds default to `123` for both owner training and the attack. `results/`
carries the summary JSON for the three configurations in the paper, so the
numbers can be checked without rerunning. `DATASET_MODEL_GUIDE.md` lists every
supported dataset and architecture.

## 3. LLM watermarking

```bash
cd experiments/03_llm_watermarking

cp .env.example .env    # add a Hugging Face token if you use a gated model
python -m llm_watermark.main --config config.yaml
```

`examples/` holds ready configurations for CPU, CUDA, and MPS. `output/` carries
the per-epoch metric CSVs behind the appendix table.

---

## What is not included, and why

Model checkpoints (`.pt`, `.pth`, `.pkl`) are omitted. They total roughly 350 MB
and every one of them is regenerated by the commands above. Training logs are
omitted as well, since they record local filesystem paths and carry no
information the result files do not already hold.

## License

MIT, see `LICENSE`.
