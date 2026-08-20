# Dataset root

`experiments/01_chained_watermarking` resolves its data root to this directory
(`../../../dataset/<name>` relative to `code/`). `dataset.py` calls torchvision
with `download=False`, so the data has to be here before the first run.

```bash
python - <<'PY'
import torchvision
torchvision.datasets.CIFAR10('dataset/cifar10',  train=True,  download=True)
torchvision.datasets.CIFAR10('dataset/cifar10',  train=False, download=True)
torchvision.datasets.CIFAR100('dataset/cifar100', train=True,  download=True)
torchvision.datasets.CIFAR100('dataset/cifar100', train=False, download=True)
PY
```

TinyImageNet is loaded with `ImageFolder` and expects `dataset/imagenet/train`
and `dataset/imagenet/val`. AG News is loaded through torchtext into
`dataset/news`.
