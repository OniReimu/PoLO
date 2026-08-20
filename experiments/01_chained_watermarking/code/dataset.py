
import torch
# import torch.nn as nn
# import torch.optim as optim
import torchvision
# from torch.utils.data import DataLoader, random_split
# from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
# from torchvision.transforms import AutoAugment, RandAugment
# import copy
import numpy as np
# import matplotlib.pyplot as plt
# import torchtext
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset

# select the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataset(dataset, dataset_rot):
    # load the dataset
    if dataset == 'cifar10':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),  # random horizontal mirror
            transforms.Pad(4),
            torchvision.transforms.RandomCrop(32, padding=4),  # crop
            transforms.RandomGrayscale(),
            torchvision.transforms.ToTensor(),  # convert a PIL image or numpy.ndarray to a Tensor
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        train_set = torchvision.datasets.CIFAR10(dataset_rot,
                                                 train=True,
                                                 download=False,
                                                 transform=transform_train
                                                 )

        test_set = torchvision.datasets.CIFAR10(dataset_rot,
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )

    elif dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # random crop with padding
            transforms.RandomHorizontalFlip(),  # random horizontal flip
            transforms.AutoAugment(),  # AutoAugment
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),  # CIFAR-100 mean and standard deviation
        ])
        # test-set preprocessing
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),  # same CIFAR-100 mean and standard deviation
        ])
        train_set = torchvision.datasets.CIFAR100(dataset_rot,
                                                  train=True,
                                                  download=False,
                                                  transform=train_transform
                                                  )

        test_set = torchvision.datasets.CIFAR100(dataset_rot,
                                                 train=False,
                                                 download=False,
                                                 transform=test_transform
                                                 )
    elif dataset == 'imagenet':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }

        # load the training and validation sets
        train_set = torchvision.datasets.ImageFolder(root=dataset_rot + '/train',
                                                         transform=data_transforms['train'])
        test_set = torchvision.datasets.ImageFolder(root=dataset_rot + '/val',
                                                        transform=data_transforms['val'])

    elif dataset == 'news':
        # load the data
        train_iter, test_iter = AG_NEWS(split=('train', 'test'))
        # tokenizer used for splitting text
        tokenizer = get_tokenizer("basic_english")

        # build the vocabulary
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        # hyperparameters
        MAX_LEN = 200  # keep at most 200 tokens per article

        # preprocessing
        def text_pipeline(x):
            return vocab(tokenizer(x))[:MAX_LEN]  # truncate overlong text

        def label_pipeline(x):
            return int(x) - 1  # labels are zero-based

        class AGNewsDataset(Dataset):
            def __init__(self, data_iter):
                self.data = list(data_iter)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                label, text = self.data[idx]
                return torch.tensor(text_pipeline(text), dtype=torch.long), torch.tensor(label_pipeline(label),
                                                                                         dtype=torch.long)
        # build the dataset & DataLoader
        train_set = AGNewsDataset(train_iter)
        test_set = AGNewsDataset(test_iter)
    else:
        raise ValueError('dataset not supported')

    return train_set, test_set

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

