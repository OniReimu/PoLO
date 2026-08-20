
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128 * 3, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = self.embedding(text)  # (batch, seq_len, embed_dim)
        text = text.permute(0, 2, 1)  # (batch, embed_dim, seq_len)

        x1 = self.global_max_pool(self.relu(self.conv1(text))).squeeze(2)
        x2 = self.global_max_pool(self.relu(self.conv2(text))).squeeze(2)
        x3 = self.global_max_pool(self.relu(self.conv3(text))).squeeze(2)

        out = torch.cat([x1, x2, x3], dim=1)  # (batch, 128*3)
        out = self.dropout(out)
        return self.fc(out)

if __name__ == '__main__':
    # hyperparameters
    EMBED_DIM = 128  # embedding dimension
    NUM_CLASSES = 4  # AG News has 4 classes
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 10
    MAX_LEN = 200  # keep at most 200 tokens per article
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer used for splitting text
    tokenizer = get_tokenizer("basic_english")
    # load the data
    train_iter, test_iter = AG_NEWS(split=('train', 'test'))
    # build the vocabulary
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    # initialise the model
    VOCAB_SIZE = len(vocab)
    model = TextCNN(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES).to(device)
    # print(model)
    # read the model state_dict
    state_dict = model.state_dict()

    # print all keys
    for key in state_dict.keys():
        # print(key)
        if 'bias' in key:
            continue

        weight_tensor = state_dict[key]
        num_elements = weight_tensor.numel()  # element count
        print(f'{key}: {num_elements} elements')