
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchvision import datasets, transforms
import numpy as np
import random
import hashlib
import time

from args import parser_args
# from dataset import make_dataset
from models.resnet18 import ResNet18
from models.alexnet import AlexNet
# from models.resnet34 import ResNet34
from models.resnet50 import ResNet50
from models.wideresnet import WideResNet28_10
from models.vgg import vgg13, vgg16_bn
from models.resnet import resnet18, resnet50, resnet34
from models.TextCnn import TextCNN
from models.Bert import MiniBERT
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
# from torch.nn.utils.rnn import pad_sequence

args = parser_args()
embed_dim = args.embed_dim
max_len = args.max_len

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigmoid = nn.Sigmoid()
args = parser_args()
redund_rate = args.redund_rate

# build the vocabulary
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)
# tokenizer used for splitting text
tokenizer = get_tokenizer("basic_english")
# load the data
train_iter, test_iter = AG_NEWS(split=('train', 'test'))
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
# initialise the model
vocab_size = len(vocab)

def build_model(networks, num_classes, image_channel, learning_rate):
    if networks == 'alexnet':
        target_model = AlexNet(image_channel, num_classes)
    elif networks == 'vgg16':
        target_model = vgg16_bn(pretrained=False)
    elif networks == 'vgg13':
        target_model = vgg13(pretrained=False)
    elif networks == 'resnet18':
        target_model = ResNet18(image_channel, num_classes)
    elif networks == 'resnet34':
        target_model = resnet34(pretrained=False)
    elif networks == 'wideresnet':
        target_model = WideResNet28_10(num_classes=num_classes)
    elif networks == 'textcnn':
        target_model = TextCNN(vocab_size, embed_dim, num_classes).to(device)
    elif networks == 'bert':
        target_model = MiniBERT(vocab_size, hidden_dim=embed_dim, num_heads=4, num_layers=4, num_classes=num_classes,
                     max_seq_length=max_len).to(device)
    else:
        target_model = ResNet50(num_classes)
    if networks == 'resnet18':
        target_optimizer = optim.SGD(target_model.parameters(), learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
    elif networks == 'textcnn':
        target_optimizer = optim.AdamW(target_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif networks == 'bert':
        target_optimizer = optim.AdamW(target_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    else:
        target_optimizer = optim.SGD(target_model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
    target_model = target_model.to(device)
    target_loss_func = nn.CrossEntropyLoss()
    return target_model, target_optimizer, target_loss_func

def get_random_indices(total_size, num_indices):
    return random.sample(range(total_size), num_indices)

def np_sort(arr):
    return -np.sort(-arr)

def get_time_location(time_location):
    """
    epochs at which a watermark is embedded
    :param time_location:
    :return:
    """
    time_watermark_epoch = []
    for i in time_location:
        temp = []
        for j in range(i[0], i[1]):
            temp.append(j)
        time_watermark_epoch.append(temp)
    return time_watermark_epoch

def get_spatial_location_fine_tune(model, num_indices, network):
    # record the weight positions each client owns
    """
    :param model:
    :param num_indices:
    :param network:
    :return:
    """
    if network == 'alexnet': ###select weights from features.3.weight and features.6.weight
        total_weight = sum(p.numel() for name, p in model.named_parameters() if
                           'features.3.weight' in name or 'features.6.weight' in name or 'features.10.weight' in name)
    elif network == 'resnet18' or network == 'resnet34':
        total_weight = sum(p.numel() for name, p in model.named_parameters() if
                           'layer1.0.conv1.weight' in name or 'layer1.1.conv1.weight' in name or 'layer2.0.conv2.weight' in name)
    elif network == 'vgg16':
        total_weight = sum(p.numel() for name, p in model.named_parameters() if
                           'features.3.weight' in name or 'features.10.weight' in name or 'features.14.weight' in name)
    elif network == 'wideresnet':
        total_weight = sum([p.numel() for name, p in model.named_parameters() if
                             'block1.layer.2.conv1.weight' in name or 'block2.layer.0.conv1.weight' in name])
    elif network == 'textcnn':
        total_weight = sum([p.numel() for name, p in model.named_parameters() if
                             'conv2.weight' in name or 'conv3.weight' in name])
    elif network == 'bert':
        total_weight = sum([p.numel() for name, p in model.named_parameters() if
                             'encoder_layers.layers.0.linear1.weight' in name or
                            'encoder_layers.layers.1.linear1.weight' in name or 'encoder_layers.layers.2.linear2.weight' in name])
    else:
        total_weight = None
        print('Network Error')
    positions = get_random_indices(total_weight, num_indices)
    spatial_positions = [i for i in positions]
    return spatial_positions

def get_watermark_layer_weight_fine_tune(model, network):
    if network == 'alexnet':
        # take the weights of model.features[3].weight and model.features[6].weight
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'features.3.weight' in name or 'features.8.weight' in name or 'features.10.weight' in name])
    elif network == 'resnet18' or network == 'resnet34':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'layer1.0.conv1.weight' in name or 'layer1.1.conv1.weight' in name or 'layer2.0.conv2.weight' in name])
    elif network == 'vgg16':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'features.3.weight' in name or 'features.10.weight' in name or 'features.14.weight' in name])
    elif network == 'wideresnet':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'block1.layer.2.conv1.weight' in name or 'block2.layer.0.conv1.weight' in name])
    elif network == 'textcnn':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'conv2.weight' in name or 'conv3.weight' in name])
    elif network == 'bert':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'encoder_layers.layers.0.linear1.weight' in name or
                            'encoder_layers.layers.1.linear1.weight' in name or 'encoder_layers.layers.2.linear2.weight' in name])
    else:
        print('Network Error')
        weights = None
    return weights

def get_selected_weight_fine_tune(model, network, positions_key):
    if network == 'alexnet':
        # take the weights of model.features[3].weight and model.features[6].weight
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'features.3.weight' in name or 'features.8.weight' in name or 'features.10.weight' in name])
    elif network == 'resnet18' or network == 'resnet34':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'layer1.0.conv1.weight' in name or 'layer1.1.conv1.weight' in name or 'layer2.0.conv2.weight' in name])
    elif network == 'vgg16':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'features.3.weight' in name or 'features.10.weight' in name or 'features.14.weight' in name])
    elif network == 'wideresnet':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'block1.layer.2.conv1.weight' in name or 'block2.layer.0.conv1.weight' in name])
    elif network == 'textcnn':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'conv2.weight' in name or 'conv3.weight' in name])
    elif network == 'bert':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'encoder_layers.layers.0.linear1.weight' in name or
                            'encoder_layers.layers.1.linear1.weight' in name or 'encoder_layers.layers.2.linear2.weight' in name])
    else:
        print('Network Error')
        weights = None
    selected_weights = weights[positions_key['spatial_positions']]
    return selected_weights

def make_reference_weights_fine_tune(reference_model_path, network, num_class, num_models, positions_key):
    reference_weights = {}
    model, _, _ = build_model(network, num_class, 3, 0.01)
    for i in range(num_models):
        model_path = reference_model_path + str(i+1) + '.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        # reference_weights[i+1] = get_watermark_layer_weight(model, network)
        reference_weights[i] = get_selected_weight_fine_tune(model, network, positions_key)
    return reference_weights

def embed_spatial_watermark_fine_tune(model, b_key_data, positions_key, watermark, alpha, network):
    #cite:: RIGA-Covert and Robust White-BoxWatermarking of Deep Neural Networks (2021 www conference)
    # example usage
    selected_weights = get_selected_weight_fine_tune(model, network, positions_key)
    loss = 0.0
    b_prime = torch.matmul(b_key_data['spatial_watermark_key'], selected_weights)
    # apply a sigmoid and binarise
    # b_prime_binary = torch.sigmoid(b_prime)
    # loss = alpha * (F.binary_cross_entropy(b_prime_binary, b_key_data['spatial_watermark'], reduction='sum'))
    loss += alpha * (torch.nn.functional.mse_loss(b_prime, watermark, reduction='mean'))
    return loss

def extract_spatial_watermark_fine_tune(model, network, b_key_data, positions_key, watermark):
    selected_weights = get_selected_weight_fine_tune(model, network, positions_key)
    # compute b'
    watermark_prime = torch.matmul(b_key_data['spatial_watermark_key'], selected_weights)
    #compute the similarity
    mse = torch.nn.functional.mse_loss(watermark, watermark_prime)
    #compute the bit error rate
    differences = torch.abs(watermark_prime - watermark)
    error_bits = torch.sum(differences >= 0.5).item()
    total_bits = watermark_prime.numel()
    """
    total_bits = b_key_data['spatial_watermark'].numel() / redund_rate
    if redund_rate == 1 or redund_rate == 2:
        count_threshold = redund_rate // 2
    elif redund_rate == 4 or redund_rate == 8:
        count_threshold = redund_rate // 2 + 1
    elif redund_rate == 12:
        count_threshold = redund_rate // 2 + 3
    else:
        count_threshold = redund_rate
        print('illegal redund_rate: ', count_threshold)
    error_bits = 0
    for i in range(0, len(differences), redund_rate):
        segment = differences[i:i+redund_rate]
        count_05 = torch.sum(segment >= 0.5).item()
        if count_05 > count_threshold:
            error_bits += 1
    """
    error_rate = error_bits / total_bits

    return mse.item(), 1 - error_rate