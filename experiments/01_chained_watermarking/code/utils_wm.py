
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import random
import piq
from PIL import Image
from utils_im import get_selected_weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def embed_spatial_watermark_1(model, b_key_data, positions_key, watermark, alpha, network):
    #cite:: RIGA-Covert and Robust White-BoxWatermarking of Deep Neural Networks (2021 www conference)
    # example usage
    selected_weights = get_selected_weight(model, network, positions_key)
    loss = 0.0
    b_prime = torch.matmul(b_key_data['spatial_watermark_key'], selected_weights)
    # apply a sigmoid and binarise
    # b_prime_binary = torch.sigmoid(b_prime)
    # loss = alpha * (F.binary_cross_entropy(b_prime_binary, b_key_data['spatial_watermark'], reduction='sum'))
    loss += alpha * (torch.nn.functional.mse_loss(b_prime, watermark, reduction='mean'))
    return loss

# function that samples watermark positions at random
def get_random_indices(total_size, num_indices):
    return random.sample(range(total_size), num_indices)

# watermark embedding routine
def embed_watermark_EDNN(model, b_key_data, positions_key, watermark, alpha, network):
    #cite::EmbeddingWatermarks into Deep Neural Networks (2017 ACM on International Conference on Multimedia Retrieval)
    # example usage
    selected_weights = get_selected_weight(model, network, positions_key)
    loss = 0.0
    b_prime = torch.matmul(b_key_data['spatial_watermark_key'], selected_weights)
    # apply a sigmoid and binarise
    b_prime_binary = torch.sigmoid(b_prime)
    loss += alpha * (F.binary_cross_entropy(b_prime_binary, watermark, reduction='mean'))
    return loss

# watermark embedding routine
def embed_watermark_RIGA(model, b_key_data, positions_key, watermark, alpha, network):
    # cite:: RIGA-Covert and Robust White-BoxWatermarking of Deep Neural Networks (2021 www conference)
    # example usage
    selected_weights = get_selected_weight(model, network, positions_key)
    loss = 0.0
    b_prime = torch.matmul(b_key_data['spatial_watermark_key'], selected_weights)
    # apply a sigmoid and binarise
    # b_prime_binary = torch.sigmoid(b_prime)
    # loss = alpha * (F.binary_cross_entropy(b_prime_binary, b_key_data['spatial_watermark'], reduction='sum'))
    loss += alpha * (torch.nn.functional.mse_loss(b_prime, watermark, reduction='mean'))
    return loss

class embed_watermark_FedIPR():
    # cite:: FedIPR: Ownership verification for federated deep neural network models
    # (2022 IEEE Transactions on Pattern Analysis and Machine Intelligence)
    def __init__(self, network, kwargs, model, alpha):
        super(embed_watermark_FedIPR, self).__init__()
        self.alpha = alpha #self.sl_ratio
        # self.loss = 0
        self.network = network
        self.model = model
        self.kwargs = kwargs

    def get_loss(self):
        # print('self.kwargs====', self.kwargs)
        # self.reset()
        loss_1 = 0
        for m in self.kwargs:
            if self.kwargs[m]['flag'] == True:
                b = self.kwargs[m]['b']
                # print('b======', b.shape)
                M = self.kwargs[m]['M']

                b = b.to(torch.device('cuda'))
                M = M.to(torch.device('cuda'))

                if self.network == 'alexnet':
                    idx = int(m.split('.')[1])  # index of the features layer
                    weights = self.model.features[int(idx)].weight.view([1, -1])[:, :2048]
                elif self.network == 'resnet18' or self.network == 'resnet34':
                    layer_idx, block_idx, conv_idx, _ = m.split('.')  # parse the string #'layer1.0.conv1.weight', 'layer1.0.conv2.weight'
                    block_idx = int(block_idx)  # e.g. 0
                    conv_layer = int(conv_idx[-1])  # parse conv1 or conv2
                    if conv_layer == 1:
                        weights = self.model.layer1[block_idx].conv1.weight.view([1, -1])[:, :2048]
                    elif conv_layer == 2:
                        weights = self.model.layer1[block_idx].conv2.weight.view([1, -1])[:, :2048]
                elif self.network == 'vgg16':
                    idx = int(m.split('.')[1])  # index of the features layer
                    weights = self.model.features[int(idx)].weight.view([1, -1])[:, :2048]
                elif self.network == 'wideresnet':
                    # idx = int(m.split('.')[1])  # index of the layer
                    parts = m.split('.')  # ['block1', 'layer', '0', 'conv2', 'weight']
                    block_name = parts[0]  # 'block1'
                    layer_idx = int(parts[2])  # parse '0' as an integer
                    conv_name = parts[3]  # 'conv2'
                    weights = self.model.block1.layer[int(layer_idx)].conv2.weight.view([1, -1])[:, :2048] #block1.layer.0.conv2.weight
                elif self.network == 'textcnn':
                    idx = int(m.split('.')[1])  # index of the layer, e.g. conv2.weight
                    weights = self.model.conv2.weight.view([1, -1])[:, :2048]
                elif self.network == 'bert':  # 'encoder_layers.layers.0.linear1.weight', 'encoder_layers.layers.0.linear2.weight'
                    parts = m.split('.')  # ['encoder_layers', 'layers', '0', 'linear1', 'weight']

                    layer_idx = int(parts[2])  # '0' -> 0
                    linear_name = parts[3]  # 'linear1' or 'linear2'

                    # access the corresponding linear layer
                    weights = getattr(self.model.encoder_layers.layers[layer_idx], linear_name).weight.view([1, -1])[:, :2048]
                else:
                    print('\033[1;31mNetwork Error\033[0m')
                    weights = None
                # print('weights', weights.shape)
                # print('M', M)
                # print('b', b.view(-1).shape)
                loss_1 += (self.alpha*F.relu(-(weights.mm(M).mul(b.view(-1))))).sum()
                # print('(F.relu(-(weights.mm(M).mul(b.view(-1))))).sum()', (F.relu(-(weights.mm(M).mul(b.view(-1))))).sum())
                # print('loss_111', loss_1)

        return loss_1

    # def reset(self):
    #     self.loss = 0

    def add_resnet_module_loss(self, sublayer, module, b, M):

        if module == "convbnrelu_1":
            scale = self.model.layer4[int(sublayer)].convbnrelu_1.scale
            conv_w = torch.mean(self.model.layer4[int(sublayer)].convbnrelu_1.conv.weight, dim = 0)
        if module == "convbn_2":
            scale = self.model.layer4[int(sublayer)].convbn_2.scale
            conv_w = torch.mean(self.model.layer4[int(sublayer)].convbn_2.conv.weight, dim = 0)

        self.loss += (self.alpha * F.relu(-scale.view([1, -1]).mm(M).mul(b.view(-1)))).sum()

# watermark extraction routine
def extract_watermark_EDNN(model, b_key_data, positions_key, watermark, network):
    selected_weights = get_selected_weight(model, network, positions_key)
    b_prime = torch.matmul(b_key_data['spatial_watermark_key'], selected_weights)
    # apply a sigmoid and binarise
    b_prime_binary = torch.sigmoid(b_prime)

    #compute the similarity
    mse = torch.nn.functional.mse_loss(b_prime_binary, watermark)
    #compute the bit error rate
    differences = torch.abs(b_prime_binary - watermark)
    error_bits = torch.sum(differences >= 0.5).item()
    total_bits = watermark.numel()
    error_rate = error_bits / total_bits

    return mse.item(), 1 - error_rate

def extract_watermark_RIGA(model, network, b_key_data, positions_key, watermark):
    selected_weights = get_selected_weight(model, network, positions_key)
    # compute b'
    watermark_prime = torch.matmul(b_key_data['spatial_watermark_key'], selected_weights)
    # compute the similarity
    mse = torch.nn.functional.mse_loss(watermark, watermark_prime)
    # compute the bit error rate
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

def extract_watermark_FedIPR(model, kwargs, network):  #model, device,
    model.eval()
    avg_private = 0
    count_private = 0

    with torch.no_grad():
        if kwargs != None:
            for m in kwargs:
                if kwargs[m]['flag'] == True:
                    b = kwargs[m]['b']
                    M = kwargs[m]['M']

                    M = M.to(device)
                    if network == 'alexnet':
                        idx = int(m.split('.')[1])  # index of the features layer
                        weights = model.features[int(idx)].weight.view([1, -1])[:, :2048]
                    elif network == 'resnet18' or network == 'resnet34':
                        layer_idx, block_idx, conv_idx, _ = m.split('.')  # parse the string
                        block_idx = int(block_idx)  # e.g. 0
                        conv_layer = int(conv_idx[-1])  # parse conv1 or conv2
                        if conv_layer == 1:
                            weights = model.layer1[block_idx].conv1.weight.view([1, -1])[:, :2048]
                        elif conv_layer == 2:
                            weights = model.layer1[block_idx].conv2.weight.view([1, -1])[:, :2048]
                    elif network == 'vgg16':
                        idx = int(m.split('.')[1])  # index of the features layer
                        weights = model.features[int(idx)].weight.view([1, -1])[:, :2048]
                    elif network == 'wideresnet':
                        # idx = int(m.split('.')[1])  # index of the layer
                        parts = m.split('.')  # ['block1', 'layer', '0', 'conv2', 'weight']

                        block_name = parts[0]  # 'block1'
                        layer_idx = int(parts[2])  # parse '0' as an integer
                        conv_name = parts[3]  # 'conv2'
                        weights = model.block1.layer[int(layer_idx)].conv2.weight.view([1, -1])[:,:2048]  # block1.layer.0.conv2.weight
                    elif network == 'textcnn':
                        # idx = int(m.split('.')[1])  # index of the layer, e.g. conv2.weight
                        weights = model.conv2.weight.view([1, -1])[:, :2048]
                    elif network == 'bert':  # 'encoder_layers.layers.0.linear1.weight', 'encoder_layers.layers.0.linear2.weight'
                        parts = m.split('.')  # ['encoder_layers', 'layers', '0', 'linear1', 'weight']

                        layer_idx = int(parts[2])  # '0' -> 0
                        linear_name = parts[3]  # 'linear1' or 'linear2'

                        # access the corresponding linear layer
                        weights = getattr(model.encoder_layers.layers[layer_idx], linear_name).weight.view([1, -1])[:, :2048]
                    else:
                        print('\033[1;31mNetwork Error\033[0m')
                        weights = None
                    signbit = weights.mm(M).sign().to(device)
                    privatebit = b
                    privatebit = privatebit.sign().to(device)

                    # print(privatebit)

                    detection = (signbit == privatebit).float().mean().item()
                    avg_private += detection
                    count_private += 1
    if kwargs == None:
        avg_private = None
    if count_private != 0:
        avg_private /= count_private

    return avg_private