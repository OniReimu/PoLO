
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

def get_spatial_location(model, num_indices, network):
    # record the weight positions each client owns
    """
    :param model:
    :param num_indices:
    :param network:
    :return:
    """
    if network == 'alexnet': ###select weights from features.3.weight and features.6.weight
        total_weight = sum(p.numel() for name, p in model.named_parameters() if
                           'features.3.weight' in name or 'features.6.weight' in name)
    elif network == 'resnet18' or network == 'resnet34':
        total_weight = sum(p.numel() for name, p in model.named_parameters() if
                           'layer1.0.conv1.weight' in name or 'layer1.0.conv2.weight' in name)
    elif network == 'vgg16':
        total_weight = sum(p.numel() for name, p in model.named_parameters() if
                           'features.3.weight' in name or 'features.7.weight' in name)
    elif network == 'wideresnet':
        total_weight = sum([p.numel() for name, p in model.named_parameters() if
                             'block1.layer.0.conv2.weight' in name])
    elif network == 'textcnn':
        total_weight = sum([p.numel() for name, p in model.named_parameters() if
                             'conv2.weight' in name])
    elif network == 'bert':
        total_weight = sum([p.numel() for name, p in model.named_parameters() if
                             'encoder_layers.layers.0.linear1.weight' in name or
                            'encoder_layers.layers.0.linear2.weight' in name])
    else:
        total_weight = None
        print('Network Error')
    positions = get_random_indices(total_weight, num_indices)
    spatial_positions = [i for i in positions]
    return spatial_positions

def get_watermark_layer_weight(model, network):
    if network == 'alexnet':
        # take the weights of model.features[3].weight and model.features[6].weight
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'features.3.weight' in name or 'features.6.weight' in name])
    elif network == 'resnet18' or network == 'resnet34':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'layer1.0.conv1.weight' in name or 'layer1.0.conv2.weight' in name])
    elif network == 'vgg16':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'features.3.weight' in name or 'features.7.weight' in name])
    elif network == 'wideresnet':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'block1.layer.0.conv2.weight' in name])
    elif network == 'textcnn':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'conv2.weight' in name])
    elif network == 'bert':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'encoder_layers.layers.0.linear1.weight' in name or
                             'encoder_layers.layers.0.linear2.weight' in name])
    else:
        print('Network Error')
        weights = None
    return weights

def get_selected_weight(model, network, positions_key):
    if network == 'alexnet':
        # take the weights of model.features[3].weight and model.features[6].weight
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'features.3.weight' in name or 'features.6.weight' in name])
    elif network == 'resnet18' or network == 'resnet34':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'layer1.0.conv1.weight' in name or 'layer1.0.conv2.weight' in name])
    elif network == 'vgg16':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'features.3.weight' in name or 'features.7.weight' in name])
    elif network == 'wideresnet':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'block1.layer.0.conv2.weight' in name])
    elif network == 'textcnn':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'conv2.weight' in name])
    elif network == 'bert':
        weights = torch.cat([p.view(-1) for name, p in model.named_parameters() if
                             'encoder_layers.layers.0.linear1.weight' in name or
                             'encoder_layers.layers.0.linear2.weight' in name])
    else:
        print('Network Error')
        weights = None
    selected_weights = weights[positions_key['spatial_positions']]
    return selected_weights

def make_reference_weights(reference_model_path, network, num_class, num_models, positions_key):
    reference_weights = {}
    model, _, _ = build_model(network, num_class, 3, 0.01)
    for i in range(num_models):
        model_path = reference_model_path + str(i+1) + '.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        # reference_weights[i+1] = get_watermark_layer_weight(model, network)
        reference_weights[i] = get_selected_weight(model, network, positions_key)
    return reference_weights

def embed_spatial_watermark(model, b_key_data, positions_key, alpha, network):
    #cite::EmbeddingWatermarks into Deep Neural Networks (2017 ACM on International Conference on Multimedia Retrieval)
    # example usage
    selected_weights = get_selected_weight(model, network, positions_key)
    loss = 0.0
    b_prime = torch.matmul(b_key_data['spatial_watermark_key'], selected_weights)
    # apply a sigmoid and binarise
    b_prime_binary = torch.sigmoid(b_prime)
    loss += alpha * (F.binary_cross_entropy(b_prime_binary, b_key_data['spatial_watermark'], reduction='sum'))
    return loss

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

def embed_spatial_watermark_im(model, b_key_data, positions_key, alpha, network, beta, reference_weights):
    #cite:: RIGA-Covert and Robust White-BoxWatermarking of Deep Neural Networks (2021 www conference)
    # example usage
    selected_weights = get_selected_weight(model, network, positions_key)
    loss1 = 0.0
    b_prime = torch.matmul(b_key_data['spatial_watermark_key'], selected_weights)
    # apply a sigmoid and binarise
    # b_prime_binary = torch.sigmoid(b_prime)
    # loss = alpha * (F.binary_cross_entropy(b_prime_binary, b_key_data['spatial_watermark'], reduction='sum'))
    loss1 += alpha * (torch.nn.functional.mse_loss(b_prime, b_key_data['spatial_watermark'], reduction='mean'))

    watermark_layer_weight = get_watermark_layer_weight(model, network)
    loss2 = 0.0
    loss2 += beta * (torch.nn.functional.mse_loss(watermark_layer_weight, reference_weights, reduction='mean'))
    # loss2 += beta * sum(np.abs(np.array(watermark_layer_weight.to('cpu').detach().numpy()) - np.array(reference_weights.to('cpu').detach().numpy())))
    return loss1 + loss2

def extract_spatial_watermark(model, network, b_key_data, positions_key):
    # cite::EmbeddingWatermarks into Deep Neural Networks (2017 ACM on International Conference on Multimedia Retrieval)
    # example usage
    selected_weights = get_selected_weight(model, network, positions_key)
    b_prime = torch.matmul(b_key_data['spatial_watermark_key'], selected_weights)
    # apply a sigmoid and binarise
    b_prime_binary = torch.sigmoid(b_prime)

    #compute the similarity
    mse = torch.nn.functional.mse_loss(b_prime_binary, b_prime)
    #compute the bit error rate
    differences = torch.abs(b_prime_binary - b_key_data['spatial_watermark'])
    error_bits = torch.sum(differences >= 0.5).item()
    total_bits = b_key_data['spatial_watermark'].numel()
    error_rate = error_bits / total_bits

    return mse.item(), 1 - error_rate

def extract_spatial_watermark_1(model, network, b_key_data, positions_key, watermark):
    selected_weights = get_selected_weight(model, network, positions_key)
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

def watermark_hash_function(x, selected_weight, watermark_len):
    # take the current timestamp
    timestamp = int(time.time())  # current timestamp in seconds

    # join the custom integer, the timestamp and the sequence into one string
    input_string = f"{x}-{timestamp}-" + ",".join(map(str, selected_weight))

    # derive the digest with SHA-256
    hash_object = hashlib.sha256(input_string.encode())
    binary_hash = bin(int(hash_object.hexdigest(), 16))[2:].zfill(256)
    # print('hash_object', binary_hash)
    # Generate variable-length hash by iterative hashing
    extended_binary_hash = binary_hash
    while len(extended_binary_hash) < watermark_len:
        hash_object = hashlib.sha256(extended_binary_hash.encode())
        extended_binary_hash += bin(int(hash_object.hexdigest(), 16))[2:].zfill(256)
    return torch.tensor([float(bit) for bit in extended_binary_hash[:watermark_len]], device=device)  #, timestamp

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)

def laplacian_noise(shape, epsilon, sensitivity=1.0):
    scale = sensitivity / epsilon  # scale parameter b
    return torch.tensor(torch.distributions.laplace.Laplace(0, scale).sample(shape), dtype=torch.float32)

def add_laplacian_noise(model, network, epsilon, sensitivity=1.0):
    """
    add Laplace noise only to the layers named in target_layers
    :param model: the network to be noised
    :param target_layers: list of layers to be noised (e.g. ['features.3.weight', 'features.4.weight'])
    :param epsilon: DP privacy budget; ε larger values mean less noise
    :param sensitivity: sensitivity used for the scale parameter, default 1.0
    """
    if network == 'alexnet':
        target_layers = ['features.8.weight']
    elif network == 'resnet18' or network == 'resnet34':
        target_layers = ['layer1.1.conv1.weight', 'layer2.0.conv1.weight']
    elif network == 'vgg16':
        target_layers = ['features.0.weight', 'features.10.weight']
    elif network == 'wideresnet':
        target_layers = ['block1.layer.0.conv1.weight', 'block1.layer.1.conv1.weight']
    elif network == 'textcnn':
        target_layers = ['conv1.weight']
    elif network == 'bert':
        target_layers = ['encoder_layers.layers.1.linear1.weight', 'encoder_layers.layers.1.linear2.weight']
    else:
        print('Network Error')
        target_layers = None
    if not isinstance(target_layers, list):
        raise ValueError("target_layers must be a list, e.g. ['features.3.weight', 'features.4.weight']")

    with torch.no_grad():  # keep this out of the autograd graph
        for name, param in model.named_parameters():
            if param.requires_grad and name in target_layers:  # only the named layers are noised
                noise = laplacian_noise(param.shape, epsilon, sensitivity).to(param.device)

                if network == 'alexnet': ###for AlexNet, add noise to half of the layer
                    num_elements = param.numel()  # total parameter count
                    half_size = num_elements // 1000  # number of parameters in the first half
                    # add noise to the first half only
                    flat_param = param.view(-1)  # flatten to 1D
                    flat_noise = noise.view(-1)  # flatten as well
                    flat_param[:half_size] += flat_noise[:half_size]  # add noise to the first half_size elements only
                    # reshape back to the original tensor
                    param.copy_(flat_param.view(param.shape))
                else:  #for other networks, add noise across all named layers
                    param.add_(noise)  # add the noise directly to the weights



if __name__ == '__main__':
    datasets = 'cifar10'
    # network = 'resnet18'
    num_classes = 10
    image_channel = 3
    learning_rate = 0.01
    spatial_num_bit = 256
    time_num_bit = 10
    is_spatial_watermark = True
    n_model = 1
    # model_path_true = ('../model_files/' + str(datasets) + '/' + str(network) + '/' + 's_sign-' + str(spatial_num_bit) +
    #               '_t_sign-' + str(time_num_bit) + '_True_model_1.pth')
    # old_model_path_true = ('../model_files/' + str(datasets) + '/' + str(network) + '/old_models/' + 's_sign-' + str(spatial_num_bit) +
    #                    '_t_sign-' + str(time_num_bit) + '_True_model_1.pth')
    # model_path_false = ('../model_files/' + str(datasets) + '/' + str(network) + '/' + 's_sign-' + str(spatial_num_bit) +
    #                    '_t_sign-' + str(time_num_bit) + '_False_model_2.pth')
    # reference_model_path = ('../model_files/' + str(datasets) + '/' + str(network) + '/' + 's_sign-' + str(spatial_num_bit) +
    #                         '_t_sign-' + str(time_num_bit) + '_False_model_')
    # model_true, _, _ = make_model(network, num_classes, image_channel, learning_rate)
    # model_false, _, _ = make_model(network, num_classes, image_channel, learning_rate)
    # model_true.load_state_dict(torch.load(model_path_true, map_location=torch.device('cpu')))
    # model_false.load_state_dict(torch.load(model_path_false, map_location=torch.device('cpu')))
    # weights_true = get_watermark_layer_weight(model_true, network)
    # weights_false = get_watermark_layer_weight(model_false, network)
    # print(len(weights_false))
    # hist = histogram(weights_true.tolist())
    # run the KS test
    # result = ks_2samp(weights_false.tolist(), weights_true.tolist())

    # print(f"KS statistic: {result.statistic:.3f}")
    # print(f"P-value: {result.pvalue:.3f}")
    # reference_weights = make_reference_weights(reference_model_path, network, 10, 8)
    # print(len(reference_weights))

    # example
    x = 42  # custom integer
    selected_weight = [2, 3, 4, 5]  # input sequence
    binary_hash = watermark_hash_function(x, selected_weight, 256)
    print(binary_hash)