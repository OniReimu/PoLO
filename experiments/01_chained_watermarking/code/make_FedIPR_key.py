
import torch

from args import parser_args
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt

args = parser_args()
dataset = args.dataset
num_classes = args.num_classes
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
image_channel = args.image_channel
networks = args.networks
learning_rate = args.learning_rate
epochs = args.epochs
max_epochs = args.max_epochs
n_model = args.n_model
log_interval = args.log_interval
redund_rate = args.redund_rate
is_embed_watermark = args.is_embed_watermark
watermark_bit = args.watermark_bit
wm_methods = args.wm_methods
key_len = args.key_len
alpha = args.alpha
beta = args.beta
gamma = args.gamma
weights_batch_size = args.weights_batch_size
is_add_dp = args.is_add_dp
epsilon = args.epsilon
if is_embed_watermark == 1:
    is_embed_watermark = True
else:
    is_embed_watermark = False
if is_add_dp == 1:
    is_add_dp = True
else:
    is_add_dp = False

# check for an available GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def fedIPR_key(network, b_length):
    #layers in which each client embeds its watermark
    if network == 'alexnet':
        passport_layer = {'features.0.weight':False, 'features.3.weight': 'this is my signature',
                          'features.6.weight': 'this is my signature', 'features.8.weight':False, 'features.10.weight':False}
    elif network == 'resnet18' or network == 'resnet34':
        passport_layer = {'layer1.0.conv1.weight': 'this is my signature', 'layer1.0.conv2.weight': 'this is my signature',
                          'layer1.1.conv1.weight': False, 'layer1.1.conv2.weight': False, 'layer2.0.conv1.weight': False,
                          'layer2.0.conv2.weight': False}
    elif network == 'vgg16':  #'features.3.weight' in name or 'features.7.weight' in name
        passport_layer = {'features.3.weight': 'this is my signature', 'features.7.weight': 'this is my signature',
                          'features.10.weight': False, 'features.14.weight': False, 'features.17.weight': False}
    elif network == 'wideresnet':  #'block1.layer.0.conv2.weight'
        passport_layer = {'block1.layer.0.conv1.weight': False, 'block1.layer.0.conv2.weight': 'this is my signature',
                          'block1.layer.1.conv1.weight': False, 'block1.layer.1.conv2.weight': False,
                          'block1.layer.2.conv1.weight': False}
    elif network == 'textcnn': #'conv2.weight'
        passport_layer = {'conv1.weight': False, 'conv2.weight': 'this is my signature', 'conv3.weight': False}
    elif network == 'bert': #'encoder_layers.layers.0.linear1.weight', 'encoder_layers.layers.0.linear2.weight'
        passport_layer = {'encoder_layers.layers.0.linear1.weight': 'this is my signature',
                          'encoder_layers.layers.0.linear2.weight': 'this is my signature',
                          'encoder_layers.layers.1.linear1.weight': False,
                          'encoder_layers.layers.1.linear2.weight': False}
    else:
        print('\033[1;31mNetwork Error\033[0m')
        passport_layer = None

    key = construct_passport_kwargs(passport_layer, b_length, network)

    return key

def construct_passport_kwargs(passport_settings, bit_length, network):

    passport_kwargs = {}
    if network == 'alexnet':
        channels_1 = {
            'features.0.weight': (2048, 3456), 'features.3.weight': (2048, 2304), 'features.6.weight': (2048, 2304),
            'features.8.weight': (2048, 2304), 'features.10.weight': (2048, 2304)}
    elif network == 'resnet18' or network == 'resnet34':
        channels_1 = {'layer1.0.conv1.weight': (2048, 2304), 'layer1.0.conv2.weight': (2048, 2304),
                          'layer1.1.conv1.weight': (2048, 2304), 'layer1.1.conv2.weight': (2048, 2304),
                      'layer2.0.conv1.weight': (2048, 2304), 'layer2.0.conv2.weight':(2048, 2304)}
    elif network == 'vgg16':
        channels_1 = {'features.3.weight': (2048, 2304), 'features.7.weight': (2048, 2304),
                          'features.10.weight': (2048, 2304), 'features.14.weight': (2048, 2304), 'features.17.weight': (2048, 2304)}
    elif network == 'wideresnet':
        channels_1 = {'block1.layer.0.conv1.weight': (2048, 2304), 'block1.layer.0.conv2.weight': (2048, 2304),
                          'block1.layer.1.conv1.weight': (2048, 2304), 'block1.layer.1.conv2.weight': (2048, 2304),
                          'block1.layer.2.conv1.weight': (2048, 2304)}
    elif network == 'textcnn':
        channels_1 = {'conv1.weight': (2048, 2304), 'conv2.weight': (2048, 2304), 'conv3.weight': (2048, 2304)}
    elif network == 'bert':
        channels_1 = {'encoder_layers.layers.0.linear1.weight': (2048, 2304),
                          'encoder_layers.layers.0.linear2.weight': (2048, 2304),
                          'encoder_layers.layers.1.linear1.weight': (2048, 2304),
                          'encoder_layers.layers.1.linear2.weight': (2048, 2304)}
    else:
        print('\033[1;31mNetwork Error\033[0m')
        channels_1 = None

    channels_2 = {
        'layer2': (512, 4068),
        'layer3': (1024, 8136),
        'layer4': (2048, 16272)
    }
    for layer_key in passport_settings:
        # print('isinstance(passport_settings[layer_key], dict)', isinstance(passport_settings[layer_key], dict))
        if isinstance(passport_settings[layer_key], dict):
            passport_kwargs[layer_key] = {}
            for i in passport_settings[layer_key]:  # i = 0, 1
                passport_kwargs[layer_key][i] = {}
                for module_key in passport_settings[layer_key][i]:  # module_key = convbnrelu
                    flag = passport_settings[layer_key][i][module_key]  # flag = str in module_key
                    b = flag if isinstance(flag, str) else None
                    if b is not None:
                        flag = True

                    passport_kwargs[layer_key][i][module_key] = {
                        'flag': flag
                    }

                    #                  b = torch.sign(torch.rand(self.num_bit) - 0.5)
                    if b is not None:

                        # output_channels = int(bit_length * 512 / 2048)
                        output_channels = int(bit_length * 2048 / 2048)

                        bsign = torch.sign(torch.rand(output_channels) - 0.5)
                        # bitstring = ''.join([format(ord(c), 'b').zfill(8) for c in b])
                        # for j, c in enumerate(bitstring):
                        #     if c == '0':
                        #         bsign[j] = -1
                        #     else:
                        #         bsign[j] = 1
                        b = bsign

                        # if weight_type == 'gamma':
                        M = torch.randn(channels_2[layer_key][0], output_channels)
                        # else:
                        #     M = torch.randn(resnet_channels[layer_key][1], output_channels)

                        passport_kwargs[layer_key][i][module_key]['b'] = b
                        passport_kwargs[layer_key][i][module_key]['M'] = M

        else:  ##run this one
            flag = passport_settings[layer_key]
            b = flag if isinstance(flag, str) else None
            if b is not None:
                flag = True
            passport_kwargs[layer_key] = {
                'flag': flag
            }

            if b is not None:
                # print('b is not None?????????????????', b)
                # if model == 'alexnet':
                #     if layer_key == "4":
                #         output_channels = int(bit_length * 384 / 896)
                #     if layer_key == "5":
                #         output_channels = int(bit_length * 256 / 896)
                #     if layer_key == "6":
                #         output_channels = int(bit_length * 256 / 896)
                # output_channels = int(bit_length * 384 / 896)
                output_channels = int(bit_length * 896 / 896)
                # output_channels = len(b) * 8
                bsign = torch.sign(torch.rand(output_channels) - 0.5)
                # bitstring = ''.join([format(ord(c), 'b').zfill(8) for c in b])

                # for j, c in enumerate(bitstring):
                #     if c == '0':
                #         bsign[j] = -1
                #     else:
                #         bsign[j] = 1
                b = bsign

                # if self.weight_type == 'gamma':
                M = torch.randn(channels_1[layer_key][0], output_channels)
                # else:
                    # M = torch.randn(alexnet_channels[layer_key][1], output_channels)

                passport_kwargs[layer_key]['b'] = b
                passport_kwargs[layer_key]['M'] = M

    return passport_kwargs

if __name__ == '__main__':
    key = fedIPR_key('resnet34', 128)
    print('key', key)
