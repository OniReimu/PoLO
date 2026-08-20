

import time
import pandas as pd
import numpy as np
import hashlib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD, lr_scheduler
import torchvision
import json
import matplotlib.pyplot as plt
from PIL import Image
import random
import pickle
from args import parser_args
from utils_im import (get_spatial_location, get_selected_weight, make_reference_weights, watermark_hash_function,
                      add_laplacian_noise)
from train_gan import (build_target_model, build_detector_model, train_non_watermark, test_non_watermark,
                       train_watermark, test_watermark, test_dp_model, return_train_result)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


####################################parameters###############################################
# check for an available GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
args = parser_args()
print('args===', args)
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

model_path = '../model_files/' + str(dataset) + '/' + str(networks) + '/' + 'False_model_1.pth'

def compute_model_sha256(pth_path):
    # load the model parameters
    state_dict = torch.load(pth_path, map_location='cpu')

    # if a full model was saved, call .state_dict()
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()

    # initialise the hasher
    sha256 = hashlib.sha256()

    # concatenate all parameter data and feed it to the hash function
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if isinstance(tensor, torch.Tensor):
            sha256.update(tensor.cpu().numpy().tobytes())
        else:
            print(f"Warning: {key} is not a tensor and will be skipped.")

    # return the 256-bit digest as 64 hex characters
    return sha256.hexdigest()

if __name__ == '__main__':
    begin_time = time.time()
    hash_val = compute_model_sha256(model_path)
    print(f"Model SHA-256 Hash:\n{hash_val}")
    end_time = time.time()
    used_time = end_time - begin_time
    print(f"used_time: {used_time}")
