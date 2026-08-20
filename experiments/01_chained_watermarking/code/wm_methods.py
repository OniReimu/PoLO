
import time
import pandas as pd
import numpy as np
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
from utils_im import (get_spatial_location, get_selected_weight, watermark_hash_function,
                      add_laplacian_noise)
from train_gan import (build_target_model, train_wm, train_non_watermark, test_non_watermark,
                       train_watermark, test_watermark, test_dp_model, return_train_result)
from make_FedIPR_key import fedIPR_key


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

####################################paths###############################################
dataset_rot = '../../../dataset/' + str(dataset)
watermark_key_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                      '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_b_M.pkl')
selection_key_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                      '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_selected_indices.pkl')
init_model_path = ('../model_files/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                   '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_init_model.pth')
log_path = ('../experiments/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                 '_' + str(wm_methods) + '_')
pol_model_path = ('../proof/' + str(dataset) + '/' + str(networks) + '/' + '_sign-' + str(watermark_bit) +
                   '_model_' + str(n_model) + '_epsilon-' + str(epsilon) + '_wm_pol_model.pth')

def save_training_log(log_file, acc, train_time, epochs, wm_methods, wm_shards):
    """
    write the training log, including final accuracy and elapsed time.

    Args:
        log_file (str): path to the log file
        acc (float): the final accuracy
        train_time (float): total training time in seconds
    """
    with open(log_file, "a") as f:  # open the file in append mode
        f.write(f"Wm_methods: {wm_methods}, Epochs: {epochs}, Wm_shards: {wm_shards}, "
                f"Final Accuracy: {acc:.4f}, Training Time: {train_time:.2f} seconds\n")
    print(f"Training log saved to {log_file}")


def main():
    time_start = time.time()
    #########################################load the model#################################################
    target_model, target_optimizer, target_loss_func = build_target_model(networks, num_classes, image_channel, learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(target_optimizer, T_max=300)

    if is_embed_watermark:
        if os.path.exists(watermark_key_path):
            print(f"embedding key is already generated!")
            with open(watermark_key_path, 'rb') as f:
                b_key_data = pickle.load(f)
        else:
            b_key_data = {}
            key_spatial = 'spatial_watermark_key'
            b_key_data[key_spatial] = torch.randn(watermark_bit, key_len, device=device)
            # write to file
            with open(watermark_key_path, 'wb') as f:
                pickle.dump(b_key_data, f)

        ##########################################position-selection key#####################################
        if os.path.exists(selection_key_path):
            print(f"selection key is already generated!")
            with open(selection_key_path, 'rb') as f:
                positions_key_data = pickle.load(f)
        else:
            positions_key_data = {}
            spatial_positions = get_spatial_location(target_model, key_len, networks)
            positions_key_data['spatial_positions'] = spatial_positions
            with open(selection_key_path, 'wb') as f:
                pickle.dump(positions_key_data, f)

    #########################################start training###########################################

    # check accuracy first; an untrained model performs poorly on the test set
    test_non_watermark(target_model, 1, target_loss_func, None)
    if is_embed_watermark:
        if os.path.exists(init_model_path):
            target_model.load_state_dict(torch.load(init_model_path))  # load the first-round model for subsequent training
            target_model.to(device)
        else:
            torch.save(target_model.state_dict(), init_model_path)  # save the initial model first
            target_model.load_state_dict(torch.load(init_model_path))  # load the first-round model for subsequent training
            target_model.to(device)
        init_select_weight = get_selected_weight(target_model, networks, positions_key_data)  ##select the weights for the first round
        if wm_methods == 'FedIPR':
            b_1 = fedIPR_key(networks, watermark_bit)
            # b_1 = torch.sign(torch.rand(watermark_bit) - 0.5).to(device)
        else:
            b_1 = watermark_hash_function(1, init_select_weight, watermark_bit)  # derive the first-round watermark from the weights
        proof_dict = {}  # dictionary that collects the PoL proof
        watermark_round = 1
        for epoch in range(max_epochs + 1):
            if epoch == 0:  # train the initial model
                watermark = b_1  # start from the initial watermark
                watermark_acc_temp, selected_weight = train_wm(target_model, target_loss_func, target_optimizer, epoch,
                                                               positions_key_data, b_key_data, watermark, networks)
                test_watermark(target_model, epoch, target_loss_func, b_key_data, positions_key_data, None)
            if epoch >= 1 and watermark_acc >= 0.96:  # rotate the watermark once the embedding accuracy exceeds the threshold
                watermark_round += 1  # count the watermark rounds
                if wm_methods == 'FedIPR':
                    watermark = fedIPR_key(networks, watermark_bit)
                    # watermark = torch.sign(torch.rand(watermark_bit) - 0.5).to(device)
                else:
                    watermark = watermark_hash_function(epoch, selected_weight.tolist(), watermark_bit)  # recompute and rotate the watermark
                ########################add DP noise####################
                if is_add_dp:
                    # inject noise, save the model, then reload it
                    add_laplacian_noise(target_model, networks, epsilon=epsilon, sensitivity=1.0)
                    # measure accuracy after noise injection and record it
                    acc = test_dp_model(target_model, epoch, target_loss_func)
                    print('\033[31mwatermark_round: \033[0m' + str(watermark_round), '\033[31m-epoch:\033[0m' +
                          str(epoch), '\033[31m-acc: \033[0m' + str(acc))
                    torch.save(target_model.state_dict(), pol_model_path)  # save the noised model
                    target_model.load_state_dict(torch.load(pol_model_path))
            #############training between consecutive shards####################
            watermark_acc_temp, selected_weight = train_wm(target_model, target_loss_func, target_optimizer,epoch,
                                                           positions_key_data, b_key_data, watermark, networks)
            test_watermark(target_model, epoch, target_loss_func, b_key_data, positions_key_data, None)
            proof_dict[watermark_round] = [watermark, selected_weight]
            watermark_acc = watermark_acc_temp
            print('\033[32mwatermark_round:::\033[0m', watermark_round)

            # print('proof_dict', proof_dict)
            # stopping condition
            if epoch > epochs and watermark_acc_temp > 0.9:
                print(f"Stopping training as epoch > {epoch} and accuracy > 90%!")
                break
        # save selected_weight and reference_weight to a .pkl file
        # with open(proof_path, 'wb') as f:
        #     pickle.dump(proof_dict, f)
    time_end = time.time()
    used_time = time_end - time_start
    ######################################write the training log##################################
    train_acces, train_losses, test_acces, test_losses, watermark_acces = return_train_result()
    log_path_saved = (log_path + 'Epoch:' + str(epoch) + '_Watermark_round:' + str(watermark_round) + '_Shards_rate::'
                      + str((watermark_round-1)/epoch) + '.txt')
    save_training_log(log_path_saved, test_acces[-1], used_time, epoch, wm_methods, watermark_round)

if __name__ == '__main__':
    main()
