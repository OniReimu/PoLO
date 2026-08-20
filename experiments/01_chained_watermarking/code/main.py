
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

####################################paths###############################################
dataset_rot = '../../../dataset/' + str(dataset)
watermark_key_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                      '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_b_M.pkl')
selection_key_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                      '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_selected_indices.pkl')
model_path = ('../model_files/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
              '_' + str(is_embed_watermark) + '_rd_' + '_epsilon-' + str(epsilon) + str(redund_rate) + '_model_' + str(n_model) + '_dr_' + str(0.95) + '.pth')
ref_model_path = ('../model_files/' + str(dataset) + '/' + str(networks) + '/' + 'False_model_' + str(n_model) + '.pth')
init_model_path = ('../model_files/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
              '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_init_model.pth')
log_csv_path = ('../experiments/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_dr_' + str(0.95) + '_log.csv')
dp_csv_path = ('../experiments/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_dr_' + str(0.95) + '_dp_acc.csv')
train_image_path = ('../images/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                    '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_results.png')
proof_path = ('../proof/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
              'rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_dr_' + str(0.95) + '_proof.pkl')
reference_model_path = '../model_files/' + str(dataset) + '/' + str(networks) + '/' + 'False_model_'
wm_reference_weights_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit)
                             + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_reference_weights_part.pt')
pol_model_path = ('../proof/' + str(dataset) + '/' + str(networks) + '/' + '_sign-' + str(watermark_bit) +
                   '_model_' + str(n_model) + '_epsilon-' + str(epsilon)) + '_dr_' + str(0.95)

def main():
    dp_acc = []
    dp_epoch = []
    dp_watermark = []
    #########################################load the model#################################################
    target_model, target_optimizer, target_loss_func = build_target_model(networks, num_classes, image_channel, learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(target_optimizer, T_max=300)
    #####################################watermark embedding key######################################
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

        ##########################################reference weights#####################################
        if os.path.exists(wm_reference_weights_path):
            print(f"reference weights is already generated!")
            wm_reference_weights_data = torch.load(wm_reference_weights_path)
        else:
            wm_reference_weights_data = make_reference_weights(reference_model_path, networks, num_classes, 10, positions_key_data)
            torch.save(wm_reference_weights_data, wm_reference_weights_path)
        # print('wm_reference_weights_data', wm_reference_weights_data)

    #########################################start training###########################################

    # check the model accuracy first
    test_start_time = time.time()
    test_non_watermark(target_model, 1, target_loss_func, model_path)
    test_end_time = time.time()
    print('\n\033[1;31mThe test data for one time used is : \033[0m', test_end_time - test_start_time)
    if is_embed_watermark:
        torch.save(target_model.state_dict(), init_model_path)  # save the initial model first
        init_select_weight = get_selected_weight(target_model, networks, positions_key_data) ##select the weights for the first round
        water_gen_begin_time = time.time()
        b_1 = watermark_hash_function(1, init_select_weight, watermark_bit) #derive the first-round watermark from the weights
        water_gen_end_time = time.time()
        water_used_time = water_gen_end_time - water_gen_begin_time
        print('\n\033[1;31mThe watermark generate time used is : \033[0m', water_used_time)
        target_model.load_state_dict(torch.load(init_model_path)) #load the first-round model for subsequent training
        target_model.to(device)
        detector_model, detector_optimizer = build_detector_model(len(wm_reference_weights_data[1])) #instantiate the detector model
        proof_dict = {}  #dictionary that collects the PoL proof
        watermark_round = 1
        for epoch in range(max_epochs + 1):
            if epoch == 0:  #train the initial model
                watermark = b_1  #start from the initial watermark
                watermark_acc_temp, selected_weight = train_watermark(target_model, target_loss_func, target_optimizer,
                                            detector_model, detector_optimizer, epoch, positions_key_data, b_key_data,
                                            watermark, networks, wm_reference_weights_data)
                test_watermark(target_model, epoch, target_loss_func, b_key_data, positions_key_data, model_path)
            if epoch >= 1 and watermark_acc >= 0.95:   #rotate the watermark once the embedding accuracy exceeds the threshold
                watermark_round += 1                   #count the watermark rounds
                watermark = watermark_hash_function(epoch, selected_weight.tolist(), watermark_bit)  #recompute and rotate the watermark
                ########################add DP noise####################
                if is_add_dp:
                    #inject noise, save the model, then reload it
                    add_laplacian_noise(target_model, networks, epsilon=epsilon, sensitivity=1.0)
                    # measure accuracy after noise injection and record it
                    acc = test_dp_model(target_model, epoch, target_loss_func)
                    print('\033[31mshard_round: \033[0m' + str(watermark_round), '\033[31m-epoch:\033[0m' +
                          str(epoch), '\033[31m-acc: \033[0m' + str(acc))
                    torch.save(target_model.state_dict(), pol_model_path + '_watermark_round-' + str(watermark_round) +
                               '_epoches-' + str(epoch) + '_pol_model.pth')  # save the noised model
                    target_model.load_state_dict(torch.load(pol_model_path + '_watermark_round-' + str(watermark_round) +
                               '_epoches-' + str(epoch) + '_pol_model.pth'))
                    dp_acc.append(acc)
                    dp_epoch.append(epoch)
                    dp_watermark.append(watermark_round)
            #############training between consecutive shards####################
            watermark_acc_temp, selected_weight = train_watermark(target_model, target_loss_func, target_optimizer,
                                                                 detector_model, detector_optimizer, epoch,
                                                                 positions_key_data, b_key_data,
                                                                 watermark, networks, wm_reference_weights_data)
            test_watermark(target_model, epoch, target_loss_func, b_key_data, positions_key_data, model_path)
            proof_dict[watermark_round] = [watermark, selected_weight]
            watermark_acc = watermark_acc_temp
            print('\033[32mshard_round:::\033[0m', watermark_round)

            # print('proof_dict', proof_dict)
            # stopping condition
            if epoch > epochs and watermark_acc_temp > 0.90:
                print(f"Stopping training as epoch > {epoch} and accuracy > 90%!")
                break
        #save selected_weight and reference_weight to a .pkl file
        with open(proof_path, 'wb') as f:
            pickle.dump(proof_dict, f)
    else:
        for epoch in range(epochs+1):
            train_non_watermark(target_model, target_optimizer, target_loss_func, epoch)
            # scheduler.step()
            test_non_watermark(target_model, epoch, target_loss_func, ref_model_path)

    ######################################write the training log##################################
    train_acces, train_losses, test_acces, test_losses, watermark_acces = return_train_result()
    df = pd.DataFrame()
    df['train_acc' + str(is_embed_watermark)] = train_acces
    df['test_acc' + str(is_embed_watermark)] = test_acces[1:]
    if is_embed_watermark:
        df['watermark_acc' + str(watermark_round)] = watermark_acces
    df.to_csv(log_csv_path, index=False, encoding='utf-8')
    if is_add_dp and is_embed_watermark:
        df_dp = pd.DataFrame(columns=['dp_epoch', 'dp_watermark_round', 'dp_test_acc'])
        df_dp['dp_epoch'] = dp_epoch
        df_dp['dp_watermark_round'] = dp_watermark
        df['watermark_acc' + str(watermark_round)] = watermark_acces
        df_dp['dp_test_acc'] = dp_acc
        df_dp.to_csv(dp_csv_path, index=False, encoding='utf-8')

if __name__ == "__main__":
    begin_time = time.time()
    main()
    end_time = time.time()
    used_time = end_time - begin_time
    print('\n\033[1;31mThe time used is : \033[0m', used_time, '\n\033[1;31m for \033[0m', dataset, '+', networks,
          '+wm_bit', watermark_bit)


