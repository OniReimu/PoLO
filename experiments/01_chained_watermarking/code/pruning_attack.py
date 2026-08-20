

import torch
from torch.optim import SGD, lr_scheduler
import torchvision
from torch.utils.data import DataLoader, random_split
import copy
import random
import numpy as np
import pandas as pd
import os
import re
from args import parser_args
from dataset import make_dataset
import pickle
from torch.nn.utils.rnn import pad_sequence
import time

from utils_im import watermark_hash_function, wasserstein_loss, extract_spatial_watermark_1
from utils_fine_tune import (get_spatial_location_fine_tune, get_selected_weight_fine_tune, make_reference_weights_fine_tune,
                             embed_spatial_watermark_fine_tune, extract_spatial_watermark_fine_tune)

from train_gan import (build_target_model, build_detector_model, test_dp_model, calculate_accuracy)
from fine_tune_attack import get_shards_info


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
# learning_rate = args.learning_rate
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
is_fine_tue = args.is_fine_tue
is_pruning = args.is_pruning
prun_watermark_bit= args.prun_watermark_bit
fine_tue_epochs = args.fine_tue_epochs
fine_learning_rate =args.fine_learning_rate
if is_pruning == 1:
    is_pruning = True
else:
    is_pruning = False

####################################paths###############################################
dataset_rot = '../../../dataset/' + str(dataset)
watermark_key_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(prun_watermark_bit) +
                      '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_b_M_pruning_attack.pkl')
selection_key_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(prun_watermark_bit) +
                      '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_selected_indices_pruning_attack.pkl')
watermark_key_ori_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                          '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_b_M.pkl')
selection_key_ori_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                      '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_selected_indices.pkl')
pruning_attack_model_path = '../model_files/' + str(dataset) + '/' + str(networks) + '/'
log_csv_path = ('../experiments/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_log.csv')
dp_csv_path = ('../experiments/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_dp_acc.csv')
pruning_csv_path = ('../experiments/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(prun_watermark_bit) +
                '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model))
train_image_path = ('../images/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                    '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_results.png')
proof_path = ('../proof/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
              'rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_proof.pkl')
reference_model_path = '../model_files/' + str(dataset) + '/' + str(networks) + '/' + 'False_model_'
wm_reference_weights_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(prun_watermark_bit)
                             + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_reference_weights_part_pruning_attack.pt')
pol_model_path = ('../proof/' + str(dataset) + '/' + str(networks) + '/' )

##text preprocessing
def collate_batch(batch):
    texts, labels = zip(*batch)
    # pad_sequence equalises the text length within a batch
    texts = [torch.tensor(t, dtype=torch.long) for t in texts]
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)  # (batch, max_seq_len)

    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded.to(device), labels.to(device)

train_set, test_set = make_dataset(dataset, dataset_rot)
if dataset == 'news':
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, collate_fn=collate_batch)
else:
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=28)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=28)

def pruning_selected_weight(model, network, change_number):
    """
    in `param_names` across the layers named in `change_number` weights and replaces them with random values in [-0.1, 0.1].

    :param model: the PyTorch model
    :param network: network type (alexnet, resnet18, vgg16, ...)
    :param change_number: number of weights to change
    """

    # ** locate the target layers **
    if network == 'alexnet':
        param_names = ['features.8.weight', 'features.10.weight',
                       'classifier.1.weight', 'classifier.4.weight']
    elif network == 'resnet18' or network == 'resnet34':
        param_names = ['layer1.0.conv1.weight', 'layer1.1.conv1.weight', 'layer2.0.conv2.weight',
                       'layer3.0.conv2.weight', 'layer3.1.conv1.weight']  #, 'layer1.0.conv1.weight',
    elif network == 'vgg16':
        param_names = ['features.3.weight', 'features.10.weight', 'features.14.weight',
                       'features.17.weight', 'features.24.weight'] #'features.3.weight'
    elif network == 'wideresnet':
        param_names = ['block1.layer.2.conv1.weight', 'block2.layer.0.conv1.weight',
                       'block3.layer.1.conv2.weight']
    elif network == 'textcnn':
        param_names = ['conv2.weight', 'conv3.weight', 'conv1.weight'] #'conv2.weight'
    elif network == 'bert':
        param_names = ['encoder_layers.layers.0.linear1.weight', 'encoder_layers.layers.1.linear1.weight',
                       'encoder_layers.layers.2.linear2.weight', 'encoder_layers.layers.2.linear1.weight',
                       'encoder_layers.layers.0.linear2.weight'] #'encoder_layers.layers.0.linear1.weight'
    else:
        raise ValueError("Network Error")

    # ** locate the target layers in the model **
    target_layers = {name: p for name, p in model.named_parameters() if name in param_names}

    if not target_layers:
        raise ValueError("target layer not found, check `param_names` is correct")

    # ** total parameter count over the target layers **
    total_params = sum(p.numel() for p in target_layers.values())

    # ** ensure `change_number` does not exceed the total weight count **
    if change_number > total_params:
        raise ValueError(f"change_number {change_number} exceeds the number of replaceable weights {total_params}")

    # ** distribute the change budget across layers **
    layer_sizes = {name: p.numel() for name, p in target_layers.items()}
    layer_changes = {name: 0 for name in target_layers.keys()}  # per-layer change counter

    remaining_changes = change_number
    while remaining_changes > 0:
        # pick a layer at random
        selected_layer = random.choice(list(target_layers.keys()))

        # remaining budget for this layer
        max_change = min(layer_sizes[selected_layer] - layer_changes[selected_layer], remaining_changes)

        if max_change > 0:
            layer_changes[selected_layer] += max_change
            remaining_changes -= max_change

    # ** perform the replacement **
    with torch.no_grad():
        for name, p in target_layers.items():
            num_changes = layer_changes[name]  # number of weights to change in this layer

            if num_changes > 0:
                flat_p = p.view(-1)  # flatten to 1D
                change_indices = random.sample(range(flat_p.numel()), num_changes)  # sample random indices

                # draw random values in [-0.1, 0.1]
                # random_values = (torch.rand(num_changes) * 0.2 - 0.1).to(p.device)
                # random_values = (torch.rand(num_changes) - 100).to(p.device)
                random_values = 5

                # replace the values at the sampled indices
                flat_p[torch.tensor(change_indices, device=p.device)] = random_values

                # write back into the model
                p.data.copy_(flat_p.view(p.shape))

    print(f"In {len(target_layers)} layers, randomly replaced {change_number} weights with random values in [-0.5, 0.0]")

    return model  # return the modified model


def main():
    file_info1 = '_sign-' + str(watermark_bit) + '_model_1'
    file_info2 = '_epsilon-' + str(epsilon)
    # initialise the record lists
    pruning_acces = []
    time_list = []
    epoches = []
    ori_watermark_acces = []
    pruning_watermark_acces = []
    #########################################load the model#################################################
    target_model, target_optimizer, target_loss_func = build_target_model(networks, num_classes, image_channel,
                                                                          0.1, is_pruning)
    scheduler = lr_scheduler.CosineAnnealingLR(target_optimizer, T_max=300)
    #####################################watermark embedding key######################################
    if is_pruning:
        # load the legitimate watermark metadata
        with open(watermark_key_ori_path, 'rb') as f:  # load the watermark embedding key
            b_key_data_ori = pickle.load(f)
        with open(selection_key_ori_path, 'rb') as f:  # load the position-selection key
            positions_key_data_ori = pickle.load(f)
        with open(proof_path, 'rb') as f:  # load the legitimate watermark
            proof = pickle.load(f)

        ##########################################position-selection key#####################################
        if os.path.exists(selection_key_path):
            print(f"selection key is already generated!")
            with open(selection_key_path, 'rb') as f:
                positions_key_data = pickle.load(f)
        else:
            print(f"selection key is not generated!")
            positions_key_data = {}
            spatial_positions = get_spatial_location_fine_tune(target_model, key_len, networks)
            positions_key_data['spatial_positions'] = spatial_positions
            with open(selection_key_path, 'wb') as f:
                pickle.dump(positions_key_data, f)

    wm_shards_info, epoches_info, file_1 = get_shards_info(pol_model_path, file_info1=file_info1,
                                                           file_info2=file_info2)

    file_split = file_1.split('_epoches-')
    file_pre = file_split[0].split('round-')[0]
    file_pro = file_split[1].split('_pol')[1]

    # run the fine-tuning attack on each PoL checkpoint
    for shards, epoch_i in zip(wm_shards_info, epoches_info):
        shards_model = file_pre + 'round-' + str(shards) + '_epoches-' + str(epoch_i) + '_pol' + file_pro
        print('========================' + shards_model + '========================')
        # check accuracy first; an untrained model performs poorly on the test set
        target_model.load_state_dict(torch.load(pol_model_path + shards_model))
        target_model.to(device)

        test_dp_model(target_model, 1, target_loss_func)

        # measure the legitimate watermark accuracy after the attack
        watermark_ori = proof[int(shards) - 1][0].to(device)  # the legitimate watermark
        # print('original watermark:', watermark_ori)

        # generate the watermark
        selected_weights = get_selected_weight_fine_tune(target_model, networks, positions_key_data)  ##select the weights for the first round

        selected_weights_len = len(selected_weights)

        #write the updated weights back into the model
        time_start = time.time()
        target_model = pruning_selected_weight(target_model, networks, selected_weights_len)

        test_acc = test_dp_model(target_model, 1, target_loss_func)
        time_end = time.time()
        used_time = time_end - time_start


        _, ori_watermark_acc = extract_spatial_watermark_1(target_model, networks, b_key_data_ori,
                                                           positions_key_data_ori,
                                                           watermark_ori)

        print('\n\033[1;31m Original_watermark_acc:\033[0m', ori_watermark_acc)
        time_list.append(used_time)
        epoches.append(epoch_i)
        pruning_acces.append(test_acc)
        ori_watermark_acces.append(ori_watermark_acc)
    # save the post-attack state
    df = pd.DataFrame()
    df['epoches'] = epoches
    df['main_task_acc'] = pruning_acces
    df['ori_watermark_acc'] = ori_watermark_acces
    df['time'] = time_list
    total_time = sum(time_list)

    df.to_csv(pruning_csv_path + '_UES_TIME-' + str(total_time) + '_pruning_log.csv', index=False, encoding='utf-8')
    print('\n\033[1;31mUES_TIME:\033[0m', total_time)

if __name__ == '__main__':
    main()