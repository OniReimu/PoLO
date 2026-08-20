
import torch
import torch.optim as optim
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
fine_watermark_bit = args.fine_watermark_bit
fine_tue_epochs = args.fine_tue_epochs
fine_learning_rate =args.fine_learning_rate
if is_fine_tue == 1:
    is_fine_tue = True
else:
    is_fine_tue = False

####################################paths###############################################
dataset_rot = '../../../dataset/' + str(dataset)
watermark_key_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(fine_watermark_bit) +
                      '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_b_M_fine-tune_attack_2.pkl')
selection_key_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(fine_watermark_bit) +
                      '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_selected_indices_fine-tune_attack_2.pkl')
watermark_key_ori_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                          '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_b_M.pkl')
selection_key_ori_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                      '_rd_' + str(redund_rate) + '_model_' + str(n_model) + '_selected_indices.pkl')
fine_tune_attack_model_path = '../model_files/' + str(dataset) + '/' + str(networks) + '/'
log_csv_path = ('../experiments/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_log.csv')
dp_csv_path = ('../experiments/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_dp_acc.csv')
fine_tune_csv_path = ('../experiments/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(fine_watermark_bit) +
                '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model))
train_image_path = ('../images/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
                    '_' + str(is_embed_watermark) + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_results.png')
proof_path = ('../proof/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(watermark_bit) +
              'rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_model_' + str(n_model) + '_proof.pkl')
reference_model_path = '../model_files/' + str(dataset) + '/' + str(networks) + '/' + 'False_model_'
wm_reference_weights_path = ('../keys/' + str(dataset) + '/' + str(networks) + '/' + 's_sign-' + str(fine_watermark_bit)
                             + '_rd_' + str(redund_rate) + '_epsilon-' + str(epsilon) + '_reference_weights_part_fine-tune_attack.pt')
pol_model_path = ('../proof/' + str(dataset) + '/' + str(networks) + '/' )

###############################check for an available GPU and load the model###################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model, optimizer, loss_func = build_target_model(networks, num_classes, image_channel, fine_learning_rate)

####################################split the dataset###############################################
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

# if os.path.exists(watermark_key_path):
#     print(f"key and watermark is already generated!")
#     with open(watermark_key_path, 'rb') as f:
#         b_key_data = pickle.load(f)
# else:
#    print(f"key and watermark is not generated!")
#
# ##########################################position-selection key#####################################
# if os.path.exists(selection_key_path):
#     print(f"selection key is already generated!")
#     with open(selection_key_path, 'rb') as f:
#         positions_key_data = pickle.load(f)
# else:
#     print(f"selection key is not generated!")

##################################collectors used during training####################################
#lists used to store results
train_losses = []
train_counter = []
train_acces = []
test_losses = []
# test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]
test_acces = []
time_end_list = []  ##storage
weights_dict = {} ##stores selected_weight per epoch
watermark_acc = []
epochs_list = []

def get_shards_info(pol_model_path, file_info1='_sign-2048_model_1', file_info2='_epsilon-5'):
    files = [f for f in os.listdir(pol_model_path) if file_info1 in f and file_info2 in f and '.pth' in f and 'wm' not in f]
    epochs_list, wm_rounds_list = [], []
    for filename in files:
        wm_round = filename.split('round-')[1].split('_')[0]
        epoch = filename.split('epoches-')[1].split('_')[0]
        epochs_list.append(epoch)
        wm_rounds_list.append(wm_round)
    wm_rounds_sorted_list = sorted(wm_rounds_list, key=int)
    epochs_sorted_list = sorted(epochs_list, key=int)
    return wm_rounds_sorted_list, epochs_sorted_list, files[0]

def fine_tune(target_model, target_optimizer, target_loss_func, epoch, watermark, detector_model,detector_optimizer,
              wm_reference_weights_data, positions_key_data, b_key_data):
    target_model.train()  # switch the network to training mode
    train_correct = 0

    ##required when the RIGA scheme is used
    # detector_model.train()
    # real = torch.ones((weights_batch_size, 1), requires_grad=False).to(device)
    # fake = torch.zeros((weights_batch_size, 1), requires_grad=False).to(device)
    # selected_weights = get_selected_weight_fine_tune(target_model, networks, positions_key_data)
    # selected_weights_clone = selected_weights.detach().clone().requires_grad_(True)
    # wm_batch = [selected_weights_clone.clone() for _ in range(weights_batch_size)]
    # wm_batch = torch.stack(wm_batch)
    # nonwm_batch = random.sample(list(wm_reference_weights_data.values()), weights_batch_size)
    # nonwm_batch = torch.stack(nonwm_batch).to(device)

    # for one batch
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("Input data shape:", data.shape)
        # enumerate yields batch_id, data, and label
        # 1 - zero the gradients
        target_optimizer.zero_grad()

        # 2 - feed one batch of images and run the forward pass
        output = target_model(data.to(device))

        # 3 - compute the loss
        loss_o = torch.tensor(0.).to(device)
        loss_w = torch.tensor(0.).to(device)
        loss_g = torch.tensor(0.).to(device)

        loss_o += target_loss_func(output, target.to(device))
        loss_w += embed_spatial_watermark_fine_tune(target_model, b_key_data, positions_key_data, watermark, alpha, networks)
        # loss_g += wasserstein_loss(real, detector_model(wm_batch))

        # loss_total = loss_o + loss_w + beta * loss_g
        loss_total = loss_o + loss_w
        # loss_total = loss_o

        # 4 - backpropagate
        loss_total.backward()

        # 5 - update the parameters
        target_optimizer.step()
        # exp_lr_scheduler.step()

        train_pred = output.data.max(dim=1, keepdim=True)[1]  # take the highest-scoring class from output,
        # dim=1 takes the row-wise maximum; [1] returns the argmax index rather than the value at [0]

        train_correct += train_pred.eq(target.data.view_as(train_pred).to(device)).sum()  # count the correctly classified samples

        print('\r  {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss_total.item()), end='')

        # every 10th batch (log_interval = 10)
        if batch_idx % log_interval == 0:
            # print(batch_idx)
            # print the epoch index, the image index and the training set size, progress percentage and current loss
            '''print('\r  {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()),end = '')
            '''
            # append the current loss to train_losses, used for plotting later
            train_losses.append(loss_total.item())
            # counter
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            '''
            # save the model after each batch 
            torch.save(network.state_dict(), './model.pth')

            # save the optimizer after each batch
            torch.save(optimizer.state_dict(), './optimizer.pth')
            '''
        # ###train the discriminator
        # loss_d = torch.tensor(0.).to(device)
        # detector_optimizer.zero_grad()
        # real_loss = wasserstein_loss(real, detector_model(nonwm_batch))
        # fake_loss = wasserstein_loss(fake, detector_model(wm_batch))
        # loss_d += (fake_loss + real_loss) / 2
        # loss_d.backward()
        # detector_optimizer.step()
    mse, watermark_acc = extract_spatial_watermark_fine_tune(target_model, networks, b_key_data, positions_key_data,
                                                     watermark)
    # adjust_lr_rate(epoch, target_optimizer)
    train_acc = train_correct / len(train_loader.dataset)
    train_acces.append(train_acc.cpu().numpy().tolist())
    print('\tTrain Accuracy:{:.2f}%'.format(100. * train_acc))
    print('watermark_acc====', watermark_acc)

    return watermark_acc

def fine_tune_serval_layers():
    time_start = time.time()
    file_info1 = '_sign-' + str(watermark_bit) + '_model_1'
    file_info2 = '_epsilon-' + str(epsilon)

    #########################################load the model#################################################
    target_model, target_optimizer, target_loss_func = build_target_model(networks, num_classes, image_channel,
                                                                          fine_learning_rate, is_fine_tue)
    scheduler = lr_scheduler.CosineAnnealingLR(target_optimizer, T_max=300)
    #####################################watermark embedding key######################################
    if is_fine_tue:
        # # load the legitimate watermark metadata
        # with open(watermark_key_ori_path, 'rb') as f:  # load the watermark embedding key
        #     b_key_data_ori = pickle.load(f)
        # with open(selection_key_ori_path, 'rb') as f:  # load the position-selection key
        #     positions_key_data_ori = pickle.load(f)
        # with open(proof_path, 'rb') as f:  # load the legitimate watermark
        #     proof = pickle.load(f)

        ##load or generate the attacker watermark
        if os.path.exists(watermark_key_path):
            print(f"embedding key is already generated!")
            with open(watermark_key_path, 'rb') as f:
                b_key_data = pickle.load(f)
        else:
            print(f"embedding key is not generated!")
            b_key_data = {}
            key_spatial = 'spatial_watermark_key'
            b_key_data[key_spatial] = torch.randn(fine_watermark_bit, key_len, device=device)
            # write to file
            with open(watermark_key_path, 'wb') as f:
                pickle.dump(b_key_data, f)

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

        ##########################################reference weights#####################################
        # if os.path.exists(wm_reference_weights_path):
        #     print(f"reference weights is already generated!")
        #     wm_reference_weights_data = torch.load(wm_reference_weights_path)
        # else:
        #     print(f"reference weights is not generated!")
        #     wm_reference_weights_data = make_reference_weights_fine_tune(reference_model_path, networks, num_classes,
        #                                                                  10,
        #                                                                  positions_key_data)
        #     torch.save(wm_reference_weights_data, wm_reference_weights_path)
    # detector_model, detector_optimizer = build_detector_model(len(wm_reference_weights_data[1]))  # instantiate the detector model
    # parse each PoL checkpoint file
    wm_shards_info, epoches_info, file_1 = get_shards_info(pol_model_path, file_info1=file_info1, file_info2=file_info2)
    file_split = file_1.split('_epoches-')
    file_pre = file_split[0].split('round-')[0]
    file_pro = file_split[1].split('_pol')[1]
    # run the fine-tuning attack on each PoL checkpoint
    for shards, epoch_i in zip(wm_shards_info, epoches_info):
        shards_model = file_pre + 'round-' + str(shards) + '_epoches-' + str(epoch_i) + '_pol' + file_pro
        print('========================' + shards_model + '========================')
        # load the pretrained model and check its accuracy first; an untrained model performs poorly on the test set
        target_model.load_state_dict(torch.load(pol_model_path + shards_model))
        target_model.to(device)
        test_dp_model(target_model, 1, target_loss_func)

        # generate the watermark
        select_weight = get_selected_weight_fine_tune(target_model, networks, positions_key_data)  ##select the weights for the first round
        watermark = watermark_hash_function(1, select_weight, fine_watermark_bit)  # derive the watermark from the weights
        # print('attack watermark:', watermark)
        # path of the saved model
        saved_model_path = (
                    fine_tune_attack_model_path + 's_sign-' + str(fine_watermark_bit) + '_epsilon-' + str(epsilon)
                    + '-watermark_round-' + str(shards) + 'fine-tune-attack.pth')

        # #start the fine-tuning attack, five rounds per pass
        # ---------- 3. freeze every parameter outside the target layers ----------
        for p in target_model.parameters():
            p.requires_grad = False

        # unfreeze only the two layers to be fine-tuned, biases included#######AlexNet only for now
        if networks == 'alexnet':
            target_layers = [target_model.features[6], target_model.classifier[6]]
            for layer in target_layers:
                for p in layer.parameters():
                    p.requires_grad = True


        elif networks == 'resnet18':
            target_layers = {'layer1.0.conv1.weight', 'layer1.1.conv1.weight', 'layer2.0.conv2.weight',
                             'layer3.0.conv1.weight', 'fc.weight'} #layer1.0.conv1.weight
            for name, param in target_model.named_parameters():
                param.requires_grad = (name in target_layers)

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, target_model.parameters()),
            lr=fine_learning_rate, momentum=0.15, weight_decay=1e-4
        )
        for epoch in range(fine_tue_epochs):
            one_epoch_start_time = time.time()
            target_model.train()  # switch the network to training mode
            train_correct = 0

            ##required when the RIGA scheme is used
            # detector_model.train()
            # real = torch.ones((weights_batch_size, 1), requires_grad=False).to(device)
            # fake = torch.zeros((weights_batch_size, 1), requires_grad=False).to(device)
            # selected_weights = get_selected_weight_fine_tune(target_model, networks, positions_key_data)
            # selected_weights_clone = selected_weights.detach().clone().requires_grad_(True)
            # wm_batch = [selected_weights_clone.clone() for _ in range(weights_batch_size)]
            # wm_batch = torch.stack(wm_batch)
            # nonwm_batch = random.sample(list(wm_reference_weights_data.values()), weights_batch_size)
            # nonwm_batch = torch.stack(nonwm_batch).to(device)

            # for one batch
            for batch_idx, (data, target) in enumerate(train_loader):
                # print("Input data shape:", data.shape)
                # enumerate yields batch_id, data, and label
                # 1 - zero the gradients
                target_optimizer.zero_grad()

                # 2 - feed one batch of images and run the forward pass
                output = target_model(data.to(device))

                # 3 - compute the loss
                loss_o = torch.tensor(0.).to(device)
                loss_w = torch.tensor(0.).to(device)
                loss_g = torch.tensor(0.).to(device)

                loss_o += target_loss_func(output, target.to(device))
                loss_w += embed_spatial_watermark_fine_tune(target_model, b_key_data, positions_key_data, watermark,
                                                            alpha, networks)
                # loss_g += wasserstein_loss(real, detector_model(wm_batch))

                # loss_total = loss_o + loss_w + beta * loss_g
                loss_total = loss_o + loss_w
                # loss_total = loss_o

                # 4 - backpropagate
                loss_total.backward()

                # 5 - update the parameters
                optimizer.step()
                # exp_lr_scheduler.step()

                train_pred = output.data.max(dim=1, keepdim=True)[1]  # take the highest-scoring class from output,
                # dim=1 takes the row-wise maximum; [1] returns the argmax index rather than the value at [0]

                train_correct += train_pred.eq(target.data.view_as(train_pred).to(device)).sum()  # count the correctly classified samples

                print('\r  {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss_total.item()), end='')

                # every 10th batch (log_interval = 10)
                if batch_idx % log_interval == 0:
                    # print(batch_idx)
                    # print the epoch index, the image index and the training set size, progress percentage and current loss
                    '''print('\r  {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()),end = '')
                    '''
                    # append the current loss to train_losses, used for plotting later
                    train_losses.append(loss_total.item())
                    # counter
                    train_counter.append(
                        (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                    '''
                    # save the model after each batch 
                    torch.save(network.state_dict(), './model.pth')

                    # save the optimizer after each batch
                    torch.save(optimizer.state_dict(), './optimizer.pth')
                    '''
                ###train the discriminator
                # loss_d = torch.tensor(0.).to(device)
                # detector_optimizer.zero_grad()
                # real_loss = wasserstein_loss(real, detector_model(nonwm_batch))
                # fake_loss = wasserstein_loss(fake, detector_model(wm_batch))
                # loss_d += (fake_loss + real_loss) / 2
                # loss_d.backward()
                # detector_optimizer.step()
            mse, watermark_acc = extract_spatial_watermark_fine_tune(target_model, networks, b_key_data,
                                                                     positions_key_data,
                                                                     watermark)
            # adjust_lr_rate(epoch, target_optimizer)
            train_acc = train_correct / len(train_loader.dataset)
            train_acces.append(train_acc.cpu().numpy().tolist())
            print('\tTrain Accuracy:{:.2f}%'.format(100. * train_acc))
            one_epoch_end_time = time.time()
            one_epoch_time = one_epoch_end_time - one_epoch_start_time
            print('one_epoch_start_time=======', one_epoch_time)
            test_acc = test_model(target_model, epoch, target_loss_func, saved_model_path)
            print('\tWatermark Accuracy===========', watermark_acc)


def test_model(model, epoch, criterion, model_path):
    model.eval()  # switch the network to evaluation mode
    test_loss = 0
    correct = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))  # run the forward pass on this batch
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += criterion(output, target.to(device)).item()

            pred = output.data.max(dim=1, keepdim=True)[1]  # take the highest-scoring class from output,
            # dim=1 takes the row-wise maximum; [1] returns the argmax index rather than the value at [0]
            correct += pred.eq(target.data.view_as(pred).to(device)).sum()  # count the correctly classified samples

            if dataset == 'imagenet':
                # compute Top-1 and Top-5 accuracy
                top1, top5 = calculate_accuracy(output, target, topk=(1, 5))
                correct_top1 += top1
                correct_top5 += top5
                total += target.size(0)
    if dataset == 'imagenet':
        print(f'Test Loss: {test_loss / len(test_loader):.4f}, '
              f'Top-1 Accuracy: {correct_top1 / len(test_loader):.2f}%, '
              f'Top-5 Accuracy: {correct_top5 / len(test_loader):.2f}%')

    acc = correct / len(test_loader.dataset)
    if dataset == 'imagenet':
        # test_acces.append(correct_top5 / len(test_loader))
        test_acc = ((correct_top5 / (len(test_loader))).item()) / 100.0
        # test_acces.append(test_acc)
    else:
        test_acc = acc.cpu().numpy().tolist()
        # test_acces.append(test_acc)

    test_loss /= len(test_loader.dataset)  # mean loss over the 10000 test samples
    # test_losses.append(test_loss)  # record test_loss for this epoch
    # save the model after each batch
    torch.save(model.state_dict(), model_path)

    # if test_acces[-1] >= max(test_acces) and epoch > 1:
        # save the model after each batch
        # torch.save(model.state_dict(), model_path)

        # save the optimizer after each batch
        # torch.save(optimizer.state_dict(), optimizer_path)
        # print('\n\033[1;31mThe network Max Avg Accuracy : {:.2f}%\033[0m'.format(100. * max(test_acces)))

    # print e.g. Test set: Avg. loss: 2.3129, Accuracy: 1205/10000 (12%)
    '''
    print('\rTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),end = '')
    '''
    print('\r Test set \033[1;31m{}\033[0m : Avg. loss: {:.4f}, Accuracy: {}/{}  \033[1;31m({:.2f}%)\033[0m\n' \
          .format(epoch, test_loss, correct, len(test_loader.dataset), 100. * acc), end='')
    return test_acc

def main():
    # time_start = time.time()
    file_info1 = '_sign-' + str(watermark_bit) + '_model_1'
    file_info2 = '_epsilon-' + str(epsilon)
    #initialise the record lists
    fine_tune_acces = []
    wm_rounds = []
    epoches = []
    ori_watermark_acces = []
    fine_tune_watermark_acces = []
    #########################################load the model#################################################
    target_model, target_optimizer, target_loss_func = build_target_model(networks, num_classes, image_channel,
                                                                          fine_learning_rate, is_fine_tue)
    scheduler = lr_scheduler.CosineAnnealingLR(target_optimizer, T_max=300)
    #####################################watermark embedding key######################################
    if is_fine_tue:
        #load the legitimate watermark metadata
        with open(watermark_key_ori_path, 'rb') as f:  #load the watermark embedding key
            b_key_data_ori = pickle.load(f)
        with open(selection_key_ori_path, 'rb') as f: #load the position-selection key
            positions_key_data_ori = pickle.load(f)
        with open(proof_path, 'rb') as f:  #load the legitimate watermark
            proof = pickle.load(f)

        ##load or generate the attacker watermark
        if os.path.exists(watermark_key_path):
            print(f"embedding key is already generated!")
            with open(watermark_key_path, 'rb') as f:
                b_key_data = pickle.load(f)
        else:
            print(f"embedding key is not generated!")
            b_key_data = {}
            key_spatial = 'spatial_watermark_key'
            b_key_data[key_spatial] = torch.randn(fine_watermark_bit, key_len, device=device)
            # write to file
            with open(watermark_key_path, 'wb') as f:
                pickle.dump(b_key_data, f)

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

        ##########################################reference weights#####################################
        if os.path.exists(wm_reference_weights_path):
            print(f"reference weights is already generated!")
            wm_reference_weights_data = torch.load(wm_reference_weights_path)
        else:
            print(f"reference weights is not generated!")
            wm_reference_weights_data = make_reference_weights_fine_tune(reference_model_path, networks, num_classes, 10,
                                                               positions_key_data)
            torch.save(wm_reference_weights_data, wm_reference_weights_path)
    detector_model, detector_optimizer = build_detector_model(len(wm_reference_weights_data[1]))  # instantiate the detector model
    #parse each PoL checkpoint file
    wm_shards_info, epoches_info, file_1 = get_shards_info(pol_model_path, file_info1=file_info1, file_info2=file_info2)
    file_split = file_1.split('_epoches-')
    file_pre = file_split[0].split('round-')[0]
    file_pro = file_split[1].split('_pol')[1]
    #run the fine-tuning attack on each PoL checkpoint
    for shards, epoch_i in zip(wm_shards_info, epoches_info):
        shards_model = file_pre+'round-'+str(shards)+'_epoches-'+ str(epoch_i) +'_pol'+ file_pro
        print('========================' + shards_model + '========================')
        # check accuracy first; an untrained model performs poorly on the test set
        target_model.load_state_dict(torch.load(pol_model_path + shards_model))
        target_model.to(device)
        test_dp_model(target_model, 1, target_loss_func)
        #generate the watermark
        select_weight = get_selected_weight_fine_tune(target_model, networks, positions_key_data)  ##select the weights for the first round
        watermark = watermark_hash_function(1, select_weight, fine_watermark_bit)  # derive the watermark from the weights
        # print('attack watermark:', watermark)
        #path of the saved model
        saved_model_path = (fine_tune_attack_model_path + 's_sign-' + str(fine_watermark_bit) + '_epsilon-' + str(epsilon)
                            + '-watermark_round-' + str(shards) + 'fine-tune-attack.pth')
        # #start the fine-tuning attack, five rounds per pass
        for epoch in range(fine_tue_epochs):
            one_epoch_start_time = time.time()
            watermark_acc_fine_tune = fine_tune(target_model, target_optimizer, target_loss_func, epoch, watermark,
                                                detector_model, detector_optimizer, wm_reference_weights_data,
                                                positions_key_data, b_key_data)
            one_epoch_end_time = time.time()
            one_epoch_time = one_epoch_end_time - one_epoch_start_time
            print(f"one_epoch_time==================: {one_epoch_time}")
            test_acc = test_model(target_model, epoch, target_loss_func, saved_model_path)
        # test_acc = 0
        # watermark_acc_fine_tune = 0
        #measure the legitimate watermark accuracy after the fine-tuning attack
        watermark_ori = proof[int(shards)-1][0].to(device)  #the legitimate watermark
        # print('original watermark:', watermark_ori)
        _, ori_watermark_acc = extract_spatial_watermark_1(target_model, networks, b_key_data_ori, positions_key_data_ori,
                                                         watermark_ori)
        print('\n\033[1;31m Main task_acc:\033[0m', test_acc, '\n\033[1;31m Watermark_acc_fine_tune:\033[0m',
              watermark_acc_fine_tune, '\n\033[1;31m Original_watermark_acc:\033[0m', ori_watermark_acc)
        ##record the fine-tuning attack outcome
        wm_rounds.append(shards)
        epoches.append(epoch_i)
        fine_tune_acces.append(test_acc)
        fine_tune_watermark_acces.append(watermark_acc_fine_tune)
        ori_watermark_acces.append(ori_watermark_acc)
    #save the post-attack state
    df = pd.DataFrame()
    df['wm_rounds'] = wm_rounds
    df['epoches'] = epoches
    df['fine_tune_acc'] = fine_tune_acces
    df['watermark_acc_fine_tune'] = fine_tune_watermark_acces
    df['ori_watermark_acc_fine_tune'] = ori_watermark_acces
    time_end = time.time()
    # used_time = time_end - time_start
    # print('\n\033[1;31mUES_TIME:\033[0m', used_time)
    # df.to_csv(fine_tune_csv_path +'_UES_TIME-' + str(used_time) + '_fine_tune_log.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    ####attack variant 1######
    # main()
    ####attack variant 2######
    fine_tune_serval_layers()

    ####save the shard record
    # get_shards_info(pol_model_path, shards_csv_path, file_info1='_sign-2048_model_1', file_info2='_epsilon-5')
