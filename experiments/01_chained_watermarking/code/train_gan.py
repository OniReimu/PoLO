

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD, lr_scheduler
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import datasets, transforms
import copy
import random
import time
from args import parser_args
from dataset import make_dataset
# from models.vgg16 import VGG16
from models.resnet18 import ResNet18
from models.alexnet import AlexNet
from models.resnet34 import ResNet34
from models.resnet50 import ResNet50
from models.wideresnet import WideResNet28_10
from models.ViT import ViT
from models.vgg import vgg13, vgg16_bn
from models.resnet import resnet18, resnet50, resnet34
from models.TextCnn import TextCNN
from models.Bert import MiniBERT
from models.detector import Detector
from utils_im import watermark_hash_function, embed_spatial_watermark_1, extract_spatial_watermark_1
from utils_im import wasserstein_loss, get_selected_weight
from utils_wm import embed_watermark_EDNN, embed_watermark_FedIPR, extract_watermark_EDNN, extract_watermark_FedIPR
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

######parameters############
args = parser_args()
dataset = args.dataset
num_classes = args.num_classes
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
image_channel = args.image_channel
embed_dim = args.embed_dim
max_len = args.max_len
networks = args.networks
learning_rate = args.learning_rate
epochs = args.epochs
max_epochs = args.max_epochs
n_model = args.n_model
log_interval = args.log_interval
# redund_rate = args.redund_rate
is_embed_watermark = args.is_embed_watermark
watermark_bit = args.watermark_bit
key_len = args.key_len
alpha = args.alpha
beta = args.beta
gamma = args.gamma
wm_methods = args.wm_methods
weights_batch_size = args.weights_batch_size
if is_embed_watermark == 1:
    is_embed_watermark = True
else:
    is_embed_watermark = False

####################################paths###############################################
dataset_rot = '../../../dataset/' + str(dataset)

##################################check for an available GPU#######################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# print(f'Using device: {device}')

###############AG News preprocessing######################
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
vocab_size = len(vocab)

##################################model and optimizer#########################################
def build_target_model(networks, num_classes, image_channel, learning_rate, fine_tune=False):
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
        if fine_tune:
            target_optimizer = optim.SGD(target_model.parameters(), learning_rate, momentum=0.15, nesterov=True, weight_decay=1e-4)
        else:
            target_optimizer = optim.SGD(target_model.parameters(), learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
    elif networks == 'textcnn':
        target_optimizer = optim.AdamW(target_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif networks == 'bert':
        if fine_tune:
            target_optimizer = optim.SGD(target_model.parameters(), lr=learning_rate, momentum=0.15, weight_decay=0)
        else:
            target_optimizer = optim.SGD(target_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        if fine_tune:
            target_optimizer = optim.SGD(target_model.parameters(), learning_rate, momentum=0.5, weight_decay=1e-4)
        else:
            target_optimizer = optim.SGD(target_model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
    target_model = target_model.to(device)
    target_loss_func = nn.CrossEntropyLoss()
    return target_model, target_optimizer, target_loss_func

def build_detector_model(weights_shape):
    detector_model = Detector(weights_shape).to(device)
    detector_optimizer = optim.SGD(detector_model.parameters(), lr=0.00005)
    return detector_model, detector_optimizer

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

##################################collectors used during training####################################
#lists used to store results
train_losses = []
train_counter = []
train_acces = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]
test_acces = []
time_end_list = []  ##storage
proof_dict = {} ##stores selected_weight per epoch
watermark_acces = []
watermark_count = 0
def adjust_lr_rate(epoch, optimizer):
    lr = learning_rate
    if epoch > 180:
        lr = lr / 10000
    elif epoch > 150:
        lr = lr / 10000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_non_watermark(target_model, target_optimizer, target_loss_func, epoch):
    target_model.train()  # switch the network to training mode
    train_correct = 0
    # for one batch
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("Input data shape:", data.shape)
        # enumerate yields batch_id, data, and label
        # 1 - zero the gradients
        target_optimizer.zero_grad()

        # 2 - feed one batch of images and run the forward pass
        output = target_model(data.to(device))

        # 3 - compute the loss
        loss = torch.tensor(0.).to(device)
        loss += target_loss_func(output, target.to(device))

        # 4 - backpropagate
        loss.backward()

        # 5 - update the parameters
        target_optimizer.step()
        # exp_lr_scheduler.step()

        train_pred = output.data.max(dim=1, keepdim=True)[1]  # take the highest-scoring class from output,
        # dim=1 takes the row-wise maximum; [1] returns the argmax index rather than the value at [0]

        train_correct += train_pred.eq(target.data.view_as(train_pred).to(device)).sum()  # count the correctly classified samples

        print('\r  {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()), end='')

        # every 10th batch (log_interval = 10)
        if batch_idx % log_interval == 0:
            # print(batch_idx)
            # print the epoch index, the image index and the training set size, progress percentage and current loss
            '''print('\r  {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()),end = '')
            '''
            # append the current loss to train_losses, used for plotting later
            train_losses.append(loss.item())
            # counter
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            '''
            # save the model after each batch 
            torch.save(network.state_dict(), './model.pth')

            # save the optimizer after each batch
            torch.save(optimizer.state_dict(), './optimizer.pth')
            '''
    if networks == 'resnet18' and dataset == 'cifar10':
        pass
    # elif networks == 'textcnn':
    else:
        adjust_lr_rate(epoch, target_optimizer)
    train_acc = train_correct / len(train_loader.dataset)
    train_acces.append(train_acc.cpu().numpy().tolist())
    print('\tTrain Accuracy:{:.2f}%'.format(100. * train_acc))

#test routine
def test_non_watermark(model, epoch, criterion, model_path):
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
        print(f'Test Loss: {test_loss / len(test_loader):.4f}, ' f'Top-1 Accuracy: {correct_top1 / len(test_loader):.2f}%, '
              f''f'Top-5 Accuracy: {correct_top5 / len(test_loader):.2f}%')
    acc = correct / len(test_loader.dataset)
    if dataset == 'imagenet':
        # test_acces.append((correct_top5 / (100 * len(test_loader))).item())
        test_acces.append(((correct_top5 / (100 * len(test_loader))).item())/100.0)
    else:
        test_acces.append(acc.cpu().numpy().tolist())

    test_loss /= len(test_loader.dataset)  # mean loss over the 10000 test samples
    test_losses.append(test_loss)  # record test_loss for this epoch

    if test_acces[-1] >= max(test_acces) and epoch > 1:
        # save the model after each batch
        if model_path is not None:
            torch.save(model.state_dict(), model_path)

        # save the optimizer after each batch
        # torch.save(optimizer.state_dict(), optimizer_path)
        print('\n\033[1;31mThe network Max Avg Accuracy : {:.2f}%\033[0m'.format(100. * max(test_acces)))

    # print e.g. Test set: Avg. loss: 2.3129, Accuracy: 1205/10000 (12%)
    '''
    print('\rTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),end = '')
    '''
    print('\r Test set \033[1;31m{}\033[0m : Avg. loss: {:.4f}, Accuracy: {}/{}  \033[1;31m({:.2f}%)\033[0m\n' \
          .format(epoch, test_loss, correct, len(test_loader.dataset), 100. * acc), end='')


def train_watermark(target_model, target_loss_func, target_optimizer, detector_model, detector_optimizer,
                    epoch, positions_key_data, b_key_data, watermark, networks, wm_reference_weights_data):
    train_correct = 0
    target_model.train()
    detector_model.train()
    real = torch.ones((weights_batch_size, 1), requires_grad=False).to(device)
    fake = torch.zeros((weights_batch_size, 1), requires_grad=False).to(device)
    selected_weights = get_selected_weight(target_model, networks, positions_key_data)
    selected_weights_clone = selected_weights.detach().clone().requires_grad_(True)
    wm_batch = [selected_weights_clone.clone() for _ in range(weights_batch_size)]
    wm_batch = torch.stack(wm_batch)
    nonwm_batch = random.sample(list(wm_reference_weights_data.values()), weights_batch_size)
    nonwm_batch = torch.stack(nonwm_batch).to(device)

    for batch_idx, (data, target) in enumerate(train_loader):
        ######train the generator
        # 1 - zero the gradients
        target_optimizer.zero_grad()
        # 2 - feed one batch of images and run the forward pass
        output = target_model(data.to(device))

        # 3 - compute the loss
        loss_o = torch.tensor(0.).to(device)
        loss_g = torch.tensor(0.).to(device)
        loss_s = torch.tensor(0.).to(device)
        loss_o += target_loss_func(output, target.to(device))
        loss_g += wasserstein_loss(real, detector_model(wm_batch))
        if epoch == 1:
            wm_embed_start_time = time.time()
            loss_s += embed_spatial_watermark_1(target_model, b_key_data, positions_key_data, watermark, alpha, networks)
            wm_embed_end_time = time.time()
        else:
            loss_s += embed_spatial_watermark_1(target_model, b_key_data, positions_key_data, watermark, alpha, networks)
        # loss_g = -torch.mean(detector_model(wm_batch))
        loss_a = loss_o + loss_s + beta * loss_g

        # 4 - backpropagate
        loss_a.backward()

        # 5 - update the parameters
        target_optimizer.step()
        # exp_lr_scheduler.step()

        train_pred = output.data.max(dim=1, keepdim=True)[1]  # take the highest-scoring class from output,
        # dim=1 takes the row-wise maximum; [1] returns the argmax index rather than the value at [0]

        train_correct += train_pred.eq(target.data.view_as(train_pred).to(device)).sum()  # count the correctly classified samples

        print('\r  {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss_o.item()), end='')

        # every 10th batch (log_interval = 10)
        if batch_idx % log_interval == 0:
            # print(batch_idx)
            # print the epoch index, the image index and the training set size, progress percentage and current loss
            '''print('\r  {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()),end = '')
            '''
            # append the current loss to train_losses, used for plotting later
            train_losses.append(loss_o.item())
            # counter
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            '''
            # save the model after each batch
            torch.save(network.state_dict(), './model.pth')

            # save the optimizer after each batch
            torch.save(optimizer.state_dict(), './optimizer.pth')
            '''
        ####train the discriminator
        loss_d = torch.tensor(0.).to(device)
        detector_optimizer.zero_grad()
        real_loss = wasserstein_loss(real, detector_model(nonwm_batch))
        fake_loss = wasserstein_loss(fake, detector_model(wm_batch))
        loss_d += (fake_loss + real_loss) / 2
        loss_d.backward()
        detector_optimizer.step()
    if epoch == 1:
        ex_strat_time = time.time()
        mse, watermark_acc = extract_spatial_watermark_1(target_model, networks, b_key_data, positions_key_data,
                                                         watermark)
        ex_end_time = time.time()
        print('\n\033[1;31mThe watermark embed time used is : \033[0m', wm_embed_end_time - wm_embed_start_time)
        print('\n\033[1;31mThe watermark extract time used is : \033[0m', ex_end_time - ex_strat_time)
    else:
        mse, watermark_acc = extract_spatial_watermark_1(target_model, networks, b_key_data, positions_key_data,
                                                         watermark)
    selected_weights = get_selected_weight(target_model, networks, positions_key_data)
    adjust_lr_rate(epoch, target_optimizer)
    train_acc = train_correct / len(train_loader.dataset)
    train_acces.append(train_acc.cpu().numpy().tolist())
    print('\tTrain Accuracy:{:.2f}%\twatermark_acc{:.2f}%'.format(100. * train_acc, 100. * watermark_acc))
    print('\tspatial_watermark loss:{:.5f}\tdetector loss:{:.5f}'.format(loss_s.item(), loss_d.item()))
    watermark_acces.append(watermark_acc)

    return watermark_acc, selected_weights

def return_proof_dict():
    return proof_dict

def return_train_result():
    return train_acces, train_losses, test_acces, test_losses, watermark_acces

# test routine
def test_watermark(model, epoch, criterion, b_key_data, positions_key_data, model_path):
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
        test_acces.append(((correct_top5 / (len(test_loader))).item()) / 100.0)
    else:
        test_acces.append(acc.cpu().numpy().tolist())

    test_loss /= len(test_loader.dataset)  # mean loss over the 10000 test samples
    test_losses.append(test_loss)  # record test_loss for this epoch
    # save the model after each batch
    if model_path is not None:
        torch.save(model.state_dict(), model_path)

    if test_acces[-1] >= max(test_acces) and epoch > 1:
        # save the model after each batch
        # torch.save(model.state_dict(), model_path)

        # save the optimizer after each batch
        # torch.save(optimizer.state_dict(), optimizer_path)
        print('\n\033[1;31mThe network Max Avg Accuracy : {:.2f}%\033[0m'.format(100. * max(test_acces)))

    # print e.g. Test set: Avg. loss: 2.3129, Accuracy: 1205/10000 (12%)
    '''
    print('\rTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),end = '')
    '''
    print('\r Test set \033[1;31m{}\033[0m : Avg. loss: {:.4f}, Accuracy: {}/{}  \033[1;31m({:.2f}%)\033[0m\n' \
          .format(epoch, test_loss, correct, len(test_loader.dataset), 100. * acc), end='')

def calculate_accuracy(output, target, topk=(1, 5)):
    """compute Top-1 and Top-5 accuracy
    Args:
        output (torch.Tensor): model output (batch_size, num_classes)
        target (torch.Tensor): ground-truth labels (batch_size)
        topk (tuple): which top-k accuracies to compute
    Returns:
        list: the requested top-k accuracies
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # put target on the same device as output
    target = target.to(output.device)

    # take the top-k predictions
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # pred.shape = (batch_size, maxk)
    pred = pred.t()  # after transposing, pred.shape = (maxk, batch_size)

    # test whether the top-k predictions contain the true label
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # correct.shape = (maxk, batch_size)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)  # count the correct predictions among the top k
        res.append(correct_k / batch_size * 100)  # as a percentage
    return res

def test_dp_model(model, epoch, criterion):
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
    acc = correct / len(test_loader.dataset)
    if dataset == 'imagenet':
        print('\r \033[1;31mDP Test set {}\033[0m : Avg. loss: {:.4f}, Accuracy: {}/{}  \033[1;31m({:.2f}%)\033[0m\n' \
              .format(epoch, test_loss, correct_top5, len(test_loader.dataset), correct_top5 / len(test_loader)), end='')
        return ((correct_top5 / (len(test_loader))).item()) / 100.0
    else:
        print('\r \033[1;31mDP Test set {}\033[0m : Avg. loss: {:.4f}, Accuracy: {}/{}  \033[1;31m({:.2f}%)\033[0m\n' \
              .format(epoch, test_loss, correct, len(test_loader.dataset), 100. * acc), end='')
        return acc.cpu().numpy().tolist()

def train_wm(target_model, target_loss_func, target_optimizer, epoch, positions_key_data, b_key_data, watermark,
             networks):
    train_correct = 0
    target_model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        ######train the generator
        # 1 - zero the gradients
        target_optimizer.zero_grad()
        # 2 - feed one batch of images and run the forward pass
        output = target_model(data.to(device))

        # 3 - compute the loss
        loss_o = torch.tensor(0.).to(device)
        loss_s = torch.tensor(0.).to(device)
        loss_o += target_loss_func(output, target.to(device))
        if wm_methods == 'EDNN':
            loss_s += embed_watermark_EDNN(target_model, b_key_data, positions_key_data, watermark, alpha, networks) #
        elif wm_methods == 'RIGA':
            loss_s += embed_spatial_watermark_1(target_model, b_key_data, positions_key_data, watermark, alpha, networks)
        elif wm_methods == 'FedIPR':
            sign_loss = embed_watermark_FedIPR(networks, watermark, target_model, alpha)
            loss_s = sign_loss.get_loss()
            # loss_s = 0
        loss_a = loss_o + loss_s

        # 4 - backpropagate
        loss_a.backward()

        # 5 - update the parameters
        target_optimizer.step()

        train_pred = output.data.max(dim=1, keepdim=True)[1]  # take the highest-scoring class from output,
        # dim=1 takes the row-wise maximum; [1] returns the argmax index rather than the value at [0]

        train_correct += train_pred.eq(target.data.view_as(train_pred).to(device)).sum()  # count the correctly classified samples

        print('\r  {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss_o.item()), end='')

        # every 10th batch (log_interval = 10)
        if batch_idx % log_interval == 0:
            # print(batch_idx)
            # print the epoch index, the image index and the training set size, progress percentage and current loss
            '''print('\r  {}  Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()),end = '')
            '''
            # append the current loss to train_losses, used for plotting later
            train_losses.append(loss_o.item())
            # counter
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            '''
            # save the model after each batch
            torch.save(network.state_dict(), './model.pth')

            # save the optimizer after each batch
            torch.save(optimizer.state_dict(), './optimizer.pth')
            '''
    if wm_methods == 'EDNN':
        mse, watermark_acc = extract_watermark_EDNN(target_model, b_key_data, positions_key_data, watermark, networks)
    elif wm_methods == 'RIGA':
        mse, watermark_acc = extract_spatial_watermark_1(target_model, networks, b_key_data, positions_key_data,
                                                         watermark)
    elif wm_methods == 'FedIPR':
        watermark_acc = extract_watermark_FedIPR(target_model, watermark, networks)
    selected_weights = get_selected_weight(target_model, networks, positions_key_data)
    adjust_lr_rate(epoch, target_optimizer)
    train_acc = train_correct / len(train_loader.dataset)
    train_acces.append(train_acc.cpu().numpy().tolist())
    print('\tTrain Accuracy:{:.2f}%\twatermark_acc{:.2f}%'.format(100. * train_acc, 100. * watermark_acc))
    watermark_acces.append(watermark_acc)

    return watermark_acc, selected_weights


if __name__ == '__main__':
    pass
