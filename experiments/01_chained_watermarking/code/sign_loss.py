import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class SignLoss():
    def __init__(self, kwargs, model):
        super(SignLoss, self).__init__()
        self.alpha = 0.2  #self.sl_ratio
        self.loss = 0
        self.model = model
        self.kwargs = kwargs

    def get_loss(self):
        # print('self.kwargs====', self.kwargs)
        self.reset()
        for m in self.kwargs:
            if self.kwargs[m]['flag'] == True:
                b = self.kwargs[m]['b']
                M = self.kwargs[m]['M']

                b = b.to(torch.device('cuda'))
                M = M.to(torch.device('cuda'))

                weights = self.model.net[4].weight.view([1, -1])[:, :288]
                self.loss += (self.alpha * F.relu(-(weights.mm(M).mul(b.view(-1))))).sum()

        return self.loss

    def reset(self):
        self.loss = 0

    def add_resnet_module_loss(self, sublayer, module, b, M):

        if module == "convbnrelu_1":
            scale = self.model.layer4[int(sublayer)].convbnrelu_1.scale
            conv_w = torch.mean(self.model.layer4[int(sublayer)].convbnrelu_1.conv.weight, dim = 0) 
        if module == "convbn_2":
            scale = self.model.layer4[int(sublayer)].convbn_2.scale
            conv_w = torch.mean(self.model.layer4[int(sublayer)].convbn_2.conv.weight, dim = 0)

        self.loss += (self.alpha * F.relu(-scale.view([1, -1]).mm(M).mul(b.view(-1)))).sum()
