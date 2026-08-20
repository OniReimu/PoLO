
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, num_channels, num_classes):  # defaults to 10 classes (CIFAR-10)
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dynamically compute flattened feature size
        example_input = torch.zeros(1, num_channels, 64, 64)
        feature_output = self.features(example_input)
        flattened_size = feature_output.view(-1).shape[0]

        self.dense = nn.Sequential(
            nn.Linear(flattened_size, 4096),  # for CIFAR-10 or CIFAR-100
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.features(x)
        # print("Shape after features extraction:", out.shape)
        out = out.view(out.size(0), -1)
        # print("Shape after flattening:", out.shape)
        out = self.dense(out)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    # instantiate VGG16
    model = VGG16(num_channels=3, num_classes=200)

    # read the model state_dict
    state_dict = model.state_dict()

    # print all keys
    for key in state_dict.keys():
        # print(key)
        if 'bias' in key:
            continue

        weight_tensor = state_dict[key]
        num_elements = weight_tensor.numel()  # element count
        print(f'{key}: {num_elements} elements')
    x = torch.randn(128, 3, 64, 64)  # Batch size = 128, input size 64x64x3
    output = model(x)
    print(output.shape)  # expected [128, 200]
# features.0.weight
# features.0.bias
# features.2.weight
# features.2.bias
# features.5.weight
# features.5.bias
# features.7.weight
# features.7.bias
# features.10.weight
# features.10.bias
# features.12.weight
# features.12.bias
# features.14.weight
# features.14.bias
# features.17.weight
# features.17.bias
# features.19.weight
# features.19.bias
# features.21.weight
# features.21.bias
# features.24.weight
# features.24.bias
# features.26.weight
# features.26.bias
# features.28.weight
# features.28.bias
# classifier.0.weight
# classifier.0.bias
# classifier.3.weight
# classifier.3.bias
# classifier.6.weight
# classifier.6.bias