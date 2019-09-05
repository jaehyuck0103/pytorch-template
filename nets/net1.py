import torch
import torch.nn as nn
import torchvision


class Net1(nn.Module):
    def __init__(self):
        super().__init__()

        shuffle_net = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        modules = list(shuffle_net.children())[:-1]
        modules.append(nn.AdaptiveAvgPool2d([1, 1]))
        self.conv_net = nn.Sequential(*modules)
        self.fc = nn.Linear(1024, 10)

        '''
        res34 = torchvision.models.resnet34(pretrained=True)
        modules = list(res34.children())[:-1]
        self.conv_net = nn.Sequential(*modules)
        self.fc = nn.Linear(512, 10)
        '''

        '''
        res50 = torchvision.models.resnet50(pretrained=True)
        modules = list(res50.children())[:-1]
        self.conv_net = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, 10)
        '''

    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = self.conv_net(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        x = self.fc(x)  # (batch, 1)

        return x
