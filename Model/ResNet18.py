import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        # CNN参数
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class RESNET(nn.Module):
    def __init__(self, batch_size,num_classes=10, input_channels=1):
        super(RESNET, self).__init__()
        self.batch_size = batch_size
        self.b1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(3),
                                nn.Linear(512 * 3, 500),
                                # nn.Dropout1d(0.5),  # 添加Dropout正则化
                                nn.Flatten(),
                                nn.Linear(500, num_classes))

    def resnet_block(self, input_channels, num_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def forward(self, x):
        x = x.view(self.batch_size, 1, 2048)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        output = self.fc(x)
        return output
