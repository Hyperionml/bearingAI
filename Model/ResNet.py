import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,skip,
                 use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.skip = skip
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
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        if self.skip is True:
            Y += X
        return F.relu(Y)


class Bottleneck(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        self.expansion = 4
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=1, stride=strides)
        self.bn1 = nn.BatchNorm1d(num_channels)

        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_channels)

        self.conv3 = nn.Conv1d(num_channels, num_channels * self.expansion,
                               kernel_size=1)
        self.bn3 = nn.BatchNorm1d(num_channels * self.expansion)
        if use_1x1conv:
            self.conv4 = nn.Conv1d(input_channels, num_channels * self.expansion,
                                   kernel_size=1, stride=strides)
            self.bn4 = nn.BatchNorm1d(num_channels * self.expansion)

        else:
            self.conv4 = None

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.conv4:
            X = self.conv4(X)
            X = self.bn4(X)
        # print(X.shape, out.shape)
        out += X
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, batch_size, num_classes=10, input_channels=1, skip=True):
        super(ResNet18, self).__init__()
        self.batch_size = batch_size
        self.skip = skip  # 进行笑容实验，用于控制是否使用残差连接
        # self.use_1x1conv = use_1x1conv  # 进行消融实验，用于控制是否使用模型的残差连接块
        self.b1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                # nn.Dropout1d(0.5),  # 添加Dropout正则化
                                nn.Flatten(),
                                nn.Linear(512, num_classes))

    def resnet_block(self, input_channels, num_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2, skip=self.skip))
            else:
                blk.append(Residual(num_channels, num_channels,self.skip))
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


class ResNet34(nn.Module):
    def __init__(self, batch_size, num_classes=10, input_channels=1):
        super(ResNet34, self).__init__()
        self.batch_size = batch_size
        self.b1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*self.resnet_block(64, 64, 3, first_block=True))
        self.b3 = nn.Sequential(*self.resnet_block(64, 128, 4))
        self.b4 = nn.Sequential(*self.resnet_block(128, 256, 6))
        self.b5 = nn.Sequential(*self.resnet_block(256, 512, 3))
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                # nn.Dropout1d(0.5),  # 添加Dropout正则化
                                nn.Flatten(),
                                nn.Linear(512, num_classes))

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


class ResNet50(nn.Module):
    def __init__(self, batch_size, num_classes=10, input_channels=1):
        super(ResNet50, self).__init__()
        self.batch_size = batch_size
        self.b1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*self.resnet_block(64, 64, 3, first_block=True))
        self.b3 = nn.Sequential(*self.resnet_block(256, 128, 4))
        self.b4 = nn.Sequential(*self.resnet_block(512, 256, 6))
        self.b5 = nn.Sequential(*self.resnet_block(1024, 512, 3))
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                nn.Flatten(),
                                nn.Linear(2048, num_classes))

    def resnet_block(self, input_channels, num_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and first_block:
                blk.append(Bottleneck(input_channels, num_channels, use_1x1conv=True))
            elif i == 0 and not first_block:
                blk.append(Bottleneck(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Bottleneck(num_channels * 4, num_channels))
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


class ResNet101(nn.Module):
    def __init__(self, batch_size, num_classes=10, input_channels=1):
        super(ResNet101, self).__init__()
        self.batch_size = batch_size
        self.b1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*self.resnet_block(64, 64, 3, first_block=True))
        self.b3 = nn.Sequential(*self.resnet_block(256, 128, 4))
        self.b4 = nn.Sequential(*self.resnet_block(512, 256, 23))
        self.b5 = nn.Sequential(*self.resnet_block(1024, 512, 3))
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                nn.Flatten(),
                                nn.Linear(2048, num_classes))

    def resnet_block(self, input_channels, num_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and first_block:
                blk.append(Bottleneck(input_channels, num_channels, use_1x1conv=True))
            elif i == 0 and not first_block:
                blk.append(Bottleneck(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Bottleneck(num_channels * 4, num_channels))
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


# resnet = ResNet101(32)
# print(resnet)
# x = torch.randn(32, 1, 2048)
# X = resnet(x)
# print(X.shape)
