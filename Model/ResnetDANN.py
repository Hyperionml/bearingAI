import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


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
    def __init__(self, batch_size, input_channels=1):
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
        # x = x.view(self.batch_size, 1, 2048)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        # output = self.fc(x)
        return x


class DANN(nn.Module):
    def __init__(self, batch_size, domain_count, num_classes=10, input_channels=1):
        super(DANN, self).__init__()
        self.batch_size = batch_size
        self.domain_count = domain_count

        # Resnet特征提取
        self.features = RESNET(batch_size)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 构建源域标签分类器和域鉴别器
        self.task_classifier = nn.Sequential(
            nn.Linear(512, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, self.domain_count)
        )
        self.GRL = GRL  # 梯度反转层

    def forward(self, x, alpha):
        x = x.view(self.batch_size, 1, 2048)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        task_predict = self.task_classifier(x)  # 源域标签分类预测
        x = self.GRL.apply(x, alpha)  # ！
        domain_predict = self.domain_classifier(x)  # 域鉴别器
        # print(task_predict,domain_predict)
        return task_predict, domain_predict
