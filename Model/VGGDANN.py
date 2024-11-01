import torch
import torch.nn as nn
from torch.autograd import Function


class CNN(torch.nn.Module):
    def __init__(self, conv_archs, batch_size, input_channels=1):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        # CNN参数
        self.conv_archs = conv_archs  # 网络结构
        self.input_channels = input_channels
        self.features = self.make_layers()

    # 卷积池化结构
    def make_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_archs:
            for _ in range(num_convs):
                layers.append(torch.nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(torch.nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(torch.nn.MaxPool1d(kernel_size=2, stride=2))
        return torch.nn.Sequential(*layers)

    # 定义前向传播
    def forward(self, input_seq):
        # 改变输入形状，适应网络输入 （batch,H_in,seq_length）
        # input_seq = input_seq.view(self.batch_size, 1, 2048)
        features = self.features(input_seq)
        # X = self.avgpool(features)
        # flat_tensor = X.view(self.batch_size, -1)
        # output = self.classfier(flat_tensor)
        return features


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(nn.Module):
    def __init__(self, batch_size, domain_count, num_classes=10, input_channels=1):
        super(DANN, self).__init__()
        self.batch_size = batch_size
        self.domain_count = domain_count
        # self.conv_archs = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))  # vgg16  这个结构用不了
        # self.conv_archs = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # VGG11
        self.conv_archs = ((2, 64), (1, 256), (1, 512))  # 浅层
        # self.conv_archs = ((2, 32), (1, 64), (1, 128))  # 浅层

        # VGG16特征提取
        self.features = CNN(self.conv_archs, batch_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 构建源域标签分类器和域鉴别器
        self.task_classifier = nn.Sequential(
            nn.Linear(512, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes)
        )
        # self.task_classifier = nn.Sequential(nn.AdaptiveAvgPool1d(1),
        #                                      nn.Flatten(),
        #                                      nn.Linear(512, num_classes))
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
