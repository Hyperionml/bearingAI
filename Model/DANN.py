import torch
import torch.nn as nn
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


class DANN(nn.Module):
    def __init__(self, batch_size, domain_count, num_classes=10, input_channels=1):
        super(DANN, self).__init__()
        self.batch_size = batch_size
        self.domain_count = domain_count
        # 浅层网络特征提取
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5),
            nn.Conv1d(64, 128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(5)

        # 构建源域标签分类器和域鉴别器
        self.task_classifier = nn.Sequential(
            nn.Linear(128 * 5, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes)
        )
        # self.task_classifier = nn.Sequential(nn.AdaptiveAvgPool1d(1),
        #                                      nn.Flatten(),
        #                                      nn.Linear(512, num_classes))
        self.domain_classifier = nn.Sequential(
            nn.Linear(128 * 5, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, self.domain_count)
        )
        self.GRL = GRL  # 梯度反转层

    def forward(self, x, alpha):
        x = x.view(self.batch_size, 1, 2048)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)

        task_predict = self.task_classifier(x)  # 源域标签分类预测
        x = self.GRL.apply(x, alpha)  # ！
        domain_predict = self.domain_classifier(x)  # 域鉴别器
        # print(task_predict,domain_predict)
        return task_predict, domain_predict
