# coding = utf-8
import os

import matplotlib
import torch
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt
from torch._jit_internal import loader
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import time

def make_dataset(data_list, labels):
    if labels:
        len_ = len(data_list)
        datas = [(data_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(data_list[0].split()) > 2:
            datas = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in data_list]
        else:
            datas = [(val.split()[0], int(val.split()[1])) for val in data_list]
    return datas


def txt_loader(path):
    try:
        with open(path, 'r') as file:
            content = file.read()
        values_as_strings = content.split()
        numeric_values = [float(value) for value in values_as_strings]
        one_dimensional_array = np.array(numeric_values)
        return one_dimensional_array
    except Exception as e:
        print(f"Error: {e}")
        return None


class DataList(Dataset):
    def __init__(self, data_list, std=0.1, labels=None, transform=None, target_transform=None, mode='TXT',
                 choose='train'):
        contents = make_dataset(data_list, labels)
        # [('./DATA/CRWU_1d_2048/data/B021/B021_Fan_12k_1_240.txt', 9),
        # ('./DATA/CRWU_1d_2048/data/IR007/IR007_Fan_12k_1_534.txt', 1),...]
        if len(contents) == 0:
            raise (RuntimeError("ɶ��û��"))

        loader_dict = {'TXT': txt_loader}
        self.loader = loader_dict.get(mode)
        self.contents = contents
        self.transform = transform
        self.target_transform = target_transform

        if transform is not None:
            paths, _ = zip(*contents)
            datas = [self.loader(path) for path in paths]
            datas = np.array(datas)
            self.transform = transform(datas, std=std, ctype=mode, choose=choose)

    def __getitem__(self, index):
        path, target = self.contents[index]
        data = self.loader(path)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.contents)


# 数据载入

def data_load(args):
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.dset_path).readlines()

    dsize = len(txt_src)

    tr_size = int(dsize * 0.7)
    te_size = int(dsize * 0.15)
    test_size = int(dsize - tr_size - te_size)

    tr_txt, te_txt, txt_test = torch.utils.data.random_split(txt_src, [tr_size, te_size, test_size])

    print("原训练集长度：{}\n原测试集长度：{}\n目标测试集长度：{}".format(len(tr_txt), len(te_txt),
                                                    len(txt_test)))

    dsets["train"] = DataList(tr_txt, mode='TXT', choose='train')
    dset_loaders["train"] = DataLoader(dsets["train"], batch_size=train_bs, shuffle=True,
                                       num_workers=args.worker, drop_last=False)
    dsets["val"] = DataList(te_txt, mode='TXT', choose='test')
    dset_loaders["val"] = DataLoader(dsets["val"], batch_size=train_bs, shuffle=True,
                                     num_workers=args.worker, drop_last=False)
    dsets["test"] = DataList(txt_test, mode='TXT', choose='test')
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


class CNN(torch.nn.Module):
    def __init__(self, conv_archs, num_classes, batch_size, input_channels=1):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        # CNN参数
        self.conv_archs = conv_archs   # 网络结构
        self.input_channels = input_channels
        self.features = self.make_layers()
        self.avgpool = torch.nn.AdaptiveAvgPool1d(9)  # 9
        # 定义全连接层
        self.classfier = torch.nn.Sequential(
            torch.nn.Linear(512 * 3 * 3, 500),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(500, num_classes)
        )

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
        input_seq = input_seq.view(self.batch_size, 1, 2048)
        features = self.features(input_seq)
        X = self.avgpool(features)
        flat_tensor = X.view(self.batch_size, -1)
        output = self.classfier(flat_tensor)
        return output


# 训练函数
def train_source(args):
    dset_loaders = loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## 网络初始化

    # conv_archs = ((2, 32), (1, 64), (1, 128))  # 浅层
    conv_archs = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))  # vgg16
    batch_size = args.batch_size
    num_classes = args.class_num
    model = CNN(conv_archs, num_classes, batch_size).to(device)
    # model = model.to(args.gpu_id)
    print("CNNģ�ͽṹ��", model)

    ## 优化器设置


    learn_rate = 0.0003
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)
    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss()

    ## 参数初始化

    epochs = args.max_epoch
    best_acc = 0.0
    best_model = model
    train_loss_list = []  # 记录在训练集上每个epoch的loss的变化情况
    train_acc_list = []  # 记录在训练集上每个epoch的准确率的变化情况
    val_acc_list = []
    val_loss_list = []

    model.train()

    ## 迭代训练过程

    start_time = time.time()
    for epoch in range(epochs):
        loss_epoch = 0.
        corr_epoch = 0
        iter_source_tr = iter(dset_loaders["train"])
        iter_source_val = iter(dset_loaders["val"])
        train_size = dset_loaders["train"].__len__() * 32
        test_size = dset_loaders["val"].__len__() * 32
        model.batch_size = args.batch_size
        for batch_idx, (inputs, labels) in enumerate(iter_source_tr):
            # 输出inputs为tensor(32,2048) 32为batch_size 2048为样本维度
            # 输出labels为tensor(32,) 32为batch_size
            inputs = inputs.to(torch.float32).to(device)  # (32,2048)
            labels = labels.to(device)
            optimizer.zero_grad()
            if len(inputs) < batch_size:
                model.batch_size = len(inputs)
            pred = model(inputs)  # shape = (32,10)
            probabilities = F.softmax(pred, dim=1)
            labels_pred = torch.argmax(probabilities, dim=1)

            corr_epoch += (labels_pred == labels).sum().item()
            loss = loss_function(pred, labels)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()
            # print(inputs)
            # break
            # 准确率

        train_acc = corr_epoch / train_size
        train_loss = loss_epoch / train_size
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        print(f'Epoch:{epoch + 1:2} train_Loss:{train_loss:10.8f} train_Accuracy:{train_acc:4.4f}')

        loss_epoch = 0.
        corr_epoch = 0
        model.batch_size = args.batch_size
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(iter_source_val):
                inputs = inputs.to(torch.float32).to(device)
                labels = labels.to(device)
                if len(inputs) < batch_size:
                    model.batch_size = len(inputs)
                pred = model(inputs)
                probabilities = F.softmax(pred, dim=1)
                labels_pred = torch.argmax(probabilities, dim=1)

                corr_epoch += (labels_pred == labels).sum().item()
                loss = loss_function(pred, labels)
                loss_epoch += loss.item()
        val_acc = corr_epoch / test_size
        val_loss = loss_epoch / test_size
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print(f'Epoch:{epoch + 1:2} val_Loss:{val_loss:10.8f} val_Accuracy:{val_acc:4.4f}')
        # 保存当前最优模型参数
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model  # 更新最佳模型的参数
    # 保存最好的参数
    torch.save(best_model, 'best_model_cnn1d_vgg16.pt')
    ## 性能评估

    matplotlib.rc("font", family='Microsoft YaHei')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    ax[0].plot(range(epochs), train_loss_list, color='b', label='train_loss')
    ax[0].plot(range(epochs), val_loss_list, color='r', label='validate_loss')
    ax[0].legend()
    ax[0].set_title("模型损失变化")
    ax[1].plot(range(epochs), train_acc_list, color='b', label='train_acc')
    ax[1].plot(range(epochs), val_acc_list, color='r', label='validate_acc')
    ax[1].legend()
    ax[1].set_title("模型训练准确率变化")
    plt.savefig("图2.jpg")  # 会覆盖原来的图片
    plt.show()  # 显示 lable
    print("best_accuracy :", best_acc)

    return


# 测试函数
def test_source(args):
    dset_loaders = data_load(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_function = torch.nn.CrossEntropyLoss()
    model = torch.load('best_model_cnn1d_vgg16.pt')
    batch_size = args.batch_size * 2

    iter_source_test = iter(dset_loaders["test"])
    test_size = dset_loaders["test"].__len__() * 64
    loss_epoch = 0.
    corr_epoch = 0
    model.batch_size = batch_size
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, labels) in enumerate(iter_source_test):
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(device)
            if len(inputs) < batch_size:
                model.batch_size = len(inputs)
            pred = model(inputs)
            probabilities = F.softmax(pred, dim=1)
            labels_pred = torch.argmax(probabilities, dim=1)

            corr_epoch += (labels_pred == labels).sum().item()
            loss = loss_function(pred, labels)
            loss_epoch += loss.item()
    test_acc = corr_epoch / test_size
    test_loss = loss_epoch / test_size
    print(f'test_Loss:{test_loss:10.8f} test_Accuracy:{test_acc:4.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neighbors')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")

    args = parser.parse_args()

    args.username = 'WWY'
    args.file_name = 'CRWU_1d_Fan_12k_2048'
    args.data_name = 'CRWU_1d_Fan_12k_2048'
    names = ['Fan_12k_0', 'Fan_12k_1', 'Fan_12k_2', 'Fan_12k_3']
    args.class_num = 10
    folder = './DATA/CRWU_1d_2048/'

    # if args.dset == 'CRWU_1d_Drive_12k_2048':
    #     names = ['Drive_12k_0', 'Drive_12k_1', 'Drive_12k_2', 'Drive_12k_3']
    #     args.class_num = 10
    # if args.dset == 'CRWU_1d_Fan_12k_2048':
    #     names = ['Fan_12k_0', 'Fan_12k_1', 'Fan_12k_2', 'Fan_12k_3']
    #     args.class_num = 10
    # if args.dset == 'CRWU_1d_Drive_48k_2048':
    #     names = ['Drive_48k_0', 'Drive_48k_1', 'Drive_48k_2', 'Drive_48k_3']
    #     args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    args.dset_path = folder + names[1] + '_label.txt'
    print("源域读取路径：{}".format(args.dset_path))

    print("源域训练开始")
    train_source(args)
    print("源域训练结束")
    print("源域测试开始")
    test_source(args)
    print("源域测试结束")
