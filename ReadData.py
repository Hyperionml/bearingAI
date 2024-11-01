import argparse

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Graph.GeneralPlot import box_hit_plot, t_SNE, confusion_matrix
from Log.Logs import writeLogs
from Log.log import uselog
from AddNoise import add_gaussian_noise

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


"""
无标准化处理
"""


class DataList(Dataset):
    def __init__(self, data_list, std=0.1, labels=None, transform=None, target_transform=None, mode='TXT',
                 choose='train'):
        contents = make_dataset(data_list, labels)
        if len(contents) == 0:
            raise RuntimeError("啥都没有")

        loader_dict = {'TXT': txt_loader}
        self.loader = loader_dict.get(mode)
        self.contents = contents
        self.data_transform = transform  # 保留transform函数，不是数据
        self.target_transform = target_transform

        # 加载全部数据，用于计算标准化参数
        paths, _ = zip(*contents)
        datas = [self.loader(path) for path in paths]
        datas = np.array(datas)
        self.scaler = StandardScaler()
        self.scaler.fit(datas)  # 只在这里进行fit

    def __getitem__(self, index):
        path, target = self.contents[index]
        data = self.loader(path)
        data = data.reshape(1, -1)  # 确保data是二维的，即使是单个样本
        data_normalized = self.scaler.transform(data)  # 使用transform而不是fit_transform

        if self.data_transform is not None:
            data_normalized = self.data_transform(data_normalized)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data_normalized[0], target  # 返回原始形状的数据

    def __len__(self):
        return len(self.contents)


"""
Min-Max 标准化
"""


class MinMaxDataList(Dataset):
    def __init__(self, data_list, std=0.1, labels=None, transform=None, target_transform=None, mode='TXT',
                 choose='train'):
        contents = make_dataset(data_list, labels)
        if len(contents) == 0:
            raise RuntimeError("啥都没有")

        loader_dict = {'TXT': txt_loader}
        self.loader = loader_dict.get(mode)
        self.contents = contents
        self.data_transform = transform  # 保留transform函数，不是数据
        self.target_transform = target_transform

        # 加载全部数据，用于计算标准化参数
        paths, _ = zip(*contents)
        datas = [self.loader(path) for path in paths]
        datas = np.vstack(datas)  # 将所有一维数据堆叠成二维数组
        self.scaler = MinMaxScaler()
        self.scaler.fit(datas)  # 使用MinMaxScaler进行fit

    def __getitem__(self, index):
        path, target = self.contents[index]
        data = self.loader(path)
        data = data.reshape(1, -1)  # 确保data是二维的
        data_normalized = self.scaler.transform(data)  # 使用transform而不是fit_transform

        if self.data_transform is not None:
            data_normalized = self.data_transform(data_normalized)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data_normalized[0], target  # 返回原始形状的数据

    def __len__(self):
        return len(self.contents)


"""
Z-score标准化
"""


class NormalizedDataList(Dataset):
    def __init__(self, data_list, std=0.1, labels=None, transform=None, target_transform=None, mode='TXT',
                 choose='train'):
        contents = make_dataset(data_list, labels)
        if len(contents) == 0:
            raise RuntimeError("啥都没有")

        loader_dict = {'TXT': txt_loader}
        self.loader = loader_dict.get(mode)
        self.contents = contents
        self.data_transform = transform  # 保留transform函数，不是数据
        self.target_transform = target_transform

        # 加载全部数据，用于计算标准化参数
        paths, _ = zip(*contents)
        datas = [self.loader(path) for path in paths]
        datas = np.vstack(datas)  # 将所有一维数据堆叠成二维数组
        self.scaler = StandardScaler()
        self.scaler.fit(datas)  # 只在这里进行fit

    def __getitem__(self, index):
        path, target = self.contents[index]
        data = self.loader(path)
        data = data.reshape(1, -1)  # 确保data是二维的
        data_normalized = self.scaler.transform(data)  # 使用transform而不是fit_transform

        if self.data_transform is not None:
            data_normalized = self.data_transform(data_normalized)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data_normalized[0], target  # 返回原始形状的数据

    def __len__(self):
        return len(self.contents)


class NoiseDataList(Dataset):
    def __init__(self, data_list, snr=-7., std=0.1, labels=None, transform=None, target_transform=None, mode='TXT',
                 choose='train'):
        contents = make_dataset(data_list, labels)
        # [('./DATA/CRWU_1d_2048/data/B021/B021_Fan_12k_1_240.txt', 9),
        # ('./DATA/CRWU_1d_2048/data/IR007/IR007_Fan_12k_1_534.txt', 1),...]
        if len(contents) == 0:
            raise (RuntimeError("啥都没有"))

        loader_dict = {'TXT': txt_loader}
        self.loader = loader_dict.get(mode)
        self.contents = contents
        self.transform = transform
        self.target_transform = target_transform
        self.snr = snr
        if transform is not None:
            paths, _ = zip(*contents)
            datas = [self.loader(path) for path in paths]
            datas = np.array(datas)
            self.transform = transform(datas, std=std, ctype=mode, choose=choose)

    def __getitem__(self, index):
        path, target = self.contents[index]
        data = add_gaussian_noise(self.snr, self.loader(path))

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.contents)


# 数据载入
def data_load(args):
    """
    :param args: 全局变量,
        args.batch_size
        args.dset_path
        args.Scaler=="Min-Max"
    :return: dset_loaders
    """
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.dset_path).readlines()

    dsize = len(txt_src)

    tr_size = int(dsize * 0.7)
    te_size = int(dsize * 0.15)
    test_size = int(dsize - tr_size - te_size)

    tr_txt, te_txt, txt_test = torch.utils.data.random_split(txt_src, [tr_size, te_size, test_size])

    args.logger.info(
        "原训练集长度：{}\t原测试集长度：{}\t目标测试集长度：{}".format(len(tr_txt), len(te_txt), len(txt_test)))

    if args.Scaler == "Min-Max":
        dsets["train"] = MinMaxDataList(tr_txt, mode='TXT', choose='train')
        dset_loaders["train"] = DataLoader(dsets["train"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
        dsets["val"] = MinMaxDataList(te_txt, mode='TXT', choose='test')
        dset_loaders["val"] = DataLoader(dsets["val"], batch_size=train_bs, shuffle=True,
                                         num_workers=args.worker, drop_last=False)
        dsets["test"] = MinMaxDataList(txt_test, mode='TXT', choose='test')
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True,
                                          num_workers=args.worker, drop_last=False)
    elif args.Scaler == "Z-Score":
        dsets["train"] = NormalizedDataList(tr_txt, mode='TXT', choose='train')
        dset_loaders["train"] = DataLoader(dsets["train"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
        dsets["val"] = NormalizedDataList(te_txt, mode='TXT', choose='test')
        dset_loaders["val"] = DataLoader(dsets["val"], batch_size=train_bs, shuffle=True,
                                         num_workers=args.worker, drop_last=False)
        dsets["test"] = NormalizedDataList(txt_test, mode='TXT', choose='test')
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True,
                                          num_workers=args.worker, drop_last=False)
    elif args.Scaler == "Noise-Add":
        dsets["train"] = NoiseDataList(tr_txt, mode='TXT', choose='train')
        dset_loaders["train"] = DataLoader(dsets["train"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
        dsets["val"] = NoiseDataList(te_txt, mode='TXT', choose='test')
        dset_loaders["val"] = DataLoader(dsets["val"], batch_size=train_bs, shuffle=True,
                                         num_workers=args.worker, drop_last=False)
        dsets["test"] = NoiseDataList(txt_test, mode='TXT', choose='test')
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True,
                                          num_workers=args.worker, drop_last=False)
    else:
        dsets["train"] = DataList(tr_txt, mode='TXT', choose='train')
        dset_loaders["train"] = DataLoader(dsets["train"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
        dsets["val"] = DataList(te_txt, mode='TXT', choose='test')
        dset_loaders["val"] = DataLoader(dsets["val"], batch_size=train_bs, shuffle=True,
                                         num_workers=args.worker, drop_last=False)
        dsets["test"] = DataList(txt_test, mode='TXT', choose='test')
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True,
                                          num_workers=args.worker, drop_last=False)

    return dset_loaders


def d2d_data_load(args):
    """
    :param args: 全局变量,
        args.batch_size
        args.source_dset_path
        args.target_dset_path
        args.Scaler=="Min-Max"
    :return: dset_loaders
    """
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_source = open(args.source_dset_path).readlines()
    txt_target = open(args.target_dset_path).readlines()
    dsize_source = len(txt_source)
    dsize_target = len(txt_target)
    tr_size_source = int(dsize_source * 0.8)
    te_size_source = int(dsize_source * 0.2)
    tr_size_target = int(dsize_target * 0.8)
    te_size_target = int(dsize_target * 0.2)
    tr_txt_source, te_txt_source = torch.utils.data.random_split(txt_source, [tr_size_source, te_size_source])
    tr_txt_target, te_txt_target = torch.utils.data.random_split(txt_target, [tr_size_target, te_size_target])
    #
    args.logger.info(
        f"源域训练集长度：{tr_size_source}\t目标域训练长度：{tr_size_target}\t源域测试长度：{tr_size_target}\t目标域测试长度:{dsize_target}")

    if args.Scaler == "Min-Max":

        dsets["source_train"] = MinMaxDataList(tr_txt_source, mode='TXT', choose='train')
        dset_loaders["source_train"] = DataLoader(dsets["source_train"], batch_size=train_bs, shuffle=True,
                                                  num_workers=args.worker, drop_last=False)
        dsets["source_test"] = MinMaxDataList(te_txt_source, mode='TXT', choose='test')
        dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size=train_bs, shuffle=True,
                                                 num_workers=args.worker, drop_last=False)
        dsets["target_train"] = MinMaxDataList(tr_txt_target, mode='TXT', choose='train')
        dset_loaders["target_train"] = DataLoader(dsets["target_train"], batch_size=train_bs, shuffle=True,
                                                  num_workers=args.worker, drop_last=False)
        dsets["target_test"] = MinMaxDataList(te_txt_target, mode='TXT', choose='test')
        dset_loaders["target_test"] = DataLoader(dsets["target_test"], batch_size=train_bs, shuffle=True,
                                                 num_workers=args.worker, drop_last=False)
        dsets["target"] = MinMaxDataList(txt_target, mode='TXT', choose='test')  #
        dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                            num_workers=args.worker, drop_last=False)
    elif args.Scaler == "Z-Score":
        dsets["source_train"] = NormalizedDataList(tr_txt_source, mode='TXT', choose='train')
        dset_loaders["source_train"] = DataLoader(dsets["source_train"], batch_size=train_bs, shuffle=True,
                                                  num_workers=args.worker, drop_last=False)
        dsets["source_test"] = NormalizedDataList(te_txt_source, mode='TXT', choose='test')
        dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size=train_bs, shuffle=True,
                                                 num_workers=args.worker, drop_last=False)
        dsets["target_train"] = NormalizedDataList(tr_txt_target, mode='TXT', choose='train')
        dset_loaders["target_train"] = DataLoader(dsets["target_train"], batch_size=train_bs, shuffle=True,
                                                  num_workers=args.worker, drop_last=False)
        dsets["target_test"] = NormalizedDataList(te_txt_target, mode='TXT', choose='test')
        dset_loaders["target_test"] = DataLoader(dsets["target_test"], batch_size=train_bs, shuffle=True,
                                                 num_workers=args.worker, drop_last=False)
        dsets["target"] = NormalizedDataList(txt_target, mode='TXT', choose='test')  #
        dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                            num_workers=args.worker, drop_last=False)
    else:
        dsets["source_train"] = DataList(tr_txt_source, mode='TXT', choose='train')
        dset_loaders["source_train"] = DataLoader(dsets["source_train"], batch_size=train_bs, shuffle=True,
                                                  num_workers=args.worker, drop_last=False)
        dsets["source_test"] = DataList(te_txt_source, mode='TXT', choose='test')
        dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size=train_bs, shuffle=True,
                                                 num_workers=args.worker, drop_last=False)
        dsets["target_train"] = DataList(tr_txt_target, mode='TXT', choose='train')
        dset_loaders["target_train"] = DataLoader(dsets["target_train"], batch_size=train_bs, shuffle=True,
                                                  num_workers=args.worker, drop_last=False)
        dsets["target_test"] = DataList(te_txt_target, mode='TXT', choose='test')
        dset_loaders["target_test"] = DataLoader(dsets["target_test"], batch_size=train_bs, shuffle=True,
                                                 num_workers=args.worker, drop_last=False)
        dsets["target"] = DataList(txt_target, mode='TXT', choose='test')  #
        dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                            num_workers=args.worker, drop_last=False)
    return dset_loaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neighbors')
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    args = parser.parse_args()
    args.logger = uselog("logger")
    """测试数据读取"""
    folder = './DATA/CRWU_1d_2048/'
    names = ['Fan_12k_0', 'Fan_12k_1', 'Fan_12k_2', 'Fan_12k_3']
    args.dset_path = folder + names[0] + '_label.txt'
    args.batch_size = 32
    args.Scaler = ''
    dset_loaders = data_load(args)
    args.Scaler = 'Min-Max'
    dset_loaders1 = data_load(args)  # Min-Max 标准化
    # 训练集长度：219,验证集长度：47,测试集长度：24
    print(
        f'训练集长度：{dset_loaders["train"].__len__()},验证集长度：{dset_loaders["val"].__len__()},测试集长度：{dset_loaders["test"].__len__()}')
    iter_source_tr = iter(dset_loaders["train"])
    iter_source_val = iter(dset_loaders["val"])
    iter_source_test = iter(dset_loaders["test"])
    iter_source_tr1 = iter(dset_loaders1["train"])
    iter_source_val1 = iter(dset_loaders1["val"])
    iter_source_test1 = iter(dset_loaders1["test"])

    print("==================== 训练集样本和标签（data,label） ============================")
    inputs, labels = iter_source_tr.__next__()  # 未标准化数据
    inputs1, labels1 = iter_source_tr1.__next__()  # 标准化数据

    box_hit_plot(inputs.numpy().flatten(), inputs1.numpy().flatten(),f'./Graph/01-{args.Scaler}标准化训练数据图.png')
    # print(inputs_li, labels_li)
    # print(inputs_li.shape, labels_li.shape)
    print("----------------------------------------------------------------------------")
    print(inputs1.numpy().flatten())

    print("==================== 验证集样本和标签（data,label） ============================")
    inputs, labels = iter_source_val.__next__()  # 未标准化数据
    inputs2, labels2 = iter_source_val1.__next__()  # 未标准化数据
    box_hit_plot(inputs.numpy().flatten(), inputs2.numpy().flatten(),f'./Graph/02-{args.Scaler}标准化验证数据图.png')

    print(inputs, labels)
    print("----------------------------------------------------------------------------")
    print(inputs2.numpy().flatten())
    print("==================== 测试集样本和标签（data,label） ============================")
    inputs, labels = iter_source_test.__next__()  # 未标准化数据
    inputs3, labels3 = iter_source_test1.__next__()  # 未标准化数据
    box_hit_plot(inputs.numpy().flatten(), inputs3.numpy().flatten(),f'./Graph/03-{args.Scaler}标准化测试数据图.png')

    print(inputs, labels)
    print("----------------------------------------------------------------------------")
    print(inputs3.numpy().flatten())
#
#     """端到端数据测试"""
#     args.source_dset_path = folder + names[1] + '_label.txt'
#     args.target_dset_path = folder + names[2] + '_label.txt'
#     args.Scaler = ''
#     d2d_dset_loaders = d2d_data_load(args)
#     args.Scaler = 'Min-Max'
#     d2d_dset_loaders1 = d2d_data_load(args)
#     print(
#     f'源域训练集长度：{d2d_dset_loaders["source_train"].__len__()}，源域测试集长度：{d2d_dset_loaders["source_test"].__len__()}')
#     print(
#     f'目标域训练集长度：{d2d_dset_loaders["target_train"].__len__()}，目标域测试集长度：{d2d_dset_loaders["target_test"].__len__()}')
#     # 未处理数据集
#     source_train_iter = iter(d2d_dset_loaders["source_train"])
#     target_train_iter = iter(d2d_dset_loaders["target_train"])
#     source_test_iter = iter(d2d_dset_loaders["source_test"])
#     target_test_iter = iter(d2d_dset_loaders["target_test"])
#     # Min-Max标准化处理
#     source_train_iter1 = iter(d2d_dset_loaders1["source_train"])
#     target_train_iter1 = iter(d2d_dset_loaders1["target_train"])
#     source_test_iter1 = iter(d2d_dset_loaders1["source_test"])
#     target_test_iter1 = iter(d2d_dset_loaders1["target_test"])
#
#     train_data_len = source_train_iter.__len__()
#     test_sourcedata_len = source_test_iter.__len__()
#     test_targetdata_len = target_test_iter.__len__()
#     print("==================== 源域和目标域训练集样本和标签（data,label） ============================")
#     source_inputs, source_labels = source_train_iter.__next__()  # 未标准化数据
#     source_inputs1, source_labels1 = source_train_iter1.__next__()  # 未标准化数据
#     box_hit_plot(source_inputs.numpy().flatten(), source_inputs1.numpy().flatten(),f'./Graph/04-{args.Scaler}标准化端到端源域训练数据图.png')
#
#     print(source_inputs, source_labels)
#     print("==================== 源域测试集样本和标签（data,label） ==================================")
#     source_inputs, source_labels = source_test_iter.__next__()  # 未标准化数据
#     source_inputs2, source_labels2 = source_test_iter1.__next__()  # 未标准化数据
#     box_hit_plot(source_inputs.numpy().flatten(), source_inputs2.numpy().flatten(),f'./Graph/05-{args.Scaler}标准化端到端源域测试数据图.png')
#     print(source_inputs, source_labels)
#     print("===================== 目标域测试集样本和标签（data,label） ===============================")
#     target_inputs, target_labels = target_test_iter.__next__()  # 未标准化数据
#     source_inputs3, source_labels3 = target_test_iter1.__next__()  # 未标准化数据
#     box_hit_plot(source_inputs.numpy().flatten(), source_inputs3.numpy().flatten(),f'./Graph/06-{args.Scaler}标准化端到端目标域测试数据图.png')
#     print(target_inputs, target_labels)
