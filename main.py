import argparse
import os
import time

import torch
import numpy as np
import random

from matplotlib import pyplot as plt

import ReadData
from Graph.GeneralPlot import confusion_matrix, boxplot
from Log.Reporter import matrix_report, get_report
from training import training
from testing import testing, test_source, test_target,test_d2d_simple_source,test_d2d_simple_target
from Log.log import uselog


def exp(args):
    args.logger.info(f"======================= {args.model_name} =======================")
    start = time.time()
    x_li = []
    test_acc_str_li = []
    # label_list = ['hp0', 'hp1']
    label_list = ['hp0', 'hp1', 'hp2', 'hp3']
    color = ['lightblue', 'darkcyan', 'red', 'hotpink']
    labels1 = []
    labels2 = []
    for i in range(4):
        # 初始化结果存储
        results = []
        # 对模型进行多次测试减少偶然性
        nums = 20
        for num in range(nums):
            time1 = time.time()
            args.logger.info(f"----------- {args.model_name}-{names[i]} -----------")
            args.dset_path = folder + names[i] + '_label.txt'  # 数据加载路径
            args.model_path = f'./Model/model({args.model_name})_{names[i]}.pt'  # 模型保存路径
            args.dset_loaders = ReadData.data_load(args)  # 使用数据加载器
            args.logger.info(f'数据加载的时间：{time.time() - time1:.3f}s')
            time2 = time.time()
            training(args)
            args.logger.info(f'模型训练的时间：{time.time() - time2:.3f}s')
            time3 = time.time()
            if num == 0:
                test_acc, labels1, labels2 = testing(args, True)
            else:
                test_acc = testing(args, False)

            results.append(test_acc)
            args.logger.info(f'模型测试的时间：{time.time() - time3:.3f}s')
            args.logger.info(f'模型 {names[i]}_model({args.model_name}) 完整运行的时间：{time.time() - time1:.3f}s')
        # 计算每次测试的平均值和标准偏差
        averages = np.mean(results)
        std_devs = np.std(results)
        x_li.append(results)
        test_acc_str = f' Accuracy = {averages * 100:.2f}% ± {std_devs * 100:.2f}%'
        test_acc_str_li.append(test_acc_str)
        args.logger.info(test_acc_str)
        if args.Test is True:
            break

    if args.Test is False:
        args.logger.info(f"数据标准化方式：{args.Scaler}")
        matrix, report = matrix_report(labels1, labels2,
                                       f'./Graph/graph/{args.photo_nums}-模型（{args.model_name}-混淆矩阵（{names[0]}）.png')
        args.photo_nums += 1
        args.logger.info(f"混淆矩阵：\n"
                         f"{str(matrix)}\n"
                         f"预测报告：\n"
                         f"{report}")
        for i in range(len(test_acc_str_li)):
            args.logger.info(f'{names[i]}：{test_acc_str_li[i]}')
        # 绘制箱线图
        boxplot(x_li, label_list, color, f'./Graph/graph/{args.photo_nums}-模型（{args.model_name}）-ALL-Hp.png',
                if_avg=True)
        args.photo_nums += 1
    else:
        report = get_report(labels1, labels2)
        args.logger.info(f"预测报告：\n"
                         f"{report}")
    total_seconds = time.time() - start
    total_minutes = int(total_seconds // 60)
    remaining_seconds = int(total_seconds % 60)
    formatted_time = f'{total_minutes} 分 {remaining_seconds} 秒'
    args.logger.info(f'实验模型{args.model_name}运行的时间：{formatted_time}')


def d2d_exp(args):
    args.logger.info(f"======================= {args.model_name} =======================")
    start = time.time()
    x_li_source = []
    x_li_target = []
    test_acc_str_li_source = []
    test_acc_str_li_target = []
    labels1_source = []
    labels2_source = []
    labels1_target = []
    labels2_target = []
    label_list=['hp0->hp1','hp0->hp2','hp0->hp3',
                'hp1->hp0','hp1->hp2','hp1->hp3',
                'hp2->hp0','hp2->hp1','hp2->hp3',
                'hp3->hp0','hp3->hp1','hp3->hp2']
    color = ['lightblue', 'darkcyan', 'red', 'hotpink']

    for i in range(4):
        for j in range(4):
            if i != j:
                nums = 20
                results_source = []
                results_target = []
                for num in range(nums):
                        time1 = time.time()
                        args.logger.info(f"----------- {args.model_name}-{names[i]}->{names[j]} -----------")
                        args.source_dset_path = folder + names[i] + '_label.txt'  # 数据加载路径
                        args.target_dset_path = folder + names[j] + '_label.txt'  # 数据加载路径
                        args.model_path = f'./Model/model({args.model_name})_{names[i]}_{names[j]}.pt'  # 模型保存路径
                        args.dset_loaders = ReadData.d2d_data_load(args)  # 使用数据加载器
                        args.logger.info(f'数据加载的时间：{time.time() - time1:.3f}s')
                        time2 = time.time()
                        # 模型训练
                        training(args)
                        args.logger.info(f'模型训练的时间：{time.time() - time2:.3f}s')
                        time3 = time.time()
                        if num == 0:
                            test_acc_source, labels1_source, labels2_source = test_source(args, True)
                            test_acc_target, labels1_target, labels2_target = test_target(args, True)
                        else:
                            test_acc_source = test_source(args, False)
                            test_acc_target = test_target(args, False)
                        args.logger.info(f'模型测试的时间：{time.time() - time3:.3f}s')
                        args.logger.info(
                            f'模型 {names[i]}_model({args.model_name}) 完整运行的时间：{time.time() - time1:.3f}s')
                        results_source.append(test_acc_source)  # 源域准确率
                        results_target.append(test_acc_target)  # 目标域准确率
                        print(f'i={i},j={j},num={num}')
                    # 计算每次测试的平均值和标准偏差
                averages_source = np.mean(results_source)
                averages_target = np.mean(results_target)
                std_devs_source = np.std(results_source)
                std_devs_target = np.std(results_target)
                x_li_source.append(results_source)
                x_li_target.append(results_target)
                test_acc_str_source = f'{names[i]}--> {names[j]} Source Accuracy = {averages_source * 100:.2f}% ± {std_devs_source * 100:.2f}%'
                test_acc_str_target = f'{names[i]}--> {names[j]} Target Accuracy = {averages_target * 100:.2f}% ± {std_devs_target * 100:.2f}%'
                args.logger.info(test_acc_str_source)   # 输出nums轮训练后准确率的方差和均值
                args.logger.info(test_acc_str_target)
                test_acc_str_li_source.append(test_acc_str_source)
                test_acc_str_li_target.append(test_acc_str_target)
            if args.Test is True and i != j: # 控制单例测试
                break
    if args.Test is True:
        print('测试模式')
        report = get_report(labels1_source, labels2_source)
        args.logger.info(f"预测报告：\n"
                         f"{report}")
        report = get_report(labels1_target, labels2_target)
        args.logger.info(f"预测报告：\n"
                         f"{report}")
        for i in range(len(test_acc_str_li_source)):
            args.logger.info(f'{names[i]}：{test_acc_str_li_source[i]}')
        for i in range(len(test_acc_str_li_target)):
            args.logger.info(f'{names[i]}：{test_acc_str_li_target[i]}')
    else:
        print('非测试测试模式')

        args.logger.info(f"数据标准化方式：{args.Scaler}")
        matrix_source, report_source = matrix_report(labels1_source, labels2_source,
                                                     f'./Graph/graph/{args.photo_nums}-模型（{args.model_name}-源域混淆矩阵（{names[0]},{names[1]}）.png')
        args.photo_nums += 1
        matrix_target, report_target = matrix_report(labels1_target, labels2_target,
                                                     f'./Graph/graph/{args.photo_nums}-模型（{args.model_name}-目标域混淆矩阵（{names[0]},{names[1]}）.png')
        args.photo_nums += 1
        args.logger.info(f"混淆矩阵：\n"
                         f"{str(matrix_source)}\n"
                         f"预测报告：\n"
                         f"{report_source}")
        args.logger.info(f"混淆矩阵：\n"
                         f"{str(matrix_target)}\n"
                         f"预测报告：\n"
                         f"{report_target}")

        for i in range(4):
            for j in range(3):
                args.logger.info(f'{names[i]}：{test_acc_str_li_source[3*i+j]}')
        for i in range(4):
            for j in range(3):
                args.logger.info(f'{names[i]}：{test_acc_str_li_target[3*i+j]}')
        # 绘制箱线图
        boxplot(x_li_source, label_list, color,
                f'./Graph/graph/{args.photo_nums}-模型（{args.model_name}）-ALL-Hp.png',
                if_avg=True,autofmt_xdate=True)
        args.photo_nums += 1
        boxplot(x_li_target, label_list, color,
                f'./Graph/graph/{args.photo_nums}-模型（{args.model_name}）-ALL-Hp.png',
                if_avg=True,autofmt_xdate=True)
        args.photo_nums += 1
    total_seconds = time.time() - start
    total_minutes = int(total_seconds // 60)
    remaining_seconds = int(total_seconds % 60)
    formatted_time = f'{total_minutes} 分 {remaining_seconds} 秒'
    args.logger.info(f'实验模型{args.model_name}运行的时间：{formatted_time}')

def d2d_exp_simple(args):
    args.logger.info(f"======================= {args.model_name} =======================")
    start = time.time()
    x_li_source = []
    x_li_target = []
    test_acc_str_li_source = []
    test_acc_str_li_target = []
    labels1_source = []
    labels2_source = []
    labels1_target = []
    labels2_target = []
    label_list = ['hp0->hp1', 'hp0->hp2', 'hp0->hp3',
                  'hp1->hp0', 'hp1->hp2', 'hp1->hp3',
                  'hp2->hp0', 'hp2->hp1', 'hp2->hp3',
                  'hp3->hp0', 'hp3->hp1', 'hp3->hp2']
    color = ['lightblue', 'darkcyan', 'red', 'hotpink']

    for i in range(4):
        for j in range(4):
            if i != j:
                nums = 20
                results_source = []
                results_target = []
                for num in range(nums):
                    time1 = time.time()
                    args.logger.info(f"----------- {args.model_name}-{names[i]}->{names[j]} -----------")
                    args.source_dset_path = folder + names[i] + '_label.txt'  # 数据加载路径
                    args.target_dset_path = folder + names[j] + '_label.txt'  # 数据加载路径
                    args.model_path = f'./Model/model({args.model_name})_{names[i]}_{names[j]}.pt'  # 模型保存路径
                    args.dset_loaders = ReadData.d2d_data_load(args)  # 使用数据加载器
                    args.logger.info(f'数据加载的时间：{time.time() - time1:.3f}s')
                    time2 = time.time()
                    # 模型训练
                    training(args)
                    args.logger.info(f'模型训练的时间：{time.time() - time2:.3f}s')
                    time3 = time.time()
                    if num == 0:
                        test_acc_source, labels1_source, labels2_source = test_d2d_simple_source(args, True)
                        test_acc_target, labels1_target, labels2_target = test_d2d_simple_target(args, True)
                    else:
                        test_acc_source = test_d2d_simple_source(args, False)
                        test_acc_target = test_d2d_simple_target(args, False)
                    args.logger.info(f'模型测试的时间：{time.time() - time3:.3f}s')
                    args.logger.info(
                        f'模型 {names[i]}_model({args.model_name}) 完整运行的时间：{time.time() - time1:.3f}s')
                    results_source.append(test_acc_source)  # 源域准确率
                    results_target.append(test_acc_target)  # 目标域准确率
                    print(f'i={i},j={j},num={num}')
                # 计算每次测试的平均值和标准偏差
                averages_source = np.mean(results_source)
                averages_target = np.mean(results_target)
                std_devs_source = np.std(results_source)
                std_devs_target = np.std(results_target)
                x_li_source.append(results_source)
                x_li_target.append(results_target)
                test_acc_str_source = f'{names[i]}--> {names[j]} Source Accuracy = {averages_source * 100:.2f}% ± {std_devs_source * 100:.2f}%'
                test_acc_str_target = f'{names[i]}--> {names[j]} Target Accuracy = {averages_target * 100:.2f}% ± {std_devs_target * 100:.2f}%'
                args.logger.info(test_acc_str_source)  # 输出nums轮训练后准确率的方差和均值
                args.logger.info(test_acc_str_target)
                test_acc_str_li_source.append(test_acc_str_source)
                test_acc_str_li_target.append(test_acc_str_target)
            if args.Test is True and i != j:  # 控制单例测试
                break
    if args.Test is True:
        print('测试模式')
        report = get_report(labels1_source, labels2_source)
        args.logger.info(f"预测报告：\n"
                         f"{report}")
        report = get_report(labels1_target, labels2_target)
        args.logger.info(f"预测报告：\n"
                         f"{report}")
        for i in range(len(test_acc_str_li_source)):
            args.logger.info(f'{names[i]}：{test_acc_str_li_source[i]}')
        for i in range(len(test_acc_str_li_target)):
            args.logger.info(f'{names[i]}：{test_acc_str_li_target[i]}')
    else:
        print('非测试测试模式')

        args.logger.info(f"数据标准化方式：{args.Scaler}")
        matrix_source, report_source = matrix_report(labels1_source, labels2_source,
                                                     f'./Graph/graph/{args.photo_nums}-模型（{args.model_name}-源域混淆矩阵（{names[0]},{names[1]}）.png')
        args.photo_nums += 1
        matrix_target, report_target = matrix_report(labels1_target, labels2_target,
                                                     f'./Graph/graph/{args.photo_nums}-模型（{args.model_name}-目标域混淆矩阵（{names[0]},{names[1]}）.png')
        args.photo_nums += 1
        args.logger.info(f"混淆矩阵：\n"
                         f"{str(matrix_source)}\n"
                         f"预测报告：\n"
                         f"{report_source}")
        args.logger.info(f"混淆矩阵：\n"
                         f"{str(matrix_target)}\n"
                         f"预测报告：\n"
                         f"{report_target}")

        for i in range(4):
            for j in range(3):
                args.logger.info(f'{names[i]}：{test_acc_str_li_source[3 * i + j]}')
        for i in range(4):
            for j in range(3):
                args.logger.info(f'{names[i]}：{test_acc_str_li_target[3 * i + j]}')
        # 绘制箱线图
        boxplot(x_li_source, label_list, color,
                f'./Graph/graph/{args.photo_nums}-模型（{args.model_name}）-ALL-Hp.png',
                if_avg=True, autofmt_xdate=True)
        args.photo_nums += 1
        boxplot(x_li_target, label_list, color,
                f'./Graph/graph/{args.photo_nums}-模型（{args.model_name}）-ALL-Hp.png',
                if_avg=True, autofmt_xdate=True)
        args.photo_nums += 1
    total_seconds = time.time() - start
    total_minutes = int(total_seconds // 60)
    remaining_seconds = int(total_seconds % 60)
    formatted_time = f'{total_minutes} 分 {remaining_seconds} 秒'
    args.logger.info(f'实验模型{args.model_name}运行的时间：{formatted_time}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neighbors')
    # 选择GPU设备
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # 并行工作数
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    # 随机种子
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    # 训练轮次
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    # 训练批次
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    # 学习率
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    # 学习率衰减
    # parser.add_argument('--lr_decay', type=float, default=0.9, help="learning rate decay")
    # 权重衰减
    # parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    args = parser.parse_args()
    args.username = 'LHT'
    # 加载CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # 风扇端四个负载样本集
    names = ['Fan_12k_0', 'Fan_12k_1', 'Fan_12k_2', 'Fan_12k_3']
    # names = ['Drive_12k_0', 'Drive_12k_1', 'Drive_12k_2', 'Drive_12k_3']

    args.class_num = 10  # 分类数
    folder = './DATA/CRWU_1d_2048/'
    #folder = 'D:\文件过渡\test\CRWU_1d_2048'
    # 默认数据读取路径
    args.dset_path = folder + names[0] + '_label.txt'
    args.source_dset_path = folder + names[1] + '_label.txt'
    args.target_dset_path = folder + names[2] + '_label.txt'
    args.Scaler = 'Z-Score'  # 采用Min-Max来标准化数据
    # args.Scaler = ''  # 通过对数据的观察，原始数据已经进行了标准化处理。使用 Z-Score 与未再进行标准化的数据，其分布表现一致
    args.optimizer = 'Adam'  # 模型优化器
    args.loss_func = 'CrossEntropyLoss'  # 模型损失函数：交叉熵
    args.activation = 'ReLU'  # 网络激活函数
    args.use_resnet_shot_connection = True  # ResNet默认使用残差连接
    args.weight_lambda = 1  # DANN 域权重默认
    args.times = 10  # 迭代 max_epoch 轮 times 次，最后计算平均准确率和偏差范围
    args.Test = False # 模型单例测试
    args.photo_nums = 21  # 图片起始数
    args.logger = uselog('my_log_20240608-1.log')
    args.logger.info(f"\n========================"
                     f"\n========================"
                     f"\n======== 实验开始 ========"
                     f"\n========================"
                     f"\n========================")
    start_time = time.time()
    """
    VGG16 模型训练与测试
    """
    args.model_name = 'VGG16'
    exp(args)
    args.logger.info(f"========================================")

    """
    AlexNet 模型训练与测试
    """
    args.model_name = 'AlexNet'
    args.lr = 0.003
    # exp(args)
    args.logger.info(f"========================================")

    """
    ResNet18 模型训练与测试
    """
    args.model_name = 'ResNet18'
    args.lr = 1e-5
    # exp(args)
    args.logger.info(f"========================================")

    #
    """
    ResNet18-NoSkip 消融实验 模型训练与测试
    """
    args.use_resnet_shot_connection = False
    args.model_name = 'ResNet18-NoSkip'
    # exp(args)
    args.use_resnet_shot_connection = True
    args.logger.info(f"========================================")

    """
    ResNet18-Noise 模型训练与测试
    """
    args.Scaler = 'Noise-Add'
    args.model_name = 'ResNet18-Noise'
    # args.lr = 1e-5
    # exp(args)
    args.logger.info(f"========================================")

    #
    """
    ResNet18-NoSkip-Noise 消融实验 模型训练与测试
    """
    args.use_resnet_shot_connection = False
    args.model_name = 'ResNet18-NoSkip-Noise'
    # exp(args)
    args.use_resnet_shot_connection = True
    args.Scaler = 'Z-Score'
    args.logger.info(f"========================================")

    """
    ResNet34 模型训练与测试
    """
    args.model_name = 'ResNet34'
    # exp(args)
    args.logger.info(f"========================================")

    """
    ResNet50 模型训练与测试
    """
    args.model_name = 'ResNet50'
    # exp(args)
    args.logger.info(f"========================================")

    """
    ResNet101 模型训练与测试
    """
    args.model_name = 'ResNet101'
    # exp(args)
    args.logger.info(f"========================================")
    args.lr = 1e-3

    """
    ResNetDANN 模型训练与测试
    """
    args.model_name = 'ResNetDANN'
    # d2d_exp(args)
    args.logger.info(f"========================================")
    """
    ResNet-d2d 模型训练与测试
    """
    args.model_name = 'ResNet-d2d'
    # d2d_exp_simple(args)
    args.logger.info(f"========================================")
    #
    """
    VGGDANN 模型训练与测试
    """
    args.lr = 1e-3
    args.model_name = 'VGGDANN'
    # d2d_exp(args)
    args.logger.info(f"========================================")
    """
    VGG-d2d 模型训练与测试
    """
    args.model_name = 'VGG-d2d'
    # d2d_exp_simple(args)
    args.logger.info(f"========================================")


    total_seconds = time.time() - start_time
    total_hours = int(total_seconds // 3600)
    remaining_minutes = int((total_seconds % 3600) // 60)
    remaining_seconds = int(total_seconds % 60)
    formatted_time = f'{total_hours} 小时 {remaining_minutes} 分 {remaining_seconds} 秒'
    args.logger.info(f'实验总训练时间：{formatted_time}')
    args.logger.info(f"\n========================"
                     f"\n========================"
                     f"\n======== 实验结束 ========"
                     f"\n========================"
                     f"\n========================")
