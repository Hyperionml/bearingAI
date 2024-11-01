import numpy as np
import torch
from Model.AlexNet import AlexNet
from Model.VGG16 import CNN
from Model.ResNet import ResNet18, ResNet34, ResNet50, ResNet101
import torch.nn.functional as F
from Model.ResnetDANN import DANN as ResNetDANN
from Model.VGGDANN import DANN as VGGDANN
from Model.DANN import DANN
from torch.optim.lr_scheduler import ExponentialLR


# from Model.ResNet50 import ResNet
# 看下这个网络结构总共有多少个参数
def count_parameters(model, logger):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        logger.info(f'{item:>6}')
    logger.info(f'______\n{sum(params):>6}')


def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name
    # print(f"training {model_name}")
    logger = args.logger
    logger.info(f"training {model_name}")

    dset_loaders = args.dset_loaders
    batch_size = args.batch_size
    num_classes = args.class_num

    epochs = args.max_epoch
    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss()
    model = None
    d2d_models = ['ResNetDANN', 'VGGDANN','VGG-d2d', 'ResNet-d2d']
    d2d_simple_models=['VGG-d2d', 'ResNet-d2d']
    # 选择模型
    # if model_name != 'ResNetDANN' and model_name != 'VGGDANN' and model_name != 'DANN':
    if model_name not in d2d_models:
        # dset_loaders: ["train","val","test"]
        if model_name == 'VGG16':
            # conv_archs = ((2, 32), (1, 64), (1, 128))  # 浅层
            # conv_archs =((2, 64), (1, 256), (1, 512))
            # conv_archs= ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)) # VGG11
            conv_archs = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))  # vgg16
            model = CNN(conv_archs, batch_size, num_classes=num_classes).to(device)
            # logger.info(f"VGG16模型结构：{model}")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')  # 5680838

        if model_name == 'AlexNet':
            model = AlexNet(batch_size).to(device)
            # logger.info(f"AlexNet模型结构：{model}")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # args.lr = 1e-3
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')  # 5680838
            count_parameters(model, logger)

        if model_name == 'ResNet18':
            model = ResNet18(batch_size).to(device)
            # logger.info(f"ResNet18模型结构：{model}")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')  # 4620422
            count_parameters(model, logger)
        if model_name == 'ResNet18-NoSkip':
            model = ResNet18(batch_size, skip=False).to(device)
            # logger.info(f"ResNet18-NoSkip模型结构：{model}")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')  # 4620422
            count_parameters(model, logger)

        if model_name == 'ResNet18-Noise':
            model = ResNet18(batch_size).to(device)
            # logger.info(f"ResNet18模型结构：{model}")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')  # 4620422
            count_parameters(model, logger)
        if model_name == 'ResNet18-NoSkip-Noise':
            model = ResNet18(batch_size, skip=False).to(device)
            # logger.info(f"ResNet18-NoSkip模型结构：{model}")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')  # 4620422
            count_parameters(model, logger)

        if model_name == 'ResNet34':
            model = ResNet34(batch_size).to(device)
            # logger.info(f"ResNet34 模型结构：{model}")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')
            count_parameters(model, logger)
        if model_name == 'ResNet50':
            model = ResNet50(batch_size).to(device)
            # logger.info(f"ResNet50 模型结构：{model}")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')

        if model_name == 'ResNet101':
            model = ResNet101(batch_size).to(device)
            # logger.info(f"ResNet101 模型结构：{model}")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        # scheduler = optimizer.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

        model.train()
        best_model = model
        best_acc = 0
        for epoch in range(epochs):
            loss_epoch = 0.
            corr_epoch = 0
            iter_source_tr = iter(dset_loaders["train"])
            iter_source_val = iter(dset_loaders["val"])
            train_size = dset_loaders["train"].__len__()
            test_size = dset_loaders["val"].__len__()
            model.batch_size = args.batch_size
            last_size = 0
            # scheduler.step()
            for batch_idx, (inputs, labels) in enumerate(iter_source_tr):
                # 输出inputs为tensor(32,2048) 32为batch_size 2048为样本维度
                # 输出labels为tensor(32,) 32为batch_size
                inputs = inputs.to(torch.float32).to(device)  # (32,2048)
                labels = labels.to(device)
                optimizer.zero_grad()
                if len(inputs) < batch_size:
                    model.batch_size = len(inputs)
                    last_size = len(inputs)
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

            if last_size != 0 :
                train_acc = corr_epoch / ((train_size - 1) * batch_size + last_size)
                train_loss = loss_epoch / ((train_size - 1) * batch_size + last_size)
            else:
                train_acc = corr_epoch / (train_size * batch_size )
                train_loss = loss_epoch / (train_size * batch_size )
            # train_loss_list.append(train_loss)
            # train_acc_list.append(train_acc)
            logger.info(f'Epoch:{epoch + 1:2} train_Loss:{train_loss:10.8f} train_Accuracy:{train_acc:4.4f}')

            loss_epoch = 0.
            corr_epoch = 0
            model.batch_size = args.batch_size
            last_size = 0
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(iter_source_val):
                    inputs = inputs.to(torch.float32).to(device)
                    labels = labels.to(device)
                    if len(inputs) < batch_size:
                        model.batch_size = len(inputs)
                        last_size = len(inputs)
                    pred = model(inputs)
                    probabilities = F.softmax(pred, dim=1)
                    labels_pred = torch.argmax(probabilities, dim=1)

                    corr_epoch += (labels_pred == labels).sum().item()
                    loss = loss_function(pred, labels)
                    loss_epoch += loss.item()
            val_acc = corr_epoch / ((test_size - 1) * batch_size + last_size)
            val_loss = loss_epoch / ((test_size - 1) * batch_size + last_size)
            # val_loss_list.append(val_loss)
            # val_acc_list.append(val_acc)
            logger.info(f'Epoch:{epoch + 1:2} val_Loss:{val_loss:10.8f} val_Accuracy:{val_acc:4.4f}')
            # 保存当前最优模型参数
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model  # 更新最佳模型的参数

        model.batch_size = batch_size
        torch.save(best_model, args.model_path)
        logger.info(f'模型最高验证准确率：{best_acc}')
        return 0
    elif model_name in d2d_simple_models:
        if model_name == 'VGG-d2d':
            conv_archs =((2, 64), (1, 256), (1, 512))
            model = CNN(conv_archs, batch_size, num_classes=num_classes).to(device)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')  #
        if model_name == 'ResNet-d2d':
            model = ResNet18(batch_size).to(device)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')  #
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        model.train()
        model.batch_size = batch_size
        for epoch in range(epochs):
            loss_epoch = 0.
            corr_epoch = 0
            iter_source_tr = iter(dset_loaders["source_train"])
            train_size = dset_loaders["source_train"].__len__()
            model.batch_size = args.batch_size
            last_size = 0
            # scheduler.step()
            for batch_idx, (inputs, labels) in enumerate(iter_source_tr):
                # 输出inputs为tensor(32,2048) 32为batch_size 2048为样本维度
                # 输出labels为tensor(32,) 32为batch_size
                inputs = inputs.to(torch.float32).to(device)  # (32,2048)
                labels = labels.to(device)
                optimizer.zero_grad()
                if len(inputs) < batch_size:
                    model.batch_size = len(inputs)
                    last_size = len(inputs)
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
            if last_size != 0 :
                train_acc = corr_epoch / ((train_size - 1) * batch_size + last_size)
                train_loss = loss_epoch / ((train_size - 1) * batch_size + last_size)
            else:
                train_acc = corr_epoch / (train_size * batch_size )
                train_loss = loss_epoch / (train_size * batch_size )
            logger.info(f'Epoch:{epoch + 1:2} train_Loss:{train_loss:10.8f} train_Accuracy:{train_acc:4.4f}')
        torch.save(model, args.model_path)
        return 0
    else:
        if model_name == 'ResNetDANN':
            model = ResNetDANN(batch_size, 2).to(device)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')  #
        if model_name == 'VGGDANN':
            model = VGGDANN(batch_size, 2).to(device)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'模型参数量：{trainable_params}，模型学习率：{args.lr}')  #

        # dset_loaders: ["source_train","target_train","source_test","target"]
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        model.train()

        ## 参数初始化
        epochs = args.max_epoch
        loss_class = torch.nn.CrossEntropyLoss()
        loss_domain = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(epochs):
            source_train_iter = iter(dset_loaders["source_train"])
            target_train_iter = iter(dset_loaders["target_train"])

            source_train_size = source_train_iter.__len__()
            data_len = source_train_size

            loss_epoch = 0.
            loss_epoch_label = 0.
            loss_epoch_domain_source = 0.
            loss_epoch_domain_target = 0.
            last_size = 0
            for batchid in range(data_len):
                p = (float(batchid + epoch * data_len) / (epoch + 1)) / data_len
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                # ========================》 源域训练
                source_inputs, source_labels = source_train_iter.__next__()
                source_inputs = source_inputs.to(torch.float32).to(device)
                source_labels = source_labels.to(device)
                source_domain_labels = torch.zeros(batch_size).long().to(device)  # 初始化，将来自源域的数据都映射为0
                optimizer.zero_grad()
                if len(source_inputs) < batch_size:
                    model.batch_size = len(source_inputs)
                    last_size = len(source_inputs)
                task_predict, domain_predict = model(source_inputs, alpha=alpha)
                loss_s_label = loss_class(task_predict, source_labels)  # =============》标签分类损失
                loss_s_domain = loss_domain(domain_predict, source_domain_labels)  # =============》源域分类
                # break
                # ========================》 目标域训练
                target_inputs, _ = target_train_iter.__next__()  # 只需要目标域的未打标签的数据，这里不需要目标域的标签
                target_inputs = target_inputs.to(torch.float32).to(device)
                target_domain_labels = torch.ones(batch_size).long().to(device)  # 初始化，将来自目标域的数据都映射为1
                optimizer.zero_grad()

                _, domain_predict = model(target_inputs, alpha=alpha)  # 对应目标域只有域分类任务，所以只需要获得对域的预测
                loss_t_domain = loss_domain(domain_predict, target_domain_labels)

                # =======================》 梯度反传
                loss = loss_s_label + args.weight_lambda * (loss_s_domain + loss_t_domain)

                loss_epoch += loss.item()
                loss_epoch_label += loss_s_label.item()
                loss_epoch_domain_source += loss_s_domain.item()
                loss_epoch_domain_target += loss_t_domain.item()
                loss.backward()
                optimizer.step()
            # 求损失均值((data_len - 1) * batch_size + last_size)
            if last_size != 0 :
                loss_epoch = loss_epoch / ((data_len - 1) * batch_size + last_size)
                loss_epoch_label = loss_epoch_label / ((data_len - 1) * batch_size + last_size)
                loss_epoch_domain_source = loss_epoch_domain_source / ((data_len - 1) * batch_size + last_size)
                loss_epoch_domain_target = loss_epoch_domain_target / ((data_len - 1) * batch_size + last_size)
            else:
                loss_epoch = loss_epoch / (data_len * batch_size)
                loss_epoch_label = loss_epoch_label /(data_len * batch_size)
                loss_epoch_domain_source = loss_epoch_domain_source / (data_len * batch_size)
                loss_epoch_domain_target = loss_epoch_domain_target /(data_len * batch_size)
            print(f'epoch:{epoch + 1}:{epochs} loss总和：{loss_epoch:.8f} 源域分类Loss：{loss_epoch_label:.8f}\n '
                  f'源域-域分类Loss：{loss_epoch_domain_source:.8f}\n目标域1-域分类Loss：{loss_epoch_domain_target:.8f}')
            # print(type(loss_epoch.cpu().detach().numpy()))

        # 保存模型-finally
        torch.save(model, args.model_name)
