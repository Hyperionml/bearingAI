import numpy as np
import torch

from Log.log import uselog
import torch.nn.functional as F


def testing(args, label_return):
    # print(f"testing {model_name}")
    model_name = args.model_name
    logger = args.logger
    logger.info(f"testing {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dset_loaders = args.dset_loaders
    batch_size = args.batch_size * 2
    model = torch.load(args.model_path)
    model.eval()
    corr_epoch = 0
    iter_source_test = iter(dset_loaders["test"])
    test_size = dset_loaders["test"].__len__()
    last_size = 0
    model.batch_size = batch_size
    lables1 = []
    labels2 = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(iter_source_test):
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(device)
            if len(inputs) < batch_size:
                model.batch_size = len(inputs)
                last_size = len(inputs)
            pred = model(inputs)
            probabilities = F.softmax(pred, dim=1)
            labels_pred = torch.argmax(probabilities, dim=1)
            corr_epoch += (labels_pred == labels).sum().item()
            if label_return:
                lables1.append(labels.cpu().numpy())
                labels2.append(labels_pred.cpu().numpy())
    if last_size!=0:
        test_acc = corr_epoch / ((test_size - 1) * batch_size + last_size)
    else:
        test_acc = corr_epoch / test_size * batch_size
    logger.info(f'test_Accuracy:{test_acc:4.4f}')
    if label_return:
        return test_acc, np.concatenate(lables1), np.concatenate(labels2)
    else:
        return test_acc


def test_source(args, label_return):
    model_name = args.model_name
    logger = args.logger
    logger.info(f"testing {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dset_loaders = args.dset_loaders
    batch_size = args.batch_size
    model = torch.load(args.model_name).to(device)
    source_test_iter = iter(dset_loaders["source_test"])
    source_test_size = source_test_iter.__len__()
    model.eval()
    corr_epoch_classification = 0
    corr_epoch_domain = 0
    lables1 = []
    labels2 = []
    with torch.no_grad():
        for batchid, (inputs, labels) in enumerate(source_test_iter):
            alpha = 0.001
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(device)
            source_domain_labels = torch.ones(batch_size).long().to(device)  # 构建全1标签
            last_size = 0
            if len(inputs) < batch_size:
                model.batch_size = len(inputs)
                last_size = len(inputs)
                source_domain_labels = torch.zeros(len(inputs)).long().to(device)
            task_predict, domain_predict = model(inputs, alpha)

            probabilities1 = F.softmax(task_predict, dim=1)
            labels_pred1 = torch.argmax(probabilities1, dim=1)
            corr_epoch_classification += (labels_pred1 == labels).sum().item()  # 计算源域分类准确率

            probabilities2 = F.softmax(task_predict, dim=1)
            labels_pred2 = torch.argmax(probabilities2, dim=1)
            corr_epoch_domain += (labels_pred2 == source_domain_labels).sum().item()
            if label_return:
                lables1.append(labels.cpu().numpy())
                labels2.append(labels_pred1.cpu().numpy())
    # ((source_test_size - 1) * batch_size + last_size)
    if last_size != 0:
        test_acc_classification = corr_epoch_classification / ((source_test_size - 1) * batch_size + last_size)
        test_acc_domain = corr_epoch_domain / ((source_test_size - 1) * batch_size + last_size)
    else:
        test_acc_classification = corr_epoch_classification / (source_test_size * batch_size)
        test_acc_domain = corr_epoch_domain /(source_test_size * batch_size)
    logger.info(f'test_acc_classification:{test_acc_classification:.4f},test_acc_domain：{test_acc_domain:.4f}')
    if label_return:
        return test_acc_classification, np.concatenate(lables1), np.concatenate(labels2)
    else:
        return test_acc_classification


def test_target(args, label_return):
    model_name = args.model_name
    logger = args.logger
    logger.info(f"testing {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dset_loaders = args.dset_loaders
    batch_size = args.batch_size
    model = torch.load(args.model_name).to(device)
    target_test_iter = iter(dset_loaders["target"])
    target_test_size = target_test_iter.__len__()
    model.eval()
    corr_epoch_classification = 0
    corr_epoch_domain = 0
    lables1 = []
    labels2 = []
    with torch.no_grad():
        for batchid, (inputs, labels) in enumerate(target_test_iter):
            alpha = 0.001
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(device)
            source_domain_labels = torch.zeros(batch_size).long().to(device)
            last_size = 0
            if len(inputs) < batch_size:
                model.batch_size = len(inputs)
                last_size = len(inputs)
                source_domain_labels = torch.zeros(len(inputs)).long().to(device)  # 构建全0标签
            task_predict, domain_predict = model(inputs, alpha)

            probabilities1 = F.softmax(task_predict, dim=1)
            labels_pred1 = torch.argmax(probabilities1, dim=1)
            corr_epoch_classification += (labels_pred1 == labels).sum().item()  # 计算源域分类准确率

            probabilities2 = F.softmax(task_predict, dim=1)
            labels_pred2 = torch.argmax(probabilities2, dim=1)
            corr_epoch_domain += (labels_pred2 == source_domain_labels).sum().item()
            if label_return:
                lables1.append(labels.cpu().numpy())
                labels2.append(labels_pred1.cpu().numpy())
    if last_size != 0:
        test_acc_classification = corr_epoch_classification / ((target_test_size - 1) * batch_size + last_size)
        test_acc_domain = corr_epoch_domain / ((target_test_size - 1) * batch_size + last_size)
    else:
        test_acc_classification = corr_epoch_classification / (target_test_size * batch_size)
        test_acc_domain = corr_epoch_domain /(target_test_size * batch_size)
    logger.info(f'test_acc_classification:{test_acc_classification:.4f},test_acc_domain：{test_acc_domain:.4f}')
    if label_return:
        return test_acc_classification, np.concatenate(lables1), np.concatenate(labels2)
    else:
        return test_acc_classification

def test_d2d_simple_source(args,label_return):
    model_name = args.model_name
    logger = args.logger
    logger.info(f"testing {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dset_loaders = args.dset_loaders
    batch_size = args.batch_size
    model = torch.load(args.model_path)
    model.eval()
    corr_epoch = 0
    iter_source_test = iter(dset_loaders["source_test"])
    test_size = dset_loaders["source_test"].__len__()
    last_size = 0
    model.batch_size = batch_size
    lables1 = []
    labels2 = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(iter_source_test):
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(device)
            if len(inputs) < batch_size:
                model.batch_size = len(inputs)
                last_size = len(inputs)
            pred = model(inputs)
            probabilities = F.softmax(pred, dim=1)
            labels_pred = torch.argmax(probabilities, dim=1)
            corr_epoch += (labels_pred == labels).sum().item()
            # print((labels_pred == labels).sum().item())
            if label_return:
                lables1.append(labels.cpu().numpy())
                labels2.append(labels_pred.cpu().numpy())
    if last_size != 0:
        test_acc = corr_epoch / ((test_size - 1) * batch_size + last_size)
    else:
        test_acc = corr_epoch / test_size * batch_size
    logger.info(f'Source test_Accuracy:{test_acc:4.4f}')
    if label_return:
        return test_acc, np.concatenate(lables1), np.concatenate(labels2)
    else:
        return test_acc


def test_d2d_simple_target(args,label_return):
    model_name = args.model_name
    logger = args.logger
    logger.info(f"testing {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dset_loaders = args.dset_loaders
    batch_size = args.batch_size
    model = torch.load(args.model_path)
    model.eval()
    corr_epoch = 0
    iter_source_test = iter(dset_loaders["target_test"])
    test_size = dset_loaders["target_test"].__len__()
    last_size = 0
    model.batch_size = batch_size
    lables1 = []
    labels2 = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(iter_source_test):
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(device)
            if len(inputs) < batch_size:
                model.batch_size = len(inputs)
                last_size = len(inputs)
            pred = model(inputs)
            probabilities = F.softmax(pred, dim=1)
            labels_pred = torch.argmax(probabilities, dim=1)
            corr_epoch += (labels_pred == labels).sum().item()
            if label_return:
                lables1.append(labels.cpu().numpy())
                labels2.append(labels_pred.cpu().numpy())
    if last_size!=0:
        test_acc = corr_epoch / ((test_size - 1) * batch_size + last_size)
    else:
        test_acc = corr_epoch / test_size * batch_size
    logger.info(f'Target test_Accuracy:{test_acc:4.4f}')
    if label_return:
        return test_acc, np.concatenate(lables1), np.concatenate(labels2)
    else:
        return test_acc
