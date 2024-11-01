import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from Graph.GeneralPlot import confusion_matrix as cmatrix
from Log.log import uselog


def get_matrix(true_labels, predicted_labels):
    return confusion_matrix(true_labels, predicted_labels)


def get_report(true_labels, predicted_labels):
    """
    :param logger: 日志记录器
    :param true_labels:真实标签
    :param predicted_labels: 预测标签
    :return:
    """
    return classification_report(true_labels, predicted_labels)


def matrix_report(true_labels, predicted_labels, confusion_matrix_filename):
    """
    :param true_labels: 真实标签列表,类型为numpy.array
    :param predicted_labels: 预测标签列表,类型为numpy.array
    :param confusion_matrix_filename: 混淆矩阵图保存地址
    :param repoter_logs_filename: 预测报告日志输出地址
    :return: 返回混淆矩阵和类别预测报告

    使用说明：
    随机生成真实类别标签（0到9之间的整数）
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 5, 8, 4, 1, 5, 6, 1, 2, 5, 6, 8, 7, 9,
              5, 2, 1, 6, 7]
    随机生成预测类别标签（0到9之间的整数）
    labels1 = [0, 1, 2, 3, 4, 2, 6, 7, 8, 9, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 3, 5, 8, 4, 1, 4, 6, 1, 2, 5, 4, 8, 7, 9,
               5, 2, 1, 1, 7]
    cm, rp = reporter(labels, labels1, 'Confusion Matrix.png', 'Confusion Matrix', 'Confusion Matrix.txt')
    """
    matrix = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels)

    cmatrix(matrix, confusion_matrix_filename)

    return matrix, report
