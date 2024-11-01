import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE


def bar(acc, model_names, labels, filename):
    """
    :param acc: numpy.array 列表，每个马力负载下各模型的准确率为列表的准确率
    :param model_names:每个模型的名字
    :param labels: 横轴为各马力负载：如hp0,hp1,hp2,hp3
    :param filename: 图片保存地址
    :return:

    例子：
    hp = ['HP(0)', 'HP(1)', 'HP(2)', 'HP(3)']
    model_names = ['model1', 'model2', 'model3', 'model4']
    h0 = [0.15, 0.15, 0.15, 0.15]
    h1 = [0.6906, 0.8268, 0.6982, 0.7782]
    h2 = [0.6512, 0.8478, 0.8979, 0.9721]
    h3 = [0.6512, 0.6478, 0.8979, 0.9721]
    acc_list = [h0, h1, h2, h3]
    bar(acc_list, hp, model_names,'./tmp.png')
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red', 'skyblue', 'darkcyan', 'hotpink']  # 颜色列表
    width = 0.15  # 柱状图的宽度
    x_indexes = np.arange(len(acc))  # x轴标签位置

    for i, data in enumerate(acc):
        print(data)
        ax.bar(x_indexes + i * width, data, width, label=labels[i], color=colors[i])
    # ax.plot(source, sourceClass, linestyle='-', color='red', label=target[0])
    # ax.plot(source, targetClass, linestyle='--', color='darkcyan', label=target[1])
    # ax.plot(source, targetClass2, linestyle='--', color='skyblue', label=target[2])
    # ax.plot(source, targetClass3, linestyle='--', color='green', label=target[3])

    ax.set_ylabel('Accuracy', fontsize=20)
    ax.set_xticks(x_indexes + (len(acc) - 1) * width / 2)
    ax.set_xticklabels(model_names, fontsize=20, rotation=0)  # 默认横着
    ax.legend(loc='upper right', fontsize=15, ncol=4)
    ax.set_ylim(0, 1.2)
    plt.tick_params(axis='y', labelsize=15)  # 设置y轴刻度字体大小
    plt.savefig(filename)
    plt.show()


def single_line(values, ylabel, title, filename, linestyle='--', color='b'):
    """
    :param values: 列表（numpy.array），每个epoch的值
    :param ylabel: 纵坐标标签
    :param title: 标题
    :param filename: 图片保存路径
    :param linestyle: 默认线条为虚线
    :param color: 默认颜色为蓝色
    :return: 输出折线图，主要用于训练阶段的损失和验证精度的绘制

    例子：
    train_loss = [1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15]  # 训练损失
    val_accuracy = [0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88]  # 验证准确率
    line(train_loss, 'train_loss', 'Train Loss Over Epochs',filename='./tmp.png')
    line(val_accuracy, 'val_accuracy', 'Validation Accuracy Over Epochs',filename='./tmp.png', color='r')
    """
    epochs = list(range(1, len(values) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, values, label=ylabel, marker='s', linestyle=linestyle, color=color)
    plt.title(title, fontsize=20)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.savefig(filename)
    plt.show()


def boxplot(x_list, label_list, color_param, filename, if_avg=False, rotation=0, autofmt_xdate=False):
    """
    :param x_list: numpy.array类型的列表，列表每个元素都是一个列表
    :param label_list: 标签列表，用于设置y轴
    :param filename: 图片保存路径
    :param color_param: 颜色设置。比较多个模型的各马力负载的时候设置
    :param if_avg: 是否添加均值线
    :return:输出箱型图片

    例子：计算一个模型在不同马力负载下的分布和总平均准确率
    hp0 = np.random.uniform(low=0.7, high=0.9, size=10)
    hp1 = np.random.uniform(low=0.7, high=0.9, size=10)
    hp2 = np.random.uniform(low=0.7, high=0.9, size=10)
    hp3 = np.random.uniform(low=0.7, high=0.9, size=10)
    li = [hp0, hp1, hp2, hp3]
    label_list = ['hp0', 'hp1', 'hp2', 'hp3']
    color_param = ['lightblue', 'green']
    boxplot(li, label_list, color_param, './tmp.png', if_avg=True)
    """
    # color_param[0] = 'lightblue'
    # color_param[1] = 'green'

    fig = plt.figure(figsize=(8, 6))
    plt.boxplot(x_list, labels=label_list, patch_artist=True,
                boxprops=dict(facecolor=color_param[0], color='black'))
    if if_avg is not False:
        plt.axhline(y=np.mean(x_list), color=color_param[1], linestyle='-', linewidth=2)
    plt.title('Box Plot Comparison', fontsize=20)
    plt.xticks(fontsize=15, rotation=rotation)  # 设置x轴刻度标签字体大小为15
    plt.yticks(fontsize=15)  # 设置y轴刻度标签字体大小为15
    if autofmt_xdate:
        fig.autofmt_xdate()
    plt.savefig(filename)
    # 显示图形
    plt.show()


def hitplot(data, label, kde=True, color='blue'):
    sns.histplot(data, kde=kde, color=color, label=label)


def box_hit_plot(original_data, normalized_data, filename):
    """
    :param original_data: 未标准化的原始，类型为numpy.array
    :param normalized_data: 标准化后的数据，类型为numpy.array
    :param filename: 图像保存位置
    :return: 输出处理前和处理后的直方图和箱型图

    使用例子：
    original_data = np.random.normal(loc=50, scale=10, size=1000)  # 原始数据示例
    print(original_data,type(original_data))
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(original_data.reshape(-1, 1)).flatten()
    print(normalized_data,type(normalized_data))
    box_hit_plot(original_data,normalized_data)
    """
    plt.figure(figsize=(12, 6))
    # 绘制直方图比较
    plt.subplot(1, 2, 1)  # 1行2列的第1个
    hitplot(original_data, label='Original Data', kde=True, color='blue')
    hitplot(normalized_data, label='Normalized Data', kde=True, color='red')
    plt.title('Histogram Comparison')
    plt.legend()

    # 绘制箱线图比较
    plt.subplot(1, 2, 2)  # 1行2列的第2个
    plt.boxplot([original_data, normalized_data], labels=['Original', 'Normalized'])
    plt.title('Box Plot Comparison')
    plt.savefig(filename)
    # 显示图形
    plt.show()


def confusion_matrix(matrix, filename):
    """
    :param filename: 图像保存地址
    :param matrix: 10分类矩阵
    :return: 输出混淆矩阵图像

    from sklearn.metrics import confusion_matrix
    # 随机生成真实类别标签（0到9之间的整数）
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 5, 8, 4, 1, 5, 6, 1, 2, 5, 6, 8, 7, 9,
              5, 2, 1, 6, 7]
    # 随机生成预测类别标签（0到9之间的整数）
    labels1 = [0, 1, 2, 3, 4, 2, 6, 7, 8, 9, 0, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 3, 5, 8, 4, 1, 4, 6, 1, 2, 5, 4, 8, 7, 9,
               5, 2, 1, 1, 7]
    confusion_matrix_sample=confusion_matrix(labels,labels1)
    cmatrix(matrix, 'Confusion Matrix', './tmp.png')
    """
    # 原始标签和自定义标签的映射
    label_mapping = {
        0: "C1", 1: "C2", 2: "C3", 3: "C4", 4: "C5",
        5: "C6", 6: "C7", 7: "C8", 8: "C9", 9: "C10",
    }
    nums = 0

    plt.figure(figsize=(8, 5), dpi=300)
    sns.heatmap(matrix, xticklabels=label_mapping.values(),
                yticklabels=label_mapping.values(), annot=True, fmt='d', cmap='GnBu')  # cmap设置颜色
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    # plt.title(f'{title} Confusion Matrix',fontsize=20)
    plt.savefig(filename)
    plt.show()


def t_SNE(matrix, label, title, filename):
    """
    :param matrix: 输入矩阵
    :param label: 分类标签
    :param filename: 图片保存地址名
    :param title: 图片标题
    :return: 输出t-SNE图片

    shuffled_confusion_mat.shape=(32,1,m)
    confusion_mat.shape=(32,1,n)
    t_SNE(shuffled_confusion_mat, './tmp.png', 'confusion matrix')
    t_SNE(confusion_mat, 'confusion matrix', './tmp.png')
    """
    tsne = TSNE(n_components=2, n_iter=10000)
    t_sne = tsne.fit_transform(matrix)
    plt.scatter(t_sne[:, 0], t_sne[:, 1], c=label)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red', 'skyblue', 'darkcyan', 'hotpink']  # 颜色列表
    x = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101']
    y = [66.78, 69, 116.68, 198.42]
    plt.bar(x, y, color=colors)
    plt.ylabel('Spend Time(minute)',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
