import numpy as np
from matplotlib import pyplot as plt


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


def snr_to_variance(snr_db, signal_variance):
    """
    Convert SNR in dB to noise variance.

    Parameters:
        snr_db (float): SNR in dB.
        signal_variance (float): Signal variance.

    Returns:
        noise_variance (float): Noise variance.
    """
    snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear scale
    noise_variance = signal_variance / snr_linear
    return noise_variance


def add_gaussian_noise(snr, signal):
    """
    添加加性高斯白噪声到信号中。

    参数：
    signal: numpy 数组，输入的信号值。
    snr: float，信噪比（以分贝为单位）。

    返回值：
    noisy_signal: numpy 数组，添加噪声后的信号值。
    """
    # 计算信号的功率
    signal_power = np.mean(np.abs(signal) ** 2)

    # 计算噪声的功率
    noise_power = signal_power / (10 ** (snr / 10))

    # 生成高斯白噪声
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # 将噪声添加到信号中
    noisy_signal = signal + noise

    return noisy_signal


# 读取原数据信号
def get_source_data():
    # 读取风扇端
    names = ['Fan_12k_0', 'Fan_12k_1', 'Fan_12k_2', 'Fan_12k_3']
    label_path = 'D:\TransitionalFiles\test\CRWU_1d_2048\\' + names[0] + '_label.txt'

    data_list = open(label_path).readlines()
    contents = make_dataset(data_list, None)
    datapath_list = []
    data_list = []

    loader_dict = {'TXT': txt_loader}
    loader = loader_dict.get('TXT')

    # 将数据添加到列表中
    i = 0
    for datapath, label in contents.__iter__():
        if i >= 2:
            break
        datapath_list.append(datapath)
        data_list.append(loader(datapath))
        i += 1
    # 未加噪声数据
    plt.figure(figsize=(9, 3))  # 或(16,9)
    plt.plot(range(2048), data_list[0], color='green')
    plt.show()
    # 噪声比为 -2dB
    plt.figure(figsize=(9, 3))  # 或(16,9)
    noise = add_gaussian_noise(-2., data_list[0])
    plt.plot(range(2048), noise, color='green')
    plt.show()
    # 噪声比为 -6dB
    plt.figure(figsize=(9, 3))  # 或(16,9)
    noise = add_gaussian_noise(-6., data_list[0])
    plt.plot(range(2048), noise, color='green')
    plt.show()

    # 返回噪声数据

#
# if __name__ == '__main__':
#     get_source_data()
