import numpy as np
import pandas as pd
from scipy.io import loadmat

file_names = ['97.mat','105.mat','118.mat','130.mat','169.mat','185.mat','197.mat','209.mat','222.mat','234.mat']

for file in file_names:
    # 读取MAT文件
    data = loadmat(f'./12k Drive End Bearing Fault Data/{file}')
    print(list(data.keys()))

# # 采用驱动端数据
# data_columns = ['X097_FAN_time', 'X105_FAN_time', 'X118_FAN_time', 'X130_FAN_time', 'X169_FAN_time',
#                 'X185_FAN_time','X197_FAN_time','X209_FAN_time','X222_FAN_time','X234_FAN_time']
# columns_name = ['fan_normal','fan_7_inner','fan_7_ball',
#                 'fan_7_outer','fan_14_inner','fan_14_ball',
#                 'fan_14_outer','fan_21_inner','fan_21_ball','fan_21_outer']
# data_12k_10c = pd.DataFrame()
# for index in range(10):
#     # 读取MAT文件
#     data = loadmat(f"./'12k Drive End Bearing Fault Data'/{file}")
#     dataList = data[data_columns[index]].reshape(-1)
#     data_12k_10c[columns_name[index]] = dataList[:119808]  # 121048  min: 121265
# print(data_12k_10c.shape)
