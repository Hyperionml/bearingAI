# 使用手册

## 一、 准备数据。

打开百度网盘链接：链接: https://pan.baidu.com/s/1ap1OTgjMY-Ra617Y2iMyiw 提取码: X2f1。下载好后继续解压。

## 二、 准备实验环境。

下表为实验时所使用的软硬件基本配置。

| 实验软硬件环境配置 | 参数                                       |
| ------------------ | ------------------------------------------ |
| 处理器（CPU）      | Intel(R) Core(TM) i9-10920X  CPU @ 3.50GHz |
| 内存（RAM）        | DDR4 3200 32.00GB                          |
| 显卡（GPU）        | NVIDIA@ GeForce RTX3090-24G                |
| 操作系统           | Microsoft@ Windows 10 Pro  x64             |
| 开发环境           | Python3.11、Pytorch 2.0                    |

解压代码压缩包Code.zip，解压后，会有一个`python`环境软件包依赖文件`requirements.txt`。使用命令

```shell
pip install -r requirements.txt
```

## 三、代码文件概述

文件夹说明

| 文件夹 | 说明                                                         |
| ------ | ------------------------------------------------------------ |
| DATA   | 实验数据存放位置                                             |
| Fonts  | 实验绘图所使用的汉化字体                                     |
| Graph  | 存放实验生成图片                                             |
| Log    | Log.outputs存放实验日志文件，Log下其他文件存放日志代码和相关文件 |
| Model  | 存放模型代码，同时也是模型训练好后保存的位置                 |

文件说明

| 文件名               | 说明                                 |
| -------------------- | ------------------------------------ |
| Graph.GeneralPlot.py | 用于绘制实验图像                     |
| Log.log.py           | 实验日志数据生产                     |
| Log.Reporter.py      | 输出实验评估报告                     |
| Model.AlexNet.py     | 模型AlexNet代码                      |
| Model.DANN.py        | 模型DANN代码                         |
| Model.ResNet.py      | 模型ResNet代码                       |
| Model.ResNet18.py    | 模型ResNet18代码                     |
| Model.ResNetDANN.py  | 模型ResNetDANN代码                   |
| Model.VGG16.py       | 模型VGG16代码                        |
| Model.VGGDANN.py     | 模型VGGDANN代码                      |
| AddNoise.py          | 噪声添加代码                         |
| ReadData.py          | 从DATA中读取数据，并对数据做好预处理 |
| training.py          | 模型训练代码                         |
| testting.py          | 模型测试代码                         |
| main.py              | 项目代码运行入口文件                 |

## 四、代码运行

当环境部署好后，运行main.py即可，预计全程运行时间为40小时以上