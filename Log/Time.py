import time
from datetime import datetime


def now_time():
    """
    :return: 返回当前时间字符串
    """
    return datetime.now().strftime("%y年%m月%d日%H时%M分%S秒")


def time_difference(start_time):
    """
    :param start_time: 开始时间
    :return: 时间差

    start_time = time.time()
    j = 0
    for i in range(1000000):
        j += 1
    print(time_difference(start_time))
    """
    return time.time() - start_time



