import io
import sys
from contextlib import redirect_stdout
from Log.Time import now_time


def writeLogs(str, filename=None):
    """
    :param str:  输入到日志的字符串
    :param filename: 日志输入文件地址
    :return: 输出字符串
    """
    f = io.StringIO()
    # 捕获输出
    with redirect_stdout(f):
        print("时间：", now_time())
        print(str)

    output = f.getvalue()  # 获取输出字符串
    if filename is not None:
        # 输出可以进一步写入文件
        with open(filename, 'w') as file:
            file.write(output)
    else:
        with open('output-temp.txt', 'a') as file:
            file.write(output)
    print(str)


# writeLogs("test")
