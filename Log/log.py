import logging


def uselog(log_file):
    logger_name='my_logger'
    # 创建一个日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # 创建一个文件处理器，将日志写入文件
    # log_file = 'my_log_20240530-1.log'
    file_handler= logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # 创建一个控制台处理器，将日志输出到终端
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# logger.info('This is an info message.')
# logger.warning('This is a warning message.')
# logger.error('This is an error message.')
