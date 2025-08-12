from shared_imports import os,logging

def setup_loggers(log_dir='./cross_exchange'):
    """设置并返回日志记录器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    decision_logger = logging.getLogger('decision_logger')
    decision_logger.setLevel(logging.INFO)
    decision_handler = logging.FileHandler(os.path.join(log_dir, 'decision_log.txt'))
    decision_handler.setFormatter(formatter)
    decision_logger.addHandler(decision_handler)

    output_logger = logging.getLogger('output_logger')
    output_logger.setLevel(logging.INFO)
    output_handler = logging.FileHandler(os.path.join(log_dir, 'output_log.txt'))
    output_handler.setFormatter(formatter)
    output_logger.addHandler(output_handler)

    return decision_logger, output_logger