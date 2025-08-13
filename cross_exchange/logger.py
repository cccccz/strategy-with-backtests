import logging
from datetime import datetime

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

decision_logger = logging.getLogger('decision_logger')
decision_logger.setLevel(logging.INFO)

decision_handler = logging.FileHandler('decision_log.txt')
decision_handler.setFormatter(logging.Formatter(log_format))

decision_logger.addHandler(decision_handler)

output_logger = logging.getLogger('output_logger')
output_logger.setLevel(logging.INFO)

output_handler = logging.FileHandler('output_log.txt')
output_handler.setFormatter(logging.Formatter(log_format))

output_logger.addHandler(output_handler)
