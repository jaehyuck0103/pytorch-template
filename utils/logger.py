import os
import logging
from datetime import datetime


ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
ROOT_DIR = os.path.abspath(ROOT_DIR)


# --------------------
# Setting Root Logger
# --------------------
def init_logger(no_file_logging=False):

    begin_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    format = "%(asctime)s: %(message)s"
    level = logging.INFO
    if no_file_logging:
        handlers = [logging.StreamHandler()]
    else:
        log_dir = os.path.join(ROOT_DIR, f"Output/{begin_time}")
        log_path = os.path.join(log_dir, f"{begin_time}.log")
        os.makedirs(log_dir, exist_ok=True)
        handlers = [logging.StreamHandler(), logging.FileHandler(log_path, mode="w")]
    logging.basicConfig(format=format, level=level, handlers=handlers)

    return begin_time
