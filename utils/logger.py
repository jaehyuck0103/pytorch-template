import logging
import os


# --------------------
# Setting Root Logger
# --------------------
def init_logger(log_dir="", no_file_logging=False):

    time_format = "%(asctime)s: %(message)s"
    level = logging.INFO

    if no_file_logging:
        handlers = [logging.StreamHandler()]
    else:
        log_path = os.path.join(log_dir, "log.log")
        os.makedirs(log_dir, exist_ok=True)
        handlers = [logging.StreamHandler(), logging.FileHandler(log_path, mode="w")]
    logging.basicConfig(format=time_format, level=level, handlers=handlers)

    return log_dir
