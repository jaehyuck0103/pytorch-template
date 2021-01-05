import argparse
import logging
import os
from datetime import datetime

from agents import get_agent
from config import merge_config_from_toml
from config import settings as S
from utils.logger import init_logger

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def main(op_check):
    agent = get_agent(S.agent.name)

    # Load Pretrained
    if S.load_weights_dir:
        agent.load_snapshot(S.load_weights_dir)

    if op_check:
        agent.operation_check()
    else:
        agent.train()


if __name__ == "__main__":
    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    parser.add_argument("config_name", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--load_weights_dir", type=str)
    parser.add_argument("--op_check", action="store_true")

    args = parser.parse_args()

    # --------------------------
    # Load and update settings
    # --------------------------
    merge_config_from_toml(f"./config/{args.config_name}.toml")

    if args.lr:
        S.optim.lr = args.lr
    S.load_weights_dir = args.load_weights_dir

    begin_time = datetime.now().strftime("%y%m%d_%H%M%S")
    if args.op_check:
        begin_time = "TEST_" + begin_time
    S.agent.log_dir = os.path.join(ROOT_DIR, "Logs", args.config_name, begin_time)

    # --------------------
    # Setting Root Logger
    # --------------------
    init_logger(S.agent.log_dir, no_file_logging=args.op_check)

    logging.info(S)  # summarize settings

    # -------
    # Run
    # -------
    main(args.op_check)
