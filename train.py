import argparse
import logging
import os

from agents.agent1 import Agent1
from config import merge_config_from_toml
from config import settings as S
from utils.logger import init_logger

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def main():
    agent = Agent1()
    agent.train()


if __name__ == "__main__":
    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    parser.add_argument("CONFIG_NAME", type=str)
    parser.add_argument("--LR", type=float, default=None)
    parser.add_argument("--NO_LOG", action="store_true")

    args = parser.parse_args()

    # --------------------------
    # Load and update settings
    # --------------------------
    merge_config_from_toml(f"./config/{args.CONFIG_NAME}.toml")
    if args.LR is not None:
        S.LR = args.LR

    # --------------------
    # Setting Root Logger
    # --------------------
    log_root = os.path.join(ROOT_DIR, f"Logs/{S.NET}")
    S.CHECKPOINT_DIR = init_logger(log_root, no_file_logging=args.NO_LOG)

    logging.info(S)  # summarize settings

    # -------
    # Run
    # -------
    main()
