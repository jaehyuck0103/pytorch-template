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
    parser.add_argument("LR", type=float)
    parser.add_argument("--NO_LOG", action="store_true")

    args = parser.parse_args()

    # --------------------
    # Setting Root Logger
    # --------------------
    begin_time = init_logger(no_file_logging=args.NO_LOG)

    # --------------------------
    # Load and update settings
    # --------------------------
    merge_config_from_toml(f"./config/{args.CONFIG_NAME}.toml")
    S.LR = args.LR
    S.CHECKPOINT_DIR = os.path.join(ROOT_DIR, f"Output/{begin_time}")

    logging.info(S)  # summarize settings

    # -------
    # Run
    # -------
    main()
