from dynaconf import settings as S
import os
import logging
from datetime import datetime
import argparse
from agents.agent1 import Agent1


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def train():
    agent = Agent1()
    agent.train()


def test():
    pass


if __name__ == '__main__':
    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument('config_name', type=str)
    parser.add_argument('MODE', type=str, choices=['train', 'test'])
    parser.add_argument('LR', type=float)

    args = parser.parse_args()

    # --------------------
    # Setting Root Logger
    # --------------------
    init_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    level = logging.INFO
    format = '%(asctime)s: %(message)s'
    log_dir = os.path.join(ROOT_DIR, f'output/{init_time}')
    log_path = os.path.join(log_dir, f'{init_time}.log')
    os.makedirs(log_dir, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode='w')
    ]
    logging.basicConfig(format=format, level=level, handlers=handlers)

    # --------------------------
    # Load and Init settings
    # --------------------------
    S.load_file(path=f'{args.config_name}.toml')

    S['LR'] = args.LR

    if args.MODE == 'train':
        S.CHECKPOINT_DIR = os.path.join(ROOT_DIR, f'output/{init_time}')
    else:
        S.CHECKPOINT_DIR = os.path.join(ROOT_DIR, f'output/{args.VER_TO_LOAD}')

    logging.info(S.as_dict())   # settings summary

    # -------
    # Run
    # -------
    if args.MODE == 'train':
        train()
    else:
        test()
