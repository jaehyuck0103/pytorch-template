import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from agents.agent1 import Agent1
from config import merge_config_from_toml
from config import settings as S
from datasets.dataset1 import Dataset1
from utils.logger import init_logger

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def main():
    # 원래대로라면 test dataset용 Dataset 새로 정의해야하지만, 귀찮으므로 valid 재활용.
    test_dataset = Dataset1(mode="valid")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2 ** 32),
    )

    agent = Agent1(predict_only=True)
    agent.load_checkpoint()

    # start prediction
    for _, data in enumerate(test_loader):
        # Prepare data
        x_img = data["img"]

        # Predict
        y_pred = agent.predict(x_img)


if __name__ == "__main__":
    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    parser.add_argument("CONFIG_NAME", type=str)
    parser.add_argument("VER_TO_LOAD", type=str)

    args = parser.parse_args()

    # --------------------
    # Setting Root Logger
    # --------------------
    init_logger()

    # --------------------------
    # Load and update settings
    # --------------------------
    merge_config_from_toml(f"./config/{args.CONFIG_NAME}.toml")
    S.CHECKPOINT_DIR = os.path.join(ROOT_DIR, f"Output/{args.VER_TO_LOAD}")

    logging.info(S)  # summarize settings

    # -------
    # Run
    # -------
    main()
