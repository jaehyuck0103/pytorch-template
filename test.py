from dynaconf import settings as S
import os
import logging
import argparse
from utils.logger import init_logger

import torch
from torch.utils.data import DataLoader
from agents.agent1 import Agent1
from datasets.dataset1 import Dataset1
import numpy as np
from tqdm import tqdm


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def main():
    # 원래대로라면 test dataset용 Dataset 새로 정의해야하지만, 귀찮으므로 valid 재활용.
    test_dataset = Dataset1(mode='valid')
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1,
        shuffle=False, num_workers=16,
        worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2**32)
    )

    agent = Agent1(predict_only=True)
    agent.load_checkpoint()

    # start prediction
    tqdm_batch = tqdm(test_loader, f'Test')
    for idx, data in enumerate(tqdm_batch):
        # Prepare data
        x_img = data['img']

        # Predict
        y_pred = agent.predict(x_img)

    tqdm_batch.close()


if __name__ == '__main__':
    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    parser.add_argument('CONFIG_NAME', type=str)
    parser.add_argument('VER_TO_LOAD', type=str)

    args = parser.parse_args()

    # --------------------
    # Setting Root Logger
    # --------------------
    init_logger()

    # --------------------------
    # Load and update settings
    # --------------------------
    S.load_file(path=f'{args.CONFIG_NAME}.toml')
    S['CHECKPOINT_DIR'] = os.path.join(ROOT_DIR, f'Output/{args.VER_TO_LOAD}')

    logging.info(S.as_dict())   # summarize settings

    # -------
    # Run
    # -------
    main()
