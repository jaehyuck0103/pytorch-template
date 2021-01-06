import json
import logging
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from config import settings as S
from datasets import get_dataset
from modules import get_network
from utils.metrics import EarlyStopping

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
np.set_printoptions(linewidth=100)
cudnn.benchmark = True


class BaseAgent:
    def __init__(self, predict_only=False):
        # device configuration
        self.device = torch.device("cuda")

        # Network
        self.net = get_network(S.net.name).to(self.device)

        num_params = sum([p.data.numel() for p in self.net.parameters()])
        print(f"\nNumber of model parameters: {num_params}")

        if predict_only:
            return

        # Dataset Setting
        train_dataset, valid_dataset = get_dataset(S.dataset.name)
        print(f"{len(train_dataset)} training items and {len(valid_dataset)} validation items\n")

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=S.agent.train_batch_size,
            num_workers=S.agent.train_workers,
            shuffle=True,
            drop_last=True,
            worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2 ** 32),
        )

        self.valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=S.agent.valid_batch_size,
            num_workers=S.agent.valid_workers,
            shuffle=False,
            worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2 ** 32),
        )

        # Optimizer
        if S.optim.name == "SGD":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(), lr=S.optim.lr, momentum=0.9, weight_decay=0.0001
            )
        elif S.optim.name == "ADAM":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=S.optim.lr)
        else:
            raise ValueError(f"Unexpected Optimizer {S.optim.name}")

        # LR Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=S.optim.lr_decay_rate,
            patience=S.optim.lr_decay_patience,
            verbose=True,
            threshold=0,
        )

        self.early_stopper = EarlyStopping(mode="max", patience=S.optim.early_stop, verbose=True)

        # ETC
        self.epoch = 0

        self.save_opts()
        self.tb_writers = {
            "train": SummaryWriter(os.path.join(S.agent.log_dir, "train")),
            "val": SummaryWriter(os.path.join(S.agent.log_dir, "val")),
        }

    def train(self):
        start_epoch = self.epoch + 1
        for self.epoch in range(start_epoch, S.agent.num_epochs + 1):
            # train step
            print("\n-------------------------------------------------------------------------")
            print(f"\nEpoch {self.epoch} - LR {self.optimizer.param_groups[0]['lr']}")
            epoch_start = time.time()
            self._train_epoch()
            print("Train Epoch Duration: ", time.time() - epoch_start)

            # val step
            if self.epoch % S.agent.validation_interval == 0:
                val_acc = self._validate_epoch()
                self.lr_scheduler.step(val_acc)
                if self.early_stopper.step(val_acc):
                    break

            # save step
            self.save_snapshot()

    def operation_check(self):
        self._validate_epoch()
        # self._train_epoch(early_return_step=10)
        self.lr_scheduler.step(0.1)
        self.save_snapshot()

    def _train_epoch(self, early_return_step=None):
        raise NotImplementedError

    @torch.no_grad()
    def _validate_epoch(self):
        raise NotImplementedError

    def save_opts(self):
        os.makedirs(S.agent.log_dir, exist_ok=True)

        with open(os.path.join(S.agent.log_dir, "opt.json"), "w") as fp:
            json.dump(S, fp, indent=4)

    def save_snapshot(self):
        save_folder = os.path.join(S.agent.log_dir, "weights", f"epoch_{self.epoch}")
        os.makedirs(save_folder, exist_ok=True)

        for name, elem in self.net.snapshot_elements().items():
            save_path = os.path.join(save_folder, f"{name}.pt")
            torch.save(elem.state_dict(), save_path)
            print(f"Save snapshot at {save_path}")

        to_save = {
            "epoch": self.epoch,
            "adam_state_dict": self.optimizer.state_dict(),
        }
        save_path = os.path.join(save_folder, "info.pt")
        torch.save(to_save, save_path)

    def load_snapshot(self, weights_dir):

        logging.info(f"Loading model from folder {weights_dir}")

        for name, elem in self.net.snapshot_elements().items():
            path = os.path.join(weights_dir, f"{name}.pt")
            state_dict = torch.load(path)
            elem.load_state_dict(state_dict)
            logging.info(f"Load snapshot from {path}")

        """
        path = os.path.join(weights_dir, "info.pt")
        if os.path.isfile(path):
            info_dict = torch.load(path)
            self.optimizer.load_state_dict(info_dict["adam_state_dict"])
            print("Loading Adam weights completed")
            self.epoch = info_dict["epoch"] + 1
            print(f"Start from {self.epoch}")
            assert self.opt.width == info_dict["width"] and self.opt.height == info_dict["height"]
        else:
            print("Cannot find info.pt")
        """
