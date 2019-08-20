from dynaconf import settings as S

import os
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets.dataset1 import Dataset1
from nets.net1 import Net1
from utils.misc import AverageMeter, EarlyStopping


class Agent1():
    def __init__(self):
        # device configuration
        self.device = torch.device('cuda')

        # Dataset Setting
        train_dataset = Dataset1(mode='train')
        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=S.TRAIN_BATCH_SIZE,
            shuffle=True, num_workers=8, drop_last=True, pin_memory=True,
            worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2**32)
        )

        valid_dataset = Dataset1(mode='valid')
        self.valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=S.VALID_BATCH_SIZE,
            shuffle=False, num_workers=8,
            worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2**32)
        )

        # Network
        if S.NET == 'NET1':
            self.net = Net1().to(self.device)
        else:
            raise ValueError(f'Unexpected Network {S.NET}')

        # Optimizer
        if S.OPTIMIZER == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=S.LR,
                                             momentum=0.9, weight_decay=0.0001)
        elif S.OPTIMIZER == 'ADAM':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=S.LR)
        else:
            raise ValueError(f'Unexpected Optimizer {S.OPTIMIZER}')

        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=S.LR_DECAY_RATE,
            patience=S.PATIENCE, verbose=True, threshold=0)

        self.early_stopper = EarlyStopping(mode='max', patience=S.EARLY_STOP, verbose=True)

    def save_checkpoint(self):

        state = {
            'epoch': self.current_epoch,
            'state_dict': self.net.state_dict(),
        }

        filename = f'KFOLD_{S.KFOLD_I}.ckpt'
        logging.info("Saving checkpoint '{}'".format(filename))
        os.makedirs(S.CHECKPOINT_DIR, exist_ok=True)
        torch.save(state, os.path.join(S.CHECKPOINT_DIR, filename))

        logging.info(f'Checkpoint saved successfully at (epoch {self.current_epoch})')

    def load_checkpoint(self):

        filename = f'KFOLD_{S.KFOLD_I}.ckpt'
        logging.info("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(os.path.join(S.CHECKPOINT_DIR, filename))

        self.current_epoch = checkpoint['epoch'] + 1
        self.net.load_state_dict(checkpoint['state_dict'])

        logging.info(f'Checkpoint loaded successfully at (epoch {checkpoint["epoch"]})')

    def train(self):
        for epoch in range(S.NUM_EPOCHS):
            self.current_epoch = epoch
            self.train_one_epoch()

            if (epoch + 1) % S.VALIDATION_INTERVAL == 0:
                validate_acc = self.validate()

                self.scheduler.step(validate_acc)
                if self.early_stopper.step(validate_acc):
                    break

        self.save_checkpoint()

    def train_one_epoch(self):
        # Training mode
        self.net.train()

        # Init Average Meters
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()

        tqdm_batch = tqdm(self.train_loader, f'Epoch-{self.current_epoch}-')
        for data in tqdm_batch:
            # Prepare data
            x_img = data['img'].to(self.device, torch.float)  # (batch, 3, H, W)
            y_gt = data['gt'].to(self.device, torch.float)  # (batch, 1)
            batch_size = x_img.shape[0]

            # Forward pass
            y_pred = self.net(x_img)

            # Compute loss
            cur_loss = F.binary_cross_entropy_with_logits(y_pred, y_gt)

            # Backprop and optimize
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            # Metrics
            y_pred_t = torch.sigmoid(y_pred) > 0.5
            y_gt_t = y_gt > 0.5
            acc = (y_pred_t == y_gt_t).sum().item() / batch_size

            epoch_loss.update(cur_loss.item(), batch_size)
            epoch_acc.update(acc, batch_size)

        tqdm_batch.close()

        logging.info(f'Train at epoch- {self.current_epoch} |'
                     f'loss: {epoch_loss.val} - acc: {epoch_acc.val}')

        return epoch_acc.val

    def validate(self):
        # Eval mode
        self.net.eval()

        # Init Average Meters
        epoch_acc = AverageMeter()

        tqdm_batch = tqdm(self.valid_loader, f'Epoch-{self.current_epoch}-')
        with torch.no_grad():
            for data in tqdm_batch:
                # Prepare data
                x_img = data['img'].to(self.device, torch.float)  # (batch, 3, H, W)
                y_gt = data['gt'].to(self.device, torch.float)  # (batch, 1)
                batch_size = x_img.shape[0]

                # Forward pass
                y_pred = self.net(x_img)

                # Metrics
                y_pred_t = torch.sigmoid(y_pred) > 0.5
                y_gt_t = y_gt > 0.5
                acc = (y_pred_t == y_gt_t).sum().item() / batch_size

                epoch_acc.update(acc, batch_size)
        tqdm_batch.close()

        logging.info(f'Validate at epoch- {self.current_epoch} | acc: {epoch_acc.val}')

        return epoch_acc.val
