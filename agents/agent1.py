import logging

import torch
import torch.nn.functional as F

from utils.metrics import AverageMeter
from utils.utils import StaticPrinter

from .base import BaseAgent


class Agent1(BaseAgent):
    def _train_epoch(self):
        # Training mode
        self.net.train()

        # Init Average Meters
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()

        print("")
        sp = StaticPrinter()
        for step, inputs in enumerate(self.train_loader):
            # Prepare data
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

            # Forward pass
            y_pred = self.net(inputs)  # (batch, n_class)

            # Compute loss
            y_gt = inputs["label"]  # (batch)
            cur_loss = F.cross_entropy(y_pred, y_gt)

            # Backprop and optimize
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            # Metrics
            batch_size = y_gt.shape[0]
            _, pred_idx = torch.max(y_pred, 1)
            cur_acc = (pred_idx == y_gt).sum().item() / batch_size
            """  # binary cross entropy 일 때의 예시.
            y_pred_t = torch.sigmoid(y_pred) > 0.5
            y_gt_t = y_gt > 0.5
            cur_acc = (y_pred_t == y_gt_t).sum().item() / batch_size
            """

            epoch_loss.update(cur_loss.item(), batch_size)
            epoch_acc.update(cur_acc, batch_size)

            # Print
            sp.reset()
            sp.print(
                f"Epoch {self.epoch}| Training {step+1}/{len(self.train_loader)} | "
                f"loss: {epoch_loss.val:.4f} - acc: {epoch_acc.val:.4f}"
            )

        logging.info(
            f"Train at epoch {self.epoch} |"
            f"loss: {epoch_loss.val:.4f} - acc: {epoch_acc.val:.4f}"
        )

        return epoch_acc.val

    @torch.no_grad()
    def _validate_epoch(self):
        # Eval mode
        self.net.eval()

        # Init Average Meters
        epoch_acc = AverageMeter()

        print("")
        sp = StaticPrinter()
        for step, inputs in enumerate(self.valid_loader):
            # Prepare data
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

            # Forward pass
            y_gt = inputs["label"]  # (batch, 1)
            y_pred = self.net(inputs)

            # Metrics
            batch_size = y_gt.shape[0]
            _, pred_idx = torch.max(y_pred, 1)
            cur_acc = (pred_idx == y_gt).sum().item() / batch_size
            """  # binary cross entropy 일 때의 예시.
            y_pred_t = torch.sigmoid(y_pred) > 0.5
            y_gt_t = y_gt > 0.5
            cur_acc = (y_pred_t == y_gt_t).sum().item() / batch_size
            """
            epoch_acc.update(cur_acc, batch_size)

            # Print
            sp.reset()
            sp.print(
                f"Epoch {self.epoch}| Validation {step+1}/{len(self.valid_loader)} | "
                f"acc: {epoch_acc.val:.4f}"
            )

        logging.info(f"Validate at epoch {self.epoch} | acc: {epoch_acc.val:.4f}")

        return epoch_acc.val

    @torch.no_grad()
    def predict(self, inputs):
        self.net.eval()

        # Prepare data
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # Forward pass
        y_pred = self.net(inputs)

        return y_pred.cpu().numpy()
