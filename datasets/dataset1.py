import os

import cv2
import numpy as np
from torch.utils.data import Dataset

import pandas as pd
from config import settings as S
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
ROOT_DIR = os.path.abspath(ROOT_DIR)

TRAIN_IMG_DIR = os.path.join(ROOT_DIR, "Data/train_images")
TRAIN_CSV_PATH = os.path.join(ROOT_DIR, "Data/train.csv")
TEST_IMG_DIR = os.path.join(ROOT_DIR, "Data/test_images")


def _imread_float(f):
    img = cv2.imread(f).astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    return img


class Dataset1(Dataset):
    def __init__(self, mode):
        super().__init__()

        self.mode = mode

        df = pd.read_csv(
            TRAIN_CSV_PATH, names=["basename", "label"], dtype={"basename": str, "label": int}
        )

        num_imgs = df.shape[0]
        kf = StratifiedKFold(n_splits=S.KFOLD_N, shuffle=True, random_state=910103)
        train_idx, valid_idx = list(kf.split(np.zeros(num_imgs), df["label"]))[S.KFOLD_I]

        self.df = df
        if mode == "train":
            self.idx_map = train_idx
        elif mode == "valid":
            self.idx_map = valid_idx
        else:
            raise ValueError(f"Unknown Mode {mode}")

    def __getitem__(self, _idx):

        idx = self.idx_map[_idx]

        basename = self.df["basename"].iloc[idx]
        label = self.df["label"].iloc[idx]

        img_path = os.path.join(TRAIN_IMG_DIR, f"{basename}.jpg")
        img = _imread_float(img_path)

        img = cv2.resize(img, (224, 224))

        if self.mode == "train":
            # some augmentation
            pass
        else:
            # no augmentation
            pass

        # Add depth channels
        img = np.transpose(img, (2, 0, 1))
        # img = np.stack([img, img, img], axis=0)  # gray -> 3channel

        # Normalize
        img[0] = (img[0] - 0.485) / 0.229
        img[1] = (img[1] - 0.456) / 0.224
        img[2] = (img[2] - 0.406) / 0.225

        # sample return
        # label = np.expand_dims(label, axis=0)
        sample = {"img": img, "label": label}

        return sample

    def __len__(self):
        return len(self.idx_map)
