from dynaconf import settings as S

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset


ROOT_DIR = os.path.join(os.path.dirname(__file__), '../')
ROOT_DIR = os.path.abspath(ROOT_DIR)

TRAIN_IMG_DIR = os.path.join(ROOT_DIR, 'data/train_images')
TRAIN_CSV_PATH = os.path.join(ROOT_DIR, 'data/train.csv')
TEST_IMG_DIR = os.path.join(ROOT_DIR, 'data/test_images')


def _imread_img(f):
    img = cv2.imread(f).astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    return img


class Dataset1(Dataset):
    def __init__(self, mode):
        super().__init__()

        self.mode = mode

        df = pd.read_csv(TRAIN_CSV_PATH)

        num_imgs = df.shape[0]
        kf = StratifiedKFold(n_splits=S.KFOLD_N, shuffle=True, random_state=910103)
        train_idx, valid_idx = list(kf.split(np.zeros(num_imgs), df['gt']))[S.KFOLD_I]

        self.df = df
        if mode == 'train':
            self.idx_map = train_idx
        elif mode == 'valid':
            self.idx_map = valid_idx
        else:
            raise ValueError(f'Unknown Mode {mode}')

    def __getitem__(self, idx):

        idx = self.idx_map[idx]

        img_file_path = os.path.join(TRAIN_IMG_DIR, self.df['img_path'].iloc[idx])
        img = _imread_img(img_file_path)

        img = cv2.resize(img, (224, 224))

        gt = self.df['gt']

        if self.mode == 'train':
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
        gt = np.expand_dims(gt, axis=0)
        sample = {'img': img, 'gt': gt}

        return sample

    def __len__(self):
        return len(self.idx_map)
