from config import settings as S

from .dataset1 import Dataset1


def get_dataset(name):
    if name == "dataset1":
        train_dataset = Dataset1(mode="train")
        valid_dataset = Dataset1(mode="valid")
    else:
        raise ValueError(f"Unexpected Dataset {name}")

    return train_dataset, valid_dataset
