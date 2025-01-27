from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from torch.utils.data import DataLoader, Dataset


def load_data(
    data_dir: str, feature_cols: List[str], target_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    data_dir : str
        _description_
    feature_cols : List[str]
        _description_
    target_col : str
        _description_

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        _description_
    """
    df = pd.read_csv(data_dir)

    X = np.stack([df[col].values for col in feature_cols], axis=-1)
    y = df[target_col].values.astype(np.int32)

    return X, y


def split_data(
    X_train: np.ndarray, y_train: np.ndarray, val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    X_train : np.ndarray
        _description_
    y_train : np.ndarray
        _description_
    val_ratio : float, optional
        _description_, by default 0.1

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        _description_
    """
    valid_size = int(val_ratio * X_train.shape[1])
    logger.debug(
        f"Training size: {X_train.shape[1] - valid_size}, Validation size: {valid_size}"
    )

    permutation = np.random.RandomState(893429).permutation(X_train.shape[1])
    X_valid, y_valid = (
        X_train[:, permutation[:valid_size]],
        y_train[:, permutation[:valid_size]],
    )
    X_train, y_train = (
        X_train[:, permutation[valid_size:]],
        y_train[:, permutation[valid_size:]],
    )

    return X_train, y_train, X_valid, y_valid


class Dataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[idx]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: np.ndarray,
        valid_data: np.ndarray,
        test_data: np.ndarray,
        batch_size: int = 32,
    ):
        super().__init__()
        self.train = Dataset(train_data)
        self.val = Dataset(valid_data)
        self.test = Dataset(test_data)
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train = Dataset(self.data)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
