from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from data.utils import cut_in_sequences

SEED = 893429


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
    # logger.debug(f"Loaded data from {data_dir}: {df.shape}")

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
    valid_size = int(val_ratio * X_train.shape[0])
    logger.debug(
        f"Training size: {X_train.shape[0] - valid_size}, Validation size: {valid_size}"
    )

    permutation = np.random.RandomState(SEED).permutation(X_train.shape[0])
    X_valid, y_valid = (
        X_train[permutation[:valid_size], :],
        y_train[permutation[:valid_size], :],
    )
    X_train, y_train = (
        X_train[permutation[valid_size:], :],
        y_train[permutation[valid_size:], :],
    )

    return X_train, y_train, X_valid, y_valid


class OccupancyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> np.ndarray:
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        test_files: List[str],
        feature_cols: List[str],
        target_col: str,
        seq_len: int,
        batch_size: int = 16,
    ):
        super().__init__()
        self.train_file = train_file
        self.test_files = test_files
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load training data
        X_train, y_train = load_data(
            data_dir=self.train_file,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
        )

        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        X_train = (X_train - X_mean) / X_std
        X_train, y_train = cut_in_sequences(X_train, y_train, self.seq_len)

        # Split data for training and validation
        X_train, y_train, X_valid, y_valid = split_data(X_train, y_train)

        # Load test data
        X_tests, y_tests = [], []
        for test_data_dir in self.test_files:
            X_test, y_test = load_data(
                data_dir=test_data_dir,
                feature_cols=self.feature_cols,
                target_col=self.target_col,
            )
            X_test = (X_test - X_mean) / X_std

            X_test, y_test = cut_in_sequences(X_test, y_test, self.seq_len, 8)
            X_tests.append(X_test)
            y_tests.append(y_test)

        X_test, y_test = np.concatenate(X_tests, axis=0), np.concatenate(
            y_tests, axis=0
        )

        self.train = OccupancyDataset(X_train, y_train)
        self.valid = OccupancyDataset(X_valid, y_valid)
        self.test = OccupancyDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
