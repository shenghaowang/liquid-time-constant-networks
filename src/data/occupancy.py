from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
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


class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class DataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size=32):
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train = Dataset(self.data)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False)
