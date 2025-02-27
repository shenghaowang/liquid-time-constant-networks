import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class SmnistDataset(Dataset):
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
        train_size: float = 0.9,
        batch_size: int = 16,
    ):
        super().__init__()
        self.train_size = train_size
        self.batch_size = batch_size

    def _reshape_images(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape([-1, 28, 28])
        return np.transpose(X, (1, 0, 2))

    def setup(self, stage=None):
        # Load MNIST dataset
        mnist_train = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        mnist_test = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

        # Extract features (images) and labels
        X_train, y_train = mnist_train.data.numpy(), mnist_train.targets.numpy()
        X_test, y_test = mnist_test.data.numpy(), mnist_test.targets.numpy()

        # Split the training data into training and validation sets
        train_split = int(self.train_size * X_train.shape[0])
        X_valid, y_valid = X_train[train_split:], y_train[train_split:]
        X_train, y_train = X_train[:train_split], y_train[:train_split]

        X_train = self._reshape_images(X_train)
        X_valid = self._reshape_images(X_valid)
        X_test = self._reshape_images(X_test)

        logger.debug(f"Total number of training sequences: {X_train.shape[1]}")
        logger.debug(f"Total number of validation sequences: {X_valid.shape[1]}")
        logger.debug(f"Total number of test sequences: {X_test.shape[1]}")

        self.train = SmnistDataset(X_train, y_train)
        self.valid = SmnistDataset(X_valid, y_valid)
        self.test = SmnistDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
