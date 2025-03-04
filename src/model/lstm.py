import pytorch_lightning as pl
import torch
import torch.nn as nn


class LSTM(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        is_binary: bool = True,
        learning_rate: float = 0.001,
    ):
        super(LSTM, self).__init__()
        self.save_hyperparameters()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Fully connected layer for binary classification
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        # Loss function (Binary Cross Entropy for classification)
        self.loss_fn = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)

        if self.hparams.is_binary:
            y = y.float()  # Ensure y is (batch, 1)
            loss = self.loss_fn(logits, y)
            preds = torch.sigmoid(logits) > 0.5  # Convert logits to binary predictions

        else:  # Multi-Class Classification
            y = y.long()  # Ensure y is (batch,)
            loss = self.loss_fn(logits, y)
            preds = torch.argmax(logits, dim=1)

        # Compute accuracy
        acc = (preds == y).float().mean()

        # Logging
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        if self.hparams.is_binary:
            y = y.float()  # Ensure y is (batch, 1)
            loss = self.loss_fn(logits, y)
            preds = torch.sigmoid(logits) > 0.5  # Convert logits to binary predictions

        else:  # Multi-Class Classification
            y = y.long()  # Ensure y is (batch,)
            loss = self.loss_fn(logits, y)
            preds = torch.argmax(logits, dim=1)

        # Compute accuracy
        acc = (preds == y).float().mean()

        # Logging
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        if self.hparams.is_binary:
            y = y.float()  # Ensure y is (batch, 1)
            loss = self.loss_fn(logits, y)
            preds = torch.sigmoid(logits) > 0.5  # Convert logits to binary predictions

        else:  # Multi-Class Classification
            y = y.long()  # Ensure y is (batch,)
            loss = self.loss_fn(logits, y)
            preds = torch.argmax(logits, dim=1)

        # Compute accuracy
        acc = (preds == y).float().mean()

        # Logging
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
