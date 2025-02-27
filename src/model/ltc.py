from enum import Enum

import pytorch_lightning as pl
import torch
import torch.nn as nn


class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2


class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2


class LTCCell(nn.Module):
    def __init__(self, num_units, input_size):
        super().__init__()
        self.input_size = input_size
        self.num_units = num_units

        # Define learnable parameters
        self.W = nn.Parameter(torch.rand(input_size, num_units))
        self.b = nn.Parameter(torch.zeros(num_units))
        self.gleak = nn.Parameter(torch.ones(num_units))
        self.cm = nn.Parameter(torch.ones(num_units))

        self.ode_solver_unfolds = 6
        self.solver = ODESolver.SemiImplicit
        self.input_mapping = MappingType.Affine

    def forward(self, x, h):
        # Simple ODE-based update rule (placeholder for actual ODE solver logic)
        dh_dt = -self.gleak * h + torch.matmul(x, self.W) + self.b
        h_next = h + dh_dt / self.cm
        return h_next


class LTCModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.ltc_cell = LTCCell(hidden_size, input_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hparams.hidden_size, device=x.device)

        for t in range(seq_len):
            h = self.ltc_cell(x[:, t, :], h)

        return self.fc(h)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
