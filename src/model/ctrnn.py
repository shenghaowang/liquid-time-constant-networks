import torch
import torch.nn as nn


class CTRNN(nn.Module):
    def __init__(
        self,
        num_units: int,
        cell_clip: int = -1,
        global_feedback: bool = False,
        fix_tau: bool = True,
    ):
        self._num_units = num_units
        # Number of ODE solver steps
        self._unfolds = 6
        # Time of each ODE solver step, for variable time RNN change this
        # to a placeholder/non-trainable variable
        self._delta_t = 0.1

        self.global_feedback = global_feedback

        # Time-constant of the cell
        self.fix_tau = fix_tau
        self.tau = 1
        self.cell_clip = cell_clip

    def forward(self, x, h, global_feedback=None):
        # If global feedback is not provided, use the internal feedback
        if global_feedback is None:
            global_feedback = self.global_feedback

        # If global feedback is enabled, use the global feedback
        if global_feedback:
            h = h + x

        # If the time-constant is fixed, use the fixed time-constant
        if self.fix_tau:
            tau = self.tau
        else:
            tau = 1 + nn.functional.softplus(x[:, -1])

        # Apply the ODE solver
        for i in range(self._unfolds):
            h = h + self._delta_t / tau * (-h + nn.functional.softplus(x))
            if self.cell_clip > 0:
                h = torch.clamp(h, -self.cell_clip, self.cell_clip)
        return h
