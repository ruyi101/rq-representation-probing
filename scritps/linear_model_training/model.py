
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    """
    Simple linear model using nn.Linear without bias:

        f(x) = w^T x

    Input:  x with shape [batch_size, D]
    Output: logits with shape [batch_size]
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=False)  # <-- use torch Linear, no bias

        # Optional: small Gaussian init instead of default
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D]
        returns: logits [N]
        """
        logits = self.linear(x)      # [N, 1]
        return logits.squeeze(-1)    # [N]



