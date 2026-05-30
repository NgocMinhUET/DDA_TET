from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class AlphaMLP(nn.Module):
    """Tiny coefficient predictor for adaptive tile mixing."""

    def __init__(
        self,
        input_dim: int = 7,
        num_basis: int = 8,
        hidden_dims: Iterable[int] = (32, 32),
        activation: str = "relu",
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if num_basis < 1:
            raise ValueError("num_basis must be >= 1")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(dim)))
            layers.append(_activation(activation))
            prev_dim = int(dim)
        layers.append(nn.Linear(prev_dim, num_basis))

        output_activation = output_activation.lower()
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation in {"none", "identity"}:
            pass
        else:
            raise ValueError(f"Unsupported output_activation: {output_activation}")

        self.mlp = nn.Sequential(*layers)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        if stats.ndim != 2:
            raise ValueError(f"stats must be [N,D], got {tuple(stats.shape)}")
        return self.mlp(stats)
