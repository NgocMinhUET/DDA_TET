from __future__ import annotations

import torch
import torch.nn as nn


def residual_distill_loss(r_s: torch.Tensor, r_d: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(r_s - r_d))


def sparse_loss(r_s: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(r_s))


def orthogonality_loss(tiles: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean-squared Gram orthogonality loss for tile basis diversity."""
    if tiles.ndim != 4:
        raise ValueError(f"tiles must be [K,C,B,B], got {tuple(tiles.shape)}")
    K = tiles.shape[0]
    if K <= 1:
        return tiles.new_tensor(0.0)
    flat = tiles.view(K, -1)
    flat = flat / (flat.norm(dim=1, keepdim=True) + eps)
    gram = flat @ flat.t()
    identity = torch.eye(K, device=tiles.device, dtype=tiles.dtype)
    return torch.mean((gram - identity) ** 2)


class CoreDistillationLoss(nn.Module):
    def __init__(
        self,
        lambda_distill: float = 1.0,
        lambda_sparse: float = 0.01,
        lambda_orth: float = 0.001,
    ) -> None:
        super().__init__()
        self.lambda_distill = float(lambda_distill)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_orth = float(lambda_orth)

    def forward(
        self, r_s: torch.Tensor, r_d: torch.Tensor, tiles: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_distill = residual_distill_loss(r_s, r_d)
        l_sparse = sparse_loss(r_s)
        l_orth = orthogonality_loss(tiles)
        total = (
            self.lambda_distill * l_distill
            + self.lambda_sparse * l_sparse
            + self.lambda_orth * l_orth
        )
        logs = {
            "loss_total": float(total.detach().cpu()),
            "l_distill": float(l_distill.detach().cpu()),
            "l_sparse": float(l_sparse.detach().cpu()),
            "l_orth": float(l_orth.detach().cpu()),
        }
        return total, logs
