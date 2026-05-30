from __future__ import annotations

import torch
import torch.nn as nn

from models.alpha_mlp import AlphaMLP
from models.tile_bank import TileBank
from utils.stats_normalizer import StatsNormalizer, StatsNormalizerConfig


class DDATET(nn.Module):
    """Core DDA-TET v0.1 student.

    Online inference path:
        raw_stats -> normalize -> AlphaMLP -> alpha
        alpha + TileBank -> residual r_s
        x_m = x_c + r_s
    """

    def __init__(
        self,
        input_dim: int = 7,
        num_basis: int = 8,
        tile_size: int = 32,
        in_channels: int = 3,
        residual_scale: float = 4.0 / 255.0,
        alpha_scale: float = 1.0,
        clamp_output: bool = False,
        stats_normalizer: StatsNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.alpha_predictor = AlphaMLP(input_dim=input_dim, num_basis=num_basis)
        self.tile_bank = TileBank(
            num_basis=num_basis, tile_size=tile_size, in_channels=in_channels
        )
        self.residual_scale = float(residual_scale)
        self.alpha_scale = float(alpha_scale)
        self.clamp_output = bool(clamp_output)
        self.stats_normalizer = stats_normalizer or StatsNormalizer(StatsNormalizerConfig())

    def set_clamp_output(self, enabled: bool) -> None:
        self.clamp_output = bool(enabled)

    def forward(
        self, x_c: torch.Tensor, raw_stats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_c.ndim != 4:
            raise ValueError(f"x_c must be [N,C,H,W], got {tuple(x_c.shape)}")
        _, _, height, width = x_c.shape
        stats = self.stats_normalizer(raw_stats).to(device=x_c.device, dtype=x_c.dtype)
        alpha = self.alpha_predictor(stats) * self.alpha_scale
        r_s = self.tile_bank(alpha, height, width) * self.residual_scale
        x_m = x_c + r_s
        if self.clamp_output:
            x_m = torch.clamp(x_m, 0.0, 1.0)
        return x_m, r_s, alpha
