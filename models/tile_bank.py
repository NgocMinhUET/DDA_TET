from __future__ import annotations

import torch
import torch.nn as nn


class TileBank(nn.Module):
    """Learnable bank of RGB-domain basis tiles.

    The bank first linearly mixes K small tiles in tile space and only then
    repeats the mixed tile to the target image size. This is cheaper than
    repeating all K tiles to full resolution before mixing.
    """

    def __init__(
        self,
        num_basis: int = 8,
        tile_size: int = 32,
        in_channels: int = 3,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if num_basis < 1:
            raise ValueError("num_basis must be >= 1")
        if tile_size < 1:
            raise ValueError("tile_size must be >= 1")
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")

        self.num_basis = int(num_basis)
        self.tile_size = int(tile_size)
        self.in_channels = int(in_channels)
        self.tiles = nn.Parameter(
            torch.randn(num_basis, in_channels, tile_size, tile_size) * init_std
        )

    def forward(self, alpha: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Return full-resolution student residual.

        Args:
            alpha: Tensor with shape [N, K].
            height: Target image height.
            width: Target image width.

        Returns:
            Tensor with shape [N, C, height, width].
        """
        if alpha.ndim != 2:
            raise ValueError(f"alpha must be [N,K], got shape {tuple(alpha.shape)}")
        if alpha.shape[1] != self.num_basis:
            raise ValueError(
                f"alpha K={alpha.shape[1]} does not match num_basis={self.num_basis}"
            )
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")

        # [N,K] x [K,C,B,B] -> [N,C,B,B]
        mixed_tile = torch.einsum("nk,kchw->nchw", alpha, self.tiles)

        repeat_h = (height + self.tile_size - 1) // self.tile_size
        repeat_w = (width + self.tile_size - 1) // self.tile_size
        full_residual = mixed_tile.repeat(1, 1, repeat_h, repeat_w)
        return full_residual[:, :, :height, :width]
