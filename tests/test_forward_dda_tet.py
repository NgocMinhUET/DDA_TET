from __future__ import annotations

import torch

from losses.core_losses import CoreDistillationLoss, orthogonality_loss
from models.dda_tet import DDATET


def test_v01_pipeline() -> None:
    batch_size = 2
    height, width = 255, 341
    input_dim = 7
    num_basis = 8
    x_c = torch.rand(batch_size, 3, height, width)
    x_d = torch.clamp(x_c + torch.randn(batch_size, 3, height, width) * 0.02, 0.0, 1.0)
    r_d = x_d - x_c
    raw_stats = torch.tensor(
        [
            [42.0, 125000.0, 0.45, 0.21, 0.15, 6.2, 0.35],
            [27.0, 4500.0, 0.52, 0.18, 0.08, 4.1, 0.12],
        ],
        dtype=torch.float32,
    )
    model = DDATET(input_dim=input_dim, num_basis=num_basis, clamp_output=False)
    criterion = CoreDistillationLoss()
    x_m, r_s, alpha = model(x_c, raw_stats)
    loss, logs = criterion(r_s, r_d, model.tile_bank.tiles)
    assert x_m.shape == (batch_size, 3, height, width)
    assert r_s.shape == (batch_size, 3, height, width)
    assert alpha.shape == (batch_size, num_basis)
    assert not torch.isnan(loss)
    assert alpha.max() <= model.alpha_scale + 1e-6
    assert alpha.min() >= -model.alpha_scale - 1e-6
    assert orthogonality_loss(model.tile_bank.tiles).ndim == 0
    print("x_m", list(x_m.shape))
    print("r_s", list(r_s.shape))
    print("alpha", alpha[0].detach().numpy())
    print(logs)


if __name__ == "__main__":
    test_v01_pipeline()
