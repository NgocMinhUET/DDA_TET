from __future__ import annotations

import torch
from tqdm import trange

from losses.core_losses import CoreDistillationLoss
from models.dda_tet import DDATET


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDATET(clamp_output=False).to(device)
    criterion = CoreDistillationLoss(lambda_distill=1.0, lambda_sparse=0.01, lambda_orth=0.001)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Synthetic target residual: small periodic-ish signal.
    B, C, H, W = 4, 3, 255, 341
    x_c = torch.rand(B, C, H, W, device=device)
    raw_stats = torch.tensor(
        [[37.0, 30000.0, 0.5, 0.2, 0.1, 5.0, 0.2]] * B,
        device=device,
    )
    r_d = torch.randn(B, C, H, W, device=device) * 0.01

    for step in trange(30, desc="synthetic smoke train"):
        _, r_s, _ = model(x_c, raw_stats)
        loss, logs = criterion(r_s, r_d, model.tile_bank.tiles)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step % 10 == 0:
            print(logs)


if __name__ == "__main__":
    main()
