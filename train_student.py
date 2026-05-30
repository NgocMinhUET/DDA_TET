from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data.datasets import ResidualCacheDataset
from losses.core_losses import CoreDistillationLoss
from models.dda_tet import DDATET


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_model(cfg: dict, device: torch.device) -> DDATET:
    s = cfg["student"]
    model = DDATET(
        input_dim=s.get("input_dim", 7),
        num_basis=s.get("num_basis", 8),
        tile_size=s.get("tile_size", 32),
        in_channels=s.get("in_channels", 3),
        residual_scale=s.get("residual_scale", 4.0 / 255.0),
        alpha_scale=s.get("alpha_scale", 1.0),
        clamp_output=s.get("clamp_output_train", False),
    )
    return model.to(device)


def train(cfg: dict) -> None:
    out_dir = Path(cfg["experiment"].get("output_dir", "runs/dda_tet"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg["experiment"].get("seed", 42)))

    device_name = cfg["training"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    ds = ResidualCacheDataset(cfg["data"]["cache_root"], cfg["data"]["train_index"])
    dl = DataLoader(
        ds,
        batch_size=int(cfg["training"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )

    model = make_model(cfg, device)
    loss_cfg = cfg["loss"]
    criterion = CoreDistillationLoss(
        lambda_distill=loss_cfg.get("lambda_distill", 1.0),
        lambda_sparse=loss_cfg.get("lambda_sparse", 0.01),
        lambda_orth=loss_cfg.get("lambda_orth", 0.001),
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-5)),
    )
    use_amp = bool(cfg["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    grad_clip = float(cfg["training"].get("gradient_clip_norm", 1.0))

    for epoch in range(int(cfg["training"].get("epochs", 5))):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch+1}")
        avg = {"loss_total": 0.0, "l_distill": 0.0, "l_sparse": 0.0, "l_orth": 0.0}
        n = 0
        for batch in pbar:
            x_c = batch["x_c"].to(device, non_blocking=True)
            r_d = batch["r_d"].to(device, non_blocking=True)
            raw_stats = batch["raw_stats"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                _, r_s, _ = model(x_c, raw_stats)
                loss, logs = criterion(r_s, r_d, model.tile_bank.tiles)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
            n += 1
            for k in avg:
                avg[k] += logs[k]
            pbar.set_postfix({k: f"{avg[k]/n:.5f}" for k in avg})

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "config": cfg,
        }
        torch.save(ckpt, out_dir / f"checkpoint_epoch_{epoch+1}.pt")
    torch.save({"model": model.state_dict(), "config": cfg}, out_dir / "model_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dda_tet_pilot_v0.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
