from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ResidualCacheDataset
from models.dda_tet import DDATET


def load_model(ckpt_path: str, device: torch.device) -> DDATET:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    s = cfg.get("student", {})
    model = DDATET(
        input_dim=s.get("input_dim", 7),
        num_basis=s.get("num_basis", 8),
        tile_size=s.get("tile_size", 32),
        in_channels=s.get("in_channels", 3),
        residual_scale=s.get("residual_scale", 4.0 / 255.0),
        alpha_scale=s.get("alpha_scale", 1.0),
        clamp_output=True,
    )
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def measure_runtime(model: DDATET, loader: DataLoader, device: torch.device) -> None:
    total_images = 0
    total_time = 0.0
    for batch in tqdm(loader, desc="runtime"):
        x_c = batch["x_c"].to(device)
        raw_stats = batch["raw_stats"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(x_c, raw_stats)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        total_images += x_c.shape[0]
    print(f"images: {total_images}")
    print(f"online student time/image: {total_time / max(total_images, 1):.6f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dda_tet_pilot_v0.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ResidualCacheDataset(cfg["data"]["cache_root"], cfg["data"].get("val_index", "val.jsonl"))
    dl = DataLoader(ds, batch_size=cfg["training"].get("batch_size", 4), shuffle=False)
    model = load_model(args.checkpoint, device)
    measure_runtime(model, dl, device)
