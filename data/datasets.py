from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def _load_tensor_or_image(path: Path) -> torch.Tensor:
    if path.suffix.lower() in {".pt", ".pth"}:
        x = torch.load(path, map_location="cpu")
        if isinstance(x, dict):
            # accept common dict payloads
            for key in ("tensor", "x", "image", "xc", "rd"):
                if key in x:
                    x = x[key]
                    break
        if not torch.is_tensor(x):
            raise TypeError(f"Tensor file {path} did not contain a tensor")
        return x.float()
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        img = Image.open(path).convert("RGB")
        return pil_to_tensor(img).float() / 255.0
    raise ValueError(f"Unsupported file type: {path}")


class ResidualCacheDataset(Dataset):
    """Dataset for cached compressed reconstructions and teacher residuals."""

    def __init__(self, cache_root: str | Path, index_file: str | Path) -> None:
        self.cache_root = Path(cache_root)
        index_path = self.cache_root / index_file
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        self.samples: list[dict[str, Any]] = []
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        if not self.samples:
            raise ValueError(f"No samples found in {index_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.cache_root / p

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        x_c = _load_tensor_or_image(self._resolve(item["xc"]))
        r_d = _load_tensor_or_image(self._resolve(item["rd"]))
        if x_c.max() > 2.0:
            x_c = x_c / 255.0
        # residual may be stored as image-like offset, but recommended is tensor
        if r_d.shape != x_c.shape:
            raise ValueError(f"Shape mismatch x_c={x_c.shape}, r_d={r_d.shape}")
        stats = torch.tensor(item["stats"], dtype=torch.float32)
        sample = {"x_c": x_c, "r_d": r_d, "raw_stats": stats}
        if "xd" in item:
            sample["x_d"] = _load_tensor_or_image(self._resolve(item["xd"]))
        return sample
