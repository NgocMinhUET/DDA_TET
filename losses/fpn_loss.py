from __future__ import annotations

import torch
import torch.nn as nn

from detectors.faster_rcnn_hooks import FasterRCNNFPNFeatureExtractor


def normalize_feature(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # channel-wise mean/std over spatial dimensions
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + eps)


class FPNFeatureLoss(nn.Module):
    def __init__(
        self,
        layers: tuple[str, ...] = ("P3", "P4", "P5"),
        weights: dict[str, float] | None = None,
        distance: str = "l1",
    ) -> None:
        super().__init__()
        self.extractor = FasterRCNNFPNFeatureExtractor(layers=layers)
        self.layers = tuple(layers)
        self.weights = weights or {"P3": 0.5, "P4": 0.3, "P5": 0.2}
        self.distance = distance.lower()

    def forward(self, x_m: torch.Tensor, x_d: torch.Tensor) -> torch.Tensor:
        feats_m = self.extractor(x_m)
        feats_d = self.extractor(x_d)
        total = x_m.new_tensor(0.0)
        for layer in self.layers:
            if layer not in feats_m or layer not in feats_d:
                continue
            a = normalize_feature(feats_m[layer])
            b = normalize_feature(feats_d[layer])
            if self.distance == "l1":
                loss = torch.mean(torch.abs(a - b))
            elif self.distance in {"l2", "mse"}:
                loss = torch.mean((a - b) ** 2)
            else:
                raise ValueError(f"Unsupported distance: {self.distance}")
            total = total + float(self.weights.get(layer, 1.0)) * loss
        return total
