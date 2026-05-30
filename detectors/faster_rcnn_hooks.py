from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


class FasterRCNNFPNFeatureExtractor(nn.Module):
    """Frozen Faster R-CNN-FPN feature extractor returning selected FPN maps.

    This extracts backbone FPN outputs without running RPN/ROI heads.
    Torchvision FPN keys are usually ['0','1','2','3','pool']; they are mapped
    to P2/P3/P4/P5/P6. We expose P3/P4/P5 by default.
    """

    KEY_MAP = {"0": "P2", "1": "P3", "2": "P4", "3": "P5", "pool": "P6"}

    def __init__(self, layers: Iterable[str] = ("P3", "P4", "P5")) -> None:
        super().__init__()
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.transform = model.transform
        self.backbone = model.backbone
        self.layers = tuple(layers)

    @torch.no_grad()
    def _transform_sizes(self, images: torch.Tensor):
        # Detection transform expects list tensors. For feature loss, input is already
        # normalized [0,1]. We use model.transform to match detector preprocessing.
        img_list = [img for img in images]
        image_list, _ = self.transform(img_list, None)
        return image_list.tensors

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        tensors = self._transform_sizes(images)
        feats = self.backbone(tensors)
        if not isinstance(feats, OrderedDict):
            raise RuntimeError("Expected OrderedDict FPN outputs from torchvision backbone")
        named = {self.KEY_MAP.get(k, k): v for k, v in feats.items()}
        return {layer: named[layer] for layer in self.layers if layer in named}
