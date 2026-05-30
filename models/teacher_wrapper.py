from __future__ import annotations

import torch


class OfflineTeacherWrapper:
    """Interface for an offline diffusion restoration teacher.

    This is a placeholder adapter. In a real experiment, implement `enhance`
    using DiffBIR/IRControlNet-style restoration code and keep it offline.
    """

    def __init__(self, model=None, device: str = "cuda") -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def enhance(self, x_c: torch.Tensor, raw_stats: torch.Tensor | None = None) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError(
                "No teacher model attached. Plug in DiffBIR/IRControlNet implementation here."
            )
        return self.model(x_c.to(self.device), raw_stats)

    @torch.no_grad()
    def residual(self, x_c: torch.Tensor, raw_stats: torch.Tensor | None = None) -> torch.Tensor:
        x_d = self.enhance(x_c, raw_stats)
        return x_d - x_c.to(x_d.device)
