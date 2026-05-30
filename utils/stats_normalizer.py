from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StatsNormalizerConfig:
    qp_min: float = 22.0
    qp_max: float = 47.0
    bitrate_mode: str = "log_divisor"  # "log_divisor" or "zscore"
    bitrate_log_divisor: float = 15.0
    bitrate_log_mean: float = 0.0
    bitrate_log_std: float = 1.0
    entropy_max: float = 8.0
    clamp_min: float = -3.0
    clamp_max: float = 3.0


class StatsNormalizer:
    """Normalize raw global statistics for AlphaMLP.

    Expected raw order:
    [QP, bitrate, mean_rgb, std_rgb, edge_density, entropy, objectness_ratio]
    """

    def __init__(self, cfg: StatsNormalizerConfig | None = None) -> None:
        self.cfg = cfg or StatsNormalizerConfig()

    def __call__(self, raw_stats: torch.Tensor) -> torch.Tensor:
        if raw_stats.ndim != 2 or raw_stats.shape[1] < 7:
            raise ValueError(
                "raw_stats must be [N, >=7] with order "
                "[QP, bitrate, mean_rgb, std_rgb, edge_density, entropy, objectness_ratio]"
            )
        s = raw_stats.clone().to(dtype=torch.float32)
        cfg = self.cfg
        s[:, 0] = (s[:, 0] - cfg.qp_min) / (cfg.qp_max - cfg.qp_min + 1e-8)

        log_bitrate = torch.log1p(torch.clamp(s[:, 1], min=0.0))
        if cfg.bitrate_mode == "log_divisor":
            s[:, 1] = log_bitrate / cfg.bitrate_log_divisor
        elif cfg.bitrate_mode == "zscore":
            s[:, 1] = (log_bitrate - cfg.bitrate_log_mean) / (cfg.bitrate_log_std + 1e-8)
        else:
            raise ValueError(f"Unsupported bitrate_mode: {cfg.bitrate_mode}")

        # mean_rgb, std_rgb, edge_density are expected to be roughly [0,1]
        s[:, 5] = s[:, 5] / cfg.entropy_max
        return torch.clamp(s, cfg.clamp_min, cfg.clamp_max)
