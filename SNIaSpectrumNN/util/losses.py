import torch
from torch import Tensor, nn


class FeatureWeightedMSE(nn.Module):
    def __init__(
        self,
        feature_range: tuple[float, float] = (0.2, 0.26),
        feature_weight: float = 2.0,
    ):
        super().__init__()

        self.feature_range = feature_range
        self.feature_weight = feature_weight

        self.feature_center = 0.5 * (feature_range[1] + feature_range[0])
        self.feature_width = 0.5 * (feature_range[1] - feature_range[0])

        self.log_base = torch.tensor(10.0).log().item()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        flux_pred = y_pred.squeeze(-1)
        flux_true = y_true.squeeze(-1)

        flux_pred = torch.clamp(flux_pred, min=1e-6, max=1e4)
        flux_true = torch.clamp(flux_true, min=1e-6, max=1e4)

        log_pred = torch.log(flux_pred) / self.log_base
        log_true = torch.log(flux_true) / self.log_base
        error = (log_pred - log_true) ** 2

        seq_len = y_true.size(1)
        wave = torch.linspace(0.0, 1.0, seq_len, device=y_true.device)
        wave = wave.unsqueeze(0)  # (1, seq_len)

        wave_centered = wave - self.feature_center
        gaussian_arg = (wave_centered / self.feature_width) ** 2
        gaussian = torch.exp(-0.5 * gaussian_arg)

        weight = 1.0 + (self.feature_weight - 1.0) * gaussian

        weighted_error = error * weight
        return weighted_error.mean()
