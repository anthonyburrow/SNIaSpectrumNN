from torch import Tensor, nn

from .SpectrumModelHead import SpectrumModelHead


class ReconstructionHead(SpectrumModelHead):
    def __init__(
        self,
        embed_dim: int,
        hidden: int = 128,
        dropout: float = 0.1,
        out_dim: int = 1,
    ):
        super().__init__(embed_dim=embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
