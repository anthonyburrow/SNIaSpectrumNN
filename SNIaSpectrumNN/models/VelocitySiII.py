from torch import nn, Tensor

from .SpectrumModel import SpectrumModel
from .SpectrumModelHead import SpectrumModelHead


class ScalarHead(SpectrumModelHead):
    """Head for predicting scalar values from embeddings.
    
    This head pools the sequence embeddings and outputs a single value
    (or a small number of values) per input.
    """
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
        # Pool over sequence dimension (mean pooling)
        x = x.mean(dim=1)
        return self.net(x)


class VelocitySiII(SpectrumModel):

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
        max_len: int = 2000,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
            **kwargs,
        )
        self.set_head(
            ScalarHead(
                embed_dim,
                hidden=2 * embed_dim,
                dropout=dropout,
            )
        )
