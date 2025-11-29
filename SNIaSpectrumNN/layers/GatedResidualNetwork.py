from torch import Tensor, nn


class GatedResidualNetwork(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dense1 = nn.Linear(embed_dim, ff_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ff_dim, embed_dim)
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, padding_mask=None, training: bool = False
    ) -> Tensor:
        x_norm = self.norm(x)

        # Feedforward transformation
        ffn_out = self.dense1(x_norm)
        ffn_out = self.relu(ffn_out)
        ffn_out = self.dense2(ffn_out)
        ffn_out = self.dropout(ffn_out) if training else ffn_out

        # Gate controls how much of the transformation to apply
        gate = self.gate(x_norm)
        gate = self.sigmoid(gate)

        # Residual connection with gated transformation
        return x + gate * ffn_out
