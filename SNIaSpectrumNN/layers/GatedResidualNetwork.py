"""
Gated Residual Network (GRN) layer.

The GRN is a "smart" version of a refinement block, which is a lightweight
layer that adds depth to the model without needing to recompute heavy attention
(also saving memory). The GRN does the same thing, but with the added "trick"
that gates new information, and in doing so learns how much of the new
transformation to apply at each time step, which improves stability.
"""

from torch import Tensor, nn


class GatedResidualNetwork(nn.Module):
    """Gated residual block for sequence feature transformation.

    Parameters
    - embed_dim: Size of the input/output feature dimension.
    - ff_dim: Hidden dimension of the intermediate feed-forward projection.
    - dropout: Dropout rate applied to the transformed features during
      training.
    """

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
        """Forward pass.

        Parameters
        - x: Input tensor of shape `(batch, seq_len, embed_dim)`.
        - padding_mask: Optional mask indicating padded time steps. Not used in
          this layer but accepted for interface consistency with adjacent
          blocks.
        - training: If True, applies dropout to the transformed features.

        Returns
        - Tensor with the same shape as `x`, computed as
          `x + gate * ffn(x_norm)`.
        """

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
