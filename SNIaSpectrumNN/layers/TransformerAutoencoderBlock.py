"""
Transformer autoencoder block.

The Transformer-encoder block is made up of an attention layer and FFN layer
with some key additions:
- Before the attention/FFN layers (rather than after), the input is normalized,
  which generally makes Transformers more stable and faster to train.
- Residual connection is performed, which adds the input embedding back to the
  attention/FFN output, which allows the model to "decide" how much of the
  original info to keep vs. transform. In general, this helps gradients flow
  and stabilizes training.
- Dropout (ignoring random neurons during each training iteration) is performed
  to mitigate overfitting.
"""

from torch import Tensor, nn


class TransformerAutoencoderBlock(nn.Module):
    """Transformer encoder-style block with pre-norm.

    Parameters
    - embed_dim: Feature dimension of inputs/outputs.
    - num_heads: Number of attention heads.
    - ff_dim: Hidden size of the feed-forward network.
    - dropout: Dropout rate applied after attention and FFN.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Apply self-attention and feed-forward with pre-norm and residuals.

        Parameters
        - x: `(batch, seq_len, embed_dim)` input features.
        - padding_mask: Optional `(batch, seq_len)` boolean mask for padded
          steps.

        Returns
        - Transformed features with the same shape as `x`.
        """

        # Pre-norm: normalize BEFORE attention
        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            key_padding_mask=padding_mask,
        )
        attn_out = self.drop(attn_out)
        x = x + attn_out  # Residual connection

        # Pre-norm: normalize BEFORE FFN
        ffn_out = self.ff(self.norm2(x))
        ffn_out = self.drop(ffn_out)
        x = x + ffn_out  # Residual connection

        return x
