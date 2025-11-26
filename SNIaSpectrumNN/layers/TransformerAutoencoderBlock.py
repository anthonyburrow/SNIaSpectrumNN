from torch import nn, Tensor
from typing import Optional


class TransformerAutoencoderBlock(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout: float = 0.1
        ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
            self,
            x: Tensor,
            padding_mask: Optional[Tensor] = None
        ) -> Tensor:
        # Pre-norm: normalize BEFORE attention (more stable, faster training)
        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=padding_mask
        )
        attn_out = self.drop(attn_out)
        x = x + attn_out  # Residual connection

        # Pre-norm: normalize BEFORE FFN
        ffn_out = self.ff(self.norm2(x))
        ffn_out = self.drop(ffn_out)
        x = x + ffn_out  # Residual connection
        
        return x