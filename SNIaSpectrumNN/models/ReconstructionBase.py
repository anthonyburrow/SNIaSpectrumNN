from pathlib import Path
import sys

import torch
from torch import nn, Tensor

from ..layers.TransformerAutoencoderBlock import TransformerAutoencoderBlock
from ..layers.GatedResidualNetwork import GatedResidualNetwork


class ReconstructionBase(nn.Module):

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
        max_len: int = 2000,
        dropout: float = 0.1,
        feature_range: tuple[float, float] = (0.2, 0.26),
        feature_weight: float = 2.0,
        weights_file: Path | str | None = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_len = max_len
        self.feature_range = feature_range
        self.feature_weight = feature_weight

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Print device info
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using GPU: {gpu_name}")
        else:
            print("Using CPU (CUDA not available)")

        self.checkpoint_dir = Path(sys.argv[0]).parent / 'torch_checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.input_proj = nn.Linear(2, embed_dim)

        layers = []
        for _ in range(num_layers):
            layers.append(
                TransformerAutoencoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
            )
            layers.append(
                GatedResidualNetwork(
                    embed_dim=embed_dim,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
            )
        self.encoder_layers = nn.ModuleList(layers)

        self.to(self.device)

        # Load weights if provided
        if weights_file is not None:
            weights_path = Path(weights_file)
            if weights_path.exists():
                print(f"Loading weights from {weights_path}...")
                state = torch.load(weights_path, map_location=self.device)
                self.load_state_dict(state)
                print("Weights loaded successfully.")
            else:
                print(f"Warning: weights_file '{weights_path}' not found. Starting with random initialization.")

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.input_proj(inputs)

        mask = torch.any(inputs != 0.0, dim=-1)
        padding_mask = ~mask

        for layer in self.encoder_layers:
            x = layer(x, padding_mask)
        
        return x

    def save_weights(self) -> Path:
        path = self.checkpoint_dir / f"{self.__class__.__name__}_weights.pt"
        torch.save(self.state_dict(), path)
        return path

    def load_weights(self) -> bool:
        path = self.checkpoint_dir / f"{self.__class__.__name__}_weights.pt"
        if path.exists():
            state = torch.load(path, map_location=self.device)
            self.load_state_dict(state)
            return True
        return False
