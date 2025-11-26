from typing import Optional, Dict

import torch
from torch import nn, Tensor

from .ReconstructionBase import ReconstructionBase


class SpectrumModel(nn.Module):
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
    ):
        super().__init__()

        self.encoder = ReconstructionBase(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
            feature_range=feature_range,
            feature_weight=feature_weight,
        )

        self.head: Optional[nn.Module] = None
        self.device = self.encoder.device

    def set_head(self, head: nn.Module):
        self.head = head.to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.head is None:
            raise RuntimeError("Head not set. Call set_head() or use a subclass defining a head.")

        embeddings = self.encoder(inputs)
        return self.head(embeddings)

    def train_step(
        self,
        batch: Dict[str, Tensor],
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> float:
        self.train()
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        optimizer.zero_grad()
        pred = self.forward(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        lr: float = 2e-4,
        save_each_epoch: bool = True,
        loss: Optional[nn.Module] = None,
    ):
        if loss is None:
            raise ValueError("fit() requires a 'loss' argument (nn.Module instance).")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            total = 0.0
            for batch in train_loader:
                total += self.train_step(batch, optimizer, loss)
            train_loss = total / len(train_loader)
            log = f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f}"
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, loss)
                log += f" val_loss={val_loss:.4f}"
            print(log)
            if save_each_epoch:
                self.save_weights()

    def evaluate(self, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module) -> float:
        self.eval()
        total = 0.0
        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                pred = self.forward(x)
                total += float(loss_fn(pred, y).item())
        return total / len(dataloader)

    def predict(self, x: Tensor, batch_size: int = 32) -> Tensor:
        self.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, x.size(0), batch_size):
                chunk = x[i:i + batch_size].to(self.device)
                preds.append(self.forward(chunk).cpu())
        return torch.cat(preds, dim=0)

    def save_weights(self):
        return self.encoder.save_weights()

    def load_weights(self):
        return self.encoder.load_weights()
