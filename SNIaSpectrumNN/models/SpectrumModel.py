from typing import Optional, Dict, List
from pathlib import Path

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
        encoder_weights_file: Path | str | None = None,
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
            weights_file=encoder_weights_file,
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
        # ReduceLROnPlateau settings
        reduce_lr_on_plateau: bool = True,
        lr_factor: float = 0.5,
        lr_patience: int = 3,
        min_lr: float = 1e-6,
        # Early stopping settings
        early_stopping: bool = True,
        es_patience: int = 5,
        restore_best_weights: bool = True,
    ):
        if loss is None:
            raise ValueError("fit() requires a 'loss' argument (nn.Module instance).")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = None
        if reduce_lr_on_plateau and val_loader is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=lr_factor,
                patience=lr_patience,
                min_lr=min_lr,
            )

        best_val: float = float('inf')
        best_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_no_improve: int = 0

        history: Dict[str, List[float]] = {"loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            total = 0.0
            for batch in train_loader:
                total += self.train_step(batch, optimizer, loss)
            train_loss = total / len(train_loader)
            log = f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f}"
            history["loss"].append(train_loss)
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, loss)
                log += f" val_loss={val_loss:.4f}"
                history["val_loss"].append(val_loss)
                if scheduler is not None:
                    scheduler.step(val_loss)

                # Early stopping tracking
                if early_stopping:
                    if val_loss < best_val - 1e-8:
                        best_val = val_loss
                        epochs_no_improve = 0
                        if restore_best_weights:
                            # Keep a deep copy of weights
                            best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= es_patience:
                            print(f"Early stopping at epoch {epoch} (no improvement for {es_patience} epochs)")
                            break
            print(log)
            if save_each_epoch:
                self.save_weights()

        # Restore the best weights if requested
        if early_stopping and restore_best_weights and best_state is not None:
            self.load_state_dict(best_state)

        return history

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
