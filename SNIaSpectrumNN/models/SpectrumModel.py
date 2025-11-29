from pathlib import Path
import sys

import torch
from torch import Tensor, nn

from .ReconstructionBase import ReconstructionBase


class SpectrumModel(nn.Module):
    def __init__(
        self,
        name: str,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
        max_len: int = 2000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.name = name

        self.encoder = ReconstructionBase(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )

        self.head: nn.Module | None = None
        self.device = self.encoder.device

        self.checkpoint_dir = Path(sys.argv[0]).parent / 'torch_checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_state: dict[str, torch.Tensor] | None = None

    def set_head(self, head: nn.Module):
        self.head = head.to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.head is None:
            raise RuntimeError(
                'Head not set. Call set_head() or use a '
                'subclass defining a head.'
            )

        embeddings = self.encoder(inputs)
        return self.head(embeddings)

    def train_step(
        self,
        batch: dict[str, Tensor],
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> float:
        self.train()
        x = batch['x'].to(self.device)
        y = batch['y'].to(self.device)
        optimizer.zero_grad()
        pred = self.forward(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
        epochs: int = 10,
        lr: float = 2e-4,
        save_each_epoch: bool = True,
        loss: nn.Module | None = None,
        # ReduceLROnPlateau settings
        reduce_lr_on_plateau: bool = True,
        lr_factor: float = 0.5,
        lr_patience: int = 3,
        min_lr: float = 1e-6,
        # Early stopping settings
        es_patience: int = 5,
    ):
        if loss is None:
            raise ValueError(
                "fit() requires a 'loss' argument (nn.Module instance)."
            )

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
        epochs_no_improve: int = 0

        history: dict[str, list[float]] = {'loss': [], 'val_loss': []}

        for epoch in range(1, epochs + 1):
            total = 0.0
            for batch in train_loader:
                total += self.train_step(batch, optimizer, loss)
            train_loss = total / len(train_loader)

            log = f'Epoch {epoch}/{epochs} - train_loss={train_loss:.4f}'
            history['loss'].append(train_loss)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader, loss)
                log += f' val_loss={val_loss:.4f}'
                history['val_loss'].append(val_loss)

                if scheduler is not None:
                    scheduler.step(val_loss)

                if val_loss < best_val - 1e-8:
                    best_val = val_loss
                    epochs_no_improve = 0
                    self._save_as_best()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= es_patience:
                        print(
                            f'Early stopping at epoch '
                            f'{epoch} (no improvement for '
                            f'{es_patience} epochs)'
                        )
                        break
            print(log)
            if save_each_epoch:
                self.save_model()

        if self.best_state is not None:
            self.load_state_dict(self.best_state)

        return history

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
    ) -> float:
        self.eval()
        total = 0.0
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                pred = self.forward(x)
                total += float(loss_fn(pred, y).item())
        return total / len(dataloader)

    def predict(self, x: Tensor, batch_size: int = 32) -> Tensor:
        self.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, x.size(0), batch_size):
                chunk = x[i : i + batch_size].to(self.device)
                preds.append(self.forward(chunk).cpu())
        return torch.cat(preds, dim=0)

    def save_model(self) -> Path:
        path = self.checkpoint_dir / f'{self.name}.pt'
        torch.save(self.state_dict(), path)
        return path

    def load_model(
        self,
        path: Path | str | None = None,
        encoder_only: bool = False,
    ) -> bool:
        if path is None:
            path = self.checkpoint_dir / f'{self.name}.pt'
        else:
            path = Path(path)

        if not path.exists():
            return False

        checkpoint = torch.load(path, map_location=self.device)

        if encoder_only:
            encoder_state = {
                k.replace('encoder.', ''): v
                for k, v in checkpoint.items()
                if k.startswith('encoder.')
            }
            self.encoder.load_state_dict(encoder_state)
            print(f'Loaded encoder from {path}')
        else:
            self.load_state_dict(checkpoint)
            print(f'Loaded model from {path}')

        return True

    def _save_as_best(self):
        self.best_state = {
            k: v.detach().clone() for k, v in self.state_dict().items()
        }
