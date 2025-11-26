import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SNIaSpectrumNN.models.SpectrumModel import SpectrumModel
from SNIaSpectrumNN.models.ReconstructionHead import ReconstructionHead
from SNIaSpectrumNN.data.ReconstructionDataset import ReconstructionDataset
from SNIaSpectrumNN.util.losses import FeatureWeightedMSE


def pretrain(
    train_steps: int = 100,
    val_steps: int = 20,
    batch_size: int = 8,
    epochs: int = 5,
    lr: float = 1e-4,
    save_each_epoch: bool = True,
) -> SpectrumModel:
    model = SpectrumModel(
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        num_layers=2,
        max_len=2000,
        dropout=0.1,
        feature_range=(0.2, 0.26),
        feature_weight=2.0,
    )

    train_ds = ReconstructionDataset(
        steps=train_steps,
        max_len=model.encoder.max_len,
        batch_size=batch_size,
    )
    val_ds = ReconstructionDataset(
        steps=val_steps,
        max_len=model.encoder.max_len,
        batch_size=batch_size,
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=None)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=None)

    recon_head = ReconstructionHead(
        embed_dim=model.encoder.embed_dim,
        hidden=128,
        dropout=0.1,
        out_dim=1,
    )
    model.set_head(recon_head)

    loss = FeatureWeightedMSE(
        feature_range=model.encoder.feature_range,
        feature_weight=model.encoder.feature_weight,
    )

    print(f"Starting pretraining for {epochs} epochs...")
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        save_each_epoch=save_each_epoch,
        loss=loss,
    )
    print("Pretraining complete!")

    return model


if __name__ == "__main__":
    pretrain(
        train_steps=100,
        val_steps=20,
        batch_size=8,
        epochs=5,
        lr=1e-4,
        save_each_epoch=True,
    )
