from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

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
        name="ReconstructionModel",
        embed_dim=32,
        num_heads=2,
        ff_dim=64,
        num_layers=1,
        max_len=2000,
        dropout=0.1,
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
        hidden=64,
        dropout=0.1,
        out_dim=1,
    )
    model.set_head(recon_head)

    script_dir = Path(__file__).resolve().parent
    checkpoint_dir = script_dir / 'torch_checkpoints'
    checkpoint_path = checkpoint_dir / f'{model.name}.pt'

    if checkpoint_path.exists():
        print(f"Found checkpoint: {checkpoint_path}")
        model.load_model(path=checkpoint_path)
    else:
        print("No checkpoint found. Starting from scratch.")

    loss = FeatureWeightedMSE(
        feature_range=(0.2, 0.26),
        feature_weight=2.0,
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


def evaluate_and_plot(
    model: SpectrumModel,
    batch_size: int = 4,
    steps: int = 1,
    outdir: Path | str | None = None,
):
    model.eval()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(outdir) if outdir is not None else (script_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ReconstructionDataset(
        steps=steps,
        max_len=model.encoder.max_len,
        batch_size=batch_size,
    )
    batch = ds[0]
    x = batch["x"]  # (B, L, 2) -> [wave, flux]

    with torch.no_grad():
        y_pred = model.predict(x)  # (B, L, 1)

    x_np = x.numpy()
    y_np = y_pred.numpy()

    fig, ax = plt.subplots(1, 2, dpi=125, figsize=(12, 4.8))

    offset = 0.0
    low, high = 0.2, 0.26
    B = min(batch_size, x_np.shape[0])
    for i in range(B):
        valid = ~(np.all(x_np[i] == 0.0, axis=-1))

        wave = x_np[i][valid, 0]
        flux_in = x_np[i][valid, 1]
        flux_out = y_np[i, valid, 0]

        ax[0].plot(wave, np.log10(np.clip(flux_in, 1e-6, 1e4)) + offset, 'k-', label='Input' if i == 0 else None)
        ax[0].plot(wave, np.log10(np.clip(flux_out, 1e-6, 1e4)) + offset, 'r-', label='Reconstructed' if i == 0 else None)

        res = flux_in - flux_out
        ax[1].plot(wave, res + offset, 'k-')

        for j in range(2):
            ax[j].axvline(low, ls='--', color='k', alpha=0.25)
            ax[j].axvline(high, ls='--', color='k', alpha=0.25)
        ax[1].axhline(offset, ls='--', color='k', alpha=0.5)

        offset += 1.0

    for j in range(2):
        ax[j].set_xlim(0.0, 1.0)

    ax[0].set_xlabel('normalized wavelength')
    ax[0].set_ylabel('Log(normalized flux) + const.')
    handles, _ = ax[0].get_legend_handles_labels()
    if handles:
        ax[0].legend()

    ax[1].set_xlabel('normalized wavelength')
    ax[1].set_ylabel('residual flux')

    plt.tight_layout()
    out_path = out_dir / "pretrain_reconstruction_examples.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved reconstruction plots to {out_path}")


if __name__ == "__main__":
    model = pretrain(
        train_steps=16,
        val_steps=4,
        batch_size=16,
        epochs=32,
        lr=1e-4,
        save_each_epoch=True,
    )
    evaluate_and_plot(model, batch_size=4, steps=1)
