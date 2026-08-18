from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from citation_models.common.reproducibility import seed_everything

from .features import GCRFeatures
from .model import GCRGAN, GCRGANConfig, gradient_penalty


def train_gcr_gan(
    features: GCRFeatures,
    output_dir: str | Path,
    *,
    latent_dim: int = 150,
    noise_dim: int = 150,
    hidden_dim: int = 300,
    corruption_probability: float = 0.30,
    gradient_penalty_weight: float = 10.0,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    epochs: int = 100,
    discriminator_steps: int = 5,
    generator_steps: int = 1,
    checkpoint_every_epochs: int = 1,
    resume_from_checkpoint: bool | str | Path = False,
    max_dense_batch_mb: float = 2048.0,
    seed: int = 1203,
    device: str | None = None,
) -> list[dict[str, float]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 because the generator uses BatchNorm")
    if epochs <= 0 or discriminator_steps <= 0 or generator_steps <= 0:
        raise ValueError("epochs and discriminator/generator steps must be positive")
    if checkpoint_every_epochs <= 0:
        raise ValueError("checkpoint_every_epochs must be positive")
    if learning_rate <= 0 or gradient_penalty_weight < 0 or max_dense_batch_mb <= 0:
        raise ValueError("learning_rate must be positive and gradient penalty non-negative")
    dense_batch_mb = features.dense_batch_megabytes(batch_size)
    if dense_batch_mb > max_dense_batch_mb:
        raise MemoryError(
            f"One dense GCR-GAN input batch requires {dense_batch_mb:.1f} MiB, above "
            f"max_dense_batch_mb={max_dense_batch_mb:.1f}. Lower batch_size or raise the "
            "explicit guard after checking available accelerator memory."
        )
    seed_everything(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class FeatureDataset(Dataset):
        def __len__(self):
            return len(features.node_ids)

        def __getitem__(self, index):
            # Densification is deliberately deferred to the collator so one CSR
            # slice is converted per batch instead of one conversion per row.
            return int(index)

    def collate_feature_indices(indices):
        batch_indices = np.asarray(indices, dtype=np.int64)
        return torch.from_numpy(features.dense_rows(batch_indices))

    if len(features.node_ids) < batch_size:
        raise ValueError("batch_size exceeds the number of graph nodes; lower it for this dataset")
    config = GCRGANConfig(
        input_dim=features.input_dim,
        latent_dim=latent_dim,
        noise_dim=noise_dim,
        generator_hidden_dim=hidden_dim,
        corruption_probability=corruption_probability,
        gradient_penalty=gradient_penalty_weight,
    )
    model = GCRGAN(config).to(device_obj)
    d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate)
    g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=learning_rate)
    binary_cross_entropy = nn.BCELoss()
    history: list[dict[str, float]] = []
    start_epoch = 0

    if resume_from_checkpoint:
        resume_path = (
            output_dir / "checkpoint_last.pt"
            if resume_from_checkpoint is True
            else Path(resume_from_checkpoint)
        )
        if not resume_path.is_file():
            raise FileNotFoundError(f"GCR-GAN resume checkpoint does not exist: {resume_path}")
        payload = torch.load(resume_path, map_location=device_obj, weights_only=False)
        if payload.get("config") != model.paper_config():
            raise ValueError("Resume checkpoint architecture does not match the current features")
        if int(payload.get("seed", seed)) != seed:
            raise ValueError("Resume checkpoint seed does not match the configured seed")
        model.load_state_dict(payload["state_dict"])
        d_optimizer.load_state_dict(payload["d_optimizer"])
        g_optimizer.load_state_dict(payload["g_optimizer"])
        history = list(payload.get("history", []))
        start_epoch = int(payload["completed_epochs"])
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if device_obj.type == "cuda" and payload.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
        if start_epoch > epochs:
            raise ValueError("Resume checkpoint has completed more epochs than requested")

    def save_resume_checkpoint(completed_epochs: int) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": model.paper_config(),
                "seed": seed,
                "completed_epochs": completed_epochs,
                "history": history,
                "d_optimizer": d_optimizer.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all() if device_obj.type == "cuda" else None
                ),
            },
            output_dir / "checkpoint_last.pt",
        )

    for epoch in range(start_epoch, epochs):
        # A per-epoch sampler seed makes resumed and uninterrupted runs use the
        # same row order without replaying prior epochs.
        loader = DataLoader(
            FeatureDataset(),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            collate_fn=collate_feature_indices,
            num_workers=0,
            pin_memory=device_obj.type == "cuda",
        )
        d_total = g_total = gp_total = batches = 0.0
        for real_np in loader:
            real = real_np.to(device_obj, dtype=torch.float32, non_blocking=True)
            for _ in range(discriminator_steps):
                noise = torch.randn(real.shape[0], noise_dim, device=device_obj)
                with torch.no_grad():
                    fake = model.generator(noise)
                real_probability, _, _, _ = model.discriminator(real, corrupt=True)
                fake_probability, _, _, _ = model.discriminator(fake, corrupt=False)
                gp = gradient_penalty(model.discriminator, real, fake)
                d_loss = (
                    binary_cross_entropy(real_probability, torch.ones_like(real_probability))
                    + binary_cross_entropy(fake_probability, torch.zeros_like(fake_probability))
                    + gradient_penalty_weight * gp
                )
                d_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                d_optimizer.step()
            for _ in range(generator_steps):
                for parameter in model.discriminator.parameters():
                    parameter.requires_grad_(False)
                noise = torch.randn(real.shape[0], noise_dim, device=device_obj)
                fake = model.generator(noise)
                fake_probability, _, _, _ = model.discriminator(fake, corrupt=False)
                g_loss = binary_cross_entropy(fake_probability, torch.ones_like(fake_probability))
                g_optimizer.zero_grad(set_to_none=True)
                g_loss.backward()
                g_optimizer.step()
                for parameter in model.discriminator.parameters():
                    parameter.requires_grad_(True)
            d_total += float(d_loss.detach())
            g_total += float(g_loss.detach())
            gp_total += float(gp.detach())
            batches += 1
        row = {
            "epoch": epoch + 1,
            "discriminator_loss": d_total / batches,
            "generator_loss": g_total / batches,
            "gradient_penalty": gp_total / batches,
        }
        if not all(np.isfinite(x) for x in row.values()):
            raise FloatingPointError(f"Non-finite training statistics: {row}")
        history.append(row)
        if (epoch + 1) % checkpoint_every_epochs == 0 or epoch + 1 == epochs:
            save_resume_checkpoint(epoch + 1)

    torch.save(
        {"state_dict": model.state_dict(), "config": model.paper_config(), "seed": seed},
        output_dir / "gcr_gan.pt",
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    (output_dir / "training_report.json").write_text(
        json.dumps(
            {
                "device": str(device_obj),
                "nodes": len(features.node_ids),
                "input_dimensions": features.input_dim,
                "dense_input_batch_mib": dense_batch_mb,
                "model_parameter_count": parameter_count,
                "model_parameter_mib": parameter_bytes / (1024**2),
                "estimated_model_gradient_adam_mib": 4 * parameter_bytes / (1024**2),
                "estimate_excludes_activations": True,
                "completed_epochs": len(history),
                "resumed_from": str(resume_from_checkpoint or ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return history


def load_gcr_gan(checkpoint: str | Path, device: str | None = None):
    import torch

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(checkpoint, map_location=device_obj, weights_only=False)
    model = GCRGAN(GCRGANConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device_obj).eval()
    return model, device_obj
