from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GCRGANConfig:
    input_dim: int
    latent_dim: int = 150
    noise_dim: int = 150
    generator_hidden_dim: int = 300
    corruption_probability: float = 0.30
    gradient_penalty: float = 10.0

    def __post_init__(self) -> None:
        for name in ("input_dim", "latent_dim", "noise_dim", "generator_hidden_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.corruption_probability < 1.0:
            raise ValueError("corruption_probability must be in [0, 1)")
        if self.gradient_penalty < 0.0:
            raise ValueError("gradient_penalty must be non-negative")


class Generator(nn.Module):
    """Three-layer feed-forward generator from Section 5.5."""

    def __init__(self, noise_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class DenoisingEnergyDiscriminator(nn.Module):
    """Single-layer DAE discriminator; its hidden state is the node embedding.

    The paper defines a DAE reconstruction error and also uses D(x) as a GAN
    probability without writing the connecting operation. Following the cited
    DAE/energy discriminator lineage, D(x) is sigmoid(bias - reconstruction
    energy). This is documented as a reconstruction in PAPER_TO_CODE.md.
    """

    def __init__(self, input_dim: int, latent_dim: int, corruption_probability: float):
        super().__init__()
        self.corruption_probability = corruption_probability
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.energy_bias = nn.Parameter(torch.tensor(0.0))
        self.activation = nn.LeakyReLU(negative_slope=0.02)

    def corrupt(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.corruption_probability <= 0:
            return x
        keep = torch.rand_like(x) >= self.corruption_probability
        return x * keep

    def encode(self, x: torch.Tensor, *, corrupt: bool = False) -> torch.Tensor:
        if corrupt:
            x = self.corrupt(x)
        return self.activation(self.encoder(x))

    def reconstruct(
        self, x: torch.Tensor, *, corrupt: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(x, corrupt=corrupt)
        # Section 5.5 describes a linear decoder and reports tanh activation.
        reconstructed = torch.tanh(self.decoder(hidden))
        return hidden, reconstructed

    def forward(self, x: torch.Tensor, *, corrupt: bool = True):
        hidden, reconstructed = self.reconstruct(x, corrupt=corrupt)
        energy = torch.mean((x - reconstructed) ** 2, dim=1)
        probability = torch.sigmoid(self.energy_bias - energy)
        return probability, energy, hidden, reconstructed


class GCRGAN(nn.Module):
    def __init__(self, config: GCRGANConfig):
        super().__init__()
        self.config = config
        self.generator = Generator(
            config.noise_dim, config.generator_hidden_dim, config.input_dim
        )
        self.discriminator = DenoisingEnergyDiscriminator(
            config.input_dim, config.latent_dim, config.corruption_probability
        )

    def paper_config(self) -> dict:
        return asdict(self.config)


def gcr_gan_parameter_count(config: GCRGANConfig) -> int:
    """Exact trainable parameter count without allocating the width-dependent model."""
    width = config.input_dim
    hidden = config.generator_hidden_dim
    latent = config.latent_dim
    noise = config.noise_dim
    generator = (
        noise * hidden
        + hidden
        + 2 * hidden
        + hidden * hidden
        + hidden
        + 2 * hidden
        + hidden * width
        + width
    )
    discriminator = width * latent + latent + latent * width + width + 1
    return generator + discriminator


def gradient_penalty(
    discriminator: DenoisingEnergyDiscriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    epsilon = torch.rand(real.shape[0], 1, device=real.device)
    interpolated = epsilon * real + (1.0 - epsilon) * fake
    interpolated.requires_grad_(True)
    probability, _, _, _ = discriminator(interpolated, corrupt=False)
    gradients = torch.autograd.grad(
        probability,
        interpolated,
        grad_outputs=torch.ones_like(probability),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
