from __future__ import annotations

import torch

from citation_models import __version__
from citation_models.common.records import NamedEntity, PaperRecord
from citation_models.gcr_gan.hbn import build_hbn
from citation_models.gcr_gan.model import GCRGAN, GCRGANConfig


def test_version_and_model_core() -> None:
    records = [
        PaperRecord(id="P1", title="Graph models", authors=[NamedEntity("A1", "Ada")]),
        PaperRecord(
            id="P2",
            title="Citation models",
            authors=[NamedEntity("A2", "Ben")],
            references=["P1"],
            keywords=["citation"],
        ),
    ]
    network = build_hbn(records)
    model = GCRGAN(
        GCRGANConfig(input_dim=12, latent_dim=4, noise_dim=4, generator_hidden_dim=8)
    )
    generated = model.generator(torch.randn(2, 4))

    assert __version__ == "1.3.0"
    assert network.adjacency.nnz > 0
    assert generated.shape == (2, 12)
    assert torch.isfinite(generated).all()
