from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from citation_models.common.records import PaperRecord
from citation_models.common.reproducibility import seed_everything


class SpecterEncoder:
    """Actual SPECTER encoder used by GCR-GAN (768-dimensional CLS output)."""

    def __init__(self, model_name: str = "allenai/specter", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.last_encode_report: dict[str, int] | None = None

    def provenance(self) -> dict[str, str | int | None]:
        """Return evidence that the requested released checkpoint was loaded."""
        return {
            "model_name": self.model.config.name_or_path,
            "model_class": type(self.model).__name__,
            "hidden_size": int(self.model.config.hidden_size),
            "parameter_count": int(
                sum(parameter.numel() for parameter in self.model.parameters())
            ),
            "revision": getattr(self.model.config, "_commit_hash", None),
        }

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 16,
        max_length: int = 512,
        deduplicate: bool = True,
    ):
        import numpy as np

        if not texts:
            raise ValueError("SPECTER cannot encode an empty text collection")
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        unique_texts: list[str]
        inverse: np.ndarray | None
        if deduplicate:
            text_to_index: dict[str, int] = {}
            unique_texts = []
            positions = []
            for text in texts:
                if text not in text_to_index:
                    text_to_index[text] = len(unique_texts)
                    unique_texts.append(text)
                positions.append(text_to_index[text])
            inverse = np.asarray(positions, dtype=np.int64)
        else:
            unique_texts = list(texts)
            inverse = None
        self.last_encode_report = {
            "requested_texts": len(texts),
            "unique_texts_encoded": len(unique_texts),
        }
        self.model.eval()
        batches = []
        with self.torch.no_grad():
            for start in range(0, len(unique_texts), batch_size):
                batch = self.tokenizer(
                    unique_texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                output = self.model(**batch)
                # SPECTER's released inference recipe uses the raw [CLS] state.
                pooled = output.last_hidden_state[:, 0]
                batches.append(pooled.cpu().numpy().astype(np.float32))
        matrix = np.concatenate(batches, axis=0)
        return matrix if inverse is None else matrix[inverse]


@dataclass(frozen=True)
class CitationTrainingExample:
    query_id: str
    positive_id: str
    negative_ids: tuple[str, ...]


def build_citation_examples(
    records: Iterable[PaperRecord],
    *,
    hard_negatives: int = 2,
    easy_negatives: int = 3,
    seed: int = 1203,
) -> list[CitationTrainingExample]:
    """Construct the 1 positive + 2 hard + 3 easy SPECTER samples in the paper.

    Hard negatives are citations-of-citations not directly cited by the query.
    Easy negatives are random corpus papers outside the query neighbourhood.
    """
    records = list(records)
    by_id = {record.id: record for record in records}
    all_ids = sorted(by_id)
    rng = random.Random(seed)
    examples: list[CitationTrainingExample] = []
    for query in sorted(records, key=lambda x: x.id):
        positives = sorted(set(query.references) & set(by_id) - {query.id})
        if not positives:
            continue
        positive = rng.choice(positives)
        direct = set(positives) | {query.id}
        hard_pool = sorted(
            {
                second_hop
                for cited in positives
                for second_hop in by_id[cited].references
                if second_hop in by_id and second_hop not in direct
            }
        )
        rng.shuffle(hard_pool)
        hard = hard_pool[:hard_negatives]
        hard_neighbourhood = set(hard_pool)
        easy_pool = [x for x in all_ids if x not in direct and x not in hard_neighbourhood]
        rng.shuffle(easy_pool)
        negatives = hard + easy_pool[: hard_negatives - len(hard) + easy_negatives]
        if len(negatives) == hard_negatives + easy_negatives:
            examples.append(CitationTrainingExample(query.id, positive, tuple(negatives)))
    return examples


def fine_tune_specter(
    records: Sequence[PaperRecord],
    output_dir: str,
    *,
    model_name: str = "allenai/specter",
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    margin: float = 1.0,
    seed: int = 1203,
    max_length: int = 512,
    device: str | None = None,
) -> None:
    """Fine-tune SPECTER with the paper's citation-informed triplet objective."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    if batch_size <= 0 or epochs <= 0 or learning_rate <= 0 or margin < 0 or max_length <= 0:
        raise ValueError("Invalid SPECTER training hyperparameters")
    seed_everything(seed)
    by_id = {record.id: record for record in records}
    examples = build_citation_examples(records, seed=seed)
    if not examples:
        raise ValueError("No valid citation training examples could be constructed")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_obj)

    class TripletDataset(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            example = examples[index]
            return (
                by_id[example.query_id].specter_text,
                by_id[example.positive_id].specter_text,
                [by_id[x].specter_text for x in example.negative_ids],
            )

    def collate(batch):
        queries, positives, negative_groups = zip(*batch)
        negatives = [item for group in negative_groups for item in group]
        texts = [*queries, *positives, *negatives]
        return tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ), len(queries), len(negative_groups[0])

    loader = DataLoader(TripletDataset(), batch_size=batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        for encoded, n_queries, n_negatives in loader:
            encoded = {key: value.to(device_obj) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = output.last_hidden_state[:, 0]
            queries = pooled[:n_queries]
            positives = pooled[n_queries : 2 * n_queries]
            negatives = pooled[2 * n_queries :].reshape(n_queries, n_negatives, -1)
            positive_distance = torch.linalg.vector_norm(queries - positives, dim=-1, keepdim=True)
            negative_distance = torch.linalg.vector_norm(queries[:, None, :] - negatives, dim=-1)
            loss = torch.relu(positive_distance - negative_distance + margin).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
