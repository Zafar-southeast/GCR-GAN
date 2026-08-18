# GCR-GAN

Paper-grounded PyTorch implementation of **GCR-GAN**, a global citation recommendation model combining SPECTER document embeddings, a heterogeneous bibliographic network, and a GAN with a denoising-autoencoder discriminator.

**Paper:** Z. Ali et al., “Global Citation Recommendation employing Generative Adversarial Network,” *Expert Systems with Applications*, 2021.
https://doi.org/10.1016/j.eswa.2021.114888

## Model

The pipeline:

1. Builds a paper–author–topic heterogeneous network.
2. encodes paper titles and abstracts using the released `allenai/specter` model;
3. concatenates SPECTER embeddings with sparse adjacency rows;
4. trains the paper-described generator and denoising-autoencoder discriminator; and
5. ranks candidate papers using content, author, and topic cosine similarities.

The implementation supports personalized and text-only non-personalized recommendation. It also exposes `semantic` and `literal_equation_8` scoring modes because Equation 8 and its accompanying explanation are inconsistent.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

A CUDA-capable GPU is recommended for full experiments.

## Data

The paper evaluates GCR-GAN on filtered versions of AMiner DBLP-V12 and ACM Citation Network V9. These datasets are not redistributed.

DBLP-V12:

https://originalstatic.aminer.cn/misc/dblp.v12.7z

```bash
python scripts/download_aminer.py dblp_v12
```

After extraction, update `dataset.path` in:

```text
configs/gcr_gan/dblp_v12.yaml
configs/gcr_gan/acm_v9.yaml
```

The loader supports JSON, JSONL, and legacy AMiner citation-network files.

## Running GCR-GAN

DBLP-V12:

```bash
bash scripts/run_gcr_gan.sh configs/gcr_gan/dblp_v12.yaml cuda
```

ACM V9:

```bash
bash scripts/run_gcr_gan.sh configs/gcr_gan/acm_v9.yaml cuda
```

Individual stages can also be executed:

```bash
citation-models validate-data --config configs/gcr_gan/dblp_v12.yaml

citation-models prepare-gcr \
  --config configs/gcr_gan/dblp_v12.yaml \
  --device cuda

citation-models train-gcr \
  --config configs/gcr_gan/dblp_v12.yaml \
  --device cuda

citation-models evaluate-gcr \
  --config configs/gcr_gan/dblp_v12.yaml \
  --device cuda
```

For non-personalized evaluation:

```bash
citation-models evaluate-gcr \
  --config configs/gcr_gan/dblp_v12.yaml \
  --device cuda \
  --non-personalized
```

## Evaluation

The implementation uses a persisted paper-level split. Training papers form the HBN and candidate collection, while held-out papers are used as queries. Query citations into the training collection define relevance and are not exposed during graph construction or ranking.

Evaluation reports configurable MAP, nDCG, Recall, and cold-start metrics.

## Reproducibility boundary

Exact reproduction of the paper’s values requires the original filtered records, split IDs, graph-processing choices, and omitted training settings. Unspecified or ambiguous choices are exposed in the YAML configurations and documented rather than presented as verified paper settings.

## Citation

```bibtex
@article{ali2021gcrgan,
  author  = {Zafar Ali and Guilin Qi and Muhammad Khan and
             Pavlos Kefalas and Shah Khusro},
  title   = {Global Citation Recommendation Employing
             Generative Adversarial Network},
  journal = {Expert Systems with Applications},
  volume  = {180},
  pages   = {114888},
  year    = {2021},
  doi     = {10.1016/j.eswa.2021.114888}
}
```

## License

Released under the MIT License. Datasets and pretrained models remain subject to their respective licenses.
