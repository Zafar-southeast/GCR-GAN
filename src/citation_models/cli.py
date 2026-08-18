from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from citation_models.common.config import load_config
from citation_models.common.io import load_id_list, load_papers, save_id_list
from citation_models.common.manifest import write_run_manifest
from citation_models.common.metrics import evaluate_rankings
from citation_models.common.ranking import required_ranking_depth
from citation_models.common.split import evaluable_truth, random_paper_split
from citation_models.common.validation import validate_records, validate_split


def _records_and_split(config: dict[str, Any]):
    records = load_papers(config["dataset"]["path"], config["dataset"].get("format", "auto"))
    report = validate_records(records)
    report.raise_for_errors()
    split = random_paper_split(
        records,
        train_ratio=float(config["dataset"].get("train_ratio", 0.8)),
        seed=int(config.get("seed", 1203)),
    )
    validate_split(split.train, split.test, {record.id for record in records})
    return records, split, report


def _records_and_saved_split(config: dict[str, Any], output: Path):
    records = load_papers(config["dataset"]["path"], config["dataset"].get("format", "auto"))
    report = validate_records(records)
    report.raise_for_errors()
    train = tuple(load_id_list(output / "train_ids.txt"))
    test = tuple(load_id_list(output / "test_ids.txt"))
    validate_split(train, test, {record.id for record in records})
    from citation_models.common.split import PaperSplit

    return records, PaperSplit(train, test), report


def command_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, split, report = _records_and_split(config)
    payload = {
        "ok": report.ok,
        "errors": report.errors,
        "warnings": report.warnings,
        "statistics": report.statistics,
        "train_papers": len(split.train),
        "test_papers": len(split.test),
    }
    print(json.dumps(payload, indent=2))


def command_prepare_gcr(args: argparse.Namespace) -> None:
    from citation_models.gcr_gan.features import create_features
    from citation_models.gcr_gan.hbn import build_hbn
    from citation_models.gcr_gan.model import GCRGANConfig, gcr_gan_parameter_count
    from citation_models.gcr_gan.specter import SpecterEncoder, fine_tune_specter

    config = load_config(args.config)
    records, split, _ = _records_and_split(config)
    by_id = {record.id: record for record in records}
    train_records = [by_id[x] for x in split.train]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    save_id_list(split.train, output / "train_ids.txt")
    save_id_list(split.test, output / "test_ids.txt")
    write_run_manifest(
        output / "run_manifest.json",
        model="GCR-GAN",
        dataset_path=config["dataset"]["path"],
        config=config,
        train_ids=split.train,
        test_ids=split.test,
    )

    specter = config["specter"]
    model_name = specter.get("model_name", "allenai/specter")
    if specter.get("fine_tune", False):
        fine_tuned_path = output / "specter_finetuned"
        fine_tune_specter(
            train_records,
            str(fine_tuned_path),
            model_name=model_name,
            epochs=int(specter.get("epochs", 1)),
            batch_size=int(specter.get("batch_size", 16)),
            learning_rate=float(specter.get("learning_rate", 2e-5)),
            margin=float(specter.get("margin", 1.0)),
            seed=int(config.get("seed", 1203)),
            max_length=int(specter.get("max_length", 512)),
            device=args.device,
        )
        model_name = str(fine_tuned_path)

    network = build_hbn(
        train_records,
        undirected=bool(config.get("network", {}).get("undirected", True)),
        include_coauthor_edges=bool(
            config.get("network", {}).get("include_coauthor_edges", True)
        ),
    )
    encoder = SpecterEncoder(model_name=model_name, device=args.device)
    embeddings = encoder.encode(
        network.texts,
        batch_size=int(specter.get("batch_size", 16)),
        max_length=int(specter.get("max_length", 512)),
        deduplicate=bool(specter.get("deduplicate_texts", True)),
    )
    expected_dimension = specter.get("expected_dimension")
    if expected_dimension is not None and embeddings.shape[1] != int(expected_dimension):
        raise ValueError(
            f"SPECTER produced {embeddings.shape[1]} dimensions; "
            f"expected {int(expected_dimension)}"
        )
    features = create_features(network, embeddings)
    features.save(output / "features")
    configured_gan_batch = int(config["gcr_gan"].get("batch_size", 64))
    gan_settings = config["gcr_gan"]
    gan_config = GCRGANConfig(
        input_dim=features.input_dim,
        latent_dim=int(gan_settings.get("latent_dim", 150)),
        noise_dim=int(gan_settings.get("noise_dim", 150)),
        generator_hidden_dim=int(gan_settings.get("generator_hidden_dim", 300)),
        corruption_probability=float(gan_settings.get("corruption_probability", 0.30)),
        gradient_penalty=float(gan_settings.get("gradient_penalty", 10.0)),
    )
    estimated_gan_parameters = gcr_gan_parameter_count(gan_config)
    (output / "feature_report.json").write_text(
        json.dumps(
            {
                "nodes": len(network.node_ids),
                "node_types": dict(Counter(network.node_types)),
                "content_dimensions": int(features.content.shape[1]),
                "adjacency_dimensions": int(features.adjacency.shape[1]),
                "adjacency_nonzero": int(features.adjacency.nnz),
                "combined_input_dimensions": int(features.input_dim),
                "content_storage_mib": features.content.nbytes / (1024**2),
                "sparse_adjacency_storage_mib": (
                    features.adjacency.data.nbytes
                    + features.adjacency.indices.nbytes
                    + features.adjacency.indptr.nbytes
                )
                / (1024**2),
                "configured_dense_batch_mib": features.dense_batch_megabytes(
                    configured_gan_batch
                ),
                "estimated_gan_parameter_count": estimated_gan_parameters,
                "estimated_fp32_model_mib": estimated_gan_parameters * 4 / (1024**2),
                "estimated_model_gradient_adam_mib": (
                    estimated_gan_parameters * 4 * 4 / (1024**2)
                ),
                "estimate_excludes_activations": True,
                "full_dense_feature_matrix_not_materialized": True,
                "finite": bool(features.all_finite()),
                "content_encoder": encoder.provenance(),
                "content_encoding": encoder.last_encode_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "resolved.json").write_text(
        json.dumps({"specter_model": model_name, "config": config}, indent=2), encoding="utf-8"
    )


def command_train_gcr(args: argparse.Namespace) -> None:
    from citation_models.gcr_gan.features import GCRFeatures
    from citation_models.gcr_gan.train import train_gcr_gan

    config = load_config(args.config)
    output = Path(config["output_dir"])
    settings = config["gcr_gan"]
    train_gcr_gan(
        GCRFeatures.load(output / "features"),
        output / "checkpoint",
        latent_dim=int(settings.get("latent_dim", 150)),
        noise_dim=int(settings.get("noise_dim", 150)),
        hidden_dim=int(settings.get("generator_hidden_dim", 300)),
        corruption_probability=float(settings.get("corruption_probability", 0.30)),
        gradient_penalty_weight=float(settings.get("gradient_penalty", 10.0)),
        learning_rate=float(settings.get("learning_rate", 0.001)),
        batch_size=int(settings.get("batch_size", 64)),
        epochs=int(settings.get("epochs", 100)),
        discriminator_steps=int(settings.get("discriminator_steps", 5)),
        generator_steps=int(settings.get("generator_steps", 1)),
        checkpoint_every_epochs=int(settings.get("checkpoint_every_epochs", 1)),
        resume_from_checkpoint=settings.get("resume_from_checkpoint", False),
        max_dense_batch_mb=float(settings.get("max_dense_batch_mb", 2048.0)),
        seed=int(config.get("seed", 1203)),
        device=args.device,
    )


def _evaluation_payload(rankings, test_records, train_ids, evaluation):
    missing_rankings = {record.id for record in test_records} - set(rankings)
    if missing_rankings:
        raise ValueError(
            f"Rankings are missing {len(missing_rankings)} held-out queries: "
            f"{sorted(missing_rankings)[:10]}"
        )
    candidates = set(train_ids)
    truth = {record.id: evaluable_truth(record, candidates) for record in test_records}
    metrics = evaluate_rankings(
        rankings,
        truth,
        recall_ks=tuple(evaluation.get("recall_ks", [20, 40, 60, 80, 100])),
        map_k=int(evaluation.get("map_k", 10)),
        ndcg_k=int(evaluation.get("ndcg_k", 100)),
    )
    cold_ids = {record.id for record in test_records if record.cold_start and truth[record.id]}
    cold_metrics = None
    if cold_ids:
        cold_metrics = evaluate_rankings(
            {x: rankings[x] for x in cold_ids},
            {x: truth[x] for x in cold_ids},
            recall_ks=(20,),
            map_k=int(evaluation.get("map_k", 10)),
            ndcg_k=int(evaluation.get("ndcg_k", 100)),
        )
    evaluable = sum(bool(values) for values in truth.values())
    return {
        "protocol": {
            "held_out_queries": len(test_records),
            "evaluable_queries": evaluable,
            "skipped_without_relevant_training_citations": len(test_records) - evaluable,
            "candidate_papers": len(candidates),
            "map_k": int(evaluation.get("map_k", 10)),
            "ndcg_k": int(evaluation.get("ndcg_k", 100)),
            "recall_ks": list(evaluation.get("recall_ks", [20, 40, 60, 80, 100])),
            "materialized_ranking_depth": required_ranking_depth(
                len(candidates),
                recall_ks=evaluation.get("recall_ks", [20, 40, 60, 80, 100]),
                map_k=int(evaluation.get("map_k", 10)),
                ndcg_k=int(evaluation.get("ndcg_k", 100)),
            ),
            "cold_start_definition": "held-out paper with no author metadata",
        },
        "overall": metrics,
        "cold_start_missing_author": cold_metrics,
    }


def command_evaluate_gcr(args: argparse.Namespace) -> None:
    from citation_models.gcr_gan.features import GCRFeatures
    from citation_models.gcr_gan.recommend import encode_training_nodes, rank_gcr_queries
    from citation_models.gcr_gan.specter import SpecterEncoder
    from citation_models.gcr_gan.train import load_gcr_gan

    config = load_config(args.config)
    output = Path(config["output_dir"])
    records, split, _ = _records_and_saved_split(config, output)
    by_id = {record.id: record for record in records}
    test_records = [by_id[x] for x in split.test]
    features = GCRFeatures.load(output / "features")
    model, device = load_gcr_gan(output / "checkpoint" / "gcr_gan.pt", args.device)
    node_embeddings = encode_training_nodes(model, features, device)
    resolved = json.loads((output / "resolved.json").read_text(encoding="utf-8"))
    encoder = SpecterEncoder(resolved["specter_model"], device=args.device)
    query_content = encoder.encode(
        [record.specter_text for record in test_records],
        batch_size=int(config["specter"].get("batch_size", 16)),
        max_length=int(config["specter"].get("max_length", 512)),
        deduplicate=bool(config["specter"].get("deduplicate_texts", True)),
    )
    evaluation = config["evaluation"]
    ranking_depth = required_ranking_depth(
        len(split.train),
        recall_ks=evaluation.get("recall_ks", [20, 40, 60, 80, 100]),
        map_k=int(evaluation.get("map_k", 10)),
        ndcg_k=int(evaluation.get("ndcg_k", 100)),
    )
    rankings = rank_gcr_queries(
        model,
        features,
        node_embeddings,
        test_records,
        query_content,
        [by_id[x] for x in split.train],
        device,
        personalized=not args.non_personalized,
        score_variant=evaluation.get("score_variant", "semantic"),
        top_k=ranking_depth,
        query_batch_size=int(evaluation.get("query_batch_size", 32)),
    )
    payload = _evaluation_payload(rankings, test_records, split.train, evaluation)
    payload["protocol"].update(
        {
            "personalized": not args.non_personalized,
            "score_variant": config["evaluation"].get("score_variant", "semantic"),
        }
    )
    filename = "evaluation_non_personalized.json" if args.non_personalized else "evaluation.json"
    path = output / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def command_compare_reported(args: argparse.Namespace) -> None:
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    observed = json.loads(Path(args.observed).read_text(encoding="utf-8"))
    observed_metrics = observed.get("overall", observed)
    rows = []
    for key, reported in reference["metrics"].items():
        observed_key = reference.get("metric_mapping", {}).get(key, key)
        value = observed_metrics.get(observed_key)
        rows.append(
            {
                "metric": key,
                "reported": reported,
                "observed": value,
                "absolute_difference": None if value is None else abs(value - reported),
            }
        )
    print(
        json.dumps(
            {
                "paper": reference["paper"],
                "comparison": rows,
                "unresolved_cutoffs": reference.get("unresolved_cutoffs", []),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gcr-gan")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def configured(name, function, *, device=True):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        if device:
            command.add_argument("--device")
        command.set_defaults(function=function)
        return command

    configured("validate-data", command_validate, device=False)
    configured("prepare", command_prepare_gcr)
    configured("train", command_train_gcr)
    evaluate = configured("evaluate", command_evaluate_gcr)
    evaluate.add_argument("--non-personalized", action="store_true")
    compare = subparsers.add_parser("compare-reported")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--observed", required=True)
    compare.set_defaults(function=command_compare_reported)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
