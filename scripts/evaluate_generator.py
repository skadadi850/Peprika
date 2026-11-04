"""
Generator evaluation on the unified corpus.

select      Checkpoint comparison. Per benchmark task an oracle (ESM-2, composition and dipeptide
            features, boosted trees) is fitted on the train split of the evidence-matched subset, stopped
            on val and scored on test. Every checkpoint samples sequences conditioned on each task state
            and on no task, and the oracle scores them. The control effect of a task is the mean oracle
            probability of sequences sampled for active minus those sampled for inactive. Writes
            generator_evaluation.csv and oracle_auroc.csv.
sample      Samples one checkpoint into results/samples/<name>__<condition>.fasta, one file per condition
            ("unconditional", "<task>_active", "<task>_inactive"), with the length distribution of the
            train split in every file. The output is written raw, without deduplication.
score       Scores every sample file of every model in results/samples identically, with the test split
            and a train sample as reference rows: validity, uniqueness, novelty against train, similarity
            to test, diversity, property distributions, Frechet distance to test in ESM-2 space, coverage
            of test, oracle scores and control within bins of similarity to train. Writes
            sample_evaluation<tag>.csv and sample_control_binned<tag>.csv.
likelihood  Exact negative log-likelihood per residue of the test peptides under each checkpoint, given the
            length and the conditions of an arm (length; length and properties; length, properties and task
            labels), with 95% intervals that resample homology components and paired differences to the
            first checkpoint. The difference between the arms with and without task labels is the label gain
            in likelihood. Writes likelihood<tag>.csv and likelihood_summary<tag>.csv and
            likelihood_label_gain<tag>.csv.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/evaluate_generator.py select --count 500 --settings 2.0:all
    PYTHONUNBUFFERED=1 uv run python scripts/evaluate_generator.py sample --name unified_epoch020 \
        --checkpoint runs/unified/final/epoch_020.pt
    PYTHONUNBUFFERED=1 uv run python scripts/evaluate_generator.py score --tag _binned
    PYTHONUNBUFFERED=1 uv run python scripts/evaluate_generator.py likelihood \
        --checkpoints wheel=runs/unified/final_s0/masked_diffusion_best.pt
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scipy import linalg, stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ampx.compliance import read_fasta, write_fasta  # noqa: E402
from ampx.evaluation import (  # noqa: E402
    BENCHMARK_TASKS,
    combined_features,
    component_groups,
    corpus_embeddings,
    esmc_lookup,
    fit_oracles,
    load_frame,
    max_similarity,
    percentile_interval,
    resample_components,
    sequence_features,
)
from ampx.models.data import (  # noqa: E402
    MU_H_CEILING,
    TASK_AXES,
    TASKS,
    compute_sequence_features,
    empirical_length_counts,
    is_valid_sequence,
    length_grid,
    normalize_charge_per_residue,
)
from ampx.models.masked_diffusion import MaskedDiffusionModel, encode_sequence  # noqa: E402
from ampx.predictor import ESM2Encoder, condition_matrix, resolve_device, sample_library  # noqa: E402

STATE_CODES = {"active": 2, "inactive": 1}
SELECTION_TASKS = list(BENCHMARK_TASKS)

DEFAULT_CHECKPOINTS = (
    f"pretrain={ROOT / 'runs/unified/pretrain/masked_diffusion_best.pt'}",
    f"final_best={ROOT / 'runs/unified/final_s0/masked_diffusion_best.pt'}",
    f"final_last={ROOT / 'runs/unified/final_s0/masked_diffusion_final.pt'}",
)

PROPERTY_COLUMNS = ("length", "net_charge", "mu_h", "gravy_raw")
DIVERSITY_SAMPLE = 1000
MIN_SEQUENCES = 100
MIN_FRECHET = 1000
SIMILARITY_BINS = (0.5, 0.7)
BIN_NAMES = ("lt050", "050_070", "ge070")
MIN_BIN_SAMPLES = 100


def generate(model, species_vocab, request, lengths, count, setting, steps, batch_size, seed, device):
    guidance, null_axes = setting
    counts = empirical_length_counts(lengths, count)
    condition = length_grid(request, counts, species_vocab, device=device)

    return sample_library(
        model,
        condition,
        guidance_weight=guidance,
        steps=steps,
        temperature=1.0,
        batch_size=batch_size,
        seed=seed,
        autocast_dtype=torch.bfloat16 if device == "cuda" else None,
        guidance_null_axes=null_axes,
    )


def parse_setting(text: str):
    weight, _, mode = text.partition(":")

    if mode not in ("all", "tasks"):
        raise SystemExit(f"setting {text!r} must be <weight>:all or <weight>:tasks")

    return text, float(weight), (TASK_AXES if mode == "tasks" else None)


def record(checkpoint, label, task, state, scores, stats_row) -> dict:
    return {
        "checkpoint": checkpoint,
        "setting": label,
        "task": task,
        "state": state,
        "mean_p_active": float(scores.mean()),
        "se": float(scores.std(ddof=1) / np.sqrt(len(scores))),
        **stats_row,
    }


def describe(sequences, train_sequences, novelty_ceiling=None):
    sequences = [sequence for sequence in sequences if sequence]
    unique = list(dict.fromkeys(sequences))
    similarity = max_similarity(unique, train_sequences)

    row = {
        "n": len(sequences),
        "unique_fraction": len(unique) / max(len(sequences), 1),
        "median_max_train_similarity": float(np.median(similarity)),
        "fraction_ge_070": float((similarity >= 0.7).mean()),
        "fraction_exact_copy": float((similarity >= 1.0).mean()),
    }

    if novelty_ceiling is not None:
        unique = [
            sequence for sequence, value in zip(unique, similarity) if value < novelty_ceiling
        ]

    row["n_scored"] = len(unique)

    return unique, row


def run_select(args) -> None:
    settings = [parse_setting(item) for item in args.settings]
    corpus = Path(args.corpus)
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    frame = load_frame(corpus)
    sequences = frame["sequence"].tolist()
    train_sequences = frame.loc[frame["split"] == "train", "sequence"].tolist()

    encoder = ESM2Encoder(device=device)
    embeddings = corpus_embeddings(sequences, corpus / "esm_embeddings.npz", device, encoder)
    oracles, oracle_table = fit_oracles(
        frame, combined_features(frame["sequence"], embeddings), args.tasks, matched=True
    )
    oracle_table.to_csv(logs / "oracle_auroc.csv", index=False)

    state_lengths = {}

    for task in oracles:
        state_lengths[task] = {}

        for name, code in STATE_CODES.items():
            rows = (frame["split"] == "train") & (frame[f"act_{task}_idx"] == code)
            state_lengths[task][name] = frame.loc[rows, "sequence"].str.len().to_numpy()

    all_lengths = frame.loc[frame["split"] == "train", "sequence"].str.len().to_numpy()
    records = []
    counter = 0

    for item in args.checkpoints:
        name, _, path = item.partition("=")
        model = MaskedDiffusionModel.load(path, device=device)
        species_vocab = json.loads((Path(path).parent / "species_vocab.json").read_text())

        for label, guidance, null_axes in settings:
            print(f"{name} {label}", flush=True)

            requests = [
                ("none_amp", None, {"is_amp": "amp"}, all_lengths),
                ("none_plain", None, {}, all_lengths),
            ]

            for task in oracles:
                for state_name in STATE_CODES:
                    lengths = state_lengths[task][state_name]

                    if not len(lengths):
                        continue

                    request = {"tasks": {task: state_name}}

                    if task != "antiinflammatory":
                        request["is_amp"] = "amp"

                    requests.append(
                        (state_name, task, request, lengths if args.state_lengths else all_lengths)
                    )

            for state_name, task, request, lengths in requests:
                counter += 1
                samples = generate(
                    model, species_vocab, request, lengths, args.count, (guidance, null_axes),
                    args.steps, args.batch_size, args.seed + counter, device,
                )
                unique, row = describe(samples, train_sequences, args.novelty_ceiling)

                if len(unique) < 30:
                    continue

                _, generated = sequence_features(unique, encoder)

                for target_task in ([task] if task else list(oracles)):
                    scores = oracles[target_task].score(generated)
                    records.append(record(name, label, target_task, state_name, scores, row))

        del model
        torch.cuda.empty_cache()

    table = pd.DataFrame(records)
    table.to_csv(logs / "generator_evaluation.csv", index=False)

    key = ["checkpoint", "setting", "task"]
    mean = table.pivot_table(index=key, columns="state", values="mean_p_active")
    error = table.pivot_table(index=key, columns="state", values="se")

    result = pd.DataFrame(
        {
            "control": mean["active"] - mean["inactive"],
            "se": np.sqrt(error["active"] ** 2 + error["inactive"] ** 2),
        }
    )
    result["z"] = result["control"] / result["se"]

    real = oracle_table.set_index("task")
    real_gap = real["test_p_active_for_active"] - real["test_p_active_for_inactive"]
    result["fraction_of_real_gap"] = (
        result["control"].to_numpy() / result.index.get_level_values("task").map(real_gap).to_numpy()
    )

    novelty = (
        table[table["state"].isin(STATE_CODES)]
        .groupby(["checkpoint", "setting"])[
            ["unique_fraction", "median_max_train_similarity", "fraction_ge_070", "fraction_exact_copy"]
        ]
        .mean()
    )

    summary = (
        result.reset_index()
        .query("task in @SELECTION_TASKS")
        .groupby(["checkpoint", "setting"])["control"]
        .mean()
        .rename("mean_control")
        .to_frame()
        .join(novelty[["fraction_exact_copy", "fraction_ge_070"]])
    )
    summary["qualifies"] = (summary["fraction_exact_copy"] <= 0.02) & (
        summary["fraction_ge_070"] <= 0.15
    )

    print(oracle_table.round(3).to_string(index=False), flush=True)
    print(result.round(3).to_string(), flush=True)
    print(novelty.round(3).to_string(), flush=True)
    print("selection summary (pretrain is a reference, not a candidate)", flush=True)
    print(summary.sort_values("mean_control", ascending=False).round(3).to_string(), flush=True)
    print(f"wrote {logs / 'generator_evaluation.csv'} and {logs / 'oracle_auroc.csv'}", flush=True)


def write_samples(out_directory: Path, name: str, label: str, samples) -> None:
    kept = [sequence for sequence in samples if sequence]
    path = out_directory / f"{name}__{label}.fasta"
    write_fasta(kept, path, prefix=f"{name}_{label}_")

    print(
        f"{label}: {len(samples)} sampled, {len(kept)} non-empty, {len(set(kept))} unique -> {path}",
        flush=True,
    )


def subset_condition(train, task, state, args, species_vocab, device, seed):
    rows = train[train[f"act_{task}_idx"] == STATE_CODES[state]]
    chosen = rows.sample(n=args.count, replace=True, random_state=seed)["sequence"].tolist()
    properties = compute_sequence_features(chosen)

    requests = []
    pool = train["sequence"].str.len().to_numpy()
    generator = np.random.default_rng(seed)

    for length, charge, moment, gravy in zip(
        properties["length"], properties["net_charge"], properties["mu_h"], properties["gravy_raw"]
    ):
        request = {
            "length": int(length),
            "net_charge": float(charge),
            "mu_h": float(moment),
            "gravy": float(gravy),
        }

        if args.common_length:
            request = {
                "length": int(generator.choice(pool)),
                "charge_per_res": normalize_charge_per_residue(float(charge), int(length)),
                "mu_h": float(moment),
                "gravy": float(gravy),
            }

        if not args.no_task_labels:
            request["tasks"] = {task: state}

        if task != "antiinflammatory":
            request["is_amp"] = "amp"

        requests.append(request)

    return condition_matrix(requests, [1] * len(requests), species_vocab, device)


def run_sample(args) -> None:
    device = resolve_device(args.device)
    out_directory = Path(args.out_dir)
    out_directory.mkdir(parents=True, exist_ok=True)

    frame = load_frame(Path(args.corpus))
    train = frame[frame["split"] == "train"]
    lengths = train["sequence"].str.len().to_numpy()
    null_axes = TASK_AXES if args.null_axes == "tasks" else None

    checkpoint = Path(args.checkpoint)
    species_vocab = json.loads((checkpoint.parent / "species_vocab.json").read_text())
    model = MaskedDiffusionModel.load(str(checkpoint), device=device)

    index = 0

    if args.conditioning == "label":
        conditions = [("unconditional", {"is_amp": "amp"})]

        for task in args.tasks:
            for state in ("active", "inactive"):
                request = {"tasks": {task: state}}

                if task != "antiinflammatory":
                    request["is_amp"] = "amp"

                conditions.append((f"{task}_{state}", request))

        for label, request in conditions:
            samples = generate(
                model, species_vocab, request, lengths, args.count, (args.guidance, null_axes),
                args.steps, args.batch_size, args.seed + index, device,
            )
            write_samples(out_directory, args.name, label, samples)
            index += 1

        return

    for task in args.tasks:
        for state in ("active", "inactive"):
            condition = subset_condition(
                train, task, state, args, species_vocab, device, args.seed + index
            )
            samples = sample_library(
                model,
                condition,
                guidance_weight=args.guidance,
                steps=args.steps,
                temperature=1.0,
                batch_size=args.batch_size,
                seed=args.seed + index,
                autocast_dtype=torch.bfloat16 if device == "cuda" else None,
                guidance_null_axes=null_axes,
            )
            write_samples(out_directory, args.name, f"{task}_{state}", samples)
            index += 1


def load_sample_files(directory: Path) -> dict[str, dict[str, list[str]]]:
    models: dict[str, dict[str, list[str]]] = {}

    for path in sorted(directory.glob("*__*.fasta")):
        model, _, condition = path.stem.rpartition("__")
        models.setdefault(model, {})[condition] = read_fasta(path)[1]

    return models


def frechet_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64)
    second = second.astype(np.float64)

    covariance_first = np.cov(first, rowvar=False)
    covariance_second = np.cov(second, rowvar=False)
    root = linalg.sqrtm(covariance_first @ covariance_second).real
    difference = first.mean(axis=0) - second.mean(axis=0)

    return float(
        difference @ difference + np.trace(covariance_first) + np.trace(covariance_second)
        - 2.0 * np.trace(root)
    )


def diversity(sequences: list[str], seed: int) -> float:
    if len(sequences) < 2:
        return float("nan")

    if len(sequences) > DIVERSITY_SAMPLE:
        chosen = np.random.default_rng(seed).choice(len(sequences), DIVERSITY_SAMPLE, replace=False)
        sequences = [sequences[index] for index in chosen]

    distance = process.cdist(
        sequences, sequences, scorer=Levenshtein.normalized_distance, dtype=np.float32, workers=-1
    )

    return float(distance[np.triu_indices(len(sequences), k=1)].mean())


SET_SUFFIX = {"matched": "", "alllabels": "_alllabels", "esmc": "_esmc"}


def evaluate_set(model, condition, sequences, context, compare_to_test=True, seed=0) -> dict:
    cleaned = [str(sequence).upper().strip() for sequence in sequences]
    valid = [sequence for sequence in cleaned if is_valid_sequence(sequence)]
    unique = list(dict.fromkeys(valid))

    row = {
        "model": model,
        "condition": condition,
        "n": len(cleaned),
        "valid_fraction": len(valid) / max(len(cleaned), 1),
        "unique_fraction": len(unique) / max(len(valid), 1),
        "n_scored": len(unique),
    }

    if len(unique) < MIN_SEQUENCES:
        print(f"{model} {condition}: only {len(unique)} valid unique sequences, not scored", flush=True)
        return {name: dict(row) for name in context["oracle_sets"]}

    train_similarity = max_similarity(unique, context["train_sequences"])
    properties = compute_sequence_features(unique)
    embeddings, features = sequence_features(unique, context["encoder"])

    row["median_max_train_similarity"] = float(np.median(train_similarity))
    row["fraction_ge_070"] = float((train_similarity >= 0.7).mean())
    row["fraction_exact_copy"] = float((train_similarity >= 1.0).mean())
    row["diversity"] = diversity(unique, seed)

    for column in PROPERTY_COLUMNS:
        row[f"median_{column}"] = float(properties[column].median())

    if compare_to_test:
        test_similarity = max_similarity(unique, context["test_sequences"])
        coverage = max_similarity(context["test_sequences"], unique)

        row["fraction_ge_070_test"] = float((test_similarity >= 0.7).mean())
        row["coverage_ge_050"] = float((coverage >= 0.5).mean())
        row["coverage_ge_070"] = float((coverage >= 0.7).mean())
        row["frechet_test"] = (
            frechet_distance(embeddings, context["test_embeddings"])
            if len(unique) >= MIN_FRECHET
            else float("nan")
        )

        for column in PROPERTY_COLUMNS:
            row[f"ks_{column}"] = float(
                stats.ks_2samp(
                    properties[column].to_numpy(dtype=np.float64),
                    context["test_properties"][column].to_numpy(dtype=np.float64),
                ).statistic
            )

    assignment = np.digitize(train_similarity, SIMILARITY_BINS)

    for index, name in enumerate(BIN_NAMES):
        row[f"n_{name}"] = int((assignment == index).sum())

    rows = {name: dict(row) for name in context["oracle_sets"]}

    for name, (oracles, lookup) in context["oracle_sets"].items():
        used = features if lookup is None else combined_features(pd.Series(unique), lookup(unique))

        for task, oracle in oracles.items():
            scores = oracle.score(used)
            rows[name][f"p_active_{task}"] = float(scores.mean())
            rows[name][f"p_active_{task}_se"] = float(scores.std(ddof=1) / np.sqrt(len(scores)))

            for index, bin_name in enumerate(BIN_NAMES):
                binned = scores[assignment == index]

                if len(binned) >= MIN_BIN_SAMPLES:
                    rows[name][f"p_active_{task}_{bin_name}"] = float(binned.mean())
                    rows[name][f"p_active_{task}_{bin_name}_se"] = float(binned.std(ddof=1) / np.sqrt(len(binned)))

    print(
        f"{model} {condition}: {len(unique)} scored, median max train similarity "
        f"{row['median_max_train_similarity']:.3f}, diversity {row['diversity']:.3f}",
        flush=True,
    )

    return rows


def control_binned(table: pd.DataFrame, tasks) -> pd.DataFrame:
    rows = []

    for model, group in table.groupby("model"):
        for task in tasks:
            active = group[group["condition"] == f"{task}_active"]
            inactive = group[group["condition"] == f"{task}_inactive"]

            if active.empty or inactive.empty:
                continue

            first, second = active.iloc[0], inactive.iloc[0]

            for name in BIN_NAMES:
                key = f"p_active_{task}_{name}"

                if key not in table or pd.isna(first[key]) or pd.isna(second[key]):
                    continue

                control = first[key] - second[key]
                error = float(np.hypot(first[f"{key}_se"], second[f"{key}_se"]))

                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "bin": name,
                        "n_active": int(first[f"n_{name}"]),
                        "n_inactive": int(second[f"n_{name}"]),
                        "control": control,
                        "se": error,
                        "z": control / error if error > 0 else float("nan"),
                    }
                )

    return pd.DataFrame(rows)


def run_choose(args) -> None:
    table = pd.read_csv(args.evaluation)
    table = table[table["state"].isin(STATE_CODES)]
    copies = table.groupby("checkpoint")[["fraction_exact_copy", "fraction_ge_070"]].mean()
    runs = dict(item.partition("=")[::2] for item in args.runs)
    rows = []

    for run, directory in runs.items():
        with open(Path(directory) / "masked_diffusion_history.pkl", "rb") as handle:
            validation = pickle.load(handle)["history"]["val"]

        for name, values in copies.iterrows():
            owner, _, kind = name.partition(":")

            if owner != run:
                continue

            if kind == "masked_diffusion_best":
                loss = min(validation)
            elif kind == "masked_diffusion_final":
                loss = validation[-1]
            else:
                loss = validation[int(kind.rsplit("_", 1)[1]) - 1]

            rows.append(
                {
                    "run": run,
                    "checkpoint": kind,
                    "validation_loss": loss,
                    "exact_copy": values["fraction_exact_copy"],
                    "ge_070": values["fraction_ge_070"],
                    "path": str(Path(directory) / f"{kind}.pt"),
                }
            )

    if not rows:
        raise SystemExit("no checkpoint of --runs appears in the evaluation file")

    result = pd.DataFrame(rows)
    result["qualifies"] = (result["exact_copy"] <= 0.02) & (result["ge_070"] <= 0.15)
    print(result.drop(columns="path").round(4).to_string(index=False), flush=True)

    chosen = (
        result[result["qualifies"]]
        .sort_values(["run", "validation_loss"])
        .groupby("run", sort=False)
        .head(1)
    )

    if chosen.empty:
        raise SystemExit("no checkpoint satisfies the copy limits")

    print("selected: lowest validation loss among checkpoints within the copy limits", flush=True)
    print(chosen.drop(columns="path").round(4).to_string(index=False), flush=True)

    missing = sorted(set(runs) - set(chosen["run"]))

    if missing:
        print(f"no admissible checkpoint for: {missing}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{row.run} {row.path}\n" for row in chosen.itertuples()))
    print(f"wrote {output}", flush=True)


def run_score(args) -> None:
    corpus = Path(args.corpus)
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    frame = load_frame(corpus)
    sequences = frame["sequence"].tolist()

    encoder = ESM2Encoder(device=device)
    embeddings = corpus_embeddings(sequences, corpus / "esm_embeddings.npz", device, encoder)
    oracle_sets = {}

    for name in args.oracle_sets:
        if name == "esmc":
            stored = np.load(corpus / "esmc_embeddings.npz", allow_pickle=False)

            if stored["sequences"].tolist() != sequences:
                raise SystemExit("esmc_embeddings.npz has a different row order from the corpus")

            lookup = esmc_lookup(sequences, stored["embeddings"], corpus / "esmc_sample_cache.npz")
            oracles, _ = fit_oracles(
                frame, combined_features(frame["sequence"], stored["embeddings"]), args.tasks, matched=True
            )
        else:
            lookup = None
            oracles, _ = fit_oracles(
                frame, combined_features(frame["sequence"], embeddings), args.tasks, matched=name == "matched"
            )

        oracle_sets[name] = (oracles, lookup)
        print(f"oracles fitted: {name}", flush=True)

    test_index = np.flatnonzero(frame["split"].to_numpy() == "test")
    train_index = np.flatnonzero(frame["split"].to_numpy() == "train")
    test_sequences = [sequences[index] for index in test_index]
    train_sequences = [sequences[index] for index in train_index]

    context = {
        "encoder": encoder,
        "oracle_sets": oracle_sets,
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "test_embeddings": embeddings[test_index],
        "test_properties": compute_sequence_features(test_sequences),
    }

    rows = {name: [] for name in oracle_sets}

    def add(parts: dict) -> None:
        for name, row in parts.items():
            rows[name].append(row)

    add(evaluate_set("reference_test", "all", test_sequences, context, compare_to_test=False))

    chosen = np.random.default_rng(args.seed).choice(
        len(train_sequences), min(args.reference_size, len(train_sequences)), replace=False
    )
    add(evaluate_set("reference_train", "all", [train_sequences[index] for index in chosen], context))

    for model, conditions in load_sample_files(Path(args.samples_dir)).items():
        for condition, samples in conditions.items():
            add(evaluate_set(model, condition, samples, context, seed=args.seed))

    tasks = list(next(iter(oracle_sets.values()))[0])

    for name, items in rows.items():
        suffix = f"{SET_SUFFIX[name]}{args.tag}"
        table = pd.DataFrame(items)
        output = logs / f"sample_evaluation{suffix}.csv"
        table.to_csv(output, index=False)

        binned = control_binned(table, tasks)
        binned_output = logs / f"sample_control_binned{suffix}.csv"
        binned.to_csv(binned_output, index=False)
        print(f"wrote {output} and {binned_output}", flush=True)


HIT_PROPERTIES = ("charge_per_residue", "mu_h", "gravy_raw")
GRAVY_LIMIT = 4.5


def property_table(sequences) -> pd.DataFrame:
    features = compute_sequence_features(list(sequences))
    length = features["length"].to_numpy(dtype=np.float64)
    charge = features["net_charge"].to_numpy(dtype=np.float64)

    return pd.DataFrame(
        {
            "length": length,
            "charge_per_residue": np.divide(charge, length, out=np.full(len(length), np.nan), where=length > 0),
            "mu_h": features["mu_h"].to_numpy(dtype=np.float64),
            "gravy_raw": features["gravy_raw"].to_numpy(dtype=np.float64),
        }
    )


def fit_density(properties: pd.DataFrame, bandwidth):
    return stats.gaussian_kde(properties[list(HIT_PROPERTIES)].to_numpy(dtype=np.float64).T, bw_method=bandwidth)


def draw_vectors(density, count: int, seed: int) -> np.ndarray:
    vectors = density.resample(count, seed=seed).T
    vectors[:, 0] = np.clip(vectors[:, 0], -1.0, 1.0)
    vectors[:, 1] = np.clip(vectors[:, 1], 0.0, MU_H_CEILING)
    vectors[:, 2] = np.clip(vectors[:, 2], -GRAVY_LIMIT, GRAVY_LIMIT)

    return vectors


def build_requests(task: str, state: str, vectors: np.ndarray, lengths: np.ndarray, labels: bool):
    requests = []

    for length, (per_residue, moment, gravy) in zip(lengths, vectors):
        request = {"length": int(length), "charge_per_res": float((per_residue + 1.0) / 2.0), "mu_h": float(moment), "gravy": float(gravy)}

        if labels:
            request["tasks"] = {task: state}

        if task != "antiinflammatory":
            request["is_amp"] = "amp"

        requests.append(request)

    return requests


def property_hit(vectors, lengths, generated: pd.DataFrame, reference: pd.DataFrame, seed: int):
    valid = generated["length"].to_numpy() > 0
    generator = np.random.default_rng(seed)
    rows = []

    for column, name in enumerate(HIT_PROPERTIES):
        spread = float(reference[name].std())
        wanted, obtained = vectors[valid, column], generated.loc[valid, name].to_numpy()
        rows.append(
            {
                "property": name, "n": int(valid.sum()),
                "mean_abs_error_sd": float(np.abs(obtained - wanted).mean() / spread),
                "shuffled_mean_abs_error_sd": float(np.abs(obtained - generator.permutation(wanted)).mean() / spread),
                "bias_sd": float((obtained - wanted).mean() / spread),
                "spearman": float(stats.spearmanr(wanted, obtained).statistic),
            }
        )

    rows.append({"property": "length", "n": int(valid.sum()), "exact_fraction": float((generated.loc[valid, "length"].to_numpy() == lengths[valid]).mean())})

    return rows


def run_property(args) -> None:
    if args.density == "pooled" and args.no_task_labels:
        raise SystemExit("--density pooled with --no-task-labels gives identical requests in both states")

    device = resolve_device(args.device)
    out_directory, logs = Path(args.out_dir), Path(args.logs)
    out_directory.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    frame = load_frame(Path(args.corpus))
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    reference = property_table(train["sequence"])
    pool = reference["length"].to_numpy().astype(np.int64)

    checkpoint = Path(args.checkpoint)
    species_vocab = json.loads((checkpoint.parent / "species_vocab.json").read_text())
    model = MaskedDiffusionModel.load(str(checkpoint), device=device)
    null_axes = TASK_AXES if args.null_axes == "tasks" else None
    pooled = fit_density(reference, args.bandwidth)
    records, index = [], 0

    for task_index, task in enumerate(args.tasks):
        draw_seed = args.seed + task_index
        lengths = np.random.default_rng(draw_seed).choice(pool, args.count)

        for state in ("active", "inactive"):
            if args.density == "state":
                density = fit_density(reference[(train[f"act_{task}_idx"] == STATE_CODES[state]).to_numpy()], args.bandwidth)
            else:
                density = pooled

            vectors = draw_vectors(density, args.count, draw_seed)
            condition = condition_matrix(build_requests(task, state, vectors, lengths, not args.no_task_labels), [1] * args.count, species_vocab, device)
            samples = sample_library(
                model, condition, guidance_weight=args.guidance, steps=args.steps, temperature=1.0, batch_size=args.batch_size,
                seed=args.seed + index, autocast_dtype=torch.bfloat16 if device == "cuda" else None, guidance_null_axes=null_axes,
            )
            index += 1
            label = f"{task}_{state}"
            write_samples(out_directory, args.name, label, samples)

            saved = pd.DataFrame(vectors, columns=HIT_PROPERTIES)
            saved.insert(0, "length", lengths)
            saved["sequence"] = samples
            saved.to_csv(out_directory / f"{args.name}__{label}.requests.tsv", sep="\t", index=False)

            for row in property_hit(vectors, lengths, property_table(samples), reference, args.seed):
                records.append({"model": args.name, "task": task, "state": state, **row})

            print(f"{label}: property-hit rows written", flush=True)

    table = pd.DataFrame(records)
    output = logs / f"property_hit_{args.name}.csv"
    table.to_csv(output, index=False)
    print(table.round(3).to_string(index=False), flush=True)
    print(f"wrote {output}", flush=True)


def run_compare(args) -> None:
    from itertools import product

    from scipy.spatial.distance import jensenshannon

    frame = load_frame(Path(args.corpus))
    train = frame.loc[frame["split"] == "train", "sequence"].tolist()
    test = frame.loc[frame["split"] == "test", "sequence"].tolist()
    amino = "ACDEFGHIKLMNPQRSTVWY"
    single = {residue: index for index, residue in enumerate(amino)}
    pair = {a + b: index for index, (a, b) in enumerate(product(amino, repeat=2))}

    def composition(sequences):
        counts = np.zeros(len(single))

        for sequence in sequences:
            for residue in sequence:
                counts[single[residue]] += 1.0

        return counts / counts.sum()

    def dipeptides(sequences):
        counts = np.zeros(len(pair))

        for sequence in sequences:
            for position in range(len(sequence) - 1):
                counts[pair[sequence[position : position + 2]]] += 1.0

        return counts / counts.sum()

    test_properties = property_table(test)
    test_composition, test_dipeptides = composition(test), dipeptides(test)
    columns = ("length", "charge_per_residue", "mu_h", "gravy_raw")

    def describe_set(name, sequences, reference=False) -> dict:
        cleaned = (str(item).upper().strip() for item in sequences)
        unique = list(dict.fromkeys(item for item in cleaned if is_valid_sequence(item)))
        properties = property_table(unique)
        net = properties["charge_per_residue"].to_numpy() * properties["length"].to_numpy()
        row = {
            "model": name,
            "n_valid_unique": len(unique),
            "diversity": diversity(unique, args.seed),
            "share_charge_ge_2": float((net >= 2.0).mean()),
        }

        if not reference:
            row["js_composition"] = float(jensenshannon(composition(unique), test_composition, base=2) ** 2)
            row["js_dipeptide"] = float(jensenshannon(dipeptides(unique), test_dipeptides, base=2) ** 2)

            for column in columns:
                row[f"w1_{column}"] = float(
                    stats.wasserstein_distance(properties[column].dropna(), test_properties[column].dropna())
                    / test_properties[column].std()
                )

        return row

    rows = [describe_set("reference_test", test, reference=True)]
    chosen = np.random.default_rng(args.seed).choice(len(train), min(5000, len(train)), replace=False)
    rows.append(describe_set("reference_train", [train[index] for index in chosen]))

    for path in sorted((ROOT / "results" / "samples_published").glob("*__unconditional.fasta")):
        rows.append(describe_set(path.stem.split("__")[0], read_fasta(path)[1]))

    for seed in (0, 1, 2):
        path = ROOT / "results" / "samples" / f"gen_s{seed}__unconditional.fasta"

        if path.exists():
            rows.append(describe_set(f"ours_s{seed}", read_fasta(path)[1]))

    table = pd.DataFrame(rows)
    output = ROOT / "results" / "tables" / "generator_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.round(3).to_string(index=False), flush=True)
    print(f"wrote {output}", flush=True)


def run_headtohead(args) -> None:
    # control of generators with the same oracles: mean oracle probability of active for the "on" samples minus the "off"
    # samples, restricted to peptides of 8 to 25 residues. strat_control reweights to the training length distribution in
    # four length bins. matched_control stratifies within each pair on length bin, charge-per-residue quintile and GRAVY
    # tertile (edges from the pooled pair) and weights each stratum with at least 20 samples on both sides by its pooled
    # count; matched_kept is the share of samples that lie in those strata
    corpus = Path(args.corpus)
    device = resolve_device(args.device)
    frame = load_frame(corpus)
    encoder = ESM2Encoder(device=device)
    embeddings = corpus_embeddings(frame["sequence"].tolist(), corpus / "esm_embeddings.npz", device, encoder)
    oracles, _ = fit_oracles(frame, combined_features(frame["sequence"], embeddings), args.tasks, matched=True)

    edges = np.array([8, 12, 16, 20, 26])
    reference = frame.loc[frame["split"] == "train", "sequence"].str.len().to_numpy()
    reference = reference[(reference >= 8) & (reference <= 25)]
    weights = np.histogram(reference, bins=edges)[0].astype(np.float64)
    weights /= weights.sum()
    cache = {}

    def scored(path: str):
        if path not in cache:
            names = (str(item).upper().strip() for item in read_fasta(Path(path))[1])
            sequences = [
                item for item in dict.fromkeys(names) if is_valid_sequence(item) and 8 <= len(item) <= 25
            ]
            _, features = sequence_features(sequences, encoder)
            length = np.array([len(item) for item in sequences])
            cache[path] = (length, property_table(sequences), {task: oracles[task].score(features) for task in oracles})
            print(f"scored {path}: {len(sequences)} sequences", flush=True)

        return cache[path]

    def summary(values, length):
        bins = np.digitize(length, edges) - 1
        value, variance, used = 0.0, 0.0, 0.0

        for index, weight in enumerate(weights):
            chosen = values[bins == index]

            if len(chosen) < 20:
                continue

            value += weight * chosen.mean()
            variance += weight**2 * chosen.var(ddof=1) / len(chosen)
            used += weight

        matched = (value / used, float(np.sqrt(variance)) / used) if used else (float("nan"), float("nan"))

        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))), matched[0], matched[1]

    def stratified(on, on_properties, off, off_properties, minimum=20):
        pooled = pd.concat([on_properties, off_properties], ignore_index=True)
        charge_cuts = np.quantile(pooled["charge_per_residue"], [0.2, 0.4, 0.6, 0.8])
        gravy_cuts = np.quantile(pooled["gravy_raw"], [1.0 / 3.0, 2.0 / 3.0])

        def cells(properties):
            return (
                (np.digitize(properties["length"].to_numpy(), edges) - 1) * 15
                + np.digitize(properties["charge_per_residue"].to_numpy(), charge_cuts) * 3
                + np.digitize(properties["gravy_raw"].to_numpy(), gravy_cuts)
            )

        on_cells, off_cells = cells(on_properties), cells(off_properties)
        value, variance, used = 0.0, 0.0, 0.0

        for cell in np.unique(np.concatenate([on_cells, off_cells])):
            first, second = on[on_cells == cell], off[off_cells == cell]

            if len(first) < minimum or len(second) < minimum:
                continue

            weight = float(len(first) + len(second))
            value += weight * (first.mean() - second.mean())
            variance += weight**2 * (first.var(ddof=1) / len(first) + second.var(ddof=1) / len(second))
            used += weight

        if not used:
            return float("nan"), float("nan"), 0.0

        return value / used, float(np.sqrt(variance)) / used, used / (len(on) + len(off))

    rows = []

    for item in args.pairs:
        label, on_template, off_template = item.split("|")

        for task in oracles:
            on_path, off_path = on_template.format(task=task), off_template.format(task=task)

            if not (Path(on_path).exists() and Path(off_path).exists()):
                continue

            (length_on, properties_on, on), (length_off, properties_off, off) = scored(on_path), scored(off_path)
            mean_on, se_on, strat_on, strat_se_on = summary(on[task], length_on)
            mean_off, se_off, strat_off, strat_se_off = summary(off[task], length_off)
            matched_value, matched_se, matched_kept = stratified(on[task], properties_on, off[task], properties_off)
            rows.append(
                {
                    "label": label, "task": task, "n_on": len(length_on), "n_off": len(length_off),
                    "p_on": mean_on, "p_off": mean_off, "control": mean_on - mean_off,
                    "control_se": float(np.hypot(se_on, se_off)),
                    "strat_control": strat_on - strat_off, "strat_se": float(np.hypot(strat_se_on, strat_se_off)),
                    "matched_control": matched_value, "matched_se": matched_se, "matched_kept": matched_kept,
                }
            )

    table = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.round(4).to_string(index=False), flush=True)
    print(f"wrote {output}", flush=True)


def run_manifold(args) -> None:
    # precision and recall of generated sequences against the test split in ESM-2 space with k-nearest-neighbour
    # manifolds (k = --neighbors), on random subsets of equal size so that the numbers compare between generators:
    # precision is the share of samples inside the manifold of the test peptides, recall the share of test peptides
    # inside the manifold of the samples
    from scipy.spatial.distance import cdist

    corpus = Path(args.corpus)
    device = resolve_device(args.device)
    frame = load_frame(corpus)
    sequences = frame["sequence"].tolist()
    encoder = ESM2Encoder(device=device)
    embeddings = corpus_embeddings(sequences, corpus / "esm_embeddings.npz", device, encoder)
    generator = np.random.default_rng(args.seed)
    train_index = np.flatnonzero(frame["split"].to_numpy() == "train")
    real = np.asarray(embeddings[np.flatnonzero(frame["split"].to_numpy() == "test")], dtype=np.float32)

    sets = {"reference_train": [sequences[index] for index in generator.choice(train_index, 5000, replace=False)]}

    for path in sorted((ROOT / "results" / "samples_published").glob("*__unconditional.fasta")):
        sets[path.stem.split("__")[0]] = read_fasta(path)[1]

    for seed in (0, 1, 2):
        path = ROOT / "results" / "samples" / f"gen_s{seed}__unconditional.fasta"

        if path.exists():
            sets[f"ours_s{seed}"] = read_fasta(path)[1]

    def radius(points):
        distance = cdist(points, points)

        return np.partition(distance, args.neighbors, axis=1)[:, args.neighbors]

    rows = []

    for name, items in sets.items():
        cleaned = (str(item).upper().strip() for item in items)
        unique = [item for item in dict.fromkeys(cleaned) if is_valid_sequence(item)]
        count = min(args.count, len(real), len(unique))
        chosen = [unique[index] for index in generator.choice(len(unique), count, replace=False)]
        samples, _ = sequence_features(chosen, encoder)
        samples = np.asarray(samples, dtype=np.float32)
        reference = real[generator.choice(len(real), count, replace=False)]
        between = cdist(samples, reference)
        precision = float((between <= radius(reference)[None, :]).any(axis=1).mean())
        recall = float((between <= radius(samples)[:, None]).any(axis=0).mean())
        rows.append({"model": name, "n_valid_unique": len(unique), "n_used": count, "precision": precision, "recall": recall})
        print(f"{name}: precision {precision:.3f}, recall {recall:.3f} on {count} against {count}", flush=True)

    table = pd.DataFrame(rows)
    output = ROOT / "results" / "tables" / "generator_manifold.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.round(3).to_string(index=False), flush=True)
    print(f"wrote {output}", flush=True)


def run_omegamp_hit(args) -> None:
    # how closely generated net charge follows the requested one, in training standard deviations of charge per residue,
    # with a shuffled-request null; the request was rounded to an integer net charge
    frame = load_frame(Path(args.corpus))
    train = frame[frame["split"] == "train"]
    spread = float(property_table(train["sequence"])["charge_per_residue"].std())
    rows = []
    dropped = []

    for path in sorted((ROOT / "results" / "samples_omegamp").glob("*.tsv")):
        table = pd.read_csv(path, sep="\t").dropna(subset=["sequence"])
        before = len(table)
        table = table[(table["sequence"].str.len() > 0) & (table["sequence"].str.len() <= 50)].reset_index(drop=True)
        dropped.append(before - len(table))
        generated = property_table(table["sequence"])
        wanted = (table["net_requested"] / table["length_requested"]).to_numpy()
        obtained = generated["charge_per_residue"].to_numpy()
        shuffled = np.random.default_rng(args.seed).permutation(wanted)
        rows.append(
            {
                "model": path.stem.split("__")[0], "condition": path.stem.split("__")[1], "n": len(table),
                "mean_abs_error_sd": float(np.abs(obtained - wanted).mean() / spread),
                "shuffled_mean_abs_error_sd": float(np.abs(obtained - shuffled).mean() / spread),
                "bias_sd": float((obtained - wanted).mean() / spread),
                "spearman": float(stats.spearmanr(wanted, obtained).statistic),
                "length_exact": float((generated["length"].to_numpy() == table["length_requested"].to_numpy()).mean()),
            }
        )

    detail = pd.DataFrame(rows)
    summary = detail.groupby("model")[["mean_abs_error_sd", "shuffled_mean_abs_error_sd", "bias_sd", "spearman", "length_exact"]].mean()
    output = ROOT / "results" / "tables" / "omegamp_property_hit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output)
    print(f"sequences empty or longer than 50 residues, dropped: {sum(dropped)}", flush=True)
    print(summary.round(3).to_string(), flush=True)
    print(f"wrote {output}", flush=True)


LIKELIHOOD_ARMS = ("length", "properties", "properties_labels")


def likelihood_conditions(test: pd.DataFrame, features: pd.DataFrame, arm: str) -> dict:
    # conditions of one arm for every test peptide; the length is always given, because the canvas carries it
    count = len(test)
    condition = {"is_amp_idx": torch.as_tensor(test["is_amp_idx"].to_numpy(dtype=np.int64))}
    continuous = {"length_norm": features["length_norm"]}

    if arm in ("properties", "properties_labels"):
        continuous["charge_per_res"] = features["charge_per_res"]
        continuous["mu_h"] = features["mu_h"]
        continuous["gravy"] = features["gravy"]
        condition["cys_class_idx"] = torch.as_tensor(features["cys_class_idx"].to_numpy(dtype=np.int64))

    if arm == "properties_labels":
        for task in TASKS:
            condition[f"act_{task}_idx"] = torch.as_tensor(test[f"act_{task}_idx"].to_numpy(dtype=np.int64))

    for axis, values in continuous.items():
        condition[axis] = torch.as_tensor(values.to_numpy(dtype=np.float32)).reshape(-1, 1)
        condition[f"{axis}_mask"] = torch.ones(count, 1)

    return condition


def run_likelihood(args) -> None:
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    autocast_dtype = torch.bfloat16 if args.bf16 and device == "cuda" else None

    frame = load_frame(Path(args.corpus), ("sequence", "split", "component"))
    test = frame[frame["split"] == "test"].sort_values("sequence").reset_index(drop=True)
    sequences = test["sequence"].tolist()
    features = compute_sequence_features(sequences)
    length = features["length"].to_numpy()
    tokens = torch.tensor([encode_sequence(sequence) for sequence in sequences], dtype=torch.long)
    order = np.argsort(length, kind="stable")
    names, parts = [], []

    for item in args.checkpoints:
        name, _, path = item.partition("=")
        names.append(name)
        model = MaskedDiffusionModel.load(path, device=device)
        print(f"{name}: order {model.order}", flush=True)

        for arm in args.arms:
            condition = likelihood_conditions(test, features, arm)
            values = np.zeros(len(sequences))

            for start in range(0, len(order), args.batch_size):
                index = torch.as_tensor(order[start : start + args.batch_size])
                chunk = {key: value[index].to(device) for key, value in condition.items()}
                log_probability, _ = model.exact_log_likelihood(
                    tokens[index], chunk, autocast_dtype=autocast_dtype,
                    random_orders=args.random_orders, order_seed=args.seed,
                )
                values[index.numpy()] = log_probability.double().cpu().numpy()

            parts.append(
                pd.DataFrame(
                    {
                        "model": name, "arm": arm, "order": model.order, "sequence": sequences,
                        "component": test["component"].to_numpy(), "length": length, "log_probability": values,
                    }
                )
            )
            print(f"{name} {arm}: {-values.sum() / length.sum():.4f} nats per residue", flush=True)

        del model
        torch.cuda.empty_cache()

    table = pd.concat(parts, ignore_index=True)
    table.to_csv(logs / f"likelihood{args.tag}.csv", index=False)

    groups = component_groups(test["component"].to_numpy())
    generator = np.random.default_rng(args.seed)
    draws = [resample_components(groups, generator) for _ in range(args.draws)]
    everything = np.arange(len(sequences))

    def per_residue(log_probability, index):
        return float(-log_probability[index].sum() / length[index].sum())

    def column(name, arm):
        return table[(table["model"] == name) & (table["arm"] == arm)]["log_probability"].to_numpy()

    summary = []

    for arm in args.arms:
        boot = {name: np.array([per_residue(column(name, arm), index) for index in draws]) for name in names}

        for name in names:
            low, high = percentile_interval(boot[name].tolist())
            row = {
                "arm": arm, "model": name, "nll_per_residue": per_residue(column(name, arm), everything),
                "lo": low, "hi": high,
            }

            if name != names[0]:
                gap_low, gap_high = percentile_interval((boot[name] - boot[names[0]]).tolist())
                row.update(
                    difference=row["nll_per_residue"] - per_residue(column(names[0], arm), everything),
                    difference_lo=gap_low, difference_hi=gap_high,
                )

            summary.append(row)

    result = pd.DataFrame(summary)
    result.to_csv(logs / f"likelihood_summary{args.tag}.csv", index=False)
    print("negative log-likelihood per residue; difference is the model minus the first checkpoint (positive: worse)", flush=True)
    print(result.round(4).to_string(index=False), flush=True)

    if "properties" in args.arms and "properties_labels" in args.arms:
        gain_rows = []

        for name in names:
            first, second = column(name, "properties"), column(name, "properties_labels")
            boot = np.array([(-first[index].sum() + second[index].sum()) / length[index].sum() for index in draws])
            low, high = percentile_interval(boot.tolist())
            gain_rows.append(
                {
                    "model": name,
                    "label_gain": float((-first.sum() + second.sum()) / length.sum()),
                    "lo": low, "hi": high,
                }
            )

        gain = pd.DataFrame(gain_rows)
        gain.to_csv(logs / f"likelihood_label_gain{args.tag}.csv", index=False)
        print("label gain: nll per residue without task labels minus with them (positive: labels help)", flush=True)
        print(gain.round(4).to_string(index=False), flush=True)

    print(f"wrote {logs / f'likelihood{args.tag}.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator evaluation on the unified corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", default=str(ROOT / "data" / "processed" / "corpus"))
    common.add_argument("--tasks", nargs="+", default=list(BENCHMARK_TASKS))
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--device", default=None)

    sampling = argparse.ArgumentParser(add_help=False)
    sampling.add_argument("--count", type=int, default=1000)
    sampling.add_argument("--steps", type=int, default=128)
    sampling.add_argument("--batch-size", type=int, default=512)

    select = subparsers.add_parser("select", parents=[common, sampling])
    select.add_argument("--logs", default=str(ROOT / "logs"))
    select.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINTS))
    select.add_argument(
        "--settings", nargs="+", default=["1.0:all", "2.0:all", "2.0:tasks"],
        help="guidance weight and the axes the unconditional branch removes: all or tasks",
    )
    select.add_argument(
        "--novelty-ceiling", type=float, default=None,
        help="score only samples whose maximum similarity to the train split is below this value",
    )
    select.add_argument(
        "--state-lengths", action="store_true",
        help="draw lengths from the train rows of each task state; by default both states use the "
             "same length distribution so only the task label differs",
    )

    sample = subparsers.add_parser("sample", parents=[common, sampling])
    sample.add_argument("--name", required=True)
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--out-dir", default=str(ROOT / "results" / "samples"))
    sample.add_argument("--guidance", type=float, default=2.0)
    sample.add_argument(
        "--conditioning", choices=["label", "subset"], default="label",
        help="label: task labels only; subset: the length, charge, moment and GRAVY of train "
             "peptides in the requested task state, with the task label unless --no-task-labels",
    )
    sample.add_argument("--no-task-labels", action="store_true")
    sample.add_argument(
        "--null-axes", choices=["all", "tasks"], default="all",
        help="axes removed from the unconditional guidance branch; tasks removes only the task "
             "labels, so the guidance direction is the label effect alone",
    )
    sample.add_argument(
        "--common-length", action="store_true",
        help="draw the length from all train peptides instead of the train peptides in the "
             "requested state; charge per residue, moment and GRAVY stay state-specific",
    )
    sample.set_defaults(count=5000)

    score = subparsers.add_parser("score", parents=[common])
    score.add_argument("--logs", default=str(ROOT / "logs"))
    score.add_argument("--samples-dir", default=str(ROOT / "results" / "samples"))
    score.add_argument("--reference-size", type=int, default=5000)
    score.add_argument("--tag", default="", help="suffix for the output file names")
    score.add_argument(
        "--oracle-sets", nargs="+", choices=list(SET_SUFFIX), default=["matched"],
        help="matched: ESM-2 oracles on matched labels; alllabels: ESM-2 on all labels; esmc: ESM C on matched "
             "labels (reads esmc_embeddings.npz and esmc_sample_cache.npz from the corpus directory)",
    )

    density = subparsers.add_parser("property", parents=[common, sampling])
    density.add_argument("--name", required=True)
    density.add_argument("--checkpoint", required=True)
    density.add_argument("--out-dir", default=str(ROOT / "results" / "samples"))
    density.add_argument("--logs", default=str(ROOT / "logs"))
    density.add_argument("--density", choices=["state", "pooled"], default="state")
    density.add_argument("--no-task-labels", action="store_true")
    density.add_argument("--guidance", type=float, default=2.0)
    density.add_argument("--null-axes", choices=["all", "tasks"], default="all")
    density.add_argument("--bandwidth", type=float, default=None)
    density.set_defaults(count=5000)

    subparsers.add_parser("compare", parents=[common])

    head = subparsers.add_parser("headtohead", parents=[common])
    head.add_argument("--pairs", nargs="+", required=True, help="<label>|<on file>|<off file>; {task} is replaced")
    head.add_argument("--output", default=str(ROOT / "results" / "tables" / "generator_flags.csv"))
    subparsers.add_parser("omegamphit", parents=[common])

    manifold = subparsers.add_parser("manifold", parents=[common])
    manifold.add_argument("--count", type=int, default=4000)
    manifold.add_argument("--neighbors", type=int, default=5)

    choose = subparsers.add_parser("choose")
    choose.add_argument("--evaluation", default=str(ROOT / "logs" / "generator_evaluation.csv"))
    choose.add_argument("--runs", nargs="+", required=True, help="<name>=<run directory>")
    choose.add_argument("--output", required=True)

    likelihood = subparsers.add_parser("likelihood", parents=[common])
    likelihood.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="<name>=<checkpoint>; the first one is the reference of the paired differences",
    )
    likelihood.add_argument("--arms", nargs="+", choices=list(LIKELIHOOD_ARMS), default=list(LIKELIHOOD_ARMS))
    likelihood.add_argument("--logs", default=str(ROOT / "logs"))
    likelihood.add_argument("--batch-size", type=int, default=128)
    likelihood.add_argument("--draws", type=int, default=2000)
    likelihood.add_argument("--random-orders", type=int, default=16, help="orders averaged for a random-order model")
    likelihood.add_argument("--bf16", action="store_true", help="bfloat16 autocast; faster, not bit-exact")
    likelihood.add_argument("--tag", default="")

    args = parser.parse_args()

    {
        "select": run_select, "sample": run_sample, "score": run_score, "property": run_property,
        "choose": run_choose, "compare": run_compare, "headtohead": run_headtohead, "omegamphit": run_omegamp_hit,
        "manifold": run_manifold, "likelihood": run_likelihood,
    }[args.command](args)


if __name__ == "__main__":
    main()