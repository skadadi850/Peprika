"""
Fits the predictors of Peprika from the corpus and writes them in the layout of the released weights.

oracles     boosted trees on ESM-2 (matched and all labels) and ESM C (matched labels) embeddings with composition and
            dipeptide fractions, one per benchmark task, saved as weights/oracles/<family>/<task>.ubj
multitask   five Multi-task networks on the 1,861-dimensional features with per-task stopping, saved with the weights of the
            epoch of each task, the standardization of the training rows and the 5% false-positive thresholds on validation
            inactives, in weights/multitask

The ESM-2 and ESM C embeddings of the corpus are read from data/corpus and computed there when a file is missing.

Usage:
    PYTHONUNBUFFERED=1 python scripts/train_predictors.py --corpus data/corpus --output weights
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ampx.evaluation import BENCHMARK_TASKS, combined_features, corpus_embeddings, fit_oracles, load_frame, task_selection  # noqa: E402
from ampx.models.data import TASKS  # noqa: E402
from ampx.predictor import ESMCEncoder, resolve_device  # noqa: E402
from milk import ACTIVITY_TASKS, FPR_QUANTILE, MIN_VALIDATION_INACTIVES  # noqa: E402
from multitask import (  # noqa: E402
    CLASSIFICATION_TASKS,
    MultiTaskNetwork,
    build_targets,
    class_weights,
    input_features,
    masked_loss,
    predict,
    standardize_inputs,
    stored_matrix,
    validation_columns,
    validation_score,
)


def snapshot(network) -> dict:
    return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}


def train_snapshots(features, target, observed, train, val, columns, args, device, seed):
    # train_network of multitask.py with per-task stopping, keeping the weights of every epoch whose scores it keeps
    torch.manual_seed(seed)

    network = MultiTaskNetwork(features.shape[1], target.shape[1], args.hidden, args.dropout, args.input_dropout).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = torch.as_tensor(class_weights(target, observed, train), device=device)
    target_tensor = torch.as_tensor(target, device=device)
    observed_tensor = torch.as_tensor(observed.astype(np.float32), device=device)

    best_value = -1.0
    best_state = None
    best_columns = np.full(len(columns), -1.0)
    column_states = {}
    stale = 0

    for epoch in range(args.epochs):
        network.train()
        order = np.random.default_rng(seed * 1000 + epoch).permutation(train)

        for start in range(0, len(order), args.batch_size):
            batch = torch.as_tensor(order[start : start + args.batch_size], device=device)
            loss = masked_loss(network(features[batch]), target_tensor[batch], observed_tensor[batch], weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scores = predict(network, features)
        value = validation_score(scores, target, observed, val, columns)
        improved = False

        if value > best_value:
            best_value, best_state, improved = value, snapshot(network), True

        current = validation_columns(scores, target, observed, val, columns)

        for position, column in enumerate(columns):
            if current[position] > best_columns[position]:
                best_columns[position] = current[position]
                column_states[column] = snapshot(network)
                improved = True

        if improved:
            stale = 0
        else:
            stale += 1

            if stale >= args.patience:
                break

    return best_state, column_states, best_value


def assemble(best_state, column_states, features, args, device) -> np.ndarray:
    network = MultiTaskNetwork(features.shape[1], len(TASKS), args.hidden, args.dropout, 0.0).to(device)
    network.load_state_dict(best_state)
    merged = predict(network, features)

    for column, state in column_states.items():
        network.load_state_dict(state)
        merged[:, column] = predict(network, features)[:, column]

    return merged


def esmc_matrix(corpus: Path, sequences: list, device: str) -> np.ndarray:
    path = corpus / "esmc_embeddings.npz"

    if path.exists():
        return stored_matrix(path, sequences)

    print(f"{path} is missing; embedding {len(sequences):,} sequences with ESM C", flush=True)
    embeddings = ESMCEncoder(device=device).encode(sequences, batch_size=128)

    if not np.isfinite(embeddings).all():
        raise SystemExit("non-finite values in the ESM C embeddings")

    np.savez(path, sequences=np.array(sequences), embeddings=embeddings)

    return embeddings


def fit_oracle_families(frame, esm2, esmc, output: Path, logs: Path) -> None:
    families = {
        "esm2_matched": (combined_features(frame["sequence"], esm2), True),
        "esm2_alllabels": (combined_features(frame["sequence"], esm2), False),
        "esmc_matched": (combined_features(frame["sequence"], esmc), True),
    }

    for name, (features, matched) in families.items():
        oracles, table = fit_oracles(frame, features, BENCHMARK_TASKS, matched=matched)
        directory = output / "oracles" / name
        directory.mkdir(parents=True, exist_ok=True)
        rounds = {}

        for task, oracle in oracles.items():
            oracle.booster.save_model(str(directory / f"{task}.ubj"))
            rounds[task] = int(oracle.rounds)

        (directory / "rounds.json").write_text(json.dumps(rounds))
        table.to_csv(logs / f"oracle_auroc_{name}.csv", index=False)
        print(f"oracles {name}: {rounds}", flush=True)


def fit_multitask(frame, corpus: Path, output: Path, args, device: str) -> None:
    raw = input_features(corpus, frame, "stabilized", "esm2")
    split = frame["split"].to_numpy()
    train = np.flatnonzero(split == "train")
    val = np.flatnonzero(split == "val")

    mean = raw[train].mean(axis=0)
    spread = raw[train].std(axis=0)
    spread[spread < 1e-6] = 1.0

    features = standardize_inputs(raw, train, args.clip, device)
    target, observed = build_targets(frame, True)
    columns = [TASKS.index(task) for task in CLASSIFICATION_TASKS]

    directory = output / "multitask"
    directory.mkdir(parents=True, exist_ok=True)
    averaged = np.zeros((len(frame), len(TASKS)), dtype=np.float64)

    for seed in range(args.seeds):
        best_state, column_states, value = train_snapshots(features, target, observed, train, val, columns, args, device, seed)
        averaged += assemble(best_state, column_states, features, args, device) / args.seeds
        torch.save({"macro": best_state, "columns": column_states}, directory / f"seed{seed}.pt")
        print(f"multitask seed {seed}: best validation macro AUROC {value:.4f}", flush=True)

    np.savez(directory / "standardization.npz", mean=mean, spread=spread)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "inputs": int(features.shape[1]), "outputs": len(TASKS), "hidden": args.hidden, "dropout": args.dropout,
                "seeds": args.seeds, "transform": "stabilized", "extra": "esm2", "clip": args.clip, "tasks": list(TASKS),
            }
        )
    )

    rows = []

    for task in ACTIVITY_TASKS:
        column = TASKS.index(task)
        keep, labels, task_split = task_selection(frame, task, True)
        inactive = keep[(task_split == "val") & (labels == 0)]
        active = keep[(task_split == "val") & (labels == 1)]
        threshold, rate = float("nan"), float("nan")

        if len(inactive) >= MIN_VALIDATION_INACTIVES:
            threshold = float(np.quantile(averaged[inactive, column], FPR_QUANTILE))

            if len(active):
                rate = float((averaged[active, column] > threshold).mean())

        rows.append({"task": task, "threshold": threshold, "n_inactive": int(len(inactive)), "n_active": int(len(active)), "validation_tpr": rate})

    thresholds = pd.DataFrame(rows)
    thresholds.to_csv(directory / "thresholds.csv", index=False)
    print("thresholds of the fitted networks", flush=True)
    print(thresholds.round(4).to_string(index=False), flush=True)

    shipped = ROOT / "results" / "milk" / "peptidomics_thresholds.csv"

    if shipped.exists():
        print("thresholds of the paper", flush=True)
        print(pd.read_csv(shipped).round(4).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the oracles and the Multi-task networks of Peprika.")
    parser.add_argument("--corpus", default=str(ROOT / "data" / "corpus"))
    parser.add_argument("--output", default=str(ROOT / "weights"))
    parser.add_argument("--logs", default=str(ROOT / "logs"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--input-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    corpus, output, logs = Path(args.corpus), Path(args.output), Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    frame = load_frame(corpus, ("sequence", "split", "component"))
    sequences = frame["sequence"].tolist()
    esm2 = corpus_embeddings(sequences, corpus / "esm_embeddings.npz", device)
    esmc = esmc_matrix(corpus, sequences, device)

    fit_oracle_families(frame, esm2, esmc, output, logs)
    fit_multitask(frame, corpus, output, args, device)


if __name__ == "__main__":
    main()
