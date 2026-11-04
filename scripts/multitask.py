"""
Multi-task tri-state predictor and the macro-average benchmark.

fit    One network over frozen ESM C embeddings, composition and dipeptide features, with one output per
       task. A label is observed only where the task state is active or inactive (and, with
       --labels matched, only inside the evidence-matched subset used by the benchmark); unknown labels
       contribute no loss. Each task's loss is class-balanced and the tasks are averaged with equal
       weight. Training stops on the validation macro AUROC over the five classification tasks. The
       scores of --seeds networks are averaged. Writes logs/multitask_scores_<mode>.csv in the format of
       benchmark_scores<tag>.csv.
macro  Macro mean test AUROC over the five classification tasks for every model in the score files,
       with a 95% interval from a bootstrap that resamples homology components jointly across tasks,
       the worst-task AUROC, and the paired difference between the reference model and every other
       model. Writes results/tables/macro_benchmark_<mode>.csv.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/multitask.py fit --labels matched
    PYTHONUNBUFFERED=1 uv run python scripts/multitask.py macro --mode subset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ampx.evaluation import (  # noqa: E402
    auroc,
    composition_matrix,
    dipeptide_matrix,
    load_frame,
    percentile_interval,
    split_indices,
    task_selection,
)
from ampx.models.data import TASKS  # noqa: E402
from ampx.predictor import resolve_device  # noqa: E402

CLASSIFICATION_TASKS = ("antibacterial", "antifungal", "anticancer", "antiinflammatory", "hemolytic")
MODE_NAMES = {"matched": "subset", "all": "all"}
MODEL_NAME = "multitask"


class MultiTaskNetwork(nn.Module):
    def __init__(
        self, inputs: int, outputs: int, hidden: int, dropout: float, input_dropout: float = 0.0
    ):
        super().__init__()

        layers = [nn.Dropout(input_dropout)] if input_dropout > 0 else []
        layers += [
            nn.Linear(inputs, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden // 2, outputs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(features))


def build_targets(frame: pd.DataFrame, matched: bool) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((len(frame), len(TASKS)), dtype=np.float32)
    observed = np.zeros((len(frame), len(TASKS)), dtype=bool)

    for column, task in enumerate(TASKS):
        keep, labels, _ = task_selection(frame, task, matched)
        observed[keep, column] = True
        target[keep, column] = labels

    return target, observed


def stored_matrix(path: Path, sequences: list) -> np.ndarray:
    stored = np.load(path, allow_pickle=False)

    if stored["sequences"].tolist() != sequences:
        raise SystemExit(f"{path.name} has a different row order from the corpus")

    return stored["embeddings"]


def input_features(corpus: Path, frame: pd.DataFrame, transform: str, extra: str) -> np.ndarray:
    sequences = frame["sequence"].tolist()
    composition = composition_matrix(frame["sequence"])
    dipeptide = dipeptide_matrix(frame["sequence"])

    if transform == "stabilized":
        composition[:, :-1] = np.sqrt(composition[:, :-1])
        composition[:, -1] = np.log(composition[:, -1])
        dipeptide = np.sqrt(dipeptide)

    blocks = [stored_matrix(corpus / "esmc_embeddings.npz", sequences), composition, dipeptide]

    if extra == "esm2":
        blocks.insert(1, stored_matrix(corpus / "esm_embeddings.npz", sequences))

    return np.hstack(blocks).astype(np.float32)


def standardize_inputs(raw: np.ndarray, train: np.ndarray, clip: float, device) -> torch.Tensor:
    mean = raw[train].mean(axis=0)
    spread = raw[train].std(axis=0)
    spread[spread < 1e-6] = 1.0

    scaled = (raw - mean) / spread

    if clip > 0:
        np.clip(scaled, -clip, clip, out=scaled)

    return torch.as_tensor(scaled.astype(np.float32), device=device)


def class_weights(target: np.ndarray, observed: np.ndarray, train: np.ndarray) -> np.ndarray:
    weights = np.ones((2, target.shape[1]), dtype=np.float32)

    for column in range(target.shape[1]):
        seen = observed[train, column]
        labels = target[train, column][seen]
        total = int(seen.sum())
        positive = int((labels == 1).sum())
        negative = total - positive

        if positive and negative:
            weights[0, column] = total / (2.0 * negative)
            weights[1, column] = total / (2.0 * positive)

    return weights


def masked_loss(logits, target, observed, weights) -> torch.Tensor:
    element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = torch.where(target > 0.5, weights[1], weights[0]) * observed
    per_task = (element * weight).sum(dim=0) / observed.sum(dim=0).clamp_min(1.0)

    return per_task.mean()


@torch.no_grad()
def predict(network: nn.Module, features: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
    network.eval()
    blocks = []

    for start in range(0, features.shape[0], batch_size):
        blocks.append(torch.sigmoid(network(features[start : start + batch_size])).cpu().numpy())

    return np.concatenate(blocks)


def validation_score(scores, target, observed, rows, columns) -> float:
    values = []

    for column in columns:
        chosen = rows[observed[rows, column]]
        labels = target[chosen, column].astype(np.int64)

        if len(np.unique(labels)) < 2:
            continue

        values.append(auroc(labels, scores[chosen, column]))

    return float(np.mean(values))


def validation_columns(scores, target, observed, rows, columns) -> np.ndarray:
    values = np.full(len(columns), np.nan)

    for position, column in enumerate(columns):
        chosen = rows[observed[rows, column]]
        labels = target[chosen, column].astype(np.int64)

        if len(np.unique(labels)) < 2:
            continue

        values[position] = auroc(labels, scores[chosen, column])

    return values


def train_network(features, target, observed, train, val, columns, args, device, seed):
    torch.manual_seed(seed)

    network = MultiTaskNetwork(
        features.shape[1], target.shape[1], args.hidden, args.dropout, getattr(args, "input_dropout", 0.0)
    ).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    weights = torch.as_tensor(class_weights(target, observed, train), device=device)
    target_tensor = torch.as_tensor(target, device=device)
    observed_tensor = torch.as_tensor(observed.astype(np.float32), device=device)

    per_task = getattr(args, "stopping", "macro") == "per_task"

    best_value = -1.0
    best_scores = None
    best_columns = np.full(len(columns), -1.0)
    column_scores = None
    stale = 0

    for epoch in range(args.epochs):
        network.train()
        order = np.random.default_rng(seed * 1000 + epoch).permutation(train)

        for start in range(0, len(order), args.batch_size):
            batch = torch.as_tensor(order[start : start + args.batch_size], device=device)
            loss = masked_loss(
                network(features[batch]), target_tensor[batch], observed_tensor[batch], weights
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scores = predict(network, features)
        value = validation_score(scores, target, observed, val, columns)

        if epoch % 10 == 0:
            print(f"  seed {seed} epoch {epoch}: validation macro AUROC {value:.4f}", flush=True)

        improved = False

        if value > best_value:
            best_value, best_scores, improved = value, scores, True

        if per_task:
            current = validation_columns(scores, target, observed, val, columns)

            if column_scores is None:
                column_scores = np.zeros_like(scores)

            for position, column in enumerate(columns):
                if current[position] > best_columns[position]:
                    best_columns[position] = current[position]
                    column_scores[:, column] = scores[:, column]
                    improved = True

        if improved:
            stale = 0
        else:
            stale += 1

            if stale >= args.patience:
                break

    if per_task:
        merged = best_scores.copy()
        valid = best_columns >= 0

        for position, column in enumerate(columns):
            if valid[position]:
                merged[:, column] = column_scores[:, column]

        return merged, float(best_columns[valid].mean())

    return best_scores, best_value


def run_fit(args) -> None:
    corpus = Path(args.corpus)
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    matched = args.labels == "matched"
    mode = MODE_NAMES[args.labels]

    frame = load_frame(corpus, ("sequence", "split", "component"))
    sequences = frame["sequence"].tolist()

    raw = input_features(corpus, frame, args.transform, args.extra)

    split = frame["split"].to_numpy()
    train = np.flatnonzero(split == "train")
    val = np.flatnonzero(split == "val")

    device = resolve_device(args.device)
    features = standardize_inputs(raw, train, args.clip, device)

    target, observed = build_targets(frame, matched)
    columns = [TASKS.index(task) for task in CLASSIFICATION_TASKS]

    ensemble = np.zeros((len(frame), len(TASKS)), dtype=np.float64)

    for seed in range(args.seeds):
        scores, value = train_network(
            features, target, observed, train, val, columns, args, device, seed
        )
        ensemble += scores / args.seeds
        print(f"seed {seed}: best validation macro AUROC {value:.4f}", flush=True)

    print(
        f"ensemble validation macro AUROC "
        f"{validation_score(ensemble, target, observed, val, columns):.4f}",
        flush=True,
    )

    saved = []

    for task in args.tasks:
        keep, labels, task_split = task_selection(frame, task, matched)
        parts = split_indices(task_split, labels)

        if parts is None:
            raise SystemExit(f"{task}: a split lacks one of the two classes")

        test = parts[2]
        rows = keep[test]

        saved.append(
            pd.DataFrame(
                {
                    "mode": mode,
                    "task": task,
                    "model": MODEL_NAME,
                    "component": frame["component"].to_numpy()[rows],
                    "target": labels[test],
                    "score": ensemble[rows, TASKS.index(task)],
                }
            )
        )

    output = logs / f"multitask_scores_{mode}{args.tag}.csv"
    pd.concat(saved, ignore_index=True).to_csv(output, index=False)
    print(f"wrote {output}", flush=True)


def auroc_with_weights(target: np.ndarray, scores: np.ndarray, weight: np.ndarray) -> float:
    positive = target == 1

    if not positive.any() or positive.all():
        return float("nan")

    unique, inverse = np.unique(scores, return_inverse=True)
    negative_mass = np.bincount(inverse, weights=weight * (~positive), minlength=len(unique))
    positive_mass = np.bincount(inverse, weights=weight * positive, minlength=len(unique))
    below = np.cumsum(negative_mass) - negative_mass

    numerator = float((positive_mass * (below + 0.5 * negative_mass)).sum())

    return numerator / float(positive_mass.sum() * negative_mass.sum())


def component_lookup(components: np.ndarray) -> dict:
    unique, inverse = np.unique(components, return_inverse=True)

    return {value: np.flatnonzero(inverse == index) for index, value in enumerate(unique)}


def draw_index(chosen, lookup: dict) -> np.ndarray:
    parts = [lookup[value] for value in chosen if value in lookup]

    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


CHUNK = 50


class ScoreRanking:
    def __init__(self, scores: np.ndarray, target: np.ndarray):
        self.order = np.argsort(scores, kind="stable")
        ordered = scores[self.order]
        self.starts = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
        self.positive = (target[self.order] == 1).astype(np.float64)

    def auroc(self, weights: np.ndarray) -> np.ndarray:
        ordered = weights[:, self.order]
        positive = ordered * self.positive
        negative = ordered - positive

        positive_group = np.add.reduceat(positive, self.starts, axis=1)
        negative_group = np.add.reduceat(negative, self.starts, axis=1)
        below = np.cumsum(negative_group, axis=1) - negative_group

        numerator = (positive_group * (below + 0.5 * negative_group)).sum(axis=1)
        denominator = positive_group.sum(axis=1) * negative_group.sum(axis=1)
        safe = np.where(denominator > 0, denominator, 1.0)

        return np.where(denominator > 0, numerator / safe, np.nan)


def joint_bootstrap(tasks, group, models, universe, draws, seed):
    # One resample of the components in `universe` is applied to every task of `group` and every model. A draw in
    # which a task of the group lacks a class is dropped for all of them.
    metrics = ("peptide", "family")
    generator = np.random.default_rng(seed)
    probabilities = np.full(len(universe), 1.0 / len(universe))
    collected = {(metric, name, task): [] for metric in metrics for name in models for task in group}
    checked = False
    done = 0

    while done < draws:
        size = min(CHUNK, draws - done)
        counts = generator.multinomial(len(universe), probabilities, size=size)
        values = {}
        valid = np.ones(size, dtype=bool)

        for task in group:
            data = tasks[task]
            rows = counts[:, data["index"]].astype(np.float64)
            weights = {"peptide": rows, "family": rows * data["family"][None, :]}

            for metric in metrics:
                for name in models:
                    result = data["ranking"][name].auroc(weights[metric])
                    values[(metric, name, task)] = result
                    valid &= ~np.isnan(result)

            if not checked:
                copies = rows[0].astype(np.int64)
                explicit = np.repeat(np.arange(len(copies)), copies)
                difference = 0.0

                for name in models:
                    scores = data["scores"][name][explicit]
                    labels = data["target"][explicit]

                    if len(np.unique(labels)) < 2:
                        continue

                    difference = max(
                        difference,
                        abs(auroc(labels, scores) - values[("peptide", name, task)][0]),
                        abs(
                            auroc_with_weights(labels, scores, data["family"][explicit])
                            - values[("family", name, task)][0]
                        ),
                    )

                print(f"bootstrap check {task}: maximum difference {difference:.2e}", flush=True)

                if difference > 1e-9:
                    raise SystemExit("the vectorized bootstrap disagrees with explicit resampling")

        checked = True

        for key, result in values.items():
            collected[key].append(result[valid])

        done += size

    return {key: np.concatenate(parts) for key, parts in collected.items()}


def run_blend(args) -> None:
    from scipy.stats import rankdata

    boosted = pd.read_csv(args.boosted)
    boosted = boosted[(boosted["mode"] == args.mode) & (boosted["model"] == args.partner)]
    network = pd.read_csv(args.network)
    network = network[network["mode"] == args.mode]
    parts = []

    for task in network["task"].unique():
        x = boosted[boosted["task"] == task].reset_index(drop=True)
        y = network[network["task"] == task].reset_index(drop=True)

        if not (np.array_equal(x["target"], y["target"]) and np.array_equal(x["component"], y["component"])):
            raise SystemExit(f"{task}: the two score files are on different rows")

        part = y.copy()
        part["model"] = "blend_network_esmc"
        part["score"] = 0.5 * (rankdata(x["score"]) + rankdata(y["score"])) / len(x)
        parts.append(part)

    pd.concat(parts).to_csv(args.output, index=False)
    print(f"wrote {args.output}", flush=True)


def run_macro(args) -> None:
    logs = Path(args.logs)
    paths = args.scores or [
        str(logs / "benchmark_scores_final.csv"),
        str(logs / f"multitask_scores_{args.mode}.csv"),
    ]
    macro_tasks = list(args.tasks)
    also = [task for task in args.also if task not in macro_tasks]
    every = macro_tasks + also
    width = len(macro_tasks)

    table = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    table = table[(table["mode"] == args.mode) & table["task"].isin(every)]
    models = sorted(table["model"].unique())

    if args.reference not in models:
        raise SystemExit(f"reference model {args.reference!r} not in {models}")

    loaded = {}
    folds = {}

    for task in every:
        part = table[table["task"] == task]
        by_model = {name: group.reset_index(drop=True) for name, group in part.groupby("model")}
        absent = [name for name in models if name not in by_model]

        if absent:
            raise SystemExit(f"{task}: no scores for {absent}")

        anchor = by_model[args.reference]

        for name, group in by_model.items():
            if (
                len(group) != len(anchor)
                or not np.array_equal(group["target"].to_numpy(), anchor["target"].to_numpy())
                or not np.array_equal(group["component"].to_numpy(), anchor["component"].to_numpy())
            ):
                raise SystemExit(f"{task}: {name} was scored on different test rows")

        loaded[task] = (
            anchor["target"].to_numpy(),
            anchor["component"].to_numpy(),
            {name: group["score"].to_numpy() for name, group in by_model.items()},
        )

        if "fold" in anchor.columns and anchor["fold"].notna().all():
            folds[task] = anchor["fold"].to_numpy()

    universe = np.unique(np.concatenate([loaded[task][1] for task in macro_tasks]))

    tasks = {}

    for task in every:
        target, components, scores = loaded[task]
        sizes = pd.Series(components).map(pd.Series(components).value_counts()).to_numpy()
        own = universe if task in macro_tasks else np.unique(components)

        tasks[task] = {
            "target": target,
            "index": np.searchsorted(own, components),
            "family": 1.0 / sizes.astype(np.float64),
            "scores": scores,
            "ranking": {name: ScoreRanking(values, target) for name, values in scores.items()},
        }

    metrics = ("peptide", "family")
    point = {metric: {} for metric in metrics}

    for name in models:
        point["peptide"][name] = np.array(
            [auroc(tasks[task]["target"], tasks[task]["scores"][name]) for task in every]
        )
        point["family"][name] = np.array(
            [
                auroc_with_weights(tasks[task]["target"], tasks[task]["scores"][name], tasks[task]["family"])
                for task in every
            ]
        )

    collected = joint_bootstrap(tasks, macro_tasks, models, universe, args.draws, args.seed)

    for number, task in enumerate(also):
        own = np.unique(loaded[task][1])
        collected.update(joint_bootstrap(tasks, [task], models, own, args.draws, args.seed + 1 + number))

    macro = {
        metric: {
            name: np.mean([collected[(metric, name, task)] for task in macro_tasks], axis=0)
            for name in models
        }
        for metric in metrics
    }

    rows = []
    detail = []

    for name in models:
        row = {"model": name}

        for metric in metrics:
            low, high = percentile_interval(macro[metric][name])
            row[f"{metric}_macro"] = float(point[metric][name][:width].mean())
            row[f"{metric}_lo"] = low
            row[f"{metric}_hi"] = high
            row[f"{metric}_worst"] = float(point[metric][name][:width].min())

            if name != args.reference:
                difference = macro[metric][args.reference] - macro[metric][name]
                gap_low, gap_high = percentile_interval(difference)
                row[f"{metric}_gap"] = float(
                    point[metric][args.reference][:width].mean() - point[metric][name][:width].mean()
                )
                row[f"{metric}_gap_lo"] = gap_low
                row[f"{metric}_gap_hi"] = gap_high
                row[f"{metric}_gap_excludes_zero"] = bool(gap_low > 0 or gap_high < 0)

            for position, task in enumerate(every):
                row[f"{metric}_{task}"] = float(point[metric][name][position])
                task_low, task_high = percentile_interval(collected[(metric, name, task)])
                detail.append(
                    {
                        "metric": metric,
                        "task": task,
                        "model": name,
                        "auroc": float(point[metric][name][position]),
                        "lo": task_low,
                        "hi": task_high,
                        "n_active": int(tasks[task]["target"].sum()),
                        "n_inactive": int((tasks[task]["target"] == 0).sum()),
                    }
                )

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("peptide_macro", ascending=False).set_index("model")

    directory = ROOT / "results" / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    result.to_csv(directory / f"macro_benchmark_{args.mode}{args.tag}.csv")
    pd.DataFrame(detail).to_csv(directory / f"macro_per_task_{args.mode}{args.tag}.csv", index=False)

    fold_rows = []

    for position, task in enumerate(every):
        if task not in folds:
            continue

        fold = folds[task]

        for name in models:
            per_fold = np.array(
                [
                    auroc(tasks[task]["target"][fold == index], tasks[task]["scores"][name][fold == index])
                    for index in np.unique(fold)
                ]
            )
            fold_rows.append(
                {
                    "task": task,
                    "model": name,
                    "pooled": float(point["peptide"][name][position]),
                    "fold_mean": float(np.nanmean(per_fold)),
                    "fold_sd": float(np.nanstd(per_fold, ddof=1)),
                    "fold_min": float(np.nanmin(per_fold)),
                    "fold_max": float(np.nanmax(per_fold)),
                }
            )

    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(directory / f"macro_folds_{args.mode}{args.tag}.csv", index=False)

    print(
        f"macro mean test AUROC over {width} tasks, mode {args.mode}, "
        f"{len(macro['peptide'][args.reference])} valid draws; reference {args.reference}; "
        f"scored per task outside the macro: {also}",
        flush=True,
    )
    print(result.round(3).to_string(), flush=True)

    if fold_rows:
        print(pd.DataFrame(fold_rows).round(3).to_string(index=False), flush=True)

    print(f"wrote {directory}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-task tri-state predictor and macro benchmark.")
    modes = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--logs", default=str(ROOT / "logs"))

    fit = modes.add_parser("fit", parents=[common])
    fit.add_argument("--corpus", default=str(ROOT / "data" / "processed" / "corpus"))
    fit.add_argument("--labels", choices=["matched", "all"], default="matched")
    fit.add_argument("--seeds", type=int, default=5)
    fit.add_argument("--tag", default="", help="suffix for the output file name")
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--patience", type=int, default=10)
    fit.add_argument("--batch-size", type=int, default=512)
    fit.add_argument("--hidden", type=int, default=512)
    fit.add_argument("--dropout", type=float, default=0.3)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--weight-decay", type=float, default=1e-2)
    fit.add_argument("--device", default=None)
    fit.add_argument("--transform", choices=["standard", "stabilized"], default="standard")
    fit.add_argument("--extra", choices=["none", "esm2"], default="none")
    fit.add_argument("--clip", type=float, default=0.0)
    fit.add_argument("--input-dropout", type=float, default=0.0)
    fit.add_argument("--stopping", choices=["macro", "per_task"], default="macro")
    fit.add_argument("--tasks", nargs="+", default=list(CLASSIFICATION_TASKS), help="tasks written to the score file")

    macro = modes.add_parser("macro", parents=[common])
    macro.add_argument("--mode", choices=["subset", "all"], default="subset")
    macro.add_argument("--scores", nargs="+", default=None)
    macro.add_argument("--reference", default=MODEL_NAME)
    macro.add_argument("--draws", type=int, default=2000)
    macro.add_argument("--tag", default="", help="suffix for the output file name")
    macro.add_argument("--seed", type=int, default=0)
    macro.add_argument("--tasks", nargs="+", default=list(CLASSIFICATION_TASKS), help="tasks of the macro mean")
    macro.add_argument("--also", nargs="*", default=[], help="tasks scored per task with their own bootstrap, outside the macro mean")

    blend = modes.add_parser("blend")
    blend.add_argument("--boosted", required=True)
    blend.add_argument("--network", required=True)
    blend.add_argument("--mode", choices=["subset", "all"], required=True)
    blend.add_argument("--output", required=True)
    blend.add_argument("--partner", default="esmc_combined")

    args = parser.parse_args()

    {"fit": run_fit, "macro": run_macro, "blend": run_blend}[args.command](args)


if __name__ == "__main__":
    main()